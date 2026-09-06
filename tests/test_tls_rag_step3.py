import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from tri_rag_harness.tls_rag_step2 import (
    Action,
    FixedScheduleController,
    FORBIDDEN_DECISION_FIELDS,
    assert_deployable_only,
    build_evidence_label_store,
    join_phase_b,
    run_phase_a as run_step2_phase_a,
    same_distance_different_angle_fixture,
)
from tri_rag_harness.tls_rag_step3 import (
    BASE_FEATURE_NAMES,
    PLAN_FACET_FEATURE_NAMES,
    PORTABLE_ARTIFACTS,
    TIMING_FIELDS,
    TRI_LAW_FEATURE_NAMES,
    WORK_FIELDS,
    CalibrationCell,
    CalibrationTable,
    CalibratedStep3Controller,
    LinearScoreModel,
    build_calibration_tables,
    build_observed_pair_risk_profile,
    build_step3_environment,
    build_synthetic_label_store,
    clopper_pearson_one_sided,
    feature_names_for_row,
    fit_score_models,
    join_evaluation_phase_b,
    load_step3_config,
    make_deployable_state,
    prepare_queries,
    reconstruct_synthetic_supervision,
    run_evaluation_phase_a,
    run_step3,
    stable_synthetic_partition,
    step2_semantic_trajectory_fingerprint,
)
from tri_rag_harness.tri_law import tri_law_probability


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "tls_rag_step3_synthetic_v1.json"
MODULE_PATH = ROOT / "src" / "tri_rag_harness" / "tls_rag_step3.py"


def _profile(vectors, projected, original, *, context=("a",), shell=("b",), previous=None):
    return build_observed_pair_risk_profile(
        query_embedding=np.zeros(2),
        exposed_candidate_ids=tuple(vectors),
        context_ids=context,
        shell_candidate_ids=shell,
        candidate_embeddings_by_id=vectors,
        projected_squared_distances_by_id=projected,
        original_squared_distances_by_id=original,
        m_prime=2,
        previous_profile=previous,
    )


class TlsRagStep3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_step3_config(CONFIG_PATH)
        cls.bundle = build_step3_environment(cls.config)
        cls.environment = cls.bundle.environment
        prepared_tuple = prepare_queries(cls.environment, cls.config)
        cls.prepared = {item.query_id: item for item in prepared_tuple}
        cls.fit_ids = cls.environment.partition("synthetic_model_fit")
        cls.bound_ids = cls.environment.partition("synthetic_bound_fit")
        cls.evaluation_ids = cls.environment.partition("synthetic_evaluation")
        fit_store = build_synthetic_label_store(cls.bundle, cls.prepared, cls.fit_ids)
        cls.fit_supervision = {
            query_id: reconstruct_synthetic_supervision(
                cls.prepared[query_id],
                {query.query_id: query for query in cls.environment.queries}[query_id],
                cls.environment,
                fit_store,
            )
            for query_id in cls.fit_ids
        }
        cls.models = fit_score_models(
            cls.config.candidates,
            cls.prepared,
            cls.fit_supervision,
            cls.fit_ids,
            cls.config,
        )
        bound_store = build_synthetic_label_store(
            cls.bundle, cls.prepared, cls.bound_ids
        )
        cls.bound_supervision = {
            query_id: reconstruct_synthetic_supervision(
                cls.prepared[query_id],
                {query.query_id: query for query in cls.environment.queries}[query_id],
                cls.environment,
                bound_store,
            )
            for query_id in cls.bound_ids
        }
        cls.tables, cls.bound_records = build_calibration_tables(
            candidates=cls.config.candidates,
            models=cls.models,
            prepared=cls.prepared,
            supervision=cls.bound_supervision,
            bound_query_ids=cls.bound_ids,
            config=cls.config,
        )
        cls.temp_one = tempfile.TemporaryDirectory()
        cls.temp_two = tempfile.TemporaryDirectory()
        cls.run_one = Path(cls.temp_one.name) / "run"
        cls.run_two = Path(cls.temp_two.name) / "run"
        run_step3(cls.config, cls.run_one)
        run_step3(cls.config, cls.run_two)

    @classmethod
    def tearDownClass(cls):
        cls.temp_one.cleanup()
        cls.temp_two.cleanup()

    def test_step2_frozen_schedule_and_fingerprints_are_unchanged(self):
        upstream = self.environment.upstream
        self.assertEqual(
            upstream.config.config_fingerprint,
            "d0506bbfe06c5a1bbb737cbfca8ccc2daacf7eef4a1ba7779df148e8713803af",
        )
        phase_a = run_step2_phase_a(upstream, FixedScheduleController(upstream.config))
        phase_b = join_phase_b(phase_a, upstream, build_evidence_label_store(upstream))
        self.assertEqual(
            phase_a.decision_fingerprint,
            "78c4e4869ffca61a7a82ab014b9a3bd9c1513824c6d7d83ad6c2180f4428c2f3",
        )
        self.assertEqual(
            phase_b.supervision_fingerprint,
            "a3d3620538c76bcc8a64b17c8dac619ac4b279be13abd43308758b43efda56e4",
        )
        self.assertEqual(
            step2_semantic_trajectory_fingerprint(
                phase_a, float_canonical_decimals=12
            ),
            self.config.section("upstream_step2")["phase_a_semantic_fingerprint"],
        )

    def test_step2_semantic_fingerprint_tolerates_only_frozen_float_lattice(self):
        upstream = self.environment.upstream
        phase_a = run_step2_phase_a(upstream, FixedScheduleController(upstream.config))
        baseline = step2_semantic_trajectory_fingerprint(
            phase_a, float_canonical_decimals=12
        )
        records = phase_a.portable_records()
        original = records[0]["decision_input"]["projected_squared_distances"][0]
        records[0]["decision_input"]["projected_squared_distances"][0] = original + 1e-14
        proxy = mock.Mock(
            config_fingerprint=phase_a.config_fingerprint,
            fixture_fingerprint=phase_a.fixture_fingerprint,
        )
        proxy.portable_records.return_value = records
        self.assertEqual(
            step2_semantic_trajectory_fingerprint(
                proxy, float_canonical_decimals=12
            ),
            baseline,
        )
        records[0]["decision_input"]["projected_squared_distances"][0] = original + 1e-6
        self.assertNotEqual(
            step2_semantic_trajectory_fingerprint(
                proxy, float_canonical_decimals=12
            ),
            baseline,
        )

    def test_hand_computed_beta_rho_and_unchanged_tri_law_value(self):
        vectors = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 2.0])}
        profile = _profile(vectors, {"a": 0.8, "b": 5.0}, {"a": 1.0, "b": 4.0})
        pair = profile.context_boundary.pairs[0]
        self.assertEqual(pair.status, "valid")
        self.assertEqual(pair.beta, 4.0)
        self.assertEqual(pair.rho, 0.0)
        self.assertEqual(pair.probability, tri_law_probability(4.0, 0.0, 2))
        self.assertAlmostEqual(pair.probability, 0.2)

    def test_pair_builder_is_current_prefix_only_and_refuses_unseen_candidates(self):
        vectors = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 2.0])}
        with self.assertRaisesRegex(ValueError, "every and only exposed"):
            _profile(
                {**vectors, "unseen": np.array([3.0, 0.0])},
                {"a": 1.0, "b": 4.0},
                {"a": 1.0, "b": 4.0},
            )
        with self.assertRaisesRegex(ValueError, "context contains an unseen"):
            _profile(vectors, {"a": 1.0, "b": 4.0}, {"a": 1.0, "b": 4.0}, context=("unseen",))

    def test_boundary_frontier_summaries_quantiles_and_previous_deltas(self):
        vectors = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 2.0]),
            "c": np.array([3.0, 1.0]),
        }
        projected = {"a": 0.5, "b": 3.0, "c": 12.0}
        original = {"a": 1.0, "b": 4.0, "c": 10.0}
        first = _profile(vectors, projected, original, shell=("b", "c"))
        probabilities = [pair.probability for pair in first.context_boundary.pairs]
        self.assertEqual(first.context_boundary.valid_pair_count, 2)
        self.assertAlmostEqual(first.context_boundary.probability_mean, np.mean(probabilities))
        self.assertAlmostEqual(first.context_boundary.probability_q90, np.quantile(probabilities, 0.9))
        self.assertTrue(first.previous_deltas.is_first_state)
        self.assertTrue(all(value == 0.0 for _, value in first.previous_deltas.values))
        self.assertTrue(all(not value for _, value in first.previous_deltas.validity))
        second = _profile(vectors, projected, original, shell=("b", "c"), previous=first)
        self.assertFalse(second.previous_deltas.is_first_state)
        self.assertTrue(all(value for _, value in second.previous_deltas.validity))
        self.assertTrue(all(abs(value) < 1e-15 for _, value in second.previous_deltas.values))

    def test_tied_duplicate_zero_nonfinite_collinear_and_empty_counts(self):
        vectors = {
            "a": np.array([1.0, 0.0]),
            "tie": np.array([0.0, 1.0]),
            "zero": np.array([0.0, 0.0]),
            "bad": np.array([np.inf, 0.0]),
            "line": np.array([2.0, 0.0]),
        }
        projected = {"a": 1.0, "tie": 1.0, "zero": 0.0, "bad": 2.0, "line": 4.0}
        original = {"a": 1.0, "tie": 1.0, "zero": 0.0, "bad": np.inf, "line": 4.0}
        profile = _profile(vectors, projected, original, shell=("tie", "zero", "bad", "line"))
        summary = profile.context_boundary
        self.assertEqual(summary.tied_pair_count, 1)
        self.assertEqual(summary.zero_pair_count, 1)
        self.assertEqual(summary.nonfinite_pair_count, 1)
        self.assertEqual(summary.collinear_pair_count, 1)
        self.assertEqual(next(pair for pair in summary.pairs if pair.collinear).probability, 0.0)
        self.assertGreater(profile.duplicate_projected_distance_pairs, 0)
        self.assertFalse(profile.required_profile_valid)
        empty = build_observed_pair_risk_profile(
            query_embedding=np.zeros(2),
            exposed_candidate_ids=(),
            context_ids=(),
            shell_candidate_ids=(),
            candidate_embeddings_by_id={},
            projected_squared_distances_by_id={},
            original_squared_distances_by_id={},
            m_prime=2,
        )
        self.assertEqual(empty.context_boundary.attempted_pair_count, 0)
        self.assertFalse(empty.context_boundary.summary_valid)

    def test_same_distances_different_angles_change_profile(self):
        fixture = same_distance_different_angle_fixture()
        query = fixture["query"]
        near = fixture["near"] - query
        far_one = fixture["far_same_plane"] - query
        far_two = fixture["far_orthogonal_plane"] - query
        beta_one = np.dot(far_one, far_one) / np.dot(near, near)
        beta_two = np.dot(far_two, far_two) / np.dot(near, near)
        rho_one = np.dot(near, far_one) / np.sqrt(np.dot(near, near) * np.dot(far_one, far_one))
        rho_two = np.dot(near, far_two) / np.sqrt(np.dot(near, near) * np.dot(far_two, far_two))
        self.assertAlmostEqual(beta_one, beta_two, places=14)
        self.assertNotAlmostEqual(rho_one, rho_two, places=8)
        self.assertNotAlmostEqual(
            tri_law_probability(beta_one, rho_one, 2),
            tri_law_probability(beta_two, rho_two, 2),
            places=8,
        )

    def test_realized_projection_changes_distortion_not_ex_ante_pair_probability(self):
        vectors = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 2.0])}
        first = _profile(vectors, {"a": 0.2, "b": 8.0}, {"a": 1.0, "b": 4.0})
        second = _profile(vectors, {"a": 5.0, "b": 0.1}, {"a": 1.0, "b": 4.0})
        self.assertEqual(
            first.context_boundary.pairs[0].probability,
            second.context_boundary.pairs[0].probability,
        )
        self.assertNotEqual(first.core_log_distortion.q50, second.core_log_distortion.q50)

    def test_rows_two_to_four_have_exact_one_factor_feature_differences(self):
        self.assertEqual(feature_names_for_row(2), BASE_FEATURE_NAMES)
        self.assertEqual(feature_names_for_row(3), BASE_FEATURE_NAMES + TRI_LAW_FEATURE_NAMES)
        self.assertEqual(
            feature_names_for_row(4),
            BASE_FEATURE_NAMES + TRI_LAW_FEATURE_NAMES + PLAN_FACET_FEATURE_NAMES,
        )
        registry = json.loads((self.run_one / "feature_registry.json").read_text())
        self.assertEqual(registry["row3_minus_row2"], list(TRI_LAW_FEATURE_NAMES))
        self.assertEqual(registry["row4_minus_row3"], list(PLAN_FACET_FEATURE_NAMES))

    def test_synthetic_partitions_are_stable_disjoint_and_not_real_roles(self):
        partitions = [set(self.environment.partition(name)) for name in (
            "synthetic_model_fit", "synthetic_bound_fit", "synthetic_evaluation"
        )]
        self.assertTrue(all(len(values) == 24 for values in partitions))
        self.assertTrue(all(not partitions[i].intersection(partitions[j]) for i in range(3) for j in range(i + 1, 3)))
        for name in ("synthetic_model_fit", "synthetic_bound_fit", "synthetic_evaluation"):
            self.assertTrue(all(stable_synthetic_partition(query_id, self.config) == name for query_id in self.environment.partition(name)))
        artifact = json.loads((self.run_one / "synthetic_partitions.json").read_text())
        self.assertTrue(artifact["pairwise_disjoint"])
        self.assertFalse(artifact["real_role_names_used"])

    def test_model_fit_labels_do_not_reach_bound_or_inference_objects(self):
        for pair in self.models:
            for model in pair:
                self.assertTrue(set(model.fit_query_ids).issubset(self.fit_ids))
                self.assertGreaterEqual(model.excluded_invalid_row_count, 9)
                self.assertEqual(
                    model.fit_row_count + model.excluded_invalid_row_count,
                    len(self.fit_ids) * len(self.environment.upstream.config.budget_grid),
                )
                self.assertFalse(set(model.fit_query_ids).intersection(self.bound_ids))
                self.assertFalse(set(model.fit_query_ids).intersection(self.evaluation_ids))
        candidate = self.config.candidates[0]
        controller = CalibratedStep3Controller(
            candidate=candidate,
            gain_model=self.models[0][0],
            sufficiency_model=self.models[0][1],
            calibration_table=self.tables[0],
            budget_grid=self.environment.upstream.config.budget_grid,
            maximum_expansions=2,
            delta_gain=0.65,
            tau_sufficient=0.15,
        )
        self.assertFalse(any("store" in key or "label" in key for key in controller.__dict__))
        phase_a_text = (self.run_one / "phase_a_decisions.jsonl").read_text()
        self.assertNotIn("target_remaining_event", phase_a_text)
        self.assertNotIn("observed_remaining_event", phase_a_text)

    def test_clopper_pearson_hand_values_and_family_wise_alpha_allocation(self):
        alpha = 0.05
        lower, upper = clopper_pearson_one_sided(0, 10, alpha)
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(upper, 1.0 - alpha ** (1.0 / 10.0))
        lower, upper = clopper_pearson_one_sided(10, 10, alpha)
        self.assertAlmostEqual(lower, alpha ** (1.0 / 10.0))
        self.assertEqual(upper, 1.0)
        all_cells = [cell for table in self.tables for cell in table.cells]
        expected_cells = len(self.config.candidates) * 3 * 2 * 2
        self.assertEqual(len(all_cells), expected_cells)
        self.assertAlmostEqual(
            sum(cell.alpha_share for cell in all_cells),
            self.config.section("calibration")["family_wise_alpha"],
        )

    def test_candidate_stage_bin_reachability_uses_unique_query_ids_and_reconstructs(self):
        saw_strict_reduction = False
        for table in self.tables:
            previous = set(self.bound_ids)
            records = [row for row in self.bound_records if row["candidate_id"] == table.candidate_id]
            for stage, ids in table.reachable_query_ids_by_stage:
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(set(ids).issubset(previous))
                if stage > 0:
                    reconstructed = {
                        row["query_id"] for row in records
                        if row["stage"] == stage - 1 and row["action"] == Action.EXPAND_TO_NEXT_GRID_VALUE.value
                    }
                    self.assertEqual(set(ids), reconstructed)
                    saw_strict_reduction |= len(ids) < len(previous)
                previous = set(ids)
            for cell in table.cells:
                self.assertEqual(len(cell.query_ids), len(set(cell.query_ids)))
        self.assertTrue(saw_strict_reduction)

    def test_empty_or_underpowered_cells_are_vacuous_and_prohibit_stop(self):
        vacuous = [cell for table in self.tables for cell in table.cells if not cell.valid]
        self.assertTrue(vacuous)
        self.assertTrue(all((cell.lower_limit, cell.upper_limit) == (0.0, 1.0) for cell in vacuous))
        candidate = self.config.candidates[0]
        state = make_deployable_state(self.prepared[self.bound_ids[0]].stages[0], candidate)
        model = self._constant_model(candidate, state, 0.5)
        table = self._manual_table(candidate, gain_upper=1.0, suff_lower=0.0, valid=False)
        decision = self._manual_controller(candidate, model, table).choose(state)
        self.assertIs(decision.action, Action.EXPAND_TO_NEXT_GRID_VALUE)
        self.assertFalse(decision.remaining_cell_valid)

    def test_dual_bounds_and_every_validity_flag_are_required_for_stop(self):
        candidate = self.config.candidates[0]
        state = make_deployable_state(self.prepared[self.bound_ids[3]].stages[0], candidate)
        self.assertTrue(state.base_state_valid and state.evidence_plan_valid)
        model = self._constant_model(candidate, state, 0.5)
        passing = self._manual_table(candidate, gain_upper=0.1, suff_lower=0.9, valid=True)
        self.assertIs(self._manual_controller(candidate, model, passing).choose(state).action, Action.STOP)
        gain_fail = self._manual_table(candidate, gain_upper=0.8, suff_lower=0.9, valid=True)
        self.assertIs(self._manual_controller(candidate, model, gain_fail).choose(state).action, Action.EXPAND_TO_NEXT_GRID_VALUE)
        suff_fail = self._manual_table(candidate, gain_upper=0.1, suff_lower=0.1, valid=True)
        self.assertIs(self._manual_controller(candidate, model, suff_fail).choose(state).action, Action.EXPAND_TO_NEXT_GRID_VALUE)
        invalid_state = replace(state, base_state_valid=False)
        self.assertIs(self._manual_controller(candidate, model, passing).choose(invalid_state).action, Action.EXPAND_TO_NEXT_GRID_VALUE)

    def test_all_invalid_states_expand_next_only_and_terminate_explicitly(self):
        invalid_query = self.evaluation_ids[1]
        candidate = self.config.candidates[2]
        gain_model, suff_model = self.models[2]
        controller = CalibratedStep3Controller(
            candidate=candidate,
            gain_model=gain_model,
            sufficiency_model=suff_model,
            calibration_table=self.tables[2],
            budget_grid=self.environment.upstream.config.budget_grid,
            maximum_expansions=2,
            delta_gain=0.65,
            tau_sufficient=0.15,
        )
        states = [make_deployable_state(stage, candidate) for stage in self.prepared[invalid_query].stages]
        first = controller.choose(states[0])
        self.assertIs(first.action, Action.EXPAND_TO_NEXT_GRID_VALUE)
        self.assertEqual(first.next_budget, self.environment.upstream.config.budget_grid[1])
        terminal = controller.choose(states[-1])
        self.assertIs(terminal.action, Action.STOP)
        self.assertIn(terminal.reason, {"invalid_evidence_plan", "invalid_state_at_corpus_exhaustion"})

    def test_recursive_forbidden_fields_and_no_label_store_reachability(self):
        state = make_deployable_state(
            self.prepared[self.evaluation_ids[3]].stages[0], self.config.candidates[0]
        )
        assert_deployable_only(state)
        serialized = json.dumps(state.to_dict(), sort_keys=True)
        for forbidden in FORBIDDEN_DECISION_FIELDS:
            self.assertNotIn(f'"{forbidden}"', serialized)
        for forbidden in FORBIDDEN_DECISION_FIELDS:
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(Exception):
                    assert_deployable_only({"nested": [{forbidden.upper(): True}]})

    def test_phase_a_serializes_before_evaluation_join_and_fingerprint_survives(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "run"
            real_builder = build_synthetic_label_store

            def guarded(bundle, prepared, allowed_ids):
                if set(allowed_ids) == set(self.evaluation_ids):
                    self.assertTrue((output / "phase_a_decisions.jsonl").is_file())
                    self.assertGreater((output / "phase_a_decisions.jsonl").stat().st_size, 0)
                return real_builder(bundle, prepared, allowed_ids)

            with mock.patch(
                "tri_rag_harness.tls_rag_step3.build_synthetic_label_store",
                side_effect=guarded,
            ):
                run_step3(self.config, output)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["phase_a_serialized_before_evaluation_label_store_opened"])
            self.assertTrue(manifest["phase_a_fingerprint_unchanged_by_join"])

    def test_complete_separate_work_timing_and_aggregate_reconstruction(self):
        work = json.loads((self.run_one / "work_counters.json").read_text())
        self.assertTrue(work["records"])
        self.assertTrue(all(set(row) - {"method_id", "query_id", "stage"} == set(WORK_FIELDS) for row in work["records"]))
        for name in WORK_FIELDS:
            self.assertEqual(work["totals"][name], sum(row[name] for row in work["records"]))
        timings = json.loads((self.run_one / "timings.json").read_text())
        self.assertFalse(timings["portable"])
        for row in timings["query_stage_records"]:
            self.assertEqual(set(row) - {"method_id", "query_id", "stage"}, set(TIMING_FIELDS))
            self.assertTrue(all(row[name] >= 0.0 for name in TIMING_FIELDS))
        decisions = [json.loads(line) for line in (self.run_one / "phase_a_decisions.jsonl").read_text().splitlines()]
        aggregates = json.loads((self.run_one / "aggregates.json").read_text())
        for method_id, values in aggregates["methods"].items():
            subset = [row for row in decisions if row["method_id"] == method_id]
            self.assertEqual(values["stage_record_count"], len(subset))
            self.assertEqual(values["expand_count"], sum(row["action"] == Action.EXPAND_TO_NEXT_GRID_VALUE.value for row in subset))

    def test_two_runs_have_byte_identical_portable_artifacts_and_denied_claims(self):
        for name in PORTABLE_ARTIFACTS:
            with self.subTest(name=name):
                self.assertEqual((self.run_one / name).read_bytes(), (self.run_two / name).read_bytes())
        self.assertNotIn("timings.json", PORTABLE_ARTIFACTS)
        manifest = json.loads((self.run_one / "manifest.json").read_text())
        self.assertEqual(
            set(manifest["claims_denied"]),
            {"real_data", "selection", "certification", "latency", "posterior_probability", "evidence_guarantee", "answer_quality"},
        )
        report = (self.run_one / "report.md").read_text().casefold()
        for phrase in ("no real", "no method was selected", "not a latency claim", "not posterior", "do not guarantee", "no approximate index", "step 4"):
            self.assertIn(phrase, report)

    def test_source_and_manifest_prohibit_network_real_approximate_gpu_llm_answer_and_step4(self):
        source = MODULE_PATH.read_text(encoding="utf-8").casefold()
        for forbidden_import in ("import requests", "import urllib", "import socket", "import faiss", "import openai"):
            self.assertNotIn(forbidden_import, source)
        manifest = json.loads((self.run_one / "manifest.json").read_text())
        scope = manifest["prohibitions_observed"]
        self.assertTrue(all(scope.values()))
        self.assertEqual(manifest["scope"], "cpu_network_free_pure_synthetic_step3_diagnostics")

    def _constant_model(self, candidate, state, score):
        names = tuple(name for name, _ in state.feature_values)
        payload = {
            "schema_version": "tls_rag_transparent_ridge_score_v1",
            "outcome": "fixture",
            "candidate_id": candidate.candidate_id,
            "feature_names": list(names),
        }
        return LinearScoreModel(
            schema_version=payload["schema_version"],
            outcome="fixture",
            candidate_id=candidate.candidate_id,
            feature_names=names,
            means=tuple(0.0 for _ in names),
            scales=tuple(1.0 for _ in names),
            coefficients=tuple(0.0 for _ in names),
            intercept=score,
            regularization=0.01,
            fit_query_ids=("synthetic-model-fit-fixture",),
            fit_row_count=1,
            excluded_invalid_row_count=0,
            model_fingerprint="manual-test-model",
        )

    def _manual_table(self, candidate, *, gain_upper, suff_lower, valid):
        cells = (
            CalibrationCell(candidate.candidate_id, 0, "remaining_useful_evidence_event", 0, 0.0, 1.0, ("q",), 0, 4, 0.0, gain_upper, valid, 0.01),
            CalibrationCell(candidate.candidate_id, 0, "current_context_sufficiency_event", 0, 0.0, 1.0, ("q",), 4, 4, suff_lower, 1.0, valid, 0.01),
        )
        return CalibrationTable(
            "tls_rag_reachable_clopper_pearson_v1",
            candidate.candidate_id,
            ((0, "remaining_useful_evidence_event", (0.0, 1.0)), (0, "current_context_sufficiency_event", (0.0, 1.0))),
            cells,
            ((0, ("q",)),),
            "manual-test-table",
        )

    def _manual_controller(self, candidate, model, table):
        return CalibratedStep3Controller(
            candidate=candidate,
            gain_model=model,
            sufficiency_model=model,
            calibration_table=table,
            budget_grid=self.environment.upstream.config.budget_grid,
            maximum_expansions=2,
            delta_gain=0.65,
            tau_sufficient=0.15,
        )


if __name__ == "__main__":
    unittest.main()
