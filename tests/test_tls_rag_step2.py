import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

import numpy as np

from tri_rag_harness.indexes import ExactSquaredL2Index
from tri_rag_harness.projection import dense_gaussian_projection
from tri_rag_harness.tls_rag_step2 import (
    FORBIDDEN_DECISION_FIELDS,
    PORTABLE_ARTIFACTS,
    TIMING_FIELDS,
    WORK_FIELDS,
    Action,
    EvidenceLabelStore,
    EvidencePlan,
    FacetSlot,
    FixedScheduleController,
    ForbiddenDecisionFieldError,
    PassageEvidence,
    _evidence_view,
    assert_deployable_only,
    build_evidence_label_store,
    build_step2_environment,
    join_phase_b,
    load_step2_config,
    run_phase_a,
    run_step2,
    same_distance_different_angle_fixture,
)
from tri_rag_harness.tri_law import tri_law_probability


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "tls_rag_step2_synthetic_v1.json"
MODULE_PATH = ROOT / "src" / "tri_rag_harness" / "tls_rag_step2.py"


def _records_by_query(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["query_id"], []).append(record)
    return grouped


class TlsRagStep2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_step2_config(CONFIG_PATH)
        cls.environment = build_step2_environment(cls.config)
        cls.controller = FixedScheduleController(cls.config)
        cls.phase_a = run_phase_a(cls.environment, cls.controller)
        cls.label_store = build_evidence_label_store(cls.environment)
        cls.phase_b = join_phase_b(cls.phase_a, cls.environment, cls.label_store)
        cls.phase_b_by_query = _records_by_query(cls.phase_b.supervision_records)

    def test_external_ids_normalization_projection_scale_and_no_renormalization(self):
        query_ids = {query.query_id for query in self.environment.queries}
        self.assertFalse(query_ids.intersection(self.environment.corpus_ids))
        np.testing.assert_allclose(
            np.linalg.norm(self.environment.corpus_embeddings, axis=1),
            1.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            [np.linalg.norm(query.embedding) for query in self.environment.queries],
            1.0,
            atol=1e-12,
        )

        statistical_matrix = dense_gaussian_projection(64, 4096, seed=814)
        self.assertAlmostEqual(
            float(np.std(statistical_matrix)), 1.0 / np.sqrt(64), delta=0.0015
        )
        expected = self.environment.corpus_embeddings @ self.environment.projection_matrix.T
        np.testing.assert_array_equal(self.environment.projected_corpus, expected)
        projected_norms = np.linalg.norm(self.environment.projected_corpus, axis=1)
        self.assertGreater(float(np.max(np.abs(projected_norms - 1.0))), 0.1)

    def test_squared_l2_stable_string_ties_in_original_and_projected_spaces(self):
        ids = self.environment.corpus_ids
        duplicate_ids = ("tls-passage-00", "tls-passage-01")
        for trace, query in zip(self.phase_a.queries, self.environment.queries):
            projected_positions = {
                passage_id: trace.full_projected_ranking_ids.index(passage_id)
                for passage_id in duplicate_ids
            }
            self.assertLess(projected_positions[duplicate_ids[0]], projected_positions[duplicate_ids[1]])
            self.assertEqual(
                trace.full_projected_squared_distances[projected_positions[duplicate_ids[0]]],
                trace.full_projected_squared_distances[projected_positions[duplicate_ids[1]]],
            )

            distances = np.einsum(
                "ij,ij->i",
                self.environment.corpus_embeddings - query.embedding,
                self.environment.corpus_embeddings - query.embedding,
            )
            order = np.lexsort((np.asarray(ids, dtype=str), distances))
            ranked_ids = [ids[int(row)] for row in order]
            self.assertLess(ranked_ids.index(duplicate_ids[0]), ranked_ids.index(duplicate_ids[1]))
            self.assertEqual(distances[0], distances[1])

        zero_trace = next(
            trace for trace in self.phase_a.queries if trace.query_id == "tls-query-zero-distance"
        )
        self.assertGreaterEqual(
            zero_trace.records[0].decision_input.validity.zero_original_distance_count, 1
        )

    def test_one_projected_scan_prefix_reuse_cached_original_rerank_and_context(self):
        corpus_rows = {
            passage_id: row for row, passage_id in enumerate(self.environment.corpus_ids)
        }
        query_by_id = {query.query_id: query for query in self.environment.queries}
        for trace in self.phase_a.queries:
            self.assertEqual(trace.projected_scan_count, 1)
            self.assertTrue(all(count == 1 for _, count in trace.original_evaluation_counts))
            self.assertEqual(
                trace.evaluated_original_ids,
                trace.full_projected_ranking_ids[
                    : trace.records[-1].decision_input.current_budget
                ],
            )
            previous_ids = ()
            query = query_by_id[trace.query_id]
            for record in trace.records:
                state = record.decision_input
                budget = self.config.budget_grid[record.stage]
                self.assertEqual(state.exposed_candidate_ids, trace.full_projected_ranking_ids[:budget])
                self.assertEqual(state.exposed_candidate_ids[: len(previous_ids)], previous_ids)
                rows = [corpus_rows[passage_id] for passage_id in state.exposed_candidate_ids]
                distances = np.einsum(
                    "ij,ij->i",
                    self.environment.corpus_embeddings[rows] - query.embedding,
                    self.environment.corpus_embeddings[rows] - query.embedding,
                )
                order = np.lexsort((np.asarray(state.exposed_candidate_ids), distances))
                expected_ids = tuple(state.exposed_candidate_ids[int(index)] for index in order)
                self.assertEqual(state.exact_reranked_ids, expected_ids)
                np.testing.assert_allclose(
                    state.exact_reranked_squared_distances, distances[order], atol=1e-14
                )
                self.assertEqual(state.context_ids, expected_ids[: self.config.k_ctx])
                previous_ids = state.exposed_candidate_ids

        search_calls = []
        real_search = ExactSquaredL2Index.search

        def counted_search(index, queries, k):
            search_calls.append((np.asarray(queries).shape, k))
            return real_search(index, queries, k)

        with mock.patch.object(
            ExactSquaredL2Index, "search", autospec=True, side_effect=counted_search
        ):
            run_phase_a(self.environment, self.controller)
        self.assertEqual(len(search_calls), len(self.environment.queries))
        self.assertTrue(all(k == self.config.corpus_size for _, k in search_calls))

    def test_fixed_controller_has_exactly_two_actions_and_never_skips_grid(self):
        self.assertEqual(
            {action.value for action in Action}, {"STOP", "EXPAND_TO_NEXT_GRID_VALUE"}
        )
        seen_stages = set()
        for trace in self.phase_a.queries:
            for record in trace.records:
                seen_stages.add(record.stage)
                if record.action is Action.EXPAND_TO_NEXT_GRID_VALUE:
                    self.assertEqual(
                        record.next_budget, self.config.budget_grid[record.stage + 1]
                    )
                else:
                    self.assertIsNone(record.next_budget)
            self.assertIs(trace.records[-1].action, Action.STOP)
        self.assertEqual(seen_stages, {0, 1, 2})
        pilot = next(
            trace for trace in self.phase_a.queries if trace.query_id == "tls-query-pilot-stop"
        )
        self.assertEqual(len(pilot.records), 1)
        candidate = next(
            trace
            for trace in self.phase_a.queries
            if trace.query_id == "tls-query-candidate-without-context"
        )
        self.assertEqual(len(candidate.records), 2)

    def test_controller_schema_and_recursive_forbidden_field_rejection(self):
        first_state = self.phase_a.queries[0].records[0].decision_input
        self.assertEqual(
            set(self.controller.__dict__), {"_grid", "_maximum_expansions", "_schedule"}
        )
        self.assertFalse(
            any(
                isinstance(value, EvidenceLabelStore)
                for value in self.controller.__dict__.values()
            )
        )
        with self.assertRaises(TypeError):
            self.controller.choose(first_state.to_dict())
        for forbidden in FORBIDDEN_DECISION_FIELDS:
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ForbiddenDecisionFieldError):
                    assert_deployable_only({"allowed": [{forbidden.upper(): 1}]})
        assert_deployable_only(first_state)
        serialized = json.dumps(first_state.to_dict(), sort_keys=True)
        for forbidden in FORBIDDEN_DECISION_FIELDS:
            self.assertNotIn(f'"{forbidden}"', serialized)

    def test_phase_a_is_immutable_closed_before_labels_and_join_preserves_fingerprint(self):
        before = self.phase_a.decision_fingerprint
        self.assertEqual(before, self.phase_a.recompute_decision_fingerprint())
        self.assertEqual(self.phase_b.decision_fingerprint_before_join, before)
        self.assertEqual(self.phase_b.decision_fingerprint_after_join, before)
        with self.assertRaises(FrozenInstanceError):
            self.phase_a.decision_fingerprint = "changed"
        with self.assertRaises(ValueError):
            self.environment.corpus_embeddings[0, 0] = 9.0

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "run"
            real_builder = build_evidence_label_store

            def guarded_builder(environment):
                self.assertTrue((output / "phase_a_decisions.jsonl").is_file())
                self.assertGreater((output / "phase_a_decisions.jsonl").stat().st_size, 0)
                return real_builder(environment)

            with mock.patch(
                "tri_rag_harness.tls_rag_step2.build_evidence_label_store",
                side_effect=guarded_builder,
            ):
                run_step2(self.config, output)

    def test_hand_computed_evidence_coverage_sufficiency_and_contradiction(self):
        plan = EvidencePlan(
            generator="tls_rag_test_plan_v1",
            slots=(
                FacetSlot("facet-a", 2, True, "synthetic"),
                FacetSlot("facet-b", 1, False, "synthetic"),
            ),
            block_contradictions=True,
        )
        store = EvidenceLabelStore(
            "tls_rag_evidence_label_store_v1",
            (
                PassageEvidence("p1", ("facet-a",), "source-1", (), False),
                PassageEvidence("p2", ("facet-a",), "source-2", (), False),
                PassageEvidence("p3", ("facet-b",), "source-3", (), False),
                PassageEvidence("p4", (), "source-4", ("facet-a",), False),
                PassageEvidence("p5", ("facet-b",), "source-5", (), True),
            ),
        )
        partial = _evidence_view(("p1", "p3"), plan, store)
        self.assertEqual(partial["covered_slots"], ["facet-b"])
        self.assertEqual(partial["coverage"], 0.5)
        self.assertFalse(partial["sufficient"])
        complete = _evidence_view(("p1", "p2", "p3"), plan, store)
        self.assertEqual(complete["coverage"], 1.0)
        self.assertTrue(complete["sufficient"])
        contradicted = _evidence_view(("p1", "p2", "p3", "p4"), plan, store)
        self.assertTrue(contradicted["blocking_contradiction"])
        self.assertFalse(contradicted["sufficient"])
        invalid_only = _evidence_view(("p5",), plan, store)
        self.assertEqual(invalid_only["coverage"], 0.0)

    def test_candidate_context_and_later_gain_counterexamples_are_distinct(self):
        candidate_records = self.phase_b_by_query["tls-query-candidate-without-context"]
        candidate = candidate_records[0]
        self.assertTrue(candidate["marginal_candidate_evidence_gain"])
        self.assertFalse(candidate["marginal_final_context_evidence_gain"])
        self.assertFalse(candidate["current_final_context_sufficiency"])
        after_expansion = candidate_records[1]
        self.assertNotEqual(
            after_expansion["candidate_evidence_ids"],
            after_expansion["context_evidence_ids"],
        )
        self.assertTrue(after_expansion["candidate_evidence_ids"])
        self.assertFalse(after_expansion["context_evidence_ids"])

        later = self.phase_b_by_query["tls-query-later-useful-evidence"]
        self.assertFalse(later[0]["marginal_candidate_evidence_gain"])
        self.assertFalse(later[0]["marginal_final_context_evidence_gain"])
        self.assertTrue(later[0]["remaining_useful_evidence"])
        self.assertTrue(later[1]["marginal_candidate_evidence_gain"])
        self.assertTrue(later[1]["marginal_final_context_evidence_gain"])

    def test_degenerate_terminal_and_nonattainment_cases_are_explicit(self):
        phase_a = {trace.query_id: trace for trace in self.phase_a.queries}
        empty = phase_a["tls-query-empty-plan"]
        self.assertTrue(all(not record.decision_input.validity.plan_valid for record in empty.records))
        self.assertEqual(empty.records[-1].action_reason, "invalid_evidence_plan")
        invalid = phase_a["tls-query-invalid-feature"]
        self.assertEqual(invalid.records[-1].action_reason, "invalid_state_at_corpus_exhaustion")
        zero = phase_a["tls-query-zero-distance"]
        self.assertIn("invalid_state", zero.records[-1].terminal_flags)
        self.assertIn("maximum_expansions_reached", zero.records[-1].terminal_flags)
        self.assertIn("corpus_exhausted", zero.records[-1].terminal_flags)
        nonattainment = self.phase_b_by_query["tls-query-nonattainment"][-1]
        self.assertTrue(nonattainment["terminal_evidence_nonattainment"])
        self.assertFalse(nonattainment["current_final_context_sufficiency"])
        self.assertFalse(nonattainment["remaining_useful_evidence"])
        pilot = self.phase_b_by_query["tls-query-pilot-stop"][0]
        self.assertTrue(pilot["current_final_context_sufficiency"])

    def test_observed_pair_beta_rho_preserve_angle_and_exact_tri_law_api(self):
        fixture = same_distance_different_angle_fixture()
        query = fixture["query"]
        near_displacement = fixture["near"] - query
        far_displacements = (
            fixture["far_same_plane"] - query,
            fixture["far_orthogonal_plane"] - query,
        )
        near_squared = float(np.dot(near_displacement, near_displacement))
        betas = []
        rhos = []
        for far_displacement in far_displacements:
            far_squared = float(np.dot(far_displacement, far_displacement))
            betas.append(far_squared / near_squared)
            rhos.append(
                float(np.dot(near_displacement, far_displacement))
                / np.sqrt(near_squared * far_squared)
            )
        self.assertAlmostEqual(betas[0], betas[1], places=14)
        self.assertGreater(betas[0], 1.0)
        self.assertNotAlmostEqual(rhos[0], rhos[1], places=8)
        probabilities = [
            tri_law_probability(betas[index], rhos[index], self.config.m_prime)
            for index in range(2)
        ]
        self.assertNotAlmostEqual(probabilities[0], probabilities[1], places=8)
        self.assertEqual(tri_law_probability(2.0, 1.0, 8), 0.0)
        self.assertAlmostEqual(tri_law_probability(2.0, 0.0, 2), 1.0 / 3.0)

        production_source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("from .tri_law", production_source)
        self.assertNotIn("tri_law_probability(", production_source)
        self.assertNotIn("risk_profile", production_source)

    def test_separate_complete_work_and_timing_counters(self):
        records = [record for trace in self.phase_a.queries for record in trace.records]
        for record in records:
            work = dict(record.work)
            self.assertEqual(tuple(work), WORK_FIELDS)
            self.assertEqual(work["new_original_distance_evaluations"], 3 if record.stage == 0 else (3 if record.stage == 1 else 6))
            self.assertEqual(work["exact_rerank_count"], 1)
            self.assertEqual(work["fixed_controller_evaluation_count"], 1)
            self.assertEqual(work["final_context_construction_count"], 1)
        totals = {field: sum(dict(record.work)[field] for record in records) for field in WORK_FIELDS}
        self.assertEqual(totals["query_projection_count"], len(self.environment.queries))
        self.assertEqual(totals["projected_full_scan_count"], len(self.environment.queries))
        self.assertEqual(
            totals["projected_distance_evaluations"],
            len(self.environment.queries) * self.config.corpus_size,
        )
        self.assertEqual(totals["new_original_distance_evaluations"], 69)
        self.assertEqual(totals["expansion_prefix_reuse_count"], 11)
        self.assertEqual(len(self.phase_a.timing_records), len(records))
        for timing in self.phase_a.timing_records:
            self.assertEqual(
                tuple(key for key in timing if key not in {"query_id", "stage"}),
                TIMING_FIELDS,
            )
            self.assertTrue(all(timing[field] >= 0.0 for field in TIMING_FIELDS))

    def test_two_run_portable_artifacts_are_identical_and_aggregates_reconstruct(self):
        with tempfile.TemporaryDirectory() as first_temp, tempfile.TemporaryDirectory() as second_temp:
            first = Path(first_temp) / "run"
            second = Path(second_temp) / "run"
            run_step2(self.config, first)
            run_step2(self.config, second)
            for name in PORTABLE_ARTIFACTS:
                with self.subTest(artifact=name):
                    self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            self.assertNotIn("timings.json", PORTABLE_ARTIFACTS)

            decisions = [json.loads(line) for line in (first / "phase_a_decisions.jsonl").read_text().splitlines()]
            supervision = [json.loads(line) for line in (first / "phase_b_supervision.jsonl").read_text().splitlines()]
            aggregate = json.loads((first / "aggregates.json").read_text())
            self.assertEqual(aggregate["stage_count"], len(decisions))
            self.assertEqual(
                aggregate["actions"],
                {
                    action.value: sum(row["action"] == action.value for row in decisions)
                    for action in Action
                },
            )
            self.assertEqual(
                aggregate["candidate_gain_stage_count"],
                sum(row["marginal_candidate_evidence_gain"] for row in supervision),
            )
            self.assertEqual(
                aggregate["context_gain_stage_count"],
                sum(row["marginal_final_context_evidence_gain"] for row in supervision),
            )
            final = {row["query_id"]: row for row in supervision}
            self.assertAlmostEqual(
                aggregate["mean_final_context_coverage"],
                float(np.mean([row["context_coverage"] for row in final.values()])),
            )
            manifest = json.loads((first / "manifest.json").read_text())
            self.assertFalse(manifest["timings_portable"])
            self.assertTrue(manifest["phase_a_closed_before_label_store_opened"])
            report = (first / "report.md").read_text(encoding="utf-8")
            for disclaimer in (
                "not a learned or calibrated controller result",
                "real-data claim",
                "certificate",
                "latency claim",
                "answer-quality result",
            ):
                self.assertIn(disclaimer, report)

    def test_runner_scope_is_offline_exact_synthetic_and_dependency_free(self):
        source = MODULE_PATH.read_text(encoding="utf-8").casefold()
        for forbidden in (
            "requests",
            "urllib",
            "socket",
            "openai",
            "query_cal",
            "query_tune",
            "query_cert",
            "query_latency",
            "query_test",
            "answer_generation(",
            "faiss",
        ):
            self.assertNotIn(forbidden, source)
        config = self.config.to_dict()
        self.assertEqual(config["retrieval"]["backend"], "numpy_exact_squared_l2")
        self.assertFalse(config["retrieval"]["post_projection_normalized"])


if __name__ == "__main__":
    unittest.main()
