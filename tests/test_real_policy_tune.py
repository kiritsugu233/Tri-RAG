import json
import tempfile
import unittest
from pathlib import Path

from tri_rag_harness.policies import PolicyDecision, TriPredictPolicy
from tri_rag_harness.real_policy_tune import (
    RealPolicyTuneError,
    _SCIENTIFIC_RESULT_NAMES,
    _canonical_lid_float,
    _choose_tri_from_raw,
    _coordinate_work,
    _evaluate_decisions,
    _validate_tune_only,
    load_real_policy_tune_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "real_scifact_policy_tune.json"


class RealPolicyTuneTests(unittest.TestCase):
    def test_checked_in_protocol_is_tune_only_and_frozen(self):
        config = load_real_policy_tune_config(CONFIG_PATH)
        self.assertEqual(config.evaluation_split, "query_tune")
        self.assertEqual(config.projection_seed, 27011)
        self.assertEqual(config.m_prime, 192)
        self.assertEqual(config.m_pilot, 32)
        self.assertEqual(config.m_grid[-1], 5183)
        self.assertEqual(config.fallback_budget, 5183)
        self.assertEqual(config.selection_target, 0.95)
        self.assertEqual(config.lid_decimal_places, 9)
        self.assertEqual(
            config.feature_version, "pilot_rerank_lid_rounded_9_v2"
        )
        self.assertEqual(config.tri_target_grid[0], 0.95)
        self.assertEqual(config.tri_target_grid[-1], 1.0)
        self.assertIn(0.99999, config.tri_target_grid)
        self.assertEqual(
            config.config_fingerprint,
            "47d37917974869641951a0155e71ffbb76f676d8229ff606fef56fabc83ba812",
        )

    def test_config_rejects_protected_scope_and_cost_mutation(self):
        original = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        mutations = (
            ("evaluation_split", "query_cert", "query_tune only"),
            ("cost", "mean_budget", "selection contract"),
            ("fallback", 4096, "terminal M_grid"),
            ("lid_precision", 12, "determinism contract"),
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            for index, (kind, value, message) in enumerate(mutations):
                raw = json.loads(json.dumps(original))
                if kind == "evaluation_split":
                    raw["evaluation_split"] = value
                elif kind == "cost":
                    raw["selection"]["cost_formula"] = value
                elif kind == "lid_precision":
                    raw["determinism"]["lid_decimal_places"] = value
                else:
                    raw["monotone_binned"]["fallback_budget"] = value
                path = directory / f"mutation-{index}.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.subTest(kind=kind):
                    with self.assertRaisesRegex(RealPolicyTuneError, message):
                        load_real_policy_tune_config(path)

    def test_genoa_lid_tail_noise_is_canonicalized_before_policy_use(self):
        observed_mac_genoa_pairs = (
            (43.414943091916, 43.414943091918),
            (41.992304850850, 41.992304850851),
            (102.784422311142, 102.784422311150),
            (77.595829699649, 77.595829699642),
            (21.185334478594, 21.185334478595),
        )
        for mac_value, genoa_value in observed_mac_genoa_pairs:
            with self.subTest(mac_value=mac_value, genoa_value=genoa_value):
                self.assertEqual(
                    _canonical_lid_float(mac_value, 9),
                    _canonical_lid_float(genoa_value, 9),
                )

    def test_compiled_lookup_is_not_a_scientific_result_artifact(self):
        self.assertNotIn(
            "compiled_tri_predict_policy.json", _SCIENTIFIC_RESULT_NAMES
        )

    def test_tune_scope_guard_rejects_certification_records(self):
        _validate_tune_only(
            [
                {"split": "query_tune"},
                {"split": "query_tune"},
            ]
        )
        with self.assertRaisesRegex(RealPolicyTuneError, "query_tune only"):
            _validate_tune_only(
                [
                    {"split": "query_tune"},
                    {"split": "query_cert"},
                ]
            )

    def test_coordinate_work_uses_the_predeclared_common_objective(self):
        work = _coordinate_work(
            corpus_size=100,
            dimension=10,
            m_prime=4,
            mean_budget=20.0,
        )
        self.assertEqual(work["query_projection"], 40.0)
        self.assertEqual(work["projected_full_scan"], 400.0)
        self.assertEqual(work["mean_original_rerank"], 200.0)
        self.assertEqual(work["total"], 640.0)
        self.assertAlmostEqual(work["reduction_fraction_vs_original_full_scan"], 0.36)

    def test_cached_tri_decision_uses_exact_complete_corpus_boundary(self):
        rounded = {10: 1.0, 15: 1.0, 20: 1.0}
        exact_target = TriPredictPolicy(
            corpus_size=20,
            m_prime=8,
            k_gt=3,
            grid=[10, 15, 20],
            target=1.0,
            max_rank_samples=16,
        )
        decision = _choose_tri_from_raw(rounded, exact_target)
        self.assertEqual(decision.budget, 20)
        self.assertFalse(decision.saturated)

        corrected = TriPredictPolicy(
            corpus_size=20,
            m_prime=8,
            k_gt=3,
            grid=[10, 15, 20],
            target=0.95,
            max_rank_samples=16,
            safety_correction=0.2,
        )
        decision = _choose_tri_from_raw(rounded, corrected)
        self.assertEqual(decision.budget, 20)
        self.assertEqual(decision.predicted_retention, 1.0)
        self.assertFalse(decision.saturated)

    def test_decision_evaluation_reads_only_selected_tune_budget(self):
        config = load_real_policy_tune_config(CONFIG_PATH)
        records = [
            {
                "split": "query_tune",
                "retention_by_budget": {"32": 0.5, "768": 0.9},
            },
            {
                "split": "query_tune",
                "retention_by_budget": {"32": 1.0, "768": 1.0},
            },
        ]
        decisions = [
            PolicyDecision(32, -1, False),
            PolicyDecision(768, -1, False),
        ]
        evaluation, query_values = _evaluate_decisions(
            records,
            decisions,
            config=config,
            corpus_size=5183,
            dimension=768,
            fixed_reference_budget=768,
        )
        self.assertEqual([row["embedding_retention"] for row in query_values], [0.5, 1.0])
        self.assertEqual(evaluation["budget"]["mean"], 400.0)
        self.assertEqual(evaluation["n"], 2)


if __name__ == "__main__":
    unittest.main()
