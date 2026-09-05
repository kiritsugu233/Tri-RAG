import copy
import dataclasses
import json
import math
import unittest
from pathlib import Path

import numpy as np

from tri_rag_harness.pdctp_features import (
    FEATURE_SCHEMA,
    PilotDistanceFeatureSpec,
    PilotFeatureVector,
)
from tri_rag_harness.pdctp_policies import PDCTPDecisionInput
from tri_rag_harness.pdctp_v3 import (
    EffectiveCurveTriPredictPolicyV3,
    EffectiveTriLIDCalibratorV3,
    EffectiveTriLIDFitRecordV3,
    PDCTPV3Error,
    TriPredictPredictionGridCacheV3,
)
from tri_rag_harness.tri_predict import tri_predict_retention_grid
from tri_rag_harness.utils import fingerprint


class PDCTPV3Tests(unittest.TestCase):
    def setUp(self):
        self.spec = PilotDistanceFeatureSpec(lid_boundary=5, minimum_count=5)
        self.names = self.spec.feature_names
        self.budgets = (10, 20, 40, 80)
        self.lid_grid = (2.0, 4.0, 8.0, 16.0, 32.0)

    def _features(self, low_lid, high_lid, *, valid=True):
        values = [0.0] * len(self.names)
        if valid:
            values[self.names.index("log_pilot_lid")] = math.log(low_lid)
            values[self.names.index("log_radius")] = math.log(high_lid)
            values[self.names.index("pilot_lid_valid")] = 1.0
            values[self.names.index("valid_distance_fraction")] = 1.0
        return PilotFeatureVector(
            FEATURE_SCHEMA,
            self.spec.fingerprint,
            self.names,
            tuple(values),
            valid,
            None if valid else "synthetic_invalid",
        )

    def _curve(self, low_lid, high_lid):
        low = tri_predict_retention_grid(
            lid=low_lid,
            m_prime=8,
            k_gt=3,
            budgets=self.budgets,
            corpus_size=80,
            max_rank_samples=16,
        )
        high = tri_predict_retention_grid(
            lid=high_lid,
            m_prime=8,
            k_gt=3,
            budgets=self.budgets,
            corpus_size=80,
            max_rank_samples=16,
        )
        values = np.asarray([low[10], low[20], high[40], 1.0], dtype=np.float64)
        return tuple(np.maximum.accumulate(values).tolist())

    def _records(self):
        pairs = (
            (2.0, 8.0),
            (2.0, 16.0),
            (4.0, 8.0),
            (4.0, 32.0),
            (8.0, 4.0),
            (8.0, 16.0),
            (16.0, 2.0),
            (16.0, 8.0),
            (32.0, 4.0),
            (32.0, 16.0),
        )
        return [
            EffectiveTriLIDFitRecordV3(
                f"cal-{index}",
                "query_cal",
                self._features(low_lid, high_lid),
                self._curve(low_lid, high_lid),
            )
            for index, (low_lid, high_lid) in enumerate(pairs)
        ]

    def _calibrator(self):
        return EffectiveTriLIDCalibratorV3.fit(
            self._records(),
            regularization=0.0,
            output_min=2.0,
            output_max=32.0,
            fallback=32.0,
            m_prime=8,
            k_gt=3,
            corpus_size=80,
            budgets=self.budgets,
            max_rank_samples=16,
            low_budget_max=20,
            high_budget_min=40,
            target_lid_grid=self.lid_grid,
        )

    def _input(self, low_lid, high_lid, *, valid=True):
        return PDCTPDecisionInput(
            self._features(low_lid, high_lid, valid=valid),
            low_lid,
            valid,
        )

    def test_cache_identity_uses_exact_lid_bits_and_full_numerical_problem(self):
        cache = TriPredictPredictionGridCacheV3()
        lid = np.float64(8.0)
        adjacent = np.nextafter(lid, np.float64(np.inf))
        first = cache.cache_key(
            lid=lid,
            m_prime=8,
            k_gt=3,
            corpus_size=80,
            budgets=self.budgets,
            max_rank_samples=16,
        )
        second = cache.cache_key(
            lid=adjacent,
            m_prime=8,
            k_gt=3,
            corpus_size=80,
            budgets=self.budgets,
            max_rank_samples=16,
        )
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 7)
        cached = cache.prediction_grid(
            lid=lid,
            m_prime=8,
            k_gt=3,
            corpus_size=80,
            budgets=self.budgets,
            max_rank_samples=16,
        )
        uncached = cache.prediction_grid(
            lid=lid,
            m_prime=8,
            k_gt=3,
            corpus_size=80,
            budgets=self.budgets,
            max_rank_samples=16,
            use_cache=False,
        )
        self.assertEqual(cached, uncached)
        self.assertEqual(cache.entry_count, 1)
        self.assertFalse(hasattr(cache, "serialize"))

    def test_checked_config_is_v3_query_cal_diagnostic_only(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "pdctp_v3_network_free_foundation_v1.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["schema"], "pdctp_effective_curve_shape_foundation_v3")
        self.assertEqual(config["version"], 3)
        self.assertEqual(config["access"]["fit_role"], "query_cal")
        self.assertFalse(config["access"]["selection_allowed"])
        self.assertEqual(
            config["repair"]["modes"],
            ["scalar_effective_lid", "two_regime_effective_lid"],
        )
        self.assertFalse(
            config["cache_contract"]["serialized_into_scientific_artifacts"]
        )
        self.assertEqual(
            config["artifact_directory"], "artifacts/pdctp_v3_network_free"
        )

    def test_fit_is_query_cal_only_and_artifact_is_new_cache_free_schema(self):
        rows = self._records()
        forbidden = list(rows)
        forbidden[0] = dataclasses.replace(forbidden[0], role="query_tune")
        with self.assertRaisesRegex(PDCTPV3Error, "query_cal-only"):
            EffectiveTriLIDCalibratorV3.fit(
                forbidden,
                regularization=0.0,
                output_min=2.0,
                output_max=32.0,
                fallback=32.0,
                m_prime=8,
                k_gt=3,
                corpus_size=80,
                budgets=self.budgets,
                max_rank_samples=16,
                low_budget_max=20,
                high_budget_min=40,
                target_lid_grid=self.lid_grid,
            )

        calibrator = self._calibrator()
        artifact = calibrator.serialize()
        self.assertEqual(artifact["schema"], "calibrated_tri_predict_effective_lid_v3")
        self.assertEqual(artifact["fit"]["role"], "query_cal")
        self.assertNotIn("cache", artifact)
        restored = EffectiveTriLIDCalibratorV3.from_serialized(artifact)
        self.assertEqual(restored.serialize(), artifact)
        tampered = copy.deepcopy(artifact)
        tampered["model"]["heads"]["high"]["intercept"] += 0.1
        with self.assertRaisesRegex(PDCTPV3Error, "fingerprint"):
            EffectiveTriLIDCalibratorV3.from_serialized(tampered)

    def test_two_regime_ablation_changes_only_curve_shape_and_fits_better(self):
        calibrator = self._calibrator()
        scalar = EffectiveCurveTriPredictPolicyV3(
            mode="scalar_effective_lid",
            calibrator=calibrator,
            target=0.8,
            minimum_budget=10,
            fallback_budget=80,
        )
        repaired = EffectiveCurveTriPredictPolicyV3(
            mode="two_regime_effective_lid",
            calibrator=calibrator,
            target=0.8,
            minimum_budget=10,
            fallback_budget=80,
        )
        self.assertEqual(
            scalar.serialize()["calibrator_fingerprint"],
            repaired.serialize()["calibrator_fingerprint"],
        )
        self.assertEqual(
            scalar.serialize()["numerical_problem"],
            repaired.serialize()["numerical_problem"],
        )
        scalar_errors = []
        repaired_errors = []
        for row in self._records():
            observation = PDCTPDecisionInput(row.features, 1.0, True)
            scalar_curve, _ = scalar.prediction_curve(observation)
            repaired_curve, _ = repaired.prediction_curve(observation)
            self.assertIsNotNone(scalar_curve)
            self.assertIsNotNone(repaired_curve)
            actual = np.asarray(row.realized_retention)
            scalar_errors.extend(
                (np.asarray(list(scalar_curve.values())) - actual).tolist()
            )
            repaired_errors.extend(
                (np.asarray(list(repaired_curve.values())) - actual).tolist()
            )
        scalar_rmse = float(np.sqrt(np.mean(np.square(scalar_errors))))
        repaired_rmse = float(np.sqrt(np.mean(np.square(repaired_errors))))
        self.assertLess(repaired_rmse, scalar_rmse)

    def test_complete_candidate_suite_is_identical_cached_and_uncached(self):
        calibrator = self._calibrator()
        observations = [
            self._input(2.0, 8.0),
            self._input(4.0, 32.0),
            self._input(16.0, 2.0),
            self._input(32.0, 16.0),
        ]

        def run(use_cache):
            shared_cache = TriPredictPredictionGridCacheV3()
            policies = {
                f"{mode}:{threshold}": EffectiveCurveTriPredictPolicyV3(
                    mode=mode,
                    calibrator=calibrator,
                    target=threshold,
                    minimum_budget=10,
                    fallback_budget=80,
                    cache=shared_cache,
                )
                for mode in EffectiveCurveTriPredictPolicyV3.MODES
                for threshold in (0.6, 0.8, 0.95)
            }
            budget_vectors = {
                name: tuple(
                    policy.choose(observation, use_cache=use_cache).budget
                    for observation in observations
                )
                for name, policy in policies.items()
            }
            candidate_values = {
                name: {
                    "budgets": list(budgets),
                    "mean_budget": float(np.mean(budgets)),
                }
                for name, budgets in budget_vectors.items()
            }
            selected_name = min(
                candidate_values,
                key=lambda name: (candidate_values[name]["mean_budget"], name),
            )
            selection = {
                "contract": "minimum_mean_budget_then_name_v3_tiny_fixture",
                "selected": selected_name,
                "selected_values": candidate_values[selected_name],
            }
            artifacts = {name: policy.serialize() for name, policy in policies.items()}
            suite = {
                "budget_vectors": budget_vectors,
                "candidate_values": candidate_values,
                "selection": selection,
                "artifacts": artifacts,
            }
            return suite, fingerprint(suite), shared_cache.entry_count

        uncached, uncached_fingerprint, uncached_entries = run(False)
        cached, cached_fingerprint, cached_entries = run(True)
        self.assertEqual(cached, uncached)
        self.assertEqual(cached_fingerprint, uncached_fingerprint)
        self.assertEqual(uncached_entries, 0)
        # Four observations produce three exact LID grids each; all thresholds
        # reuse those entries instead of recomputing threshold-specific curves.
        self.assertLessEqual(cached_entries, len(observations) * 3)

    def test_inference_input_stays_narrow_and_invalid_features_terminate(self):
        fields = {field.name for field in dataclasses.fields(PDCTPDecisionInput)}
        self.assertEqual(fields, {"features", "pilot_lid", "pilot_lid_valid"})
        policy = EffectiveCurveTriPredictPolicyV3(
            mode="two_regime_effective_lid",
            calibrator=self._calibrator(),
            target=0.8,
            minimum_budget=10,
            fallback_budget=80,
        )
        decision = policy.choose(self._input(2.0, 8.0, valid=False))
        self.assertEqual(decision.budget, 80)
        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.failure_reason, "synthetic_invalid")


if __name__ == "__main__":
    unittest.main()
