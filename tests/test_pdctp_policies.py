import dataclasses
import unittest

from tri_rag_harness.pdctp_calibration import (
    PilotLIDCalibrator,
    TriBudgetResidualCalibrator,
)
from tri_rag_harness.pdctp_features import (
    FEATURE_SCHEMA,
    PilotDistanceFeatureSpec,
    PilotFeatureVector,
)
from tri_rag_harness.pdctp_policies import (
    CalibratedTriPredictPolicy,
    FixedPDCTPPolicy,
    MonotonePDCTPPolicy,
    PDCTPDecisionInput,
    RawTriPredictPDCTPPolicy,
    validate_policy_suite,
)
from tri_rag_harness.policies import MonotoneBinnedPolicy, TriPredictPolicy


class PDCTPPolicyTests(unittest.TestCase):
    def setUp(self):
        self.spec = PilotDistanceFeatureSpec(lid_boundary=5, minimum_count=5)
        self.names = self.spec.feature_names
        self.grid = [10, 20, 40]
        self.raw = TriPredictPolicy(
            corpus_size=40,
            m_prime=8,
            k_gt=3,
            grid=self.grid,
            target=0.8,
            max_rank_samples=16,
        )
        coefficients = [0.0] * len(self.names)
        coefficients[self.names.index("log_pilot_lid")] = 1.0
        self.lid_calibrator = PilotLIDCalibrator(
            feature_names=self.names,
            feature_spec_fingerprint=self.spec.fingerprint,
            means=[0.0] * len(self.names),
            scales=[1.0] * len(self.names),
            constant_features=[False] * len(self.names),
            intercept=0.0,
            coefficients=coefficients,
            regularization=0.0,
            output_min=1.0,
            output_max=100.0,
            fallback=100.0,
            fit_ids=["cal-0", "cal-1"],
            objective_value=0.0,
        )
        self.residual = TriBudgetResidualCalibrator(
            feature_names=self.names,
            feature_spec_fingerprint=self.spec.fingerprint,
            means=[0.0] * len(self.names),
            scales=[1.0] * len(self.names),
            constant_features=[False] * len(self.names),
            intercept=0.0,
            coefficients=[0.0] * len(self.names),
            quantile=0.5,
            regularization=0.0,
            safety_offset=0.0,
            training_level=1.0,
            grid=self.grid,
            minimum_budget=10,
            fallback_budget=40,
            raw_policy_fingerprint=self.raw.serialize()["fingerprint"],
            fit_ids=["cal-0", "cal-1"],
            objective_value=0.0,
        )

    def _input(self, lid=5.0, valid=True):
        values = [0.0] * len(self.names)
        if valid:
            import math

            values[self.names.index("log_pilot_lid")] = math.log(lid)
            values[self.names.index("pilot_lid_valid")] = 1.0
            values[self.names.index("valid_distance_fraction")] = 1.0
        features = PilotFeatureVector(
            FEATURE_SCHEMA,
            self.spec.fingerprint,
            self.names,
            tuple(values),
            valid,
            None if valid else "synthetic_invalid",
        )
        return PDCTPDecisionInput(features, lid, valid)

    def test_raw_v1_serialization_loading_and_decisions_remain_unchanged(self):
        before_artifact = self.raw.serialize()
        before = [self.raw.choose(lid, True) for lid in (2.0, 5.0, 10.0)]
        adapter = RawTriPredictPDCTPPolicy(self.raw, minimum_budget=10)
        after_artifact = self.raw.serialize()
        restored = TriPredictPolicy.from_serialized(after_artifact)
        after = [self.raw.choose(lid, True) for lid in (2.0, 5.0, 10.0)]
        self.assertEqual(before_artifact, after_artifact)
        self.assertEqual(restored.serialize(), before_artifact)
        self.assertEqual(before, after)
        for lid, raw_decision in zip((2.0, 5.0, 10.0), before):
            self.assertEqual(adapter.choose(self._input(lid)).budget, raw_decision.budget)

    def test_all_ablations_share_one_decision_interface(self):
        monotone_reference = MonotoneBinnedPolicy(
            edges=[4.0, 8.0],
            budgets=[10, 20, 40],
            grid=self.grid,
            fallback_budget=40,
            target=0.8,
        )
        suite = {
            "fixed": FixedPDCTPPolicy(20, self.grid, 10),
            "monotone": MonotonePDCTPPolicy(
                monotone_reference, minimum_budget=10
            ),
            "raw_tri": RawTriPredictPDCTPPolicy(self.raw, minimum_budget=10),
            "lid_only": CalibratedTriPredictPolicy(
                mode="lid_only",
                raw_reference=self.raw,
                minimum_budget=10,
                lid_calibrator=self.lid_calibrator,
            ),
            "residual_only": CalibratedTriPredictPolicy(
                mode="residual_only",
                raw_reference=self.raw,
                minimum_budget=10,
                residual_calibrator=self.residual,
            ),
            "full_pdctp": CalibratedTriPredictPolicy(
                mode="full",
                raw_reference=self.raw,
                minimum_budget=10,
                lid_calibrator=self.lid_calibrator,
                residual_calibrator=self.residual,
            ),
        }
        validate_policy_suite(suite)
        decisions = {name: policy.choose(self._input()) for name, policy in suite.items()}
        self.assertEqual(set(decisions), set(suite))
        for decision in decisions.values():
            self.assertIn(decision.budget, self.grid)
            self.assertGreaterEqual(decision.budget, 10)
        self.assertEqual(decisions["raw_tri"].budget, decisions["lid_only"].budget)
        self.assertEqual(decisions["raw_tri"].budget, decisions["residual_only"].budget)
        self.assertEqual(decisions["raw_tri"].budget, decisions["full_pdctp"].budget)
        names = {policy.serialize()["name"] for policy in suite.values()}
        self.assertEqual(len(names), len(suite))

    def test_invalid_feature_state_has_terminal_fallback(self):
        for mode, kwargs in (
            ("lid_only", {"lid_calibrator": self.lid_calibrator}),
            ("residual_only", {"residual_calibrator": self.residual}),
            (
                "full",
                {
                    "lid_calibrator": self.lid_calibrator,
                    "residual_calibrator": self.residual,
                },
            ),
        ):
            with self.subTest(mode=mode):
                policy = CalibratedTriPredictPolicy(
                    mode=mode,
                    raw_reference=self.raw,
                    minimum_budget=10,
                    **kwargs,
                )
                decision = policy.choose(self._input(valid=False))
                self.assertEqual(decision.budget, 40)
                self.assertTrue(decision.used_fallback)

    def test_decision_contract_excludes_all_forbidden_fields(self):
        fields = {field.name for field in dataclasses.fields(PDCTPDecisionInput)}
        self.assertEqual(fields, {"features", "pilot_lid", "pilot_lid_valid"})
        self.assertTrue(
            fields.isdisjoint(
                {
                    "oracle_lid",
                    "exact_top_k",
                    "qrels",
                    "retention",
                    "answer_labels",
                    "split_role",
                }
            )
        )
        with self.assertRaises(TypeError):
            PDCTPDecisionInput(
                self._input().features,
                5.0,
                True,
                qrels=["forbidden"],  # type: ignore
            )


if __name__ == "__main__":
    unittest.main()
