import copy
import math
import unittest

from tri_rag_harness.pdctp_calibration import (
    BudgetResidualRecord,
    CalibrationError,
    LIDCalibrationRecord,
    PilotLIDCalibrator,
    TriBudgetResidualCalibrator,
    quantile_pinball_loss,
)
from tri_rag_harness.pdctp_features import (
    FEATURE_SCHEMA,
    PilotDistanceFeatureSpec,
    PilotFeatureVector,
)


class PDCTPCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.spec = PilotDistanceFeatureSpec(lid_boundary=5, minimum_count=5)
        self.names = self.spec.feature_names

    def _features(self, *, log_lid=0.0, log_radius=0.0, valid=True):
        values = [0.0] * len(self.names)
        values[self.names.index("log_pilot_lid")] = log_lid
        values[self.names.index("log_radius")] = log_radius
        values[self.names.index("pilot_lid_valid")] = 1.0 if valid else 0.0
        values[self.names.index("valid_distance_fraction")] = 1.0
        return PilotFeatureVector(
            FEATURE_SCHEMA,
            self.spec.fingerprint,
            self.names,
            tuple(values),
            valid,
            None if valid else "synthetic_invalid",
        )

    def test_lid_fit_enforces_nonnegative_pilot_coefficient_and_cal_role(self):
        records = [
            LIDCalibrationRecord(
                f"cal-{index}",
                "query_cal",
                self._features(log_lid=float(index)),
                oracle_lid=float(20 - index),
            )
            for index in range(6)
        ]
        calibrator = PilotLIDCalibrator.fit(
            records,
            regularization=0.01,
            output_min=1.0,
            output_max=100.0,
            fallback=100.0,
        )
        coefficient = calibrator.coefficients[
            calibrator.feature_names.index("log_pilot_lid")
        ]
        self.assertEqual(coefficient, 0.0)
        self.assertTrue(1.0 <= calibrator.predict(self._features()).value <= 100.0)
        bad = list(records)
        bad[0] = LIDCalibrationRecord(
            "tune-0", "query_tune", bad[0].features, bad[0].oracle_lid
        )
        with self.assertRaisesRegex(CalibrationError, "query_cal"):
            PilotLIDCalibrator.fit(
                bad,
                regularization=0.0,
                output_min=1.0,
                output_max=100.0,
                fallback=100.0,
            )

    def test_lid_artifact_round_trip_domain_and_tamper_validation(self):
        records = [
            LIDCalibrationRecord(
                f"cal-{index}",
                "query_cal",
                self._features(log_lid=math.log(index + 2.0)),
                oracle_lid=2.0 * (index + 2.0),
            )
            for index in range(5)
        ]
        calibrator = PilotLIDCalibrator.fit(
            records,
            regularization=0.1,
            output_min=1.0,
            output_max=12.0,
            fallback=12.0,
        )
        artifact = calibrator.serialize()
        restored = PilotLIDCalibrator.from_serialized(artifact)
        self.assertEqual(restored.serialize(), artifact)
        prediction = restored.predict(self._features(log_lid=math.log(1000.0)))
        self.assertEqual(prediction.value, 12.0)
        lower_prediction = restored.predict(
            self._features(log_lid=math.log(1.0e-12))
        )
        self.assertEqual(lower_prediction.value, 1.0)
        invalid = restored.predict(self._features(valid=False))
        self.assertEqual(invalid.value, 12.0)
        self.assertTrue(invalid.used_fallback)
        for mutation in ("coefficient", "domain", "schema", "fingerprint"):
            tampered = copy.deepcopy(artifact)
            if mutation == "coefficient":
                tampered["model"]["coefficients"][0] += 0.5
            elif mutation == "domain":
                tampered["output_domain"]["minimum"] = -1.0
            elif mutation == "schema":
                tampered["schema"] = "wrong"
            else:
                tampered["fingerprint"] = "0" * 64
            with self.subTest(mutation=mutation):
                with self.assertRaises(CalibrationError):
                    PilotLIDCalibrator.from_serialized(tampered)

    def test_quantile_objective_and_hand_computed_median_optimum(self):
        self.assertAlmostEqual(
            quantile_pinball_loss([-2.0, 0.0, 1.0], 0.5), 0.5
        )
        rows = [
            BudgetResidualRecord(
                query_id=f"cal-{index}",
                role="query_cal",
                features=self._features(),
                raw_budget=20,
                required_budget=required,
                training_level=1.0,
            )
            for index, required in enumerate((10, 20, 40))
        ]
        calibrator = TriBudgetResidualCalibrator.fit(
            rows,
            quantile=0.5,
            regularization=0.0,
            safety_offset=0.0,
            grid=[10, 20, 40],
            minimum_budget=10,
            fallback_budget=40,
            raw_policy_fingerprint="raw-policy",
        )
        self.assertAlmostEqual(calibrator.intercept, 0.0, places=7)
        self.assertAlmostEqual(calibrator.objective_value, math.log(2.0) / 3.0)

    def _manual_residual(self, intercept=0.0, radius_coefficient=0.0):
        coefficients = [0.0] * len(self.names)
        coefficients[self.names.index("log_radius")] = radius_coefficient
        return TriBudgetResidualCalibrator(
            feature_names=self.names,
            feature_spec_fingerprint=self.spec.fingerprint,
            means=[0.0] * len(self.names),
            scales=[1.0] * len(self.names),
            constant_features=[False] * len(self.names),
            intercept=intercept,
            coefficients=coefficients,
            quantile=0.5,
            regularization=0.0,
            safety_offset=0.0,
            training_level=1.0,
            grid=[10, 20, 40],
            minimum_budget=10,
            fallback_budget=40,
            raw_policy_fingerprint="raw-policy",
            fit_ids=["cal-0", "cal-1"],
            objective_value=0.0,
        )

    def test_residual_grid_ceiling_lower_bound_fallback_and_ties(self):
        upward = self._manual_residual(intercept=math.log(2.0))
        exact_tie = upward.choose_budget(10, self._features())
        self.assertEqual(exact_tie.budget, 20)
        self.assertGreaterEqual(exact_tie.budget, 10)

        downward = self._manual_residual(intercept=-math.log(4.0))
        clipped_lower = downward.choose_budget(20, self._features())
        self.assertEqual(clipped_lower.budget, 10)

        terminal = self._manual_residual(intercept=math.log(10.0)).choose_budget(
            20, self._features()
        )
        self.assertEqual(terminal.budget, 40)
        self.assertTrue(terminal.saturated)

        invalid = upward.choose_budget(10, self._features(valid=False))
        self.assertEqual(invalid.budget, 40)
        self.assertTrue(invalid.used_fallback)
        self.assertTrue(invalid.saturated)

    def test_residual_corrections_can_be_positive_low_and_negative_high(self):
        calibrator = self._manual_residual(radius_coefficient=-math.log(2.0))
        low = calibrator.choose_budget(10, self._features(log_radius=-1.0))
        high = calibrator.choose_budget(40, self._features(log_radius=1.0))
        self.assertGreater(low.residual, 0.0)
        self.assertEqual(low.budget, 20)
        self.assertLess(high.residual, 0.0)
        self.assertEqual(high.budget, 20)

    def test_residual_round_trip_and_tamper_rejection(self):
        calibrator = self._manual_residual(intercept=0.1)
        artifact = calibrator.serialize()
        restored = TriBudgetResidualCalibrator.from_serialized(artifact)
        self.assertEqual(restored.serialize(), artifact)
        tampered = copy.deepcopy(artifact)
        tampered["budget_contract"]["grid"] = [10, 20, 30, 40]
        with self.assertRaises(CalibrationError):
            TriBudgetResidualCalibrator.from_serialized(tampered)
        forbidden = BudgetResidualRecord(
            "cert-0", "query_cert", self._features(), 20, 20, 1.0
        )
        with self.assertRaisesRegex(CalibrationError, "query_cal"):
            TriBudgetResidualCalibrator.fit(
                [forbidden],
                quantile=0.5,
                regularization=0.0,
                safety_offset=0.0,
                grid=[10, 20, 40],
                minimum_budget=10,
                fallback_budget=40,
                raw_policy_fingerprint="raw-policy",
            )


if __name__ == "__main__":
    unittest.main()
