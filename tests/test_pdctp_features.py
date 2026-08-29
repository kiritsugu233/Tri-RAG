import dataclasses
import math
import unittest

import numpy as np

from tri_rag_harness.pdctp_features import (
    PilotDistanceFeatureExtractor,
    PilotDistanceFeatureSpec,
    PilotDistanceObservation,
    PilotFeatureError,
    stable_sort_pilot_distances,
)


class PDCTPFeatureTests(unittest.TestCase):
    def setUp(self):
        self.spec = PilotDistanceFeatureSpec(
            lid_boundary=5,
            minimum_count=5,
            gap_quantiles=(0.25, 0.5, 0.75),
            epsilon=0.0,
            duplicate_tolerance=1e-12,
        )
        self.extractor = PilotDistanceFeatureExtractor(self.spec)

    def _observation(self, scale=1.0):
        original = scale * np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
        projected = scale * np.asarray([2.0, 2.0, 8.0, 4.0, 32.0])
        return PilotDistanceObservation.from_arrays(
            np.square(original),
            np.square(projected),
            pilot_lid=2.0,
            pilot_lid_valid=True,
            pilot_lid_failure_reason=None,
            valid_distance_count=5,
        )

    def test_hand_computed_feature_profile(self):
        result = self.extractor.extract(self._observation())
        self.assertTrue(result.valid)
        values = dict(zip(result.names, result.values))
        profile = np.log(np.asarray([16.0, 8.0, 4.0, 2.0]))
        gaps = np.asarray([1.0, 2.0, 4.0, 8.0]) / 16.0
        distortion = np.log(np.asarray([2.0, 1.0, 2.0, 0.5, 2.0]))
        self.assertAlmostEqual(values["log_pilot_lid"], math.log(2.0))
        self.assertAlmostEqual(values["log_radius"], math.log(16.0))
        self.assertAlmostEqual(values["log_ratio_mean"], float(np.mean(profile)))
        self.assertAlmostEqual(values["log_ratio_std"], float(np.std(profile)))
        self.assertAlmostEqual(values["inner_half_slope"], -3.0 * math.log(2.0))
        self.assertAlmostEqual(values["outer_half_slope"], -3.0 * math.log(2.0))
        self.assertAlmostEqual(values["profile_curvature"], 0.0)
        for quantile in (0.25, 0.5, 0.75):
            self.assertAlmostEqual(
                values[f"normalized_gap_q{int(100 * quantile):02d}"],
                float(np.quantile(gaps, quantile)),
            )
        self.assertAlmostEqual(
            values["projection_log_distortion_mean"], float(np.mean(distortion))
        )
        self.assertAlmostEqual(
            values["projection_log_distortion_std"], float(np.std(distortion))
        )
        self.assertEqual(values["pilot_lid_valid"], 1.0)
        self.assertEqual(values["valid_distance_fraction"], 1.0)

    def test_ratio_features_are_scale_invariant_but_radius_is_not(self):
        base = self.extractor.extract(self._observation(1.0))
        scaled = self.extractor.extract(self._observation(7.0))
        base_values = dict(zip(base.names, base.values))
        scaled_values = dict(zip(scaled.names, scaled.values))
        invariant = set(base.names) - {"log_radius"}
        for name in invariant:
            with self.subTest(name=name):
                self.assertAlmostEqual(base_values[name], scaled_values[name], places=12)
        self.assertAlmostEqual(
            scaled_values["log_radius"] - base_values["log_radius"], math.log(7.0)
        )

    def test_feature_values_use_the_frozen_cross_platform_lattice(self):
        baseline = self.extractor.extract(self._observation())
        observation = self._observation()
        perturbed = PilotDistanceObservation.from_arrays(
            np.asarray(observation.original_squared_distances) * (1.0 + 1.0e-15),
            np.asarray(observation.projected_squared_distances) * (1.0 + 1.0e-15),
            pilot_lid=observation.pilot_lid * (1.0 + 1.0e-15),
            pilot_lid_valid=True,
            pilot_lid_failure_reason=None,
            valid_distance_count=observation.valid_distance_count,
        )
        self.assertEqual(self.extractor.extract(perturbed), baseline)
        self.assertTrue(
            all(value == round(value, self.spec.output_decimals) for value in baseline.values)
        )
        self.assertEqual(self.spec.serialize()["output_decimals"], 10)

    def test_squared_l2_is_converted_after_stable_original_sort(self):
        ids, original, projected = stable_sort_pilot_distances(
            ["b", "a", "c"],
            [4.0, 4.0, 1.0],
            [9.0, 16.0, 25.0],
        )
        self.assertEqual(ids.tolist(), ["c", "a", "b"])
        self.assertEqual(original.tolist(), [1.0, 4.0, 4.0])
        self.assertEqual(projected.tolist(), [25.0, 16.0, 9.0])

    def test_invalid_states_are_deterministic_and_trigger_fixed_fill(self):
        cases = {
            "zero": ([0.0, 4.0, 9.0, 16.0, 25.0], [1, 4, 9, 16, 25], True, None),
            "duplicate": ([1.0, 4.0, 4.0, 16.0, 25.0], [1, 4, 9, 16, 25], True, None),
            "nonfinite": ([1.0, 4.0, 9.0, 16.0, float("nan")], [1, 4, 9, 16, 25], True, None),
            "insufficient": ([1.0, 4.0, 9.0, 16.0], [1, 4, 9, 16], True, None),
            "unsorted": ([1.0, 9.0, 4.0, 16.0, 25.0], [1, 4, 9, 16, 25], True, None),
            "lid_invalid": ([1.0, 4.0, 9.0, 16.0, 25.0], [1, 4, 9, 16, 25], False, "duplicate_distances"),
        }
        reasons = {}
        for name, (original, projected, valid, failure) in cases.items():
            with self.subTest(name=name):
                observation = PilotDistanceObservation.from_arrays(
                    original,
                    projected,
                    pilot_lid=5.0,
                    pilot_lid_valid=valid,
                    pilot_lid_failure_reason=failure,
                    valid_distance_count=len(original),
                )
                one = self.extractor.extract(observation)
                two = self.extractor.extract(observation)
                self.assertEqual(one, two)
                self.assertFalse(one.valid)
                self.assertIsNotNone(one.failure_reason)
                reasons[name] = one.failure_reason
                values = dict(zip(one.names, one.values))
                self.assertEqual(values["pilot_lid_valid"], 0.0)
        self.assertEqual(reasons["zero"], "nonpositive_pilot_data")
        self.assertEqual(reasons["duplicate"], "duplicate_original_distances")
        self.assertEqual(reasons["nonfinite"], "nonfinite_pilot_data")
        self.assertEqual(reasons["insufficient"], "insufficient_pilot_data")
        self.assertEqual(reasons["unsorted"], "unsorted_original_distances")

    def test_observation_contract_cannot_accept_forbidden_inference_fields(self):
        field_names = {field.name for field in dataclasses.fields(PilotDistanceObservation)}
        self.assertTrue(
            field_names.isdisjoint(
                {"qrels", "exact_top_k", "oracle_lid", "retention", "split_role"}
            )
        )
        with self.assertRaises(TypeError):
            PilotDistanceObservation(
                (), (), 1.0, True, None, 0, split_role="query_cert"  # type: ignore
            )

    def test_feature_spec_round_trip_and_tamper_rejection(self):
        artifact = self.spec.serialize()
        self.assertEqual(PilotDistanceFeatureSpec.from_serialized(artifact), self.spec)
        tampered = dict(artifact)
        tampered["epsilon"] = 1e-9
        with self.assertRaisesRegex(PilotFeatureError, "fingerprint"):
            PilotDistanceFeatureSpec.from_serialized(tampered)


if __name__ == "__main__":
    unittest.main()
