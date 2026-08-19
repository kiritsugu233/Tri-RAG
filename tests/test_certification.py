import math
import unittest

from tri_rag_harness.certification import (
    empirical_bernstein,
    make_certificate,
    plan_sample_size,
    validate_certificate_identity,
)


class CertificationTests(unittest.TestCase):
    def test_hand_computed_fixture(self):
        values = [0.0, 1.0, 1.0]
        result = empirical_bernstein(values, alpha=0.05)
        log_term = math.log(40.0)
        expected_variance = 1.0 / 3.0
        expected_variance_term = math.sqrt(2 * expected_variance * log_term / 3)
        expected_range_term = 7 * log_term / (3 * 2)
        self.assertAlmostEqual(result.mean, 2.0 / 3.0)
        self.assertAlmostEqual(result.unbiased_variance, expected_variance)
        self.assertAlmostEqual(result.radius_variance_term, expected_variance_term)
        self.assertAlmostEqual(result.radius_range_term, expected_range_term)

    def test_certificate_identity_is_frozen(self):
        certificate = make_certificate(
            [0.8, 0.9, 1.0],
            alpha=0.05,
            target=0.5,
            policy_fingerprint="policy-a",
            split_hash="split-a",
        )
        validate_certificate_identity(
            certificate, policy_fingerprint="policy-a", split_hash="split-a"
        )
        with self.assertRaises(ValueError):
            validate_certificate_identity(
                certificate, policy_fingerprint="policy-b", split_hash="split-a"
            )
        with self.assertRaises(ValueError):
            validate_certificate_identity(
                certificate, policy_fingerprint="policy-a", split_hash="split-b"
            )

    def test_sample_size_plan_satisfies_requested_width(self):
        n = plan_sample_size(alpha=0.05, desired_radius=0.2)
        self.assertGreater(n, 2)
        worst_variance = n / (4.0 * (n - 1))
        log_term = math.log(40.0)
        radius = math.sqrt(2 * worst_variance * log_term / n) + 7 * log_term / (
            3 * (n - 1)
        )
        self.assertLessEqual(radius, 0.2)


if __name__ == "__main__":
    unittest.main()
