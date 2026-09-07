import unittest
from decimal import Decimal, localcontext
import math

import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2, f

from tri_rag_harness.tri_law import (
    tri_law_conditional_orthogonal,
    tri_law_probability,
    tri_law_threshold,
)


class TriLawTests(unittest.TestCase):
    @staticmethod
    def decimal_threshold(beta, rho):
        """Independent high-precision evaluation of the original eigenvalue law."""
        with localcontext() as context:
            context.prec = 100
            b = Decimal.from_float(float(beta))
            r = Decimal.from_float(float(rho))
            gap = b - 1
            transverse = 1 - r * r
            root = (gap * gap + 4 * b * transverse).sqrt()
            return (root + gap) ** 2 / (4 * b * transverse)

    def test_joint_near_tie_collinearity_regression(self):
        beta = np.nextafter(1.0, 2.0)
        rho = np.nextafter(1.0, 0.0)
        threshold = float(self.decimal_threshold(beta, rho))
        for signed_rho in (rho, -rho):
            self.assertAlmostEqual(tri_law_threshold(beta, signed_rho), threshold, places=14)
            self.assertGreater(tri_law_threshold(beta, signed_rho), 1.0)
            self.assertAlmostEqual(tri_law_probability(beta, signed_rho, 1),
                                   0.4999999976284066, places=14)

    def test_threshold_matches_decimal_across_float64_domain(self):
        betas = [np.nextafter(1., 2.), 1. + 1e-12, 1.0001, 1.1, 2.,
                 1e8, 1e16, 1e150, 1e200, np.finfo(np.float64).max]
        rhos = [0., 0.25, 0.9, 0.999, 1. - 1e-8, np.nextafter(1., 0.)]
        for beta in betas:
            for rho in rhos:
                expected = float(self.decimal_threshold(beta, rho))
                with self.subTest(beta=beta, rho=rho), np.errstate(all="raise"):
                    actual = tri_law_threshold(beta, rho)
                    self.assertGreaterEqual(actual, 1.)
                    if np.isinf(expected):
                        self.assertTrue(np.isposinf(actual))
                    else:
                        self.assertTrue(math.isclose(actual, expected, rel_tol=3e-15))
                    self.assertEqual(actual, tri_law_threshold(beta, -rho))
        values = np.asarray(betas)
        np.testing.assert_array_equal(tri_law_threshold(values, 0.), values)

    def test_representable_tails_survive_threshold_overflow(self):
        for beta, rho in [(1e200, 0.), (np.finfo(np.float64).max, 0.5),
                          (np.finfo(np.float64).max, np.nextafter(1., 0.))]:
            threshold = self.decimal_threshold(beta, rho)
            with localcontext() as context:
                context.prec = 100
                expected_one = 2. / math.pi * math.atan(float(1 / threshold.sqrt()))
                expected_two = float(1 / (1 + threshold))
            with self.subTest(beta=beta, rho=rho), np.errstate(all="raise"):
                p1 = tri_law_probability(beta, rho, 1)
                p2 = tri_law_probability(beta, rho, 2)
                self.assertGreater(p1, 0.)
                self.assertTrue(math.isclose(p1, expected_one, rel_tol=3e-15))
                self.assertLessEqual(abs(p2 - expected_two),
                                     max(3e-15 * expected_two, float.fromhex('0x0.0000000000001p-1022')))
                for m_prime in (1, 2, 3, 128):
                    self.assertEqual(tri_law_probability(beta, 1., m_prime), 0.)

    def test_conditional_overflow_and_subnormal_regression(self):
        tiny = float.fromhex('0x0.0000000000001p-1022')
        y = np.asarray([1e308, np.finfo(float).max, 0., tiny])
        beta = np.asarray([1e308, np.finfo(float).max, 2., 2.])
        for m_prime in (1, 2, 4, 128):
            with localcontext() as context:
                context.prec = 100
                arguments = [float(Decimal(m_prime) * Decimal.from_float(a)
                                   / Decimal.from_float(b)) for a, b in zip(y, beta)]
            with np.errstate(all="raise"):
                actual = tri_law_conditional_orthogonal(y, beta, m_prime)
            np.testing.assert_allclose(actual, chi2.cdf(arguments, m_prime), rtol=3e-15, atol=0.)
        self.assertEqual(tri_law_conditional_orthogonal(1e308, 2., 128), 1.)

    def test_boundary_broadcasting_and_probability_order(self):
        beta = np.asarray([np.nextafter(1., 2.), 2., 1e16, 1e200])[:, None]
        rho = np.asarray([-1., -np.nextafter(1., 0.), 0., np.nextafter(1., 0.), 1.])
        self.assertEqual(tri_law_threshold(beta, rho).shape, (4, 5))
        for m_prime in (1, 2, 4, 128):
            probabilities = tri_law_probability(beta, rho, m_prime)
            self.assertEqual(probabilities.shape, (4, 5))
            self.assertTrue(np.all(np.isfinite(probabilities)))
            self.assertTrue(np.all((probabilities >= 0.) & (probabilities <= 0.5 + 1e-14)))
            self.assertTrue(np.all(np.diff(probabilities, axis=0) <= 0.))
            np.testing.assert_array_equal(probabilities, probabilities[:, ::-1])

    def test_algebraic_identities_and_boundaries(self):
        beta = np.asarray([1.1, 2.0, 5.0])
        np.testing.assert_allclose(tri_law_threshold(beta, 0.0), beta, rtol=1e-14)
        self.assertEqual(tri_law_probability(2.0, 1.0, 8), 0.0)
        self.assertEqual(tri_law_probability(2.0, -1.0, 8), 0.0)
        for rho in (0.25, 0.8, 0.999999):
            self.assertAlmostEqual(
                tri_law_probability(1.7, rho, 12),
                tri_law_probability(1.7, -rho, 12),
                places=14,
            )
            self.assertLessEqual(
                tri_law_probability(1.7, rho, 12),
                tri_law_probability(1.7, 0.0, 12),
            )
        probabilities = tri_law_probability(np.asarray([1.1, 2.0, 10.0]), 0.3, 9)
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        self.assertTrue(np.all(np.diff(probabilities) < 0))
        self.assertIsInstance(tri_law_probability(2.0, 0.0, 8), float)

    def test_orthogonal_specialization(self):
        for beta in (1.2, 2.0, 4.0):
            self.assertAlmostEqual(
                tri_law_probability(beta, 0.0, 7), f.sf(beta, 7, 7), places=14
            )

    def test_conditional_marginalization(self):
        for m_prime, beta in ((3, 1.3), (8, 2.0), (15, 4.0)):
            integral, error = quad(
                lambda x: float(tri_law_conditional_orthogonal(x / m_prime, beta, m_prime))
                * chi2.pdf(x, df=m_prime),
                0.0,
                np.inf,
                epsabs=2e-11,
                epsrel=2e-11,
                limit=300,
            )
            self.assertLess(error, 1e-9)
            self.assertAlmostEqual(integral, tri_law_probability(beta, 0.0, m_prime), places=10)

    def test_input_validation(self):
        invalid_calls = [
            lambda: tri_law_probability(1.0, 0.0, 3),
            lambda: tri_law_probability(2.0, 1.2, 3),
            lambda: tri_law_probability(2.0, 0.0, 0),
            lambda: tri_law_probability(np.nan, 0.0, 3),
            lambda: tri_law_probability(np.inf, 0.0, 3),
            lambda: tri_law_conditional_orthogonal(-1.0, 2.0, 3),
            lambda: tri_law_conditional_orthogonal(1.0, np.asarray([2.0, 3.0]), 2.5),
            lambda: tri_law_probability(np.ones(2) * 2, np.ones(3) * 0.2, 3),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

    def test_monte_carlo_conformance(self):
        cases = [
            (1.2, 0.0, 4),
            (2.0, 0.55, 4),
            (2.0, -0.55, 4),
            (4.0, 0.2, 12),
            (1.1, 0.92, 12),
            (1.1, 0.999, 4),
        ]
        trials = 30_000
        for case_index, (beta, rho, m_prime) in enumerate(cases):
            rng = np.random.default_rng(700 + case_index)
            e_plus = np.asarray([1.0, 0.0])
            e_minus = np.asarray([rho, np.sqrt(1.0 - rho**2)])
            matrices = rng.normal(
                scale=1.0 / np.sqrt(m_prime), size=(trials, m_prime, 2)
            )
            projected_plus = matrices @ e_plus
            projected_minus = matrices @ (np.sqrt(beta) * e_minus)
            empirical = float(
                np.mean(
                    np.sum(projected_minus**2, axis=1)
                    < np.sum(projected_plus**2, axis=1)
                )
            )
            exact = tri_law_probability(beta, rho, m_prime)
            tolerance = 5.0 * np.sqrt(exact * (1.0 - exact) / trials) + 0.002
            with self.subTest(beta=beta, rho=rho, m_prime=m_prime):
                self.assertLessEqual(abs(empirical - exact), tolerance)


if __name__ == "__main__":
    unittest.main()
