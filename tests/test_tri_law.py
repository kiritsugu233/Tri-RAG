import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2, f

from tri_rag_harness.tri_law import (
    tri_law_conditional_orthogonal,
    tri_law_probability,
    tri_law_threshold,
)


class TriLawTests(unittest.TestCase):
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
