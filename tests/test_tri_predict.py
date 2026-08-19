import unittest

import numpy as np

from tri_rag_harness.policies import MonotoneBinnedPolicy, TriPredictPolicy
from tri_rag_harness.tri_law import tri_law_conditional_orthogonal
from tri_rag_harness.tri_predict import (
    deterministic_rank_quadrature,
    solve_y_star,
    tri_predict_h_j,
    tri_predict_retention,
    tri_predict_retention_grid,
)


class TriPredictTests(unittest.TestCase):
    def test_exact_finite_rank_sum_matches_conditional_law_terms(self):
        y = 0.8
        neighbor_rank = 3
        lid = 7.0
        m_prime = 12
        k_gt = 5
        corpus_size = 30
        ranks = np.arange(k_gt + 1, corpus_size)
        beta = (ranks / neighbor_rank) ** (2.0 / lid)
        expected = float(
            np.sum(tri_law_conditional_orthogonal(y, beta, m_prime))
        )
        actual = tri_predict_h_j(
            y,
            neighbor_rank=neighbor_rank,
            lid=lid,
            m_prime=m_prime,
            k_gt=k_gt,
            corpus_size=corpus_size,
        )
        self.assertAlmostEqual(actual, expected, places=12)

    def test_rank_quadrature_is_deterministic_and_conserves_population(self):
        exact = deterministic_rank_quadrature(6, 30, max_samples=30)
        self.assertTrue(exact.exact)
        np.testing.assert_array_equal(exact.ranks, np.arange(6, 30))
        approximate_one = deterministic_rank_quadrature(6, 10_000, max_samples=32)
        approximate_two = deterministic_rank_quadrature(6, 10_000, max_samples=32)
        self.assertFalse(approximate_one.exact)
        self.assertEqual(float(np.sum(approximate_one.weights)), 9_994.0)
        np.testing.assert_array_equal(approximate_one.ranks, approximate_two.ranks)
        np.testing.assert_array_equal(approximate_one.weights, approximate_two.weights)

    def test_h_j_is_monotone_in_conditioning_scale(self):
        values = [
            tri_predict_h_j(
                y,
                neighbor_rank=2,
                lid=9.0,
                m_prime=8,
                k_gt=4,
                corpus_size=80,
            )
            for y in (0.0, 0.1, 0.5, 1.0, 4.0, float("inf"))
        ]
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 75.0)

    def test_bounded_root_solves_mean_field_equation(self):
        parameters = {
            "neighbor_rank": 3,
            "lid": 8.0,
            "m_prime": 16,
            "k_gt": 5,
            "budget": 20,
            "corpus_size": 160,
        }
        root = solve_y_star(**parameters)
        expected_count = parameters["budget"] - parameters["neighbor_rank"]
        actual_count = tri_predict_h_j(
            root,
            neighbor_rank=parameters["neighbor_rank"],
            lid=parameters["lid"],
            m_prime=parameters["m_prime"],
            k_gt=parameters["k_gt"],
            corpus_size=parameters["corpus_size"],
        )
        self.assertAlmostEqual(actual_count, expected_count, places=9)

    def test_full_budget_boundary_has_infinite_roots_and_unit_retention(self):
        for neighbor_rank in (1, 2, 3):
            root = solve_y_star(
                neighbor_rank=neighbor_rank,
                lid=6.0,
                m_prime=8,
                k_gt=3,
                budget=19,
                corpus_size=20,
            )
            self.assertTrue(np.isposinf(root))
        self.assertEqual(
            tri_predict_retention(
                lid=6.0,
                m_prime=8,
                k_gt=3,
                budget=19,
                corpus_size=20,
            ),
            1.0,
        )

    def test_predicted_retention_is_budget_monotone_and_bounded(self):
        predictions = tri_predict_retention_grid(
            lid=8.0,
            m_prime=16,
            k_gt=5,
            budgets=[12, 20, 32, 48, 80],
            corpus_size=160,
        )
        values = list(predictions.values())
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))

    def test_geometric_approximation_agrees_with_exact_sum_and_prediction(self):
        for y in (0.5, 1.0, 2.0, 5.0):
            exact_h = tri_predict_h_j(
                y,
                neighbor_rank=3,
                lid=8.0,
                m_prime=16,
                k_gt=5,
                corpus_size=1000,
            )
            approximate_h = tri_predict_h_j(
                y,
                neighbor_rank=3,
                lid=8.0,
                m_prime=16,
                k_gt=5,
                corpus_size=1000,
                max_rank_samples=32,
            )
            self.assertLess(abs(approximate_h - exact_h) / max(1.0, exact_h), 0.002)
        exact_retention = tri_predict_retention(
            lid=16.0,
            m_prime=16,
            k_gt=5,
            budget=100,
            corpus_size=1000,
        )
        approximate_retention = tri_predict_retention(
            lid=16.0,
            m_prime=16,
            k_gt=5,
            budget=100,
            corpus_size=1000,
            max_rank_samples=32,
        )
        self.assertLess(abs(approximate_retention - exact_retention), 0.001)

    def test_analytic_policy_is_grid_safe_lid_monotone_and_logs_saturation(self):
        policy = TriPredictPolicy(
            corpus_size=160,
            m_prime=16,
            k_gt=5,
            grid=[12, 20, 32, 48, 80],
            target=0.9,
            max_rank_samples=256,
        )
        decisions = [policy.choose(lid, True) for lid in (2.0, 4.0, 8.0, 16.0, 32.0)]
        budgets = [decision.budget for decision in decisions]
        self.assertTrue(all(budget in policy.grid for budget in budgets))
        self.assertTrue(all(left <= right for left, right in zip(budgets, budgets[1:])))
        self.assertTrue(decisions[-1].saturated)
        fallback = policy.choose(100.0, False)
        self.assertTrue(fallback.used_fallback)
        self.assertEqual(fallback.budget, 80)

    def test_safety_correction_is_fit_only_from_supplied_tune_records(self):
        base = TriPredictPolicy(
            corpus_size=80,
            m_prime=12,
            k_gt=4,
            grid=[8, 16, 32],
            target=0.8,
            max_rank_samples=128,
        )
        records = []
        for lid in (4.0, 8.0, 12.0):
            predictions = base.raw_predictions(lid)
            records.append(
                {
                    "lid": lid,
                    "lid_valid": True,
                    "retention_by_budget": {
                        str(budget): max(0.0, prediction - 0.1)
                        for budget, prediction in predictions.items()
                    },
                }
            )
        fitted = TriPredictPolicy.fit(
            records,
            corpus_size=80,
            m_prime=12,
            k_gt=4,
            grid=[8, 16, 32],
            target=0.8,
            max_rank_samples=128,
            fit_safety_correction=True,
            safety_quantile=0.9,
        )
        self.assertAlmostEqual(fitted.safety_correction, 0.1, places=10)
        serialized = fitted.serialize()
        self.assertEqual(serialized["correction_source_split"], "query_tune")
        self.assertEqual(serialized["correction_fit_observations"], 9)

    def test_analytic_and_empirical_policies_share_decision_interface(self):
        empirical = MonotoneBinnedPolicy(
            edges=[5.0],
            budgets=[20, 32],
            grid=[12, 20, 32],
            fallback_budget=32,
            target=0.8,
        )
        analytic = TriPredictPolicy(
            corpus_size=80,
            m_prime=12,
            k_gt=4,
            grid=[12, 20, 32],
            target=0.8,
            max_rank_samples=128,
        )
        for policy in (empirical, analytic):
            decision = policy.choose(6.0, True)
            self.assertIn(decision.budget, [12, 20, 32])
            self.assertFalse(decision.used_fallback)

    def test_invalid_inputs_are_rejected(self):
        invalid_calls = [
            lambda: tri_predict_retention(
                lid=0.0, m_prime=8, k_gt=3, budget=5, corpus_size=20
            ),
            lambda: tri_predict_retention(
                lid=5.0, m_prime=0, k_gt=3, budget=5, corpus_size=20
            ),
            lambda: tri_predict_retention(
                lid=5.0, m_prime=8, k_gt=3, budget=2, corpus_size=20
            ),
            lambda: tri_predict_retention_grid(
                lid=5.0,
                m_prime=8,
                k_gt=3,
                budgets=[10, 5],
                corpus_size=20,
            ),
            lambda: deterministic_rank_quadrature(5, 5, max_samples=2),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
