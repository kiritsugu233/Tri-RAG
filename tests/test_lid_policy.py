import unittest

import numpy as np

from tri_rag_harness.lid import estimate_lid_from_squared_distances
from tri_rag_harness.policies import FixedBudgetPolicy, MonotoneBinnedPolicy


def lid_estimate(values):
    return estimate_lid_from_squared_distances(
        np.asarray(values, dtype=float),
        s_lid=6,
        min_neighbors=4,
        clip_min=1.0,
        clip_max=100.0,
        duplicate_tolerance=1e-12,
        fallback=100.0,
    )


class LIDPolicyTests(unittest.TestCase):
    def test_valid_lid_uses_euclidean_distances_and_excludes_boundary(self):
        squared = np.asarray([1.0, 4.0, 9.0, 16.0])
        result = lid_estimate(squared)
        expected = -1.0 / np.mean(np.log(np.asarray([1.0, 2.0, 3.0]) / 4.0))
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.raw, expected)

    def test_lid_rejects_zero_duplicate_and_insufficient_distances(self):
        cases = [
            ([0.0, 1.0, 2.0, 3.0], "nonpositive_distance"),
            ([1.0, 4.0, 4.0, 9.0], "duplicate_distances"),
            ([1.0, 4.0, 9.0], "insufficient_distances"),
        ]
        for values, reason in cases:
            with self.subTest(reason=reason):
                result = lid_estimate(values)
                self.assertFalse(result.valid)
                self.assertEqual(result.reason, reason)
                self.assertEqual(result.clipped, 100.0)

    def test_budget_policies_are_grid_safe_and_monotone(self):
        grid = [12, 20, 32, 48]
        records = []
        for index, lid in enumerate(np.linspace(2, 30, 40)):
            required_budget = grid[min(index // 10, 3)]
            records.append(
                {
                    "lid": lid,
                    "lid_valid": True,
                    "retention_by_budget": {
                        str(budget): 1.0 if budget >= required_budget else 0.0
                        for budget in grid
                    },
                }
            )
        policy = MonotoneBinnedPolicy.fit(
            records,
            grid=grid,
            n_bins=4,
            target=0.9,
            safety_margin=0.0,
            fallback_budget=48,
        )
        self.assertTrue(all(a <= b for a, b in zip(policy.budgets, policy.budgets[1:])))
        for lid in np.linspace(1, 40, 100):
            decision = policy.choose(float(lid), True)
            self.assertIn(decision.budget, grid)
            self.assertGreaterEqual(decision.budget, 12)
        self.assertEqual(policy.choose(5.0, False).budget, 48)
        self.assertEqual(FixedBudgetPolicy(20, grid, 12).choose(999.0).budget, 20)

    def test_policy_fingerprint_canonicalizes_cross_platform_float_noise(self):
        common = {
            "budgets": [32, 32, 32, 48],
            "grid": [12, 20, 32, 48, 80],
            "fallback_budget": 80,
            "target": 0.9,
        }
        local = MonotoneBinnedPolicy(
            edges=[5.332204714742856, 7.367760559001278, 9.311463422877734],
            **common,
        )
        cluster = MonotoneBinnedPolicy(
            edges=[5.332204714742856, 7.36776055900128, 9.31146342287773],
            **common,
        )
        self.assertEqual(local.serialize(), cluster.serialize())
        self.assertEqual(local.edges.tolist(), cluster.edges.tolist())


if __name__ == "__main__":
    unittest.main()
