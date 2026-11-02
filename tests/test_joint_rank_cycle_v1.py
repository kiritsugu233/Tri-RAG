"""Deterministic scientific certificates; no sampled matrices or probability estimates."""
from fractions import Fraction as F
from itertools import permutations, product
import unittest

from mechanism.joint_rank_cycle_v1 import (
    cycle_determinant_certificate,
    rank_count_from_squared_norms,
    rank_tail_gap_lower_bound,
    signed_cycle_gram,
    validate_gram_and_lift,
)


def leibniz_determinant(matrix):
    """Independent reference: permutation expansion, not Gaussian elimination."""
    total = F(0)
    for order in permutations(range(len(matrix))):
        inversions = sum(order[i] > order[j]
                         for i in range(len(order)) for j in range(i + 1, len(order)))
        term = F((-1) ** inversions)
        for i, j in enumerate(order):
            term *= matrix[i][j]
        total += term
    return total


def target_gram(residual, beta=F(2)):
    return ((F(1),) + (F(0),) * len(residual),) + tuple(
        (F(0),) + tuple(beta * x for x in row) for row in residual)


class JointRankCycleTests(unittest.TestCase):
    def test_full_determinants_and_cycle_difference_have_exact_fixtures(self):
        for n, expected_plus, expected_minus, difference in (
                (3, F(27, 32), F(25, 32), F(1, 16)),
                (4, F(3, 4), F(49, 64), -F(1, 64))):
            with self.subTest(n=n):
                certificate = cycle_determinant_certificate(n, F(1, 4))
                full = (1 << n) - 1
                self.assertEqual(certificate['plus_coefficients'][full], expected_plus)
                self.assertEqual(certificate['minus_coefficients'][full], expected_minus)
                self.assertEqual(certificate['full_coefficient'], difference)
                self.assertTrue(all(value == 0 for mask, value in
                                    certificate['differences'].items() if mask != full))

    def test_multivariate_polynomial_matches_independent_permutation_expansion(self):
        for n, r in product((3, 4), (F(0), F(1, 4))):
            certificate = cycle_determinant_certificate(n, r)
            for sign in (-1, 1):
                residual = signed_cycle_gram(n, r, cycle_sign=sign)
                coefficients = certificate['plus_coefficients' if sign == 1 else 'minus_coefficients']
                for s in product((0, 1), repeat=n):
                    matrix = tuple(tuple(F(i == j) + residual[i][j] * s[j]
                                         for j in range(n)) for i in range(n))
                    polynomial = F(0)
                    for mask, coefficient in coefficients.items():
                        term = coefficient
                        for j in range(n):
                            if mask & (1 << j):
                                term *= s[j]
                        polynomial += term
                    self.assertEqual(polynomial, leibniz_determinant(matrix))

    def test_all_proper_subsets_have_actual_signature_equivalence(self):
        for n, r in product((3, 4), (F(0), F(1, 4))):
            plus = signed_cycle_gram(n, r)
            minus = signed_cycle_gram(n, r, cycle_sign=-1)
            witnesses = cycle_determinant_certificate(n, r)['proper_subset_signatures']
            self.assertEqual(set(witnesses), set(range((1 << n) - 1)))
            for mask, signs in witnesses.items():
                ids = [j for j in range(n) if mask & (1 << j)]
                for i, u in enumerate(ids):
                    for j, v in enumerate(ids):
                        self.assertEqual(minus[u][v], signs[i] * plus[u][v] * signs[j])
            whole_graph_equivalent = any(
                all(minus[i][j] == signs[i] * plus[i][j] * signs[j]
                    for i in range(n) for j in range(n))
                for signs in product((-1, 1), repeat=n))
            self.assertEqual(whole_graph_equivalent, r == 0)

    def test_gap_lower_bounds_are_exact_and_zero_correlation_collapses(self):
        self.assertEqual(rank_tail_gap_lower_bound(3, 2, F(1, 4)), F(3, 16384))
        self.assertEqual(rank_tail_gap_lower_bound(4, 2, F(1, 4)), F(3, 400000))
        for n in (3, 4):
            self.assertEqual(rank_tail_gap_lower_bound(n, 2, 0), 0)
            certificate = cycle_determinant_certificate(n, 0)
            self.assertEqual(certificate['plus_coefficients'], certificate['minus_coefficients'])
            self.assertEqual(certificate['theta_minus_minus_theta_plus_sign'], 0)

    def test_common_scale_unit_sphere_gram_preserves_every_displacement_product(self):
        for n, r in product((3, 4), (F(0), F(1, 4))):
            scale = 1 / (1 + 2 * n / (1 - 2 * r))
            for sign in (-1, 1):
                gram = target_gram(signed_cycle_gram(n, r, cycle_sign=sign))
                witness = validate_gram_and_lift(gram, scale_squared=scale, ambient_dimension=n + 2)
                self.assertTrue(witness['lift_feasible'])
                self.assertEqual(witness['rank'], n + 1)
                self.assertFalse(witness['singular'])
                self.assertLessEqual(witness['scaled_minimum_query_norm_squared'], F(1, 4))
                points = witness['point_gram']
                self.assertTrue(all(points[i][i] == 1 for i in range(len(points))))
                for i in range(n + 1):
                    for j in range(n + 1):
                        actual = points[i + 1][j + 1] - points[i + 1][0] - points[0][j + 1] + points[0][0]
                        self.assertEqual(actual, scale * gram[i][j])

    def test_old_independent_example_needs_scaling_and_extra_dimension(self):
        gram = ((1, 0, 0), (0, 2, 0), (0, 0, 2))
        raw = validate_gram_and_lift(gram)
        self.assertEqual(raw['scaled_minimum_query_norm_squared'], F(5, 4))
        self.assertEqual(raw['lift_reason'], 'minimum_query_norm_exceeds_one')
        scaled = validate_gram_and_lift(gram, scale_squared=F(1, 2))
        self.assertTrue(scaled['lift_feasible'])
        self.assertEqual(scaled['scaled_minimum_query_norm_squared'], F(5, 8))
        self.assertEqual(validate_gram_and_lift(gram, scale_squared=F(1, 2),
                                              ambient_dimension=3)['lift_reason'],
                         'no_orthogonal_dimension_to_complete_query')

    def test_antipodal_old_example_cannot_be_lifted_by_any_common_scale(self):
        gram = ((1, 0, 0), (0, 2, -2), (0, -2, 2))
        for scale in (F(1), F(1, 2)):
            witness = validate_gram_and_lift(gram, scale_squared=scale)
            self.assertTrue(witness['singular'])
            self.assertEqual(witness['rank'], 2)
            self.assertFalse(witness['range_consistent'])
            self.assertEqual(witness['lift_reason'], 'c_outside_gram_range')
            self.assertIsNone(witness['point_gram'])

    def test_consistent_singular_geometry_is_classified_without_jitter(self):
        witness = validate_gram_and_lift(((1, 0, 0), (0, 2, 2), (0, 2, 2)))
        self.assertTrue(witness['singular'])
        self.assertEqual(witness['rank'], 2)
        self.assertTrue(witness['range_consistent'])
        self.assertEqual(witness['scaled_minimum_query_norm_squared'], F(3, 4))
        self.assertTrue(witness['lift_feasible'])
        self.assertEqual(witness['principal_minors'][7], 0)

    def test_query_norm_one_works_without_an_orthogonal_complement(self):
        witness = validate_gram_and_lift(((1, 0), (0, 3)), ambient_dimension=2)
        self.assertEqual(witness['scaled_minimum_query_norm_squared'], 1)
        self.assertTrue(witness['lift_feasible'])

    def test_invalid_gram_and_domain_inputs_are_refused(self):
        for gram in (((1, 2), (2, 2)), ((1, 0), (1, 2)), ((0, 0), (0, 2)),
                     ((1, 0), (0, 1)), ((1, 0, 0), (0, 2, 0))):
            with self.subTest(gram=gram), self.assertRaises(ValueError):
                validate_gram_and_lift(gram)
        for r in (-F(1, 4), F(1, 2)):
            with self.assertRaises(ValueError):
                signed_cycle_gram(3, r)
        for n in (1, 2, 5, True):
            with self.assertRaises(ValueError):
                signed_cycle_gram(n, F(1, 4))
        for beta in (0, 1):
            with self.assertRaises(ValueError):
                rank_tail_gap_lower_bound(3, beta, F(1, 4))
        with self.assertRaises(ValueError):
            validate_gram_and_lift(((1, 0), (0, 2)), ambient_dimension=1)
        with self.assertRaises(ValueError):
            validate_gram_and_lift(((1, 0), (0, 2)), scale_squared=0)
        with self.assertRaises(ValueError):
            signed_cycle_gram(3, F(1, 4), cycle_sign=0)

    def test_approximate_inputs_cannot_silently_enter_exact_certificates(self):
        for value in (0.25, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(TypeError):
                signed_cycle_gram(3, value)
        with self.assertRaises(TypeError):
            validate_gram_and_lift(((1.0, 0), (0, 2)))
        with self.assertRaises(ValueError):
            signed_cycle_gram(3, F(1, 4), cycle_sign=1.0)

    def test_rank_event_matches_direct_order_and_common_scaling(self):
        for scores in permutations((F(0), F(1), F(3), F(4))):
            z, rank = rank_count_from_squared_norms(2, scores)
            order = sorted([(F(2), 'target')] + [(x, str(j)) for j, x in enumerate(scores)])
            expected_rank = next(i + 1 for i, (_, name) in enumerate(order) if name == 'target')
            self.assertEqual(rank, expected_rank)
            for budget in range(1, 5):
                self.assertEqual(z >= budget, rank > budget)
            self.assertEqual((z, rank), rank_count_from_squared_norms(
                F(2, 13), [x / 13 for x in scores]))
        self.assertEqual(rank_count_from_squared_norms(1, (0, 0, 2, 2)), (2, 3))

    def test_target_ties_and_negative_squared_norms_are_refused(self):
        for target, scores in ((1, (1, 2)), (0, (0, 1)), (-1, (2,)), (1, (-1,))):
            with self.subTest(target=target, scores=scores), self.assertRaises(ValueError):
                rank_count_from_squared_norms(target, scores)

    def test_old_analytic_probabilities_and_collinear_zero_fixture(self):
        # U~Exp(1); integrating exp(-rate*U) gives 1/(1+rate).
        one = 1 - 1 / (1 + F(1, 2))
        both_independent_residuals = 1 - 2 / (1 + F(1, 2)) + 1 / (1 + F(1))
        self.assertEqual(one, F(1, 3))
        self.assertEqual(both_independent_residuals, F(1, 6))
        for t in (F(1, 2), F(1), F(2)):
            self.assertEqual(rank_count_from_squared_norms(t, (4 * t,)), (0, 1))
        collinear = validate_gram_and_lift(((1, 2), (2, 4)))
        self.assertEqual(collinear['rank'], 1)
        self.assertTrue(collinear['singular'])


if __name__ == '__main__':
    unittest.main()
