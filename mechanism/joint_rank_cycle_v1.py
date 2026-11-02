"""Exact certificates for the approved, deterministic joint-rank T1 study.

No projection sampler, probability estimator, numerical integration, dependency
on the retrieval system, or floating-point decision tolerance is provided.
See docs/JOINT_RANK_CYCLE_T1_PROTOCOL_V1.md for the finite validation scope.
"""
from __future__ import annotations

from fractions import Fraction
from math import factorial


def _fraction(value):
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("exact inputs must be int or Fraction, not floats/bools")
    return Fraction(value)


def _matrix(value):
    rows = tuple(tuple(_fraction(x) for x in row) for row in value)
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError("a nonempty square matrix is required")
    if any(rows[i][j] != rows[j][i]
           for i in range(len(rows)) for j in range(i)):
        raise ValueError("Gram matrix must be symmetric")
    return rows


def _minor(matrix, indices):
    return tuple(tuple(matrix[i][j] for j in indices) for i in indices)


def _determinant(matrix):
    rows = [list(row) for row in matrix]
    result = Fraction(1)
    for col in range(len(rows)):
        pivot = next((i for i in range(col, len(rows)) if rows[i][col]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            rows[col], rows[pivot] = rows[pivot], rows[col]
            result = -result
        diagonal = rows[col][col]
        result *= diagonal
        for i in range(col + 1, len(rows)):
            ratio = rows[i][col] / diagonal
            for j in range(col + 1, len(rows)):
                rows[i][j] -= ratio * rows[col][j]
    return result


def _principal_minors(matrix):
    size = len(matrix)
    return {mask: _determinant(_minor(matrix, tuple(
        i for i in range(size) if mask & (1 << i))))
        for mask in range(1 << size)}


def _solve(matrix, rhs):
    """Return exact rank and one solution, or None for an inconsistent RHS."""
    rows = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    pivots = []
    for col in range(len(matrix)):
        pivot = next((i for i in range(len(pivots), len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        row_index = len(pivots)
        rows[row_index], rows[pivot] = rows[pivot], rows[row_index]
        diagonal = rows[row_index][col]
        rows[row_index] = [x / diagonal for x in rows[row_index]]
        for i in range(len(rows)):
            if i != row_index:
                ratio = rows[i][col]
                rows[i] = [x - ratio * y for x, y in zip(rows[i], rows[row_index])]
        pivots.append(col)
    if any(not any(row[:-1]) and row[-1] for row in rows):
        return len(pivots), None
    solution = [Fraction(0)] * len(matrix)
    for i, col in enumerate(pivots):
        solution[col] = rows[i][-1]
    return len(pivots), tuple(solution)


def _cycle_domain(n, r):
    if isinstance(n, bool) or not isinstance(n, int) or n not in (3, 4):
        raise ValueError("the frozen T1 implementation supports n=3 or n=4")
    r = _fraction(r)
    if not 0 <= r < Fraction(1, 2):
        raise ValueError("the certified cycle family requires 0 <= r < 1/2")
    return r


def signed_cycle_gram(n, r, *, cycle_sign=1):
    """Residual Gram R: cycle edges have magnitude r; only (0,n-1) can be negative."""
    r = _cycle_domain(n, r)
    if (isinstance(cycle_sign, bool) or not isinstance(cycle_sign, int)
            or cycle_sign not in (-1, 1)):
        raise ValueError("cycle_sign must be +1 or -1")
    matrix = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for i in range(n - 1):
        matrix[i][i + 1] = matrix[i + 1][i] = r
    matrix[0][n - 1] = matrix[n - 1][0] = cycle_sign * r
    return tuple(tuple(row) for row in matrix)


def cycle_determinant_certificate(n, r):
    """Coefficients of det(I+R diag(s)) and proper-subset sign witnesses.

    Bitmask mask indexes the monomial product(s_j for j in mask). The coefficient
    is the corresponding principal minor. All coefficients are exact Fractions.
    """
    r = _cycle_domain(n, r)
    plus = signed_cycle_gram(n, r, cycle_sign=1)
    minus = signed_cycle_gram(n, r, cycle_sign=-1)
    p, q = _principal_minors(plus), _principal_minors(minus)
    differences = {mask: p[mask] - q[mask] for mask in p}
    expected = 4 * (-1) ** (n - 1) * r ** n
    full_mask = (1 << n) - 1
    signatures = {}
    for mask in range(full_mask):
        indices = tuple(i for i in range(n) if mask & (1 << i))
        signs = {}
        for root in indices:
            if root in signs:
                continue
            signs[root] = 1
            pending = [root]
            while pending:
                i = pending.pop()
                for j in indices:
                    if i == j or not plus[i][j]:
                        continue
                    proposed = signs[i] * minus[i][j] / plus[i][j]
                    if j in signs and signs[j] != proposed:
                        raise ArithmeticError("inconsistent proper-subset signature")
                    if j not in signs:
                        signs[j] = int(proposed)
                        pending.append(j)
        if any(minus[i][j] != signs[i] * plus[i][j] * signs[j]
               for i in indices for j in indices):
            raise ArithmeticError("signature does not reproduce the full submatrix")
        signatures[mask] = tuple(signs[i] for i in indices)
    if (differences[full_mask] != expected
            or any(value for mask, value in differences.items() if mask != full_mask)):
        raise ArithmeticError("cycle determinant certificate failed")
    return {"n": n, "r": r, "plus_coefficients": p, "minus_coefficients": q,
            "differences": differences, "full_coefficient": expected,
            "proper_subset_signatures": signatures,
            "theta_minus_minus_theta_plus_sign": 0 if r == 0 else (-1) ** (n - 1)}


def validate_gram_and_lift(gram, *, scale_squared=1, ambient_dimension=None):
    """Validate PSD/strict target and certify a unit-sphere lift exactly.

    The Gram columns are the target displacement followed by competitors. A
    singular PSD Gram is classified explicitly; an inconsistent lift is returned
    as infeasible. Indefinite matrices and invalid domains raise ValueError.
    The point Gram is for [q, q+epsilon*v_0, ...], epsilon^2=scale_squared.
    No coordinates with approximate square roots are used to decide feasibility.
    """
    gram = _matrix(gram)
    if not 2 <= len(gram) <= 5:
        raise ValueError("the frozen T1 fixtures have 2 to 5 displacement columns")
    principal_minors = _principal_minors(gram)
    if any(x < 0 for x in principal_minors.values()):
        raise ValueError("Gram is not positive semidefinite")
    if gram[0][0] <= 0 or any(gram[i][i] <= gram[0][0] for i in range(1, len(gram))):
        raise ValueError("target must be nonzero and strictly nearer than every competitor")
    scale_squared = _fraction(scale_squared)
    if scale_squared <= 0:
        raise ValueError("common scale squared must be positive")
    c = tuple(-gram[i][i] / 2 for i in range(len(gram)))
    rank, coefficients = _solve(gram, c)
    if ambient_dimension is None:
        ambient_dimension = rank + 1
    if (isinstance(ambient_dimension, bool) or not isinstance(ambient_dimension, int)
            or ambient_dimension < rank):
        raise ValueError("ambient dimension must be an integer at least the Gram rank")
    norm_squared = (None if coefficients is None else
                    scale_squared * sum(x * y for x, y in zip(c, coefficients)))
    if coefficients is None:
        reason = "c_outside_gram_range"
    elif norm_squared > 1:
        reason = "minimum_query_norm_exceeds_one"
    elif norm_squared < 1 and ambient_dimension == rank:
        reason = "no_orthogonal_dimension_to_complete_query"
    else:
        reason = "feasible"
    point_gram = None
    if reason == "feasible":
        point_gram = tuple(tuple(
            Fraction(1) if i == 0 and j == 0 else
            1 + scale_squared * c[j - 1] if i == 0 else
            1 + scale_squared * c[i - 1] if j == 0 else
            1 + scale_squared * (c[i - 1] + c[j - 1] + gram[i - 1][j - 1])
            for j in range(len(gram) + 1)) for i in range(len(gram) + 1))
        if (any(value < 0 for value in _principal_minors(point_gram).values())
                or any(point_gram[i][i] != 1 for i in range(len(point_gram)))):
            raise ArithmeticError("unit-sphere point Gram certificate failed")
    return {"rank": rank, "singular": rank < len(gram), "psd": True,
            "principal_minors": principal_minors, "range_consistent": coefficients is not None,
            "query_coefficients_before_scaling": coefficients,
            "scaled_minimum_query_norm_squared": norm_squared,
            "scale_squared": scale_squared, "ambient_dimension": ambient_dimension,
            "lift_feasible": reason == "feasible", "lift_reason": reason,
            "point_gram": point_gram}


def rank_tail_gap_lower_bound(n, beta, r):
    """Exact analytic lower bound for |theta_- - theta_+|, m=2 and M=n only."""
    r = _cycle_domain(n, r)
    beta = _fraction(beta)
    if beta <= 1:
        raise ValueError("beta must be strictly greater than one")
    ell = 1 - 2 * r
    return 4 * factorial(n) * beta * r ** n / (beta + n / ell) ** (n + 1)


def rank_count_from_squared_norms(target_squared, candidate_squared):
    """Return (Z, 1+Z); target ties are refused, competitor ties are allowed."""
    target_squared = _fraction(target_squared)
    candidate_squared = tuple(_fraction(x) for x in candidate_squared)
    if target_squared < 0 or any(x < 0 for x in candidate_squared):
        raise ValueError("squared norms must be nonnegative")
    if any(x == target_squared for x in candidate_squared):
        raise ValueError("target-distance tie is outside the strict rank event")
    z = sum(x < target_squared for x in candidate_squared)
    return z, 1 + z
