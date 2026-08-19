"""Tri-Predict mean-field approximation built from the orthogonal conditional law.

This module is intentionally separate from :mod:`tri_law`: the latter is an exact
single-triplet result, while this module adds an LID rank-distance model,
orthogonality, conditional independence, a structural surrogate, and mean-field
thresholding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc


ROOT_ABSOLUTE_TOLERANCE = 1e-12
ROOT_RELATIVE_TOLERANCE = 1e-12
MONOTONICITY_TOLERANCE = 1e-10


@dataclass(frozen=True)
class RankQuadrature:
    ranks: np.ndarray
    weights: np.ndarray
    exact: bool
    population_size: int


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_problem(
    *, lid: float, m_prime: int, k_gt: int, corpus_size: int, budget: Optional[int] = None
) -> Tuple[float, int, int, int, Optional[int]]:
    lid_value = float(lid)
    if not np.isfinite(lid_value) or lid_value <= 0.0:
        raise ValueError("lid must be finite and positive")
    m_value = _positive_integer(m_prime, "m_prime")
    k_value = _positive_integer(k_gt, "k_gt")
    n_value = _positive_integer(corpus_size, "corpus_size")
    if n_value < k_value + 2:
        raise ValueError("corpus_size must leave at least one modeled competitor")
    budget_value: Optional[int] = None
    if budget is not None:
        budget_value = _positive_integer(budget, "budget")
        if budget_value < k_value or budget_value > n_value:
            raise ValueError("budget must lie in [k_gt, corpus_size]")
    return lid_value, m_value, k_value, n_value, budget_value


def deterministic_rank_quadrature(
    start: int, stop: int, max_samples: Optional[int] = None
) -> RankQuadrature:
    """Represent integer ranks ``[start, stop)`` exactly or by geometric strata."""
    start_value = _positive_integer(start, "start")
    stop_value = _positive_integer(stop, "stop")
    if stop_value <= start_value:
        raise ValueError("stop must be greater than start")
    population = stop_value - start_value
    if max_samples is None or max_samples >= population:
        ranks = np.arange(start_value, stop_value, dtype=np.float64)
        return RankQuadrature(
            ranks=ranks,
            weights=np.ones(population, dtype=np.float64),
            exact=True,
            population_size=population,
        )
    sample_count = _positive_integer(max_samples, "max_samples")
    raw_boundaries = np.geomspace(
        float(start_value), float(stop_value), num=sample_count + 1
    )
    boundaries = np.unique(
        np.concatenate(
            [
                np.asarray([start_value, stop_value], dtype=np.int64),
                np.floor(raw_boundaries).astype(np.int64),
            ]
        )
    )
    boundaries = boundaries[(boundaries >= start_value) & (boundaries <= stop_value)]
    if boundaries[0] != start_value or boundaries[-1] != stop_value:
        raise AssertionError("rank quadrature failed to cover its population")
    representatives = []
    weights = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        width = int(right - left)
        if width <= 0:
            continue
        representative = (
            float(left)
            if width == 1
            else float(np.sqrt(float(left) * float(right - 1)))
        )
        representatives.append(representative)
        weights.append(float(width))
    weight_values = np.asarray(weights, dtype=np.float64)
    if int(np.sum(weight_values)) != population:
        raise AssertionError("rank quadrature weights do not cover all competitor ranks")
    return RankQuadrature(
        ranks=np.asarray(representatives, dtype=np.float64),
        weights=weight_values,
        exact=False,
        population_size=population,
    )


def _conditional_probabilities_from_ranks(
    y: float,
    *,
    ranks: np.ndarray,
    neighbor_rank: int,
    lid: float,
    m_prime: int,
) -> np.ndarray:
    y_value = float(y)
    if np.isnan(y_value) or y_value < 0.0:
        raise ValueError("y must be nonnegative and not NaN")
    if np.isposinf(y_value):
        return np.ones_like(ranks, dtype=np.float64)
    if np.isneginf(y_value):
        raise ValueError("y must be nonnegative")
    log_beta = (2.0 / lid) * np.log(ranks / neighbor_rank)
    scaled_argument = (m_prime * y_value / 2.0) * np.exp(-log_beta)
    probabilities = gammainc(m_prime / 2.0, scaled_argument)
    return np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)


def _h_from_quadrature(
    y: float,
    *,
    neighbor_rank: int,
    lid: float,
    m_prime: int,
    quadrature: RankQuadrature,
) -> float:
    probabilities = _conditional_probabilities_from_ranks(
        y,
        ranks=quadrature.ranks,
        neighbor_rank=neighbor_rank,
        lid=lid,
        m_prime=m_prime,
    )
    return float(np.dot(quadrature.weights, probabilities))


def tri_predict_h_j(
    y: float,
    *,
    neighbor_rank: int,
    lid: float,
    m_prime: int,
    k_gt: int,
    corpus_size: int,
    max_rank_samples: Optional[int] = None,
) -> float:
    """Expected modeled non-neighbor outrank count for ambient rank ``j``."""
    lid_value, m_value, k_value, n_value, _ = _validate_problem(
        lid=lid, m_prime=m_prime, k_gt=k_gt, corpus_size=corpus_size
    )
    j_value = _positive_integer(neighbor_rank, "neighbor_rank")
    if j_value > k_value:
        raise ValueError("neighbor_rank must not exceed k_gt")
    quadrature = deterministic_rank_quadrature(
        k_value + 1, n_value, max_samples=max_rank_samples
    )
    return _h_from_quadrature(
        y,
        neighbor_rank=j_value,
        lid=lid_value,
        m_prime=m_value,
        quadrature=quadrature,
    )


def solve_y_star(
    *,
    neighbor_rank: int,
    lid: float,
    m_prime: int,
    k_gt: int,
    budget: int,
    corpus_size: int,
    max_rank_samples: Optional[int] = None,
) -> float:
    """Solve ``h_j(y) = budget - j`` with explicit paper boundary handling."""
    lid_value, m_value, k_value, n_value, budget_value = _validate_problem(
        lid=lid,
        m_prime=m_prime,
        k_gt=k_gt,
        corpus_size=corpus_size,
        budget=budget,
    )
    assert budget_value is not None
    j_value = _positive_integer(neighbor_rank, "neighbor_rank")
    if j_value > k_value:
        raise ValueError("neighbor_rank must not exceed k_gt")
    target = float(budget_value - j_value)
    competitor_count = n_value - k_value - 1
    if target >= competitor_count:
        return float("inf")
    if target == 0.0:
        return 0.0
    if target < 0.0:
        raise ValueError("budget must be at least neighbor_rank")
    quadrature = deterministic_rank_quadrature(
        k_value + 1, n_value, max_samples=max_rank_samples
    )

    def objective(y_value: float) -> float:
        return _h_from_quadrature(
            y_value,
            neighbor_rank=j_value,
            lid=lid_value,
            m_prime=m_value,
            quadrature=quadrature,
        ) - target

    upper = 1.0
    while objective(upper) < 0.0:
        upper *= 2.0
        if upper > 2.0**40:
            raise FloatingPointError("could not bracket the Tri-Predict mean-field root")
    return float(
        brentq(
            objective,
            0.0,
            upper,
            xtol=ROOT_ABSOLUTE_TOLERANCE,
            rtol=ROOT_RELATIVE_TOLERANCE,
            maxiter=200,
        )
    )


def tri_predict_retention(
    *,
    lid: float,
    m_prime: int,
    k_gt: int,
    budget: int,
    corpus_size: int,
    max_rank_samples: Optional[int] = None,
) -> float:
    """Mean-field predicted retention across true-neighbor ranks 1 through k_gt."""
    lid_value, m_value, k_value, n_value, budget_value = _validate_problem(
        lid=lid,
        m_prime=m_prime,
        k_gt=k_gt,
        corpus_size=corpus_size,
        budget=budget,
    )
    assert budget_value is not None
    probabilities = []
    for neighbor_rank in range(1, k_value + 1):
        y_star = solve_y_star(
            neighbor_rank=neighbor_rank,
            lid=lid_value,
            m_prime=m_value,
            k_gt=k_value,
            budget=budget_value,
            corpus_size=n_value,
            max_rank_samples=max_rank_samples,
        )
        if np.isposinf(y_star):
            probabilities.append(1.0)
        else:
            probabilities.append(float(gammainc(m_value / 2.0, m_value * y_star / 2.0)))
    return float(np.clip(np.mean(probabilities), 0.0, 1.0))


def tri_predict_retention_grid(
    *,
    lid: float,
    m_prime: int,
    k_gt: int,
    budgets: Sequence[int],
    corpus_size: int,
    max_rank_samples: Optional[int] = None,
) -> Dict[int, float]:
    budget_values = [_positive_integer(value, "budget") for value in budgets]
    if budget_values != sorted(set(budget_values)):
        raise ValueError("budgets must be strictly increasing")
    predictions = np.asarray(
        [
            tri_predict_retention(
                lid=lid,
                m_prime=m_prime,
                k_gt=k_gt,
                budget=budget,
                corpus_size=corpus_size,
                max_rank_samples=max_rank_samples,
            )
            for budget in budget_values
        ],
        dtype=np.float64,
    )
    if np.any(np.diff(predictions) < -MONOTONICITY_TOLERANCE):
        raise FloatingPointError("Tri-Predict retention decreased as budget increased")
    predictions = np.maximum.accumulate(predictions)
    return {budget: float(value) for budget, value in zip(budget_values, predictions)}
