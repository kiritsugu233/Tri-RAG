from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .utils import fingerprint
from .tri_predict import tri_predict_retention_grid


POLICY_FLOAT_DECIMALS = 12


def _canonical_policy_float(value: float) -> float:
    result = float(np.round(float(value), decimals=POLICY_FLOAT_DECIMALS))
    if not np.isfinite(result):
        raise ValueError("policy floating-point values must be finite")
    return result


@dataclass(frozen=True)
class PolicyDecision:
    budget: int
    bin_index: int
    used_fallback: bool
    saturated: bool = False
    predicted_retention: Optional[float] = None
    raw_predicted_retention: Optional[float] = None


class FixedBudgetPolicy:
    def __init__(self, budget: int, grid: Sequence[int], minimum_budget: int):
        if budget not in grid or budget < minimum_budget:
            raise ValueError("fixed budget is not an allowed safe budget")
        self.budget = int(budget)
        self.grid = tuple(int(value) for value in grid)

    def choose(self, lid_value: float, lid_valid: bool = True) -> PolicyDecision:
        return PolicyDecision(self.budget, -1, False)

    def serialize(self) -> Dict[str, Any]:
        return {"name": "fixed", "version": 1, "budget": self.budget, "grid": list(self.grid)}


class MonotoneBinnedPolicy:
    VERSION = 2

    def __init__(
        self,
        *,
        edges: Sequence[float],
        budgets: Sequence[int],
        grid: Sequence[int],
        fallback_budget: int,
        target: float,
        feature_version: str = "pilot_rerank_lid_v1",
    ):
        self.edges = np.asarray(
            [_canonical_policy_float(value) for value in edges], dtype=np.float64
        )
        self.budgets = tuple(int(value) for value in budgets)
        self.grid = tuple(int(value) for value in grid)
        self.fallback_budget = int(fallback_budget)
        self.target = _canonical_policy_float(target)
        self.feature_version = feature_version
        if len(self.budgets) != len(self.edges) + 1:
            raise ValueError("one budget is required for each LID bin")
        if len(self.edges) and np.any(np.diff(self.edges) <= 0.0):
            raise ValueError("canonical LID edges must be strictly increasing")
        if any(value not in self.grid for value in self.budgets):
            raise ValueError("bin budgets must come from configured grid")
        if any(left > right for left, right in zip(self.budgets, self.budgets[1:])):
            raise ValueError("budgets must be nondecreasing by LID bin")
        if self.fallback_budget not in self.grid:
            raise ValueError("fallback budget must come from configured grid")

    @classmethod
    def fit(
        cls,
        tune_records: Iterable[Mapping[str, Any]],
        *,
        grid: Sequence[int],
        n_bins: int,
        target: float,
        safety_margin: float,
        fallback_budget: int,
    ) -> "MonotoneBinnedPolicy":
        records = [record for record in tune_records if bool(record["lid_valid"])]
        if not records:
            raise ValueError("cannot fit policy without valid tune-query LID values")
        lids = np.asarray([float(record["lid"]) for record in records])
        quantiles = np.arange(1, n_bins, dtype=np.float64) / n_bins
        if len(quantiles):
            try:
                raw_edges = np.quantile(lids, quantiles, method="linear")
            except TypeError:  # NumPy < 1.22 compatibility.
                raw_edges = np.quantile(lids, quantiles, interpolation="linear")
            edges = np.unique(np.round(raw_edges, decimals=POLICY_FLOAT_DECIMALS))
        else:
            edges = np.asarray([], dtype=np.float64)
        assignments = np.searchsorted(edges, lids, side="right")
        required = target + safety_margin
        selected: List[int] = []
        for bin_index in range(len(edges) + 1):
            members = [record for record, assigned in zip(records, assignments) if assigned == bin_index]
            if not members:
                selected.append(int(fallback_budget))
                continue
            chosen = int(grid[-1])
            for budget in grid:
                mean_retention = float(
                    np.mean([float(record["retention_by_budget"][str(budget)]) for record in members])
                )
                if mean_retention >= required:
                    chosen = int(budget)
                    break
            selected.append(chosen)
        monotone = np.maximum.accumulate(np.asarray(selected, dtype=np.int64)).tolist()
        return cls(
            edges=edges.tolist(),
            budgets=monotone,
            grid=grid,
            fallback_budget=fallback_budget,
            target=required,
        )

    def choose(self, lid_value: float, lid_valid: bool = True) -> PolicyDecision:
        if not lid_valid or not np.isfinite(lid_value):
            return PolicyDecision(
                self.fallback_budget,
                -1,
                True,
                self.fallback_budget == self.grid[-1],
            )
        bin_index = int(np.searchsorted(self.edges, lid_value, side="right"))
        budget = self.budgets[bin_index]
        return PolicyDecision(budget, bin_index, False, budget == self.grid[-1])

    def serialize(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "name": "monotone_binned_empirical",
            "version": self.VERSION,
            "feature_version": self.feature_version,
            "edges": self.edges.tolist(),
            "budgets": list(self.budgets),
            "grid": list(self.grid),
            "fallback_budget": self.fallback_budget,
            "tune_required_mean_retention": self.target,
        }
        result["fingerprint"] = fingerprint(result)
        return result


class TriPredictPolicy:
    """Query-local analytic policy using only deployable LID at inference time."""

    VERSION = 1

    def __init__(
        self,
        *,
        corpus_size: int,
        m_prime: int,
        k_gt: int,
        grid: Sequence[int],
        target: float,
        max_rank_samples: Optional[int],
        safety_correction: float = 0.0,
        safety_quantile: Optional[float] = None,
        correction_fit_observations: int = 0,
        feature_version: str = "pilot_rerank_lid_v1",
    ):
        self.corpus_size = int(corpus_size)
        self.m_prime = int(m_prime)
        self.k_gt = int(k_gt)
        self.grid = tuple(int(value) for value in grid)
        self.target = _canonical_policy_float(target)
        self.max_rank_samples = None if max_rank_samples is None else int(max_rank_samples)
        self.safety_correction = _canonical_policy_float(safety_correction)
        self.safety_quantile = (
            None if safety_quantile is None else _canonical_policy_float(safety_quantile)
        )
        self.correction_fit_observations = int(correction_fit_observations)
        self.feature_version = feature_version
        if self.corpus_size < self.k_gt + 2:
            raise ValueError("corpus_size must leave a modeled competitor")
        if self.m_prime < 1 or self.k_gt < 1:
            raise ValueError("m_prime and k_gt must be positive")
        if not self.grid or list(self.grid) != sorted(set(self.grid)):
            raise ValueError("grid must be nonempty and strictly increasing")
        if self.grid[0] < self.k_gt or self.grid[-1] > self.corpus_size:
            raise ValueError("grid budgets must lie in [k_gt, corpus_size]")
        if not 0.0 < self.target <= 1.0:
            raise ValueError("target must lie in (0,1]")
        if not 0.0 <= self.safety_correction <= 1.0:
            raise ValueError("safety_correction must lie in [0,1]")
        if self.safety_quantile is not None and not 0.0 < self.safety_quantile <= 1.0:
            raise ValueError("safety_quantile must lie in (0,1]")
        if self.max_rank_samples is not None and self.max_rank_samples < 1:
            raise ValueError("max_rank_samples must be positive when configured")

    @classmethod
    def fit(
        cls,
        tune_records: Iterable[Mapping[str, Any]],
        *,
        corpus_size: int,
        m_prime: int,
        k_gt: int,
        grid: Sequence[int],
        target: float,
        max_rank_samples: Optional[int],
        fit_safety_correction: bool,
        safety_quantile: float,
    ) -> "TriPredictPolicy":
        provisional = cls(
            corpus_size=corpus_size,
            m_prime=m_prime,
            k_gt=k_gt,
            grid=grid,
            target=target,
            max_rank_samples=max_rank_samples,
        )
        if not fit_safety_correction:
            return provisional
        residuals = []
        for record in tune_records:
            if not bool(record["lid_valid"]):
                continue
            predictions = provisional.raw_predictions(float(record["lid"]))
            realized = record["retention_by_budget"]
            for budget in provisional.grid:
                residuals.append(
                    predictions[budget] - float(realized[str(budget)])
                )
        if not residuals:
            raise ValueError("cannot fit Tri-Predict correction without valid tune records")
        try:
            quantile_value = float(
                np.quantile(residuals, safety_quantile, method="linear")
            )
        except TypeError:  # NumPy < 1.22 compatibility.
            quantile_value = float(
                np.quantile(residuals, safety_quantile, interpolation="linear")
            )
        correction = max(0.0, quantile_value)
        return cls(
            corpus_size=corpus_size,
            m_prime=m_prime,
            k_gt=k_gt,
            grid=grid,
            target=target,
            max_rank_samples=max_rank_samples,
            safety_correction=correction,
            safety_quantile=safety_quantile,
            correction_fit_observations=len(residuals),
        )

    def raw_predictions(self, lid_value: float) -> Dict[int, float]:
        return tri_predict_retention_grid(
            lid=lid_value,
            m_prime=self.m_prime,
            k_gt=self.k_gt,
            budgets=self.grid,
            corpus_size=self.corpus_size,
            max_rank_samples=self.max_rank_samples,
        )

    def choose(self, lid_value: float, lid_valid: bool = True) -> PolicyDecision:
        if not lid_valid or not np.isfinite(lid_value) or lid_value <= 0.0:
            return PolicyDecision(self.grid[-1], -1, True, False, None, None)
        raw = self.raw_predictions(float(lid_value))
        for budget in self.grid:
            corrected = max(0.0, raw[budget] - self.safety_correction)
            if corrected >= self.target:
                return PolicyDecision(
                    budget,
                    -1,
                    False,
                    False,
                    corrected,
                    raw[budget],
                )
        maximum = self.grid[-1]
        corrected = max(0.0, raw[maximum] - self.safety_correction)
        return PolicyDecision(
            maximum,
            -1,
            False,
            True,
            corrected,
            raw[maximum],
        )

    def serialize(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "name": "query_adaptive_tri_predict",
            "version": self.VERSION,
            "feature_version": self.feature_version,
            "corpus_size": self.corpus_size,
            "m_prime": self.m_prime,
            "k_gt": self.k_gt,
            "grid": list(self.grid),
            "target": self.target,
            "max_rank_samples": self.max_rank_samples,
            "rank_aggregation": "exact_or_deterministic_geometric_strata",
            "safety_correction": self.safety_correction,
            "safety_quantile": self.safety_quantile,
            "correction_fit_observations": self.correction_fit_observations,
            "correction_source_split": (
                None if self.safety_quantile is None else "query_tune"
            ),
        }
        result["fingerprint"] = fingerprint(result)
        return result
