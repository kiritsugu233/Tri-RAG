from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .utils import fingerprint


@dataclass(frozen=True)
class PolicyDecision:
    budget: int
    bin_index: int
    used_fallback: bool


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
        self.edges = np.asarray(edges, dtype=np.float64)
        self.budgets = tuple(int(value) for value in budgets)
        self.grid = tuple(int(value) for value in grid)
        self.fallback_budget = int(fallback_budget)
        self.target = float(target)
        self.feature_version = feature_version
        if len(self.budgets) != len(self.edges) + 1:
            raise ValueError("one budget is required for each LID bin")
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
            edges = np.unique(raw_edges)
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
            return PolicyDecision(self.fallback_budget, -1, True)
        bin_index = int(np.searchsorted(self.edges, lid_value, side="right"))
        return PolicyDecision(self.budgets[bin_index], bin_index, False)

    def serialize(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "name": "monotone_binned_empirical",
            "version": 1,
            "feature_version": self.feature_version,
            "edges": self.edges.tolist(),
            "budgets": list(self.budgets),
            "grid": list(self.grid),
            "fallback_budget": self.fallback_budget,
            "tune_required_mean_retention": self.target,
        }
        result["fingerprint"] = fingerprint(result)
        return result
