from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
        feature_version: str = "pilot_rerank_lid_v1",
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
            feature_version=feature_version,
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

    VERSION = 2

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
            if budget == self.corpus_size:
                # Retrieving the complete corpus has deterministic unit retention.
                # Do not subtract a tune-fit correction from this exact boundary.
                corrected = 1.0
            else:
                corrected = max(0.0, raw[budget] - self.safety_correction)
                # Special-function tails can round to exactly one at a finite
                # budget.  That numerical saturation is not an exact guarantee.
                if self.target == 1.0:
                    continue
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
        corrected = (
            1.0
            if maximum == self.corpus_size
            else max(0.0, raw[maximum] - self.safety_correction)
        )
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


def _positive_float64_bits(value: float, name: str) -> int:
    scalar = np.float64(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return int(np.asarray(scalar).view(np.uint64))


def _positive_float64_from_bits(bits: int) -> float:
    return float(np.asarray(np.uint64(bits)).view(np.float64))


def _decision_state(decision: PolicyDecision) -> Tuple[int, bool]:
    if decision.used_fallback:
        raise ValueError("cannot compile a fallback reference decision")
    return int(decision.budget), bool(decision.saturated)


class CompiledTriPredictPolicy:
    """Frozen float64 LID intervals that reproduce a Tri-Predict decision.

    Compilation is deliberately bounded by the configured LID clipping interval.
    It searches adjacent representable positive float64 values, rather than
    sampling a regular approximation grid. Runtime inference performs only an
    interval lookup and never evaluates the analytic retention model.
    """

    VERSION = 1

    def __init__(
        self,
        *,
        reference_policy_fingerprint: str,
        feature_version: str,
        grid: Sequence[int],
        lid_min: float,
        lid_max: float,
        upper_lids: Sequence[float],
        states: Sequence[Tuple[int, bool]],
        validation_samples: int,
        validation_points: int,
    ):
        self.reference_policy_fingerprint = str(reference_policy_fingerprint)
        self.feature_version = str(feature_version)
        self.grid = tuple(int(value) for value in grid)
        self.lid_min = float(np.float64(lid_min))
        self.lid_max = float(np.float64(lid_max))
        self.upper_lids = np.asarray(upper_lids, dtype=np.float64)
        self.states = tuple((int(budget), bool(saturated)) for budget, saturated in states)
        self.validation_samples = int(validation_samples)
        self.validation_points = int(validation_points)
        if not self.reference_policy_fingerprint:
            raise ValueError("reference_policy_fingerprint must be nonempty")
        if not self.grid or list(self.grid) != sorted(set(self.grid)):
            raise ValueError("compiled policy grid must be strictly increasing")
        _positive_float64_bits(self.lid_min, "lid_min")
        _positive_float64_bits(self.lid_max, "lid_max")
        if self.lid_min >= self.lid_max:
            raise ValueError("lid_min must be below lid_max")
        if len(self.states) != len(self.upper_lids) + 1:
            raise ValueError("one state is required for every compiled LID interval")
        if not self.states:
            raise ValueError("compiled policy must contain at least one state")
        if len(self.upper_lids) and np.any(np.diff(self.upper_lids) <= 0.0):
            raise ValueError("compiled LID upper bounds must be strictly increasing")
        if len(self.upper_lids) and (
            self.upper_lids[0] < self.lid_min or self.upper_lids[-1] >= self.lid_max
        ):
            raise ValueError("compiled LID upper bounds must lie inside the domain")
        state_orders = []
        for budget, saturated in self.states:
            if budget not in self.grid:
                raise ValueError("compiled budgets must come from the reference grid")
            if saturated and budget != self.grid[-1]:
                raise ValueError("only the maximum compiled budget may be saturated")
            state_orders.append(self.grid.index(budget) * 2 + int(saturated))
        if any(left >= right for left, right in zip(state_orders, state_orders[1:])):
            raise ValueError("compiled decision states must be strictly monotone")
        if self.validation_samples < 2 or self.validation_points < 2:
            raise ValueError("compiled validation counts must be at least two")

    @classmethod
    def compile(
        cls,
        reference: TriPredictPolicy,
        *,
        lid_min: float,
        lid_max: float,
        validation_samples: int = 65,
    ) -> "CompiledTriPredictPolicy":
        sample_count = int(validation_samples)
        if sample_count < 2:
            raise ValueError("validation_samples must be at least two")
        lower_bits = _positive_float64_bits(lid_min, "lid_min")
        upper_bits = _positive_float64_bits(lid_max, "lid_max")
        if lower_bits >= upper_bits:
            raise ValueError("lid_min must be below lid_max")
        grid = tuple(reference.grid)

        def state_at(bits: int) -> Tuple[int, bool]:
            value = _positive_float64_from_bits(bits)
            return _decision_state(reference.choose(value, True))

        def order(state: Tuple[int, bool]) -> int:
            budget, saturated = state
            if budget not in grid or (saturated and budget != grid[-1]):
                raise FloatingPointError("reference emitted an invalid decision state")
            return grid.index(budget) * 2 + int(saturated)

        current_bits = lower_bits
        current_state = state_at(current_bits)
        final_state = state_at(upper_bits)
        if order(final_state) < order(current_state):
            raise FloatingPointError("reference decisions decrease across the LID domain")
        states = [current_state]
        upper_lids: List[float] = []
        while current_state != final_state:
            left = current_bits
            right = upper_bits
            current_order = order(current_state)
            while right - left > 1:
                middle = (left + right) // 2
                middle_state = state_at(middle)
                middle_order = order(middle_state)
                if middle_order < current_order:
                    raise FloatingPointError(
                        "reference decisions are not monotone during compilation"
                    )
                if middle_state == current_state:
                    left = middle
                else:
                    right = middle
            next_state = state_at(right)
            if order(next_state) <= current_order:
                raise FloatingPointError("failed to locate a strict decision transition")
            upper_lids.append(_positive_float64_from_bits(left))
            states.append(next_state)
            current_bits = right
            current_state = next_state

        linear = np.linspace(
            _positive_float64_from_bits(lower_bits),
            _positive_float64_from_bits(upper_bits),
            sample_count,
            dtype=np.float64,
        )
        geometric = np.geomspace(
            _positive_float64_from_bits(lower_bits),
            _positive_float64_from_bits(upper_bits),
            sample_count,
            dtype=np.float64,
        )
        validation_values = list(linear) + list(geometric)
        for upper_lid in upper_lids:
            validation_values.append(np.float64(upper_lid))
            validation_values.append(
                np.nextafter(np.float64(upper_lid), np.float64(np.inf))
            )
        unique_validation = sorted(set(float(value) for value in validation_values))
        compiled = cls(
            reference_policy_fingerprint=reference.serialize()["fingerprint"],
            feature_version=reference.feature_version,
            grid=grid,
            lid_min=_positive_float64_from_bits(lower_bits),
            lid_max=_positive_float64_from_bits(upper_bits),
            upper_lids=upper_lids,
            states=states,
            validation_samples=sample_count,
            validation_points=len(unique_validation),
        )
        compiled.assert_equivalent(reference, unique_validation)
        return compiled

    @classmethod
    def from_serialized(
        cls,
        artifact: Mapping[str, Any],
        *,
        expected_reference_policy_fingerprint: Optional[str] = None,
    ) -> "CompiledTriPredictPolicy":
        raw = dict(artifact)
        stored_fingerprint = raw.pop("fingerprint", None)
        if not isinstance(stored_fingerprint, str) or not stored_fingerprint:
            raise ValueError("compiled policy artifact is missing its fingerprint")
        if fingerprint(raw) != stored_fingerprint:
            raise ValueError("compiled policy artifact fingerprint mismatch")
        if raw.get("name") != "compiled_tri_predict_lid_boundaries":
            raise ValueError("unexpected compiled policy artifact name")
        if raw.get("version") != cls.VERSION:
            raise ValueError("unsupported compiled policy artifact version")
        if raw.get("transition_search") != "adjacent_positive_float64_bisection":
            raise ValueError("unsupported compiled transition search")
        reference_fingerprint = raw.get("reference_policy_fingerprint")
        if not isinstance(reference_fingerprint, str) or not reference_fingerprint:
            raise ValueError("compiled artifact has no analytic policy fingerprint")
        if (
            expected_reference_policy_fingerprint is not None
            and reference_fingerprint != expected_reference_policy_fingerprint
        ):
            raise ValueError("compiled artifact references a different analytic policy")
        domain = raw.get("lid_domain")
        validation = raw.get("compile_validation")
        states = raw.get("states")
        numeric_upper = raw.get("upper_lids")
        hexadecimal_upper = raw.get("upper_lids_hex")
        if not isinstance(domain, Mapping) or not isinstance(validation, Mapping):
            raise ValueError("compiled artifact is missing domain or validation metadata")
        if domain.get("input_dtype") != "float64" or domain.get(
            "clipping_required"
        ) is not True:
            raise ValueError("compiled artifact requires clipped float64 LID input")
        if validation.get("mismatches") != 0 or validation.get(
            "boundary_adjacent_values_included"
        ) is not True:
            raise ValueError("compiled artifact did not pass boundary validation")
        if not isinstance(states, list) or not isinstance(numeric_upper, list):
            raise ValueError("compiled artifact states and upper_lids must be lists")
        if not isinstance(hexadecimal_upper, list) or len(hexadecimal_upper) != len(
            numeric_upper
        ):
            raise ValueError("compiled artifact hexadecimal boundaries are invalid")
        parsed_upper = []
        for numeric, hexadecimal in zip(numeric_upper, hexadecimal_upper):
            if not isinstance(hexadecimal, str):
                raise ValueError("compiled hexadecimal LID boundary must be a string")
            parsed = float.fromhex(hexadecimal)
            if _positive_float64_bits(parsed, "upper_lid") != _positive_float64_bits(
                float(numeric), "upper_lid"
            ):
                raise ValueError("numeric and hexadecimal LID boundaries disagree")
            parsed_upper.append(parsed)
        if validation.get("linear_samples") != validation.get("geometric_samples"):
            raise ValueError("compiled validation sample counts disagree")
        parsed_states = []
        for state in states:
            if not isinstance(state, Mapping):
                raise ValueError("compiled decision state must be an object")
            budget = state.get("budget")
            saturated = state.get("saturated")
            if isinstance(budget, bool) or not isinstance(budget, int):
                raise ValueError("compiled decision budget must be an integer")
            if not isinstance(saturated, bool):
                raise ValueError("compiled saturation flag must be boolean")
            parsed_states.append((budget, saturated))
        result = cls(
            reference_policy_fingerprint=str(reference_fingerprint),
            feature_version=str(raw["feature_version"]),
            grid=raw["grid"],
            lid_min=float(domain["minimum"]),
            lid_max=float(domain["maximum"]),
            upper_lids=parsed_upper,
            states=parsed_states,
            validation_samples=int(validation["linear_samples"]),
            validation_points=int(validation["unique_points"]),
        )
        if result.serialize()["fingerprint"] != stored_fingerprint:
            raise ValueError("compiled policy artifact is not canonically reproducible")
        return result

    def choose(self, lid_value: float, lid_valid: bool = True) -> PolicyDecision:
        value = float(lid_value)
        if (
            not lid_valid
            or not np.isfinite(value)
            or value < self.lid_min
            or value > self.lid_max
        ):
            return PolicyDecision(self.grid[-1], -1, True, False, None, None)
        interval = int(np.searchsorted(self.upper_lids, value, side="left"))
        budget, saturated = self.states[interval]
        return PolicyDecision(budget, -1, False, saturated, None, None)

    def assert_equivalent(
        self, reference: TriPredictPolicy, lid_values: Iterable[float]
    ) -> int:
        if reference.serialize()["fingerprint"] != self.reference_policy_fingerprint:
            raise ValueError("compiled policy does not match the supplied reference")
        count = 0
        for lid_value in lid_values:
            reference_decision = reference.choose(float(lid_value), True)
            compiled_decision = self.choose(float(lid_value), True)
            reference_state = _decision_state(reference_decision)
            compiled_state = _decision_state(compiled_decision)
            if reference_state != compiled_state:
                raise AssertionError(
                    "compiled/reference mismatch at "
                    f"lid={float(lid_value).hex()}: "
                    f"compiled={compiled_state}, reference={reference_state}"
                )
            count += 1
        return count

    def serialize(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "name": "compiled_tri_predict_lid_boundaries",
            "version": self.VERSION,
            "feature_version": self.feature_version,
            "reference_policy_fingerprint": self.reference_policy_fingerprint,
            "grid": list(self.grid),
            "lid_domain": {
                "minimum": self.lid_min,
                "maximum": self.lid_max,
                "input_dtype": "float64",
                "clipping_required": True,
            },
            "transition_search": "adjacent_positive_float64_bisection",
            "upper_lids": self.upper_lids.tolist(),
            "upper_lids_hex": [float(value).hex() for value in self.upper_lids],
            "states": [
                {"budget": budget, "saturated": saturated}
                for budget, saturated in self.states
            ],
            "compile_validation": {
                "linear_samples": self.validation_samples,
                "geometric_samples": self.validation_samples,
                "unique_points": self.validation_points,
                "boundary_adjacent_values_included": True,
                "mismatches": 0,
            },
            "runtime_prediction_fields": "omitted; decision-only serving artifact",
        }
        result["fingerprint"] = fingerprint(result)
        return result
