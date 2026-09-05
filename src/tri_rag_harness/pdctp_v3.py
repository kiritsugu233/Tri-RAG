"""Minimal Calibrated Tri-Predict v3 curve-shape repair.

This module leaves the exact Tri-Law, Raw Tri-Predict v1, and PDCTP v2
untouched.  It replaces only the v2 oracle-LID calibration target with
query-calibration-only effective Tri-LID targets for the low and high parts of
the predicted retention curve.  A scalar effective-LID mode is retained as a
one-factor ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .pdctp_features import PilotFeatureVector
from .pdctp_policies import PDCTPDecisionInput
from .tri_predict import MONOTONICITY_TOLERANCE, tri_predict_retention_grid
from .utils import fingerprint


class PDCTPV3Error(ValueError):
    """Raised when a v3 target, artifact, or decision violates its contract."""


V3_NUMERICAL_IMPLEMENTATION = "tri_predict_retention_grid_float64_v1"
V3_CALIBRATION_DECIMALS = 10


def _canonical_float(value: float) -> float:
    result = float(np.round(float(value), decimals=V3_CALIBRATION_DECIMALS))
    if not np.isfinite(result):
        raise PDCTPV3Error("v3 calibration values must be finite")
    return 0.0 if result == 0.0 else result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise PDCTPV3Error(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise PDCTPV3Error(f"{name} must be positive")
    return result


def _ordered_budgets(values: Sequence[int], corpus_size: int) -> Tuple[int, ...]:
    budgets = tuple(_positive_integer(value, "budget") for value in values)
    if not budgets or list(budgets) != sorted(set(budgets)):
        raise PDCTPV3Error("v3 budgets must be unique and strictly increasing")
    if budgets[-1] > corpus_size:
        raise PDCTPV3Error("v3 budget exceeds the corpus size")
    return budgets


def _float64_bits(value: float) -> int:
    scalar = np.float64(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise PDCTPV3Error("effective Tri-LID must be finite and positive")
    return int(np.asarray(scalar).view(np.uint64))


@dataclass(frozen=True)
class PredictionGridV3:
    """Immutable scientific values returned by the in-memory cache."""

    budgets: Tuple[int, ...]
    values: Tuple[float, ...]

    def as_dict(self) -> Dict[int, float]:
        return dict(zip(self.budgets, self.values))


class TriPredictPredictionGridCacheV3:
    """Exact float64 prediction-grid cache with no serialization surface."""

    def __init__(self) -> None:
        self._entries: Dict[Tuple[Any, ...], PredictionGridV3] = {}

    @staticmethod
    def cache_key(
        *,
        lid: float,
        m_prime: int,
        k_gt: int,
        corpus_size: int,
        budgets: Sequence[int],
        max_rank_samples: Optional[int],
    ) -> Tuple[Any, ...]:
        n_value = _positive_integer(corpus_size, "corpus_size")
        budget_values = _ordered_budgets(budgets, n_value)
        sample_value = (
            None
            if max_rank_samples is None
            else _positive_integer(max_rank_samples, "max_rank_samples")
        )
        return (
            _float64_bits(lid),
            _positive_integer(m_prime, "m_prime"),
            _positive_integer(k_gt, "k_gt"),
            n_value,
            budget_values,
            sample_value,
            V3_NUMERICAL_IMPLEMENTATION,
        )

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def prediction_grid(
        self,
        *,
        lid: float,
        m_prime: int,
        k_gt: int,
        corpus_size: int,
        budgets: Sequence[int],
        max_rank_samples: Optional[int],
        use_cache: bool = True,
    ) -> PredictionGridV3:
        key = self.cache_key(
            lid=lid,
            m_prime=m_prime,
            k_gt=k_gt,
            corpus_size=corpus_size,
            budgets=budgets,
            max_rank_samples=max_rank_samples,
        )
        if use_cache and key in self._entries:
            return self._entries[key]
        budget_values = key[4]
        computed = tri_predict_retention_grid(
            lid=float(np.float64(lid)),
            m_prime=int(m_prime),
            k_gt=int(k_gt),
            budgets=budget_values,
            corpus_size=int(corpus_size),
            max_rank_samples=max_rank_samples,
        )
        result = PredictionGridV3(
            budgets=budget_values,
            values=tuple(float(computed[budget]) for budget in budget_values),
        )
        if use_cache:
            self._entries[key] = result
        return result


@dataclass(frozen=True)
class EffectiveTriLIDFitRecordV3:
    query_id: str
    role: str
    features: PilotFeatureVector
    realized_retention: Tuple[float, ...]


@dataclass(frozen=True)
class EffectiveTriLIDPredictionV3:
    scalar_lid: float
    low_lid: float
    high_lid: float
    valid: bool
    used_fallback: bool
    failure_reason: Optional[str]


def _feature_matrix(
    features: Sequence[PilotFeatureVector],
) -> Tuple[np.ndarray, Tuple[str, ...], str]:
    if not features:
        raise PDCTPV3Error("v3 calibration requires feature records")
    names = features[0].names
    spec_fingerprint = features[0].spec_fingerprint
    if not names or len(set(names)) != len(names):
        raise PDCTPV3Error("v3 feature names must be nonempty and unique")
    for vector in features:
        if not vector.valid:
            raise PDCTPV3Error("invalid pilot features cannot fit v3")
        if vector.names != names or vector.spec_fingerprint != spec_fingerprint:
            raise PDCTPV3Error("v3 feature schemas do not align")
    matrix = np.vstack([vector.as_array() for vector in features])
    if not np.all(np.isfinite(matrix)):
        raise PDCTPV3Error("v3 feature matrix must be finite")
    return matrix, names, spec_fingerprint


def _ridge_head(
    standardized: np.ndarray,
    targets: np.ndarray,
    regularization: float,
    constrained_feature: int,
) -> Tuple[float, np.ndarray, float]:
    n, feature_count = standardized.shape
    design = np.column_stack([np.ones(n, dtype=np.float64), standardized])
    penalty = np.diag(np.concatenate([[0.0], np.full(feature_count, regularization)]))
    solution = np.linalg.pinv(design.T @ design + penalty) @ design.T @ targets
    if solution[1 + constrained_feature] < 0.0:
        keep = [index for index in range(feature_count) if index != constrained_feature]
        reduced = standardized[:, keep]
        reduced_design = np.column_stack([np.ones(n, dtype=np.float64), reduced])
        reduced_penalty = np.diag(
            np.concatenate([[0.0], np.full(len(keep), regularization)])
        )
        reduced_solution = (
            np.linalg.pinv(reduced_design.T @ reduced_design + reduced_penalty)
            @ reduced_design.T
            @ targets
        )
        coefficients = np.zeros(feature_count, dtype=np.float64)
        coefficients[keep] = reduced_solution[1:]
        intercept = float(reduced_solution[0])
    else:
        intercept = float(solution[0])
        coefficients = np.asarray(solution[1:], dtype=np.float64)
    residuals = targets - (intercept + standardized @ coefficients)
    objective = float(
        np.mean(np.square(residuals))
        + regularization * np.dot(coefficients, coefficients)
    )
    return intercept, coefficients, objective


class EffectiveTriLIDCalibratorV3:
    """Three transparent log-linear heads: scalar ablation plus low/high repair."""

    NAME = "effective_curve_shape_tri_lid_calibrator"
    SCHEMA = "calibrated_tri_predict_effective_lid_v3"
    VERSION = 3
    HEADS = ("scalar", "low", "high")

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        feature_spec_fingerprint: str,
        means: Sequence[float],
        scales: Sequence[float],
        constant_features: Sequence[bool],
        intercepts: Mapping[str, float],
        coefficients: Mapping[str, Sequence[float]],
        objectives: Mapping[str, float],
        regularization: float,
        output_min: float,
        output_max: float,
        fallback: float,
        fit_ids: Sequence[str],
        target_fingerprint: str,
        m_prime: int,
        k_gt: int,
        corpus_size: int,
        budgets: Sequence[int],
        max_rank_samples: Optional[int],
        low_budget_max: int,
        high_budget_min: int,
        target_lid_grid: Sequence[float],
    ):
        self.feature_names = tuple(str(value) for value in feature_names)
        self.feature_spec_fingerprint = str(feature_spec_fingerprint)
        self.means = np.asarray(
            [_canonical_float(value) for value in means], dtype=np.float64
        )
        self.scales = np.asarray(
            [_canonical_float(value) for value in scales], dtype=np.float64
        )
        self.constant_features = tuple(bool(value) for value in constant_features)
        self.intercepts = {
            head: _canonical_float(intercepts[head]) for head in self.HEADS
        }
        self.coefficients = {
            head: np.asarray(
                [_canonical_float(value) for value in coefficients[head]],
                dtype=np.float64,
            )
            for head in self.HEADS
        }
        self.objectives = {
            head: _canonical_float(objectives[head]) for head in self.HEADS
        }
        self.regularization = _canonical_float(regularization)
        self.output_min = _canonical_float(output_min)
        self.output_max = _canonical_float(output_max)
        self.fallback = _canonical_float(fallback)
        self.fit_ids = tuple(str(value) for value in fit_ids)
        self.target_fingerprint = str(target_fingerprint)
        self.m_prime = _positive_integer(m_prime, "m_prime")
        self.k_gt = _positive_integer(k_gt, "k_gt")
        self.corpus_size = _positive_integer(corpus_size, "corpus_size")
        self.budgets = _ordered_budgets(budgets, self.corpus_size)
        self.max_rank_samples = (
            None
            if max_rank_samples is None
            else _positive_integer(max_rank_samples, "max_rank_samples")
        )
        self.low_budget_max = int(low_budget_max)
        self.high_budget_min = int(high_budget_min)
        self.target_lid_grid = tuple(float(value) for value in target_lid_grid)
        count = len(self.feature_names)
        if (
            count == 0
            or len(set(self.feature_names)) != count
            or not self.feature_spec_fingerprint
        ):
            raise PDCTPV3Error("invalid v3 feature schema")
        if any(
            len(values) != count
            for values in (
                self.means,
                self.scales,
                self.constant_features,
                *self.coefficients.values(),
            )
        ):
            raise PDCTPV3Error("v3 calibrator arrays do not align")
        if np.any(self.scales <= 0.0):
            raise PDCTPV3Error("v3 feature scales must be positive")
        if "log_pilot_lid" not in self.feature_names:
            raise PDCTPV3Error("v3 calibrator requires log_pilot_lid")
        constrained = self.feature_names.index("log_pilot_lid")
        if any(self.coefficients[head][constrained] < 0.0 for head in self.HEADS):
            raise PDCTPV3Error("v3 log_pilot_lid coefficients must be nonnegative")
        if not self.fit_ids or len(set(self.fit_ids)) != len(self.fit_ids):
            raise PDCTPV3Error("v3 fit IDs must be nonempty and unique")
        if not self.target_fingerprint:
            raise PDCTPV3Error("v3 target fingerprint is missing")
        if self.regularization < 0.0:
            raise PDCTPV3Error("v3 regularization must be nonnegative")
        if not 0.0 < self.output_min < self.output_max:
            raise PDCTPV3Error("v3 output domain must be positive and increasing")
        if not self.output_min <= self.fallback <= self.output_max:
            raise PDCTPV3Error("v3 fallback must lie in the output domain")
        scientific = [value for value in self.budgets if value < self.corpus_size]
        low = [value for value in scientific if value <= self.low_budget_max]
        high = [value for value in scientific if value >= self.high_budget_min]
        if not low or not high or len(low) + len(high) != len(scientific):
            raise PDCTPV3Error("low/high v3 regimes must partition nonterminal budgets")
        if low[-1] >= high[0]:
            raise PDCTPV3Error("v3 low/high regimes must be ordered and disjoint")
        grid = np.asarray(self.target_lid_grid, dtype=np.float64)
        if (
            len(grid) < 2
            or not np.all(np.isfinite(grid))
            or np.any(grid <= 0.0)
            or np.any(np.diff(grid) <= 0.0)
            or grid[0] < self.output_min
            or grid[-1] > self.output_max
        ):
            raise PDCTPV3Error("invalid effective Tri-LID target grid")

    @classmethod
    def fit(
        cls,
        records: Iterable[EffectiveTriLIDFitRecordV3],
        *,
        regularization: float,
        output_min: float,
        output_max: float,
        fallback: float,
        m_prime: int,
        k_gt: int,
        corpus_size: int,
        budgets: Sequence[int],
        max_rank_samples: Optional[int],
        low_budget_max: int,
        high_budget_min: int,
        target_lid_grid: Sequence[float],
        cache: Optional[TriPredictPredictionGridCacheV3] = None,
    ) -> "EffectiveTriLIDCalibratorV3":
        rows = list(records)
        if not rows or any(row.role != "query_cal" for row in rows):
            raise PDCTPV3Error("effective Tri-LID fitting is query_cal-only")
        ids = tuple(str(row.query_id) for row in rows)
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise PDCTPV3Error("v3 fit IDs must be nonempty and unique")
        matrix, names, spec_fingerprint = _feature_matrix(
            [row.features for row in rows]
        )
        budget_values = _ordered_budgets(budgets, int(corpus_size))
        scientific_indices = np.asarray(
            [index for index, value in enumerate(budget_values) if value < corpus_size],
            dtype=np.int64,
        )
        low_indices = np.asarray(
            [
                index
                for index in scientific_indices
                if budget_values[index] <= low_budget_max
            ],
            dtype=np.int64,
        )
        high_indices = np.asarray(
            [
                index
                for index in scientific_indices
                if budget_values[index] >= high_budget_min
            ],
            dtype=np.int64,
        )
        if (
            not len(low_indices)
            or not len(high_indices)
            or len(low_indices) + len(high_indices) != len(scientific_indices)
        ):
            raise PDCTPV3Error("low/high regimes do not partition the fit budgets")
        realized = np.asarray(
            [row.realized_retention for row in rows], dtype=np.float64
        )
        if realized.shape != (len(rows), len(budget_values)):
            raise PDCTPV3Error("v3 realized curves do not align with the budget grid")
        if not np.all(np.isfinite(realized)) or np.any(
            (realized < 0.0) | (realized > 1.0)
        ):
            raise PDCTPV3Error("v3 realized retention must lie in [0,1]")
        if np.any(np.diff(realized, axis=1) < -MONOTONICITY_TOLERANCE):
            raise PDCTPV3Error("v3 realized curves must be budget-monotone")
        target_grid = tuple(float(value) for value in target_lid_grid)
        curve_cache = cache or TriPredictPredictionGridCacheV3()
        predicted = np.asarray(
            [
                curve_cache.prediction_grid(
                    lid=lid,
                    m_prime=m_prime,
                    k_gt=k_gt,
                    corpus_size=corpus_size,
                    budgets=budget_values,
                    max_rank_samples=max_rank_samples,
                ).values
                for lid in target_grid
            ],
            dtype=np.float64,
        )
        errors = np.square(realized[:, None, :] - predicted[None, :, :])
        target_indices = {
            "scalar": np.argmin(
                np.mean(errors[:, :, scientific_indices], axis=2), axis=1
            ),
            "low": np.argmin(np.mean(errors[:, :, low_indices], axis=2), axis=1),
            "high": np.argmin(np.mean(errors[:, :, high_indices], axis=2), axis=1),
        }
        targets = {
            head: np.asarray(target_grid, dtype=np.float64)[indices]
            for head, indices in target_indices.items()
        }
        means = np.mean(matrix, axis=0)
        scales = np.std(matrix, axis=0, ddof=0)
        constant = scales <= np.finfo(np.float64).eps
        scales[constant] = 1.0
        standardized = (matrix - means) / scales
        constrained = names.index("log_pilot_lid")
        intercepts: Dict[str, float] = {}
        coefficients: Dict[str, np.ndarray] = {}
        objectives: Dict[str, float] = {}
        for head in cls.HEADS:
            intercept, coefficient, objective = _ridge_head(
                standardized,
                np.log(targets[head]),
                float(regularization),
                constrained,
            )
            intercepts[head] = intercept
            coefficients[head] = coefficient
            objectives[head] = objective
        target_body = {
            "ordered_ids": list(ids),
            "budget_grid": list(budget_values),
            "target_lid_grid": list(target_grid),
            "tie_break": "lowest_effective_lid_grid_value",
            "scalar_targets": [_canonical_float(value) for value in targets["scalar"]],
            "low_targets": [_canonical_float(value) for value in targets["low"]],
            "high_targets": [_canonical_float(value) for value in targets["high"]],
        }
        return cls(
            feature_names=names,
            feature_spec_fingerprint=spec_fingerprint,
            means=means,
            scales=scales,
            constant_features=constant.tolist(),
            intercepts=intercepts,
            coefficients=coefficients,
            objectives=objectives,
            regularization=regularization,
            output_min=output_min,
            output_max=output_max,
            fallback=fallback,
            fit_ids=ids,
            target_fingerprint=fingerprint(target_body),
            m_prime=m_prime,
            k_gt=k_gt,
            corpus_size=corpus_size,
            budgets=budget_values,
            max_rank_samples=max_rank_samples,
            low_budget_max=low_budget_max,
            high_budget_min=high_budget_min,
            target_lid_grid=target_grid,
        )

    def predict(self, features: PilotFeatureVector) -> EffectiveTriLIDPredictionV3:
        if (
            not features.valid
            or features.names != self.feature_names
            or features.spec_fingerprint != self.feature_spec_fingerprint
        ):
            return EffectiveTriLIDPredictionV3(
                self.fallback,
                self.fallback,
                self.fallback,
                False,
                True,
                features.failure_reason
                if not features.valid
                else "feature_schema_mismatch",
            )
        standardized = (features.as_array() - self.means) / self.scales
        values: Dict[str, float] = {}
        for head in self.HEADS:
            log_value = self.intercepts[head] + float(
                np.dot(standardized, self.coefficients[head])
            )
            value = float(
                np.exp(
                    np.clip(log_value, np.log(self.output_min), np.log(self.output_max))
                )
            )
            if not np.isfinite(value):
                return EffectiveTriLIDPredictionV3(
                    self.fallback,
                    self.fallback,
                    self.fallback,
                    False,
                    True,
                    "nonfinite_effective_lid_prediction",
                )
            values[head] = min(
                max(_canonical_float(value), self.output_min), self.output_max
            )
        return EffectiveTriLIDPredictionV3(
            values["scalar"], values["low"], values["high"], True, False, None
        )

    def serialize(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "name": self.NAME,
            "schema": self.SCHEMA,
            "version": self.VERSION,
            "feature_schema": {
                "spec_fingerprint": self.feature_spec_fingerprint,
                "feature_names": list(self.feature_names),
            },
            "normalization": {
                "source_role": "query_cal",
                "means": self.means.tolist(),
                "scales": self.scales.tolist(),
                "constant_features": list(self.constant_features),
            },
            "model": {
                "link": "log",
                "heads": {
                    head: {
                        "intercept": self.intercepts[head],
                        "coefficients": self.coefficients[head].tolist(),
                        "objective": self.objectives[head],
                    }
                    for head in self.HEADS
                },
                "regularization": self.regularization,
                "nonnegative_coefficient": "log_pilot_lid",
                "solver": "closed_form_ridge_with_active_set_v3",
                "output_decimals": V3_CALIBRATION_DECIMALS,
            },
            "output_domain": {
                "minimum": self.output_min,
                "maximum": self.output_max,
                "fallback": self.fallback,
            },
            "target_construction": {
                "source_role": "query_cal",
                "target": "effective_tri_lid_minimum_retention_curve_mse",
                "heads": list(self.HEADS),
                "m_prime": self.m_prime,
                "k_gt": self.k_gt,
                "corpus_size": self.corpus_size,
                "budget_grid": list(self.budgets),
                "terminal_full_corpus_excluded_from_fit": True,
                "low_budget_max": self.low_budget_max,
                "high_budget_min": self.high_budget_min,
                "target_lid_grid": list(self.target_lid_grid),
                "max_rank_samples": self.max_rank_samples,
                "numerical_implementation": V3_NUMERICAL_IMPLEMENTATION,
                "target_fingerprint": self.target_fingerprint,
                "tie_break": "lowest_effective_lid_grid_value",
            },
            "fit": {
                "role": "query_cal",
                "ordered_ids": list(self.fit_ids),
                "ordered_id_hash": fingerprint(list(self.fit_ids)),
                "n": len(self.fit_ids),
            },
            "cache_contract": "in_memory_only_not_serialized_into_scientific_artifacts",
        }
        body["fingerprint"] = fingerprint(body)
        return body

    @property
    def fingerprint(self) -> str:
        return str(self.serialize()["fingerprint"])

    @classmethod
    def from_serialized(
        cls, artifact: Mapping[str, Any]
    ) -> "EffectiveTriLIDCalibratorV3":
        raw = dict(artifact)
        stored = raw.pop("fingerprint", None)
        if not isinstance(stored, str) or fingerprint(raw) != stored:
            raise PDCTPV3Error("v3 calibrator fingerprint mismatch")
        if (
            raw.get("name") != cls.NAME
            or raw.get("schema") != cls.SCHEMA
            or raw.get("version") != cls.VERSION
            or raw.get("cache_contract")
            != "in_memory_only_not_serialized_into_scientific_artifacts"
        ):
            raise PDCTPV3Error("unsupported v3 calibrator artifact")
        feature = raw.get("feature_schema")
        normalization = raw.get("normalization")
        model = raw.get("model")
        output = raw.get("output_domain")
        target = raw.get("target_construction")
        fit = raw.get("fit")
        if not all(
            isinstance(value, Mapping)
            for value in (feature, normalization, model, output, target, fit)
        ):
            raise PDCTPV3Error("v3 calibrator artifact is incomplete")
        if (
            normalization.get("source_role") != "query_cal"
            or model.get("link") != "log"
            or model.get("solver") != "closed_form_ridge_with_active_set_v3"
            or model.get("nonnegative_coefficient") != "log_pilot_lid"
            or model.get("output_decimals") != V3_CALIBRATION_DECIMALS
            or target.get("source_role") != "query_cal"
            or target.get("target") != "effective_tri_lid_minimum_retention_curve_mse"
            or target.get("heads") != list(cls.HEADS)
            or target.get("terminal_full_corpus_excluded_from_fit") is not True
            or target.get("numerical_implementation") != V3_NUMERICAL_IMPLEMENTATION
            or target.get("tie_break") != "lowest_effective_lid_grid_value"
            or fit.get("role") != "query_cal"
        ):
            raise PDCTPV3Error("v3 calibrator fit or target contract changed")
        heads = model.get("heads")
        if not isinstance(heads, Mapping) or set(heads) != set(cls.HEADS):
            raise PDCTPV3Error("v3 calibrator heads changed")
        try:
            result = cls(
                feature_names=feature["feature_names"],
                feature_spec_fingerprint=feature["spec_fingerprint"],
                means=normalization["means"],
                scales=normalization["scales"],
                constant_features=normalization["constant_features"],
                intercepts={head: heads[head]["intercept"] for head in cls.HEADS},
                coefficients={head: heads[head]["coefficients"] for head in cls.HEADS},
                objectives={head: heads[head]["objective"] for head in cls.HEADS},
                regularization=model["regularization"],
                output_min=output["minimum"],
                output_max=output["maximum"],
                fallback=output["fallback"],
                fit_ids=fit["ordered_ids"],
                target_fingerprint=target["target_fingerprint"],
                m_prime=target["m_prime"],
                k_gt=target["k_gt"],
                corpus_size=target["corpus_size"],
                budgets=target["budget_grid"],
                max_rank_samples=target["max_rank_samples"],
                low_budget_max=target["low_budget_max"],
                high_budget_min=target["high_budget_min"],
                target_lid_grid=target["target_lid_grid"],
            )
        except (KeyError, TypeError) as exc:
            raise PDCTPV3Error("invalid v3 calibrator artifact") from exc
        if result.serialize() != artifact:
            raise PDCTPV3Error("v3 calibrator schema mismatch")
        return result


@dataclass(frozen=True)
class EffectiveCurveDecisionV3:
    policy_name: str
    policy_version: int
    mode: str
    budget: int
    scalar_lid: Optional[float]
    low_lid: Optional[float]
    high_lid: Optional[float]
    predicted_retention: Optional[float]
    used_fallback: bool
    saturated: bool
    failure_reason: Optional[str]


class EffectiveCurveTriPredictPolicyV3:
    """Raw Tri-Predict curve with only its effective-LID input shape repaired."""

    NAME = "effective_curve_shape_calibrated_tri_predict"
    SCHEMA = "calibrated_tri_predict_policy_v3"
    VERSION = 3
    MODES = ("scalar_effective_lid", "two_regime_effective_lid")

    def __init__(
        self,
        *,
        mode: str,
        calibrator: EffectiveTriLIDCalibratorV3,
        target: float,
        minimum_budget: int,
        fallback_budget: int,
        cache: Optional[TriPredictPredictionGridCacheV3] = None,
    ):
        if mode not in self.MODES:
            raise PDCTPV3Error("unsupported v3 curve-shape mode")
        if not np.isfinite(target) or not 0.0 < target <= 1.0:
            raise PDCTPV3Error("v3 retention target must lie in (0,1]")
        self.mode = mode
        self.calibrator = calibrator
        self.target = float(target)
        self.grid = calibrator.budgets
        self.minimum_budget = int(minimum_budget)
        self.fallback_budget = int(fallback_budget)
        if self.minimum_budget not in self.grid or self.grid[0] < self.minimum_budget:
            raise PDCTPV3Error("v3 minimum budget violates the grid")
        if self.fallback_budget not in self.grid:
            raise PDCTPV3Error("v3 fallback budget must come from the grid")
        self.cache = cache or TriPredictPredictionGridCacheV3()

    def prediction_curve(
        self, observation: PDCTPDecisionInput, *, use_cache: bool = True
    ) -> Tuple[Optional[Dict[int, float]], EffectiveTriLIDPredictionV3]:
        prediction = self.calibrator.predict(observation.features)
        if not prediction.valid:
            return None, prediction

        def curve(lid: float) -> PredictionGridV3:
            return self.cache.prediction_grid(
                lid=lid,
                m_prime=self.calibrator.m_prime,
                k_gt=self.calibrator.k_gt,
                corpus_size=self.calibrator.corpus_size,
                budgets=self.grid,
                max_rank_samples=self.calibrator.max_rank_samples,
                use_cache=use_cache,
            )

        if self.mode == "scalar_effective_lid":
            values = np.asarray(curve(prediction.scalar_lid).values, dtype=np.float64)
        else:
            low_values = np.asarray(curve(prediction.low_lid).values, dtype=np.float64)
            high_values = np.asarray(
                curve(prediction.high_lid).values, dtype=np.float64
            )
            values = np.asarray(
                [
                    low_values[index]
                    if budget <= self.calibrator.low_budget_max
                    else high_values[index]
                    for index, budget in enumerate(self.grid)
                ],
                dtype=np.float64,
            )
            values = np.maximum.accumulate(values)
        if self.calibrator.corpus_size in self.grid:
            values[self.grid.index(self.calibrator.corpus_size)] = 1.0
        if np.any(np.diff(values) < -MONOTONICITY_TOLERANCE):
            raise FloatingPointError("v3 predicted retention decreased with budget")
        return (
            {budget: float(value) for budget, value in zip(self.grid, values)},
            prediction,
        )

    def choose(
        self, observation: PDCTPDecisionInput, *, use_cache: bool = True
    ) -> EffectiveCurveDecisionV3:
        curve, effective = self.prediction_curve(observation, use_cache=use_cache)
        if curve is None:
            return EffectiveCurveDecisionV3(
                self.NAME,
                self.VERSION,
                self.mode,
                self.fallback_budget,
                None,
                None,
                None,
                None,
                True,
                self.fallback_budget == self.grid[-1],
                effective.failure_reason,
            )
        for budget in self.grid:
            value = 1.0 if budget == self.calibrator.corpus_size else curve[budget]
            if self.target == 1.0 and budget != self.calibrator.corpus_size:
                continue
            if value >= self.target:
                return EffectiveCurveDecisionV3(
                    self.NAME,
                    self.VERSION,
                    self.mode,
                    budget,
                    effective.scalar_lid,
                    effective.low_lid,
                    effective.high_lid,
                    value,
                    False,
                    False,
                    None,
                )
        return EffectiveCurveDecisionV3(
            self.NAME,
            self.VERSION,
            self.mode,
            self.grid[-1],
            effective.scalar_lid,
            effective.low_lid,
            effective.high_lid,
            curve[self.grid[-1]],
            False,
            True,
            "target_nonattainment",
        )

    def serialize(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "name": self.NAME,
            "schema": self.SCHEMA,
            "version": self.VERSION,
            "mode": self.mode,
            "calibrator_fingerprint": self.calibrator.fingerprint,
            "target": self.target,
            "numerical_problem": {
                "m_prime": self.calibrator.m_prime,
                "k_gt": self.calibrator.k_gt,
                "corpus_size": self.calibrator.corpus_size,
                "budget_grid": list(self.grid),
                "max_rank_samples": self.calibrator.max_rank_samples,
                "numerical_implementation": V3_NUMERICAL_IMPLEMENTATION,
            },
            "regimes": {
                "low_budget_max": self.calibrator.low_budget_max,
                "high_budget_min": self.calibrator.high_budget_min,
            },
            "minimum_budget": self.minimum_budget,
            "fallback_budget": self.fallback_budget,
            "inference_fields": [
                "pilot_distances",
                "pilot_derived_features",
                "explicit_validity_fields",
            ],
            "forbidden_inference_fields": [
                "oracle_lid",
                "exact_top_k",
                "qrels",
                "realized_retention",
                "answer_labels",
                "protected_outcomes",
                "split_role",
            ],
            "cache_contract": {
                "scope": "in_memory_only",
                "serialized": False,
                "identity_fields": [
                    "float64_lid_bit_pattern",
                    "m_prime",
                    "k_gt",
                    "corpus_size",
                    "ordered_budget_grid",
                    "max_rank_samples",
                    "numerical_implementation_version",
                ],
            },
        }
        body["fingerprint"] = fingerprint(body)
        return body
