from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .pdctp_features import PilotFeatureVector
from .utils import fingerprint


class CalibrationError(ValueError):
    """Raised when calibration fit, input, or artifact validation fails."""


CALIBRATION_FLOAT_DECIMALS = 12


def _canonical_float(value: float) -> float:
    result = float(np.round(float(value), decimals=CALIBRATION_FLOAT_DECIMALS))
    if not np.isfinite(result):
        raise CalibrationError("calibration values must be finite")
    return result


def _ordered_id_hash(ids: Sequence[str]) -> str:
    return fingerprint([str(value) for value in ids])


def _validate_fit_ids(ids: Sequence[str]) -> Tuple[str, ...]:
    result = tuple(str(value) for value in ids)
    if not result or any(not value for value in result):
        raise CalibrationError("calibration fit IDs must be nonempty strings")
    if len(set(result)) != len(result):
        raise CalibrationError("calibration fit IDs must be unique")
    return result


def _feature_matrix(
    features: Sequence[PilotFeatureVector],
) -> Tuple[np.ndarray, Tuple[str, ...], str]:
    if not features:
        raise CalibrationError("calibration requires at least one feature vector")
    names = features[0].names
    spec_fingerprint = features[0].spec_fingerprint
    for vector in features:
        if not vector.valid:
            raise CalibrationError("invalid feature vectors cannot be used for fitting")
        if vector.names != names or vector.spec_fingerprint != spec_fingerprint:
            raise CalibrationError("calibration feature schemas do not match")
    matrix = np.vstack([vector.as_array() for vector in features])
    if not np.all(np.isfinite(matrix)):
        raise CalibrationError("calibration feature matrix must be finite")
    return matrix, names, spec_fingerprint


def _normalization(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(matrix, axis=0)
    raw_scales = np.std(matrix, axis=0, ddof=0)
    constant = raw_scales <= np.finfo(np.float64).eps
    scales = raw_scales.copy()
    scales[constant] = 1.0
    return means, scales, constant


def _ridge_solution(
    standardized: np.ndarray,
    targets: np.ndarray,
    regularization: float,
    *,
    constrained_feature: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    n, feature_count = standardized.shape
    design = np.column_stack([np.ones(n, dtype=np.float64), standardized])
    penalty = np.diag(
        np.concatenate([[0.0], np.full(feature_count, regularization)])
    )
    solution = np.linalg.pinv(design.T @ design + penalty) @ design.T @ targets
    if constrained_feature is not None and solution[1 + constrained_feature] < 0.0:
        keep = [index for index in range(feature_count) if index != constrained_feature]
        reduced = standardized[:, keep]
        reduced_design = np.column_stack(
            [np.ones(n, dtype=np.float64), reduced]
        )
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
        return float(reduced_solution[0]), coefficients
    return float(solution[0]), np.asarray(solution[1:], dtype=np.float64)


@dataclass(frozen=True)
class LIDCalibrationRecord:
    query_id: str
    role: str
    features: PilotFeatureVector
    oracle_lid: float


@dataclass(frozen=True)
class CalibratedLID:
    value: float
    valid: bool
    used_fallback: bool
    failure_reason: Optional[str]


class PilotLIDCalibrator:
    """Constrained log-linear LID calibrator fit only on ``query_cal``."""

    NAME = "pilot_distance_log_linear_lid_calibrator"
    SCHEMA = "pdctp_lid_calibrator_v1"
    VERSION = 1

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        feature_spec_fingerprint: str,
        means: Sequence[float],
        scales: Sequence[float],
        constant_features: Sequence[bool],
        intercept: float,
        coefficients: Sequence[float],
        regularization: float,
        output_min: float,
        output_max: float,
        fallback: float,
        fit_ids: Sequence[str],
        objective_value: float,
    ):
        self.feature_names = tuple(str(value) for value in feature_names)
        self.feature_spec_fingerprint = str(feature_spec_fingerprint)
        self.means = np.asarray(means, dtype=np.float64)
        self.scales = np.asarray(scales, dtype=np.float64)
        self.constant_features = tuple(bool(value) for value in constant_features)
        self.intercept = _canonical_float(intercept)
        self.coefficients = np.asarray(
            [_canonical_float(value) for value in coefficients], dtype=np.float64
        )
        self.regularization = _canonical_float(regularization)
        self.output_min = _canonical_float(output_min)
        self.output_max = _canonical_float(output_max)
        self.fallback = _canonical_float(fallback)
        self.fit_ids = _validate_fit_ids(fit_ids)
        self.objective_value = _canonical_float(objective_value)
        count = len(self.feature_names)
        if not self.feature_spec_fingerprint:
            raise CalibrationError("LID calibrator has no feature specification")
        if len(set(self.feature_names)) != count or count == 0:
            raise CalibrationError("LID calibrator feature names must be unique")
        if any(
            len(value) != count
            for value in (self.means, self.scales, self.constant_features, self.coefficients)
        ):
            raise CalibrationError("LID calibrator arrays do not align")
        if not np.all(np.isfinite(self.means)) or not np.all(np.isfinite(self.scales)):
            raise CalibrationError("LID normalization must be finite")
        if np.any(self.scales <= 0.0):
            raise CalibrationError("LID normalization scales must be positive")
        if "log_pilot_lid" not in self.feature_names:
            raise CalibrationError("LID calibrator requires log_pilot_lid")
        constrained = self.feature_names.index("log_pilot_lid")
        if self.coefficients[constrained] < 0.0:
            raise CalibrationError("log_pilot_lid coefficient must be nonnegative")
        if self.regularization < 0.0:
            raise CalibrationError("LID regularization must be nonnegative")
        if not 0.0 < self.output_min < self.output_max:
            raise CalibrationError("LID output domain must be positive and increasing")
        if not self.output_min <= self.fallback <= self.output_max:
            raise CalibrationError("LID fallback must lie inside the output domain")

    @classmethod
    def fit(
        cls,
        records: Iterable[LIDCalibrationRecord],
        *,
        regularization: float,
        output_min: float,
        output_max: float,
        fallback: float,
    ) -> "PilotLIDCalibrator":
        rows = list(records)
        if not rows:
            raise CalibrationError("cannot fit LID calibrator without records")
        if any(row.role != "query_cal" for row in rows):
            raise CalibrationError("LID calibrator may fit query_cal records only")
        fit_ids = _validate_fit_ids([row.query_id for row in rows])
        matrix, names, spec_fingerprint = _feature_matrix(
            [row.features for row in rows]
        )
        targets = np.asarray([row.oracle_lid for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(targets)) or np.any(targets <= 0.0):
            raise CalibrationError("oracle LID targets must be finite and positive")
        if not np.isfinite(regularization) or regularization < 0.0:
            raise CalibrationError("LID regularization must be finite and nonnegative")
        means, scales, constant = _normalization(matrix)
        standardized = (matrix - means) / scales
        target_logs = np.log(targets)
        constrained = names.index("log_pilot_lid")
        intercept, coefficients = _ridge_solution(
            standardized,
            target_logs,
            float(regularization),
            constrained_feature=constrained,
        )
        residuals = target_logs - (intercept + standardized @ coefficients)
        objective = float(
            np.mean(np.square(residuals))
            + float(regularization) * np.dot(coefficients, coefficients)
        )
        return cls(
            feature_names=names,
            feature_spec_fingerprint=spec_fingerprint,
            means=[_canonical_float(value) for value in means],
            scales=[_canonical_float(value) for value in scales],
            constant_features=constant.tolist(),
            intercept=intercept,
            coefficients=coefficients,
            regularization=regularization,
            output_min=output_min,
            output_max=output_max,
            fallback=fallback,
            fit_ids=fit_ids,
            objective_value=objective,
        )

    def predict(self, features: PilotFeatureVector) -> CalibratedLID:
        if (
            not features.valid
            or features.names != self.feature_names
            or features.spec_fingerprint != self.feature_spec_fingerprint
        ):
            reason = (
                features.failure_reason
                if not features.valid
                else "feature_schema_mismatch"
            )
            return CalibratedLID(self.fallback, False, True, reason)
        standardized = (features.as_array() - self.means) / self.scales
        log_prediction = self.intercept + float(
            np.dot(standardized, self.coefficients)
        )
        clipped_log = float(
            np.clip(log_prediction, np.log(self.output_min), np.log(self.output_max))
        )
        value = float(np.exp(clipped_log))
        if not np.isfinite(value):
            return CalibratedLID(
                self.fallback, False, True, "nonfinite_lid_prediction"
            )
        return CalibratedLID(value, True, False, None)

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
                "intercept": self.intercept,
                "coefficients": self.coefficients.tolist(),
                "nonnegative_coefficient": "log_pilot_lid",
                "solver": "closed_form_ridge_with_active_set",
            },
            "objective": {
                "name": "mean_squared_log_error_plus_l2",
                "regularization": self.regularization,
                "value": self.objective_value,
            },
            "output_domain": {
                "minimum": self.output_min,
                "maximum": self.output_max,
                "fallback": self.fallback,
            },
            "fit": {
                "role": "query_cal",
                "ordered_ids": list(self.fit_ids),
                "ordered_id_hash": _ordered_id_hash(self.fit_ids),
                "n": len(self.fit_ids),
            },
        }
        body["fingerprint"] = fingerprint(body)
        return body

    @property
    def fingerprint(self) -> str:
        return str(self.serialize()["fingerprint"])

    @classmethod
    def from_serialized(cls, artifact: Mapping[str, Any]) -> "PilotLIDCalibrator":
        raw = dict(artifact)
        if raw.get("name") != cls.NAME or raw.get("schema") != cls.SCHEMA:
            raise CalibrationError("unexpected LID calibrator artifact schema")
        if raw.get("version") != cls.VERSION:
            raise CalibrationError("unsupported LID calibrator version")
        feature = raw.get("feature_schema")
        normalization = raw.get("normalization")
        model = raw.get("model")
        objective = raw.get("objective")
        output = raw.get("output_domain")
        fit = raw.get("fit")
        if not all(
            isinstance(value, Mapping)
            for value in (feature, normalization, model, objective, output, fit)
        ):
            raise CalibrationError("LID calibrator artifact is incomplete")
        if (
            normalization.get("source_role") != "query_cal"
            or model.get("link") != "log"
            or model.get("nonnegative_coefficient") != "log_pilot_lid"
            or model.get("solver") != "closed_form_ridge_with_active_set"
            or objective.get("name") != "mean_squared_log_error_plus_l2"
            or fit.get("role") != "query_cal"
        ):
            raise CalibrationError("LID calibrator domain or fit contract mismatch")
        try:
            result = cls(
                feature_names=feature["feature_names"],
                feature_spec_fingerprint=feature["spec_fingerprint"],
                means=normalization["means"],
                scales=normalization["scales"],
                constant_features=normalization["constant_features"],
                intercept=model["intercept"],
                coefficients=model["coefficients"],
                regularization=objective["regularization"],
                output_min=output["minimum"],
                output_max=output["maximum"],
                fallback=output["fallback"],
                fit_ids=fit["ordered_ids"],
                objective_value=objective["value"],
            )
        except (KeyError, TypeError) as exc:
            raise CalibrationError("invalid LID calibrator artifact") from exc
        if result.serialize() != artifact:
            raise CalibrationError("LID calibrator fingerprint or schema mismatch")
        return result


def quantile_pinball_loss(residuals: Sequence[float], quantile: float) -> float:
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise CalibrationError("quantile residuals must be a finite nonempty vector")
    if not 0.0 < quantile < 1.0:
        raise CalibrationError("quantile must lie strictly inside (0,1)")
    return float(
        np.mean(np.where(values >= 0.0, quantile * values, (quantile - 1.0) * values))
    )


@dataclass(frozen=True)
class BudgetResidualRecord:
    query_id: str
    role: str
    features: PilotFeatureVector
    raw_budget: int
    required_budget: int
    training_level: float

    @property
    def residual(self) -> float:
        if self.raw_budget <= 0 or self.required_budget <= 0:
            raise CalibrationError("residual budgets must be positive")
        return float(np.log(self.required_budget / self.raw_budget))


@dataclass(frozen=True)
class CalibratedBudget:
    budget: int
    raw_budget: int
    residual: Optional[float]
    desired_budget: Optional[float]
    used_fallback: bool
    saturated: bool
    failure_reason: Optional[str]


class TriBudgetResidualCalibrator:
    """Linear quantile residual model retaining Raw Tri-Predict as its anchor."""

    NAME = "tri_budget_log_residual_quantile_calibrator"
    SCHEMA = "pdctp_budget_residual_calibrator_v1"
    VERSION = 1

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        feature_spec_fingerprint: str,
        means: Sequence[float],
        scales: Sequence[float],
        constant_features: Sequence[bool],
        intercept: float,
        coefficients: Sequence[float],
        quantile: float,
        regularization: float,
        safety_offset: float,
        training_level: float,
        grid: Sequence[int],
        minimum_budget: int,
        fallback_budget: int,
        raw_policy_fingerprint: str,
        fit_ids: Sequence[str],
        objective_value: float,
        anchor_lid_source: str = "calibrated_pilot_lid",
    ):
        self.feature_names = tuple(str(value) for value in feature_names)
        self.feature_spec_fingerprint = str(feature_spec_fingerprint)
        self.means = np.asarray(means, dtype=np.float64)
        self.scales = np.asarray(scales, dtype=np.float64)
        self.constant_features = tuple(bool(value) for value in constant_features)
        self.intercept = _canonical_float(intercept)
        self.coefficients = np.asarray(
            [_canonical_float(value) for value in coefficients], dtype=np.float64
        )
        self.quantile = _canonical_float(quantile)
        self.regularization = _canonical_float(regularization)
        self.safety_offset = _canonical_float(safety_offset)
        self.training_level = _canonical_float(training_level)
        self.grid = tuple(int(value) for value in grid)
        self.minimum_budget = int(minimum_budget)
        self.fallback_budget = int(fallback_budget)
        self.raw_policy_fingerprint = str(raw_policy_fingerprint)
        self.anchor_lid_source = str(anchor_lid_source)
        self.fit_ids = _validate_fit_ids(fit_ids)
        self.objective_value = _canonical_float(objective_value)
        count = len(self.feature_names)
        if count == 0 or len(set(self.feature_names)) != count:
            raise CalibrationError("residual feature names must be nonempty and unique")
        if any(
            len(value) != count
            for value in (self.means, self.scales, self.constant_features, self.coefficients)
        ):
            raise CalibrationError("residual calibrator arrays do not align")
        if not np.all(np.isfinite(self.means)) or not np.all(np.isfinite(self.scales)):
            raise CalibrationError("residual normalization must be finite")
        if np.any(self.scales <= 0.0):
            raise CalibrationError("residual normalization scales must be positive")
        if not self.feature_spec_fingerprint or not self.raw_policy_fingerprint:
            raise CalibrationError("residual calibrator fingerprints must be nonempty")
        if self.anchor_lid_source not in {
            "raw_pilot_lid",
            "calibrated_pilot_lid",
        }:
            raise CalibrationError("unsupported residual Raw Tri LID source")
        if not 0.0 < self.quantile < 1.0 or self.regularization < 0.0:
            raise CalibrationError("invalid residual quantile or regularization")
        if not 0.0 < self.training_level <= 1.0:
            raise CalibrationError("residual training level must lie in (0,1]")
        if not self.grid or list(self.grid) != sorted(set(self.grid)):
            raise CalibrationError("residual budget grid must be strictly increasing")
        if self.minimum_budget not in self.grid or self.fallback_budget != self.grid[-1]:
            raise CalibrationError("residual lower/fallback budgets violate grid contract")

    @classmethod
    def fit(
        cls,
        records: Iterable[BudgetResidualRecord],
        *,
        quantile: float,
        regularization: float,
        safety_offset: float,
        grid: Sequence[int],
        minimum_budget: int,
        fallback_budget: int,
        raw_policy_fingerprint: str,
        anchor_lid_source: str = "calibrated_pilot_lid",
    ) -> "TriBudgetResidualCalibrator":
        rows = list(records)
        if not rows:
            raise CalibrationError("cannot fit residual calibrator without records")
        if any(row.role != "query_cal" for row in rows):
            raise CalibrationError("residual calibrator may fit query_cal records only")
        levels = {float(row.training_level) for row in rows}
        if len(levels) != 1:
            raise CalibrationError("one residual calibrator requires one training level")
        fit_ids = _validate_fit_ids([row.query_id for row in rows])
        if any(row.raw_budget not in grid or row.required_budget not in grid for row in rows):
            raise CalibrationError("residual fit budgets must come from the frozen grid")
        matrix, names, spec_fingerprint = _feature_matrix(
            [row.features for row in rows]
        )
        targets = np.asarray([row.residual for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(targets)):
            raise CalibrationError("residual targets must be finite")
        if not 0.0 < quantile < 1.0:
            raise CalibrationError("quantile must lie strictly inside (0,1)")
        if not np.isfinite(regularization) or regularization < 0.0:
            raise CalibrationError("regularization must be finite and nonnegative")
        means, scales, constant = _normalization(matrix)
        standardized = (matrix - means) / scales
        n, feature_count = standardized.shape
        design = np.column_stack([np.ones(n, dtype=np.float64), standardized])
        try:
            initial_intercept = float(np.quantile(targets, quantile, method="linear"))
        except TypeError:
            initial_intercept = float(
                np.quantile(targets, quantile, interpolation="linear")
            )
        beta0 = np.concatenate([[initial_intercept], np.zeros(feature_count)])
        residual0 = targets - design @ beta0
        initial = np.concatenate(
            [beta0, np.maximum(residual0, 0.0), np.maximum(-residual0, 0.0)]
        )
        beta_count = feature_count + 1

        def objective(parameters: np.ndarray) -> float:
            beta = parameters[:beta_count]
            positive = parameters[beta_count : beta_count + n]
            negative = parameters[beta_count + n :]
            return float(
                (quantile * np.sum(positive) + (1.0 - quantile) * np.sum(negative))
                / n
                + 0.5 * regularization * np.dot(beta[1:], beta[1:])
            )

        def objective_jac(parameters: np.ndarray) -> np.ndarray:
            gradient = np.zeros_like(parameters)
            gradient[1:beta_count] = regularization * parameters[1:beta_count]
            gradient[beta_count : beta_count + n] = quantile / n
            gradient[beta_count + n :] = (1.0 - quantile) / n
            return gradient

        def equality(parameters: np.ndarray) -> np.ndarray:
            beta = parameters[:beta_count]
            positive = parameters[beta_count : beta_count + n]
            negative = parameters[beta_count + n :]
            return design @ beta + positive - negative - targets

        bounds = [(None, None)] * beta_count + [(0.0, None)] * (2 * n)
        result = minimize(
            objective,
            initial,
            jac=objective_jac,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": equality},
            options={"ftol": 1e-12, "maxiter": 5000, "disp": False},
        )
        if not result.success or np.max(np.abs(equality(result.x))) > 1e-7:
            raise CalibrationError(
                f"quantile residual optimization failed: {result.message}"
            )
        beta = result.x[:beta_count]
        canonical_intercept = _canonical_float(beta[0])
        canonical_coefficients = np.asarray(
            [_canonical_float(value) for value in beta[1:]], dtype=np.float64
        )
        fitted_residuals = targets - (
            canonical_intercept + standardized @ canonical_coefficients
        )
        canonical_objective = quantile_pinball_loss(fitted_residuals, quantile) + float(
            0.5 * regularization * np.dot(canonical_coefficients, canonical_coefficients)
        )
        return cls(
            feature_names=names,
            feature_spec_fingerprint=spec_fingerprint,
            means=[_canonical_float(value) for value in means],
            scales=[_canonical_float(value) for value in scales],
            constant_features=constant.tolist(),
            intercept=canonical_intercept,
            coefficients=canonical_coefficients,
            quantile=quantile,
            regularization=regularization,
            safety_offset=safety_offset,
            training_level=next(iter(levels)),
            grid=grid,
            minimum_budget=minimum_budget,
            fallback_budget=fallback_budget,
            raw_policy_fingerprint=raw_policy_fingerprint,
            fit_ids=fit_ids,
            objective_value=canonical_objective,
            anchor_lid_source=anchor_lid_source,
        )

    def predict_residual(self, features: PilotFeatureVector) -> Optional[float]:
        if (
            not features.valid
            or features.names != self.feature_names
            or features.spec_fingerprint != self.feature_spec_fingerprint
        ):
            return None
        standardized = (features.as_array() - self.means) / self.scales
        result = self.intercept + float(np.dot(standardized, self.coefficients))
        return result if np.isfinite(result) else None

    def choose_budget(
        self, raw_budget: int, features: PilotFeatureVector
    ) -> CalibratedBudget:
        if raw_budget not in self.grid:
            return CalibratedBudget(
                self.fallback_budget,
                int(raw_budget),
                None,
                None,
                True,
                True,
                "raw_budget_outside_grid",
            )
        residual = self.predict_residual(features)
        if residual is None:
            return CalibratedBudget(
                self.fallback_budget,
                int(raw_budget),
                None,
                None,
                True,
                True,
                features.failure_reason or "feature_schema_mismatch",
            )
        exponent = float(np.clip(residual + self.safety_offset, -700.0, 700.0))
        desired = float(raw_budget * np.exp(exponent))
        desired = max(float(self.minimum_budget), desired)
        # Serialized coefficients are canonicalized to twelve decimals.  Snap
        # only roundoff-sized deviations back to an exact configured boundary
        # so a mathematical grid tie is deterministic across round trips.
        for grid_value in self.grid:
            tolerance = 1e-12 * max(1.0, abs(desired), abs(grid_value))
            if abs(desired - grid_value) <= tolerance:
                desired = float(grid_value)
                break
        if not np.isfinite(desired) or desired > self.grid[-1]:
            return CalibratedBudget(
                self.grid[-1],
                int(raw_budget),
                residual,
                desired if np.isfinite(desired) else None,
                False,
                True,
                None,
            )
        index = int(np.searchsorted(np.asarray(self.grid), desired, side="left"))
        budget = self.grid[index]
        return CalibratedBudget(
            budget,
            int(raw_budget),
            residual,
            desired,
            False,
            False,
            None,
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
                "link": "log_budget_ratio",
                "intercept": self.intercept,
                "coefficients": self.coefficients.tolist(),
                "solver": "linear_quantile_slsqp_exact_pinball",
                "quantile": self.quantile,
                "safety_offset": self.safety_offset,
            },
            "objective": {
                "name": "mean_pinball_plus_l2",
                "regularization": self.regularization,
                "value": self.objective_value,
            },
            "anchor": {
                "name": "raw_tri_predict",
                "policy_fingerprint": self.raw_policy_fingerprint,
                "lid_source": self.anchor_lid_source,
                "target": "log(M_required/M_raw)",
                "training_level": self.training_level,
            },
            "budget_contract": {
                "grid": list(self.grid),
                "rounding": "grid_ceiling_left_inclusive_with_1e-12_boundary_snap",
                "minimum_budget": self.minimum_budget,
                "fallback_budget": self.fallback_budget,
                "fallback_is_terminal": True,
            },
            "fit": {
                "role": "query_cal",
                "ordered_ids": list(self.fit_ids),
                "ordered_id_hash": _ordered_id_hash(self.fit_ids),
                "n": len(self.fit_ids),
            },
        }
        body["fingerprint"] = fingerprint(body)
        return body

    @property
    def fingerprint(self) -> str:
        return str(self.serialize()["fingerprint"])

    @classmethod
    def from_serialized(
        cls, artifact: Mapping[str, Any]
    ) -> "TriBudgetResidualCalibrator":
        raw = dict(artifact)
        if raw.get("name") != cls.NAME or raw.get("schema") != cls.SCHEMA:
            raise CalibrationError("unexpected residual calibrator artifact schema")
        if raw.get("version") != cls.VERSION:
            raise CalibrationError("unsupported residual calibrator version")
        feature = raw.get("feature_schema")
        normalization = raw.get("normalization")
        model = raw.get("model")
        objective = raw.get("objective")
        anchor = raw.get("anchor")
        budget = raw.get("budget_contract")
        fit = raw.get("fit")
        if not all(
            isinstance(value, Mapping)
            for value in (feature, normalization, model, objective, anchor, budget, fit)
        ):
            raise CalibrationError("residual calibrator artifact is incomplete")
        if (
            normalization.get("source_role") != "query_cal"
            or model.get("link") != "log_budget_ratio"
            or model.get("solver") != "linear_quantile_slsqp_exact_pinball"
            or objective.get("name") != "mean_pinball_plus_l2"
            or anchor.get("name") != "raw_tri_predict"
            or anchor.get("target") != "log(M_required/M_raw)"
            or anchor.get("lid_source")
            not in {"raw_pilot_lid", "calibrated_pilot_lid"}
            or budget.get("rounding")
            != "grid_ceiling_left_inclusive_with_1e-12_boundary_snap"
            or budget.get("fallback_is_terminal") is not True
            or fit.get("role") != "query_cal"
        ):
            raise CalibrationError("residual calibrator fit or domain contract mismatch")
        try:
            result = cls(
                feature_names=feature["feature_names"],
                feature_spec_fingerprint=feature["spec_fingerprint"],
                means=normalization["means"],
                scales=normalization["scales"],
                constant_features=normalization["constant_features"],
                intercept=model["intercept"],
                coefficients=model["coefficients"],
                quantile=model["quantile"],
                regularization=objective["regularization"],
                safety_offset=model["safety_offset"],
                training_level=anchor["training_level"],
                grid=budget["grid"],
                minimum_budget=budget["minimum_budget"],
                fallback_budget=budget["fallback_budget"],
                raw_policy_fingerprint=anchor["policy_fingerprint"],
                fit_ids=fit["ordered_ids"],
                objective_value=objective["value"],
                anchor_lid_source=anchor["lid_source"],
            )
        except (KeyError, TypeError) as exc:
            raise CalibrationError("invalid residual calibrator artifact") from exc
        if result.serialize() != artifact:
            raise CalibrationError("residual calibrator fingerprint or schema mismatch")
        return result
