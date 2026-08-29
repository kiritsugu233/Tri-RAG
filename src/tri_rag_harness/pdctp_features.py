from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .utils import fingerprint


class PilotFeatureError(ValueError):
    """Raised when a feature artifact or specification is not trustworthy."""


FEATURE_SCHEMA = "pilot_distance_features_v1"


def _canonical_feature_float(value: float, decimals: int) -> float:
    result = float(np.round(float(value), decimals=decimals))
    if not np.isfinite(result):
        raise PilotFeatureError("pilot feature values must be finite")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class PilotDistanceFeatureSpec:
    """Frozen deployable pilot-distance feature contract.

    Inputs are already stable-sorted by Euclidean original-space distance.
    ``projected_squared_distances`` must be reordered by the same permutation.
    The default epsilon is zero because strictly positive distances are required;
    this preserves exact common-scale invariance for every ratio-only feature.
    """

    lid_boundary: int
    minimum_count: int
    gap_quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75)
    epsilon: float = 0.0
    duplicate_tolerance: float = 1e-12
    invalid_fill: float = 0.0
    output_decimals: int = 10
    schema: str = FEATURE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FEATURE_SCHEMA:
            raise PilotFeatureError("unsupported pilot feature schema")
        if (
            isinstance(self.lid_boundary, bool)
            or not isinstance(self.lid_boundary, int)
            or self.lid_boundary < 5
        ):
            raise PilotFeatureError("lid_boundary must be an integer at least five")
        if (
            isinstance(self.minimum_count, bool)
            or not isinstance(self.minimum_count, int)
            or self.minimum_count < self.lid_boundary
        ):
            raise PilotFeatureError("minimum_count must cover the LID boundary")
        quantiles = tuple(float(value) for value in self.gap_quantiles)
        if not quantiles or any(not 0.0 < value < 1.0 for value in quantiles):
            raise PilotFeatureError("gap quantiles must lie strictly inside (0,1)")
        if tuple(sorted(set(quantiles))) != quantiles:
            raise PilotFeatureError("gap quantiles must be unique and increasing")
        for name, value in (
            ("epsilon", self.epsilon),
            ("duplicate_tolerance", self.duplicate_tolerance),
            ("invalid_fill", self.invalid_fill),
        ):
            if not np.isfinite(value):
                raise PilotFeatureError(f"{name} must be finite")
        if self.epsilon < 0.0 or self.duplicate_tolerance < 0.0:
            raise PilotFeatureError("epsilon and duplicate_tolerance must be nonnegative")
        if (
            isinstance(self.output_decimals, bool)
            or not isinstance(self.output_decimals, int)
            or not 6 <= self.output_decimals <= 15
        ):
            raise PilotFeatureError("output_decimals must be an integer from 6 to 15")
        object.__setattr__(self, "gap_quantiles", quantiles)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        gap_names = tuple(
            f"normalized_gap_q{int(round(value * 100)):02d}"
            for value in self.gap_quantiles
        )
        return (
            "log_pilot_lid",
            "log_radius",
            "log_ratio_mean",
            "log_ratio_std",
            "inner_half_slope",
            "outer_half_slope",
            "profile_curvature",
            *gap_names,
            "projection_log_distortion_mean",
            "projection_log_distortion_std",
            "pilot_lid_valid",
            "valid_distance_fraction",
        )

    def serialize(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "name": "pilot_distance_feature_specification",
            "schema": self.schema,
            "version": 1,
            "input_distance": "squared_l2_converted_to_euclidean",
            "ordering": "stable_original_distance_then_candidate_id",
            "lid_boundary": self.lid_boundary,
            "minimum_count": self.minimum_count,
            "gap_quantiles": list(self.gap_quantiles),
            "epsilon": float(self.epsilon),
            "duplicate_tolerance": float(self.duplicate_tolerance),
            "invalid_fill": float(self.invalid_fill),
            "output_decimals": self.output_decimals,
            "slope_rank_coordinate": "linear_zero_to_one_with_disjoint_halves",
            "normalized_gap_denominator": "lid_boundary_radius_plus_epsilon",
            "invalid_behavior": "fixed_fill_with_validity_and_count_indicators",
            "feature_names": list(self.feature_names),
        }
        body["fingerprint"] = fingerprint(body)
        return body

    @property
    def fingerprint(self) -> str:
        return str(self.serialize()["fingerprint"])

    @classmethod
    def from_serialized(
        cls, artifact: Mapping[str, Any]
    ) -> "PilotDistanceFeatureSpec":
        raw = dict(artifact)
        stored = raw.pop("fingerprint", None)
        if not isinstance(stored, str) or fingerprint(raw) != stored:
            raise PilotFeatureError("pilot feature specification fingerprint mismatch")
        required_constants = {
            "name": "pilot_distance_feature_specification",
            "schema": FEATURE_SCHEMA,
            "version": 1,
            "input_distance": "squared_l2_converted_to_euclidean",
            "ordering": "stable_original_distance_then_candidate_id",
            "slope_rank_coordinate": "linear_zero_to_one_with_disjoint_halves",
            "normalized_gap_denominator": "lid_boundary_radius_plus_epsilon",
            "invalid_behavior": "fixed_fill_with_validity_and_count_indicators",
        }
        for key, expected in required_constants.items():
            if raw.get(key) != expected:
                raise PilotFeatureError(f"unsupported pilot feature specification {key}")
        try:
            result = cls(
                lid_boundary=raw["lid_boundary"],
                minimum_count=raw["minimum_count"],
                gap_quantiles=tuple(raw["gap_quantiles"]),
                epsilon=raw["epsilon"],
                duplicate_tolerance=raw["duplicate_tolerance"],
                invalid_fill=raw["invalid_fill"],
                output_decimals=raw["output_decimals"],
                schema=raw["schema"],
            )
        except (KeyError, TypeError) as exc:
            raise PilotFeatureError("invalid pilot feature specification") from exc
        if result.serialize() != artifact:
            raise PilotFeatureError("pilot feature specification schema mismatch")
        return result


@dataclass(frozen=True)
class PilotDistanceObservation:
    """The complete and intentionally narrow PDCTP inference input."""

    original_squared_distances: Tuple[float, ...]
    projected_squared_distances: Tuple[float, ...]
    pilot_lid: float
    pilot_lid_valid: bool
    pilot_lid_failure_reason: Optional[str]
    valid_distance_count: int

    @classmethod
    def from_arrays(
        cls,
        original_squared_distances: Sequence[float],
        projected_squared_distances: Sequence[float],
        *,
        pilot_lid: float,
        pilot_lid_valid: bool,
        pilot_lid_failure_reason: Optional[str],
        valid_distance_count: int,
    ) -> "PilotDistanceObservation":
        return cls(
            tuple(float(value) for value in original_squared_distances),
            tuple(float(value) for value in projected_squared_distances),
            float(pilot_lid),
            bool(pilot_lid_valid),
            pilot_lid_failure_reason,
            int(valid_distance_count),
        )


@dataclass(frozen=True)
class PilotFeatureVector:
    schema: str
    spec_fingerprint: str
    names: Tuple[str, ...]
    values: Tuple[float, ...]
    valid: bool
    failure_reason: Optional[str]

    def __post_init__(self) -> None:
        if self.schema != FEATURE_SCHEMA:
            raise PilotFeatureError("feature vector schema mismatch")
        if not self.spec_fingerprint:
            raise PilotFeatureError("feature vector has no specification fingerprint")
        if len(self.names) != len(self.values) or len(set(self.names)) != len(self.names):
            raise PilotFeatureError("feature names and values must align uniquely")
        if not np.all(np.isfinite(np.asarray(self.values, dtype=np.float64))):
            raise PilotFeatureError("feature values must be finite")
        if self.valid != (self.failure_reason is None):
            raise PilotFeatureError("feature validity and failure reason disagree")

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)

    def serialize(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "spec_fingerprint": self.spec_fingerprint,
            "feature_names": list(self.names),
            "values": list(self.values),
            "valid": self.valid,
            "failure_reason": self.failure_reason,
        }


def stable_sort_pilot_distances(
    candidate_ids: Sequence[str],
    original_squared_distances: Sequence[float],
    projected_squared_distances: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable-sort a raw pilot profile before constructing the observation."""
    ids = np.asarray(candidate_ids, dtype=str)
    original = np.asarray(original_squared_distances, dtype=np.float64)
    projected = np.asarray(projected_squared_distances, dtype=np.float64)
    if ids.ndim != 1 or original.ndim != 1 or projected.ndim != 1:
        raise PilotFeatureError("pilot candidate IDs and distances must be one-dimensional")
    if len(ids) != len(original) or len(ids) != len(projected):
        raise PilotFeatureError("pilot candidate IDs and distances must align")
    if len(set(ids.tolist())) != len(ids):
        raise PilotFeatureError("pilot candidate IDs must be unique")
    order = np.lexsort((ids, original))
    return ids[order], original[order], projected[order]


def _linear_slope(values: np.ndarray, positions: np.ndarray) -> float:
    centered_x = positions - float(np.mean(positions))
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0.0:
        raise PilotFeatureError("profile half has insufficient rank variation")
    centered_y = values - float(np.mean(values))
    return float(np.dot(centered_x, centered_y) / denominator)


class PilotDistanceFeatureExtractor:
    def __init__(self, spec: PilotDistanceFeatureSpec):
        self.spec = spec

    def _invalid(
        self, reason: str, *, valid_distance_count: int
    ) -> PilotFeatureVector:
        values = np.full(
            len(self.spec.feature_names), self.spec.invalid_fill, dtype=np.float64
        )
        validity_index = self.spec.feature_names.index("pilot_lid_valid")
        count_index = self.spec.feature_names.index("valid_distance_fraction")
        values[validity_index] = 0.0
        values[count_index] = float(
            np.clip(valid_distance_count / self.spec.minimum_count, 0.0, 1.0)
        )
        values = np.asarray(
            [
                _canonical_feature_float(value, self.spec.output_decimals)
                for value in values
            ],
            dtype=np.float64,
        )
        return PilotFeatureVector(
            FEATURE_SCHEMA,
            self.spec.fingerprint,
            self.spec.feature_names,
            tuple(float(value) for value in values),
            False,
            reason,
        )

    def extract(self, observation: PilotDistanceObservation) -> PilotFeatureVector:
        original_sq = np.asarray(
            observation.original_squared_distances, dtype=np.float64
        )
        projected_sq = np.asarray(
            observation.projected_squared_distances, dtype=np.float64
        )
        count = min(
            max(int(observation.valid_distance_count), 0), len(original_sq)
        )
        if original_sq.ndim != 1 or projected_sq.ndim != 1:
            return self._invalid("non_vector_pilot_data", valid_distance_count=count)
        if len(original_sq) != len(projected_sq):
            return self._invalid("misaligned_pilot_data", valid_distance_count=count)
        if len(original_sq) < self.spec.minimum_count:
            return self._invalid("insufficient_pilot_data", valid_distance_count=count)
        if observation.valid_distance_count < self.spec.minimum_count:
            return self._invalid("insufficient_valid_distances", valid_distance_count=count)
        if not np.all(np.isfinite(original_sq)) or not np.all(np.isfinite(projected_sq)):
            return self._invalid("nonfinite_pilot_data", valid_distance_count=count)
        if np.any(original_sq <= 0.0) or np.any(projected_sq <= 0.0):
            return self._invalid("nonpositive_pilot_data", valid_distance_count=count)
        original = np.sqrt(original_sq)
        projected = np.sqrt(projected_sq)
        differences = np.diff(original)
        if np.any(differences < 0.0):
            return self._invalid("unsorted_original_distances", valid_distance_count=count)
        if np.any(differences <= self.spec.duplicate_tolerance):
            return self._invalid("duplicate_original_distances", valid_distance_count=count)
        if not observation.pilot_lid_valid:
            reason = observation.pilot_lid_failure_reason or "invalid_pilot_lid"
            return self._invalid(reason, valid_distance_count=count)
        if not np.isfinite(observation.pilot_lid) or observation.pilot_lid <= 0.0:
            return self._invalid("invalid_pilot_lid", valid_distance_count=count)

        boundary_count = self.spec.lid_boundary
        original = original[:boundary_count]
        projected = projected[:boundary_count]
        radius = float(original[-1])
        epsilon = self.spec.epsilon
        profile = np.log(
            (radius + epsilon) / (original[:-1] + epsilon)
        )
        positions = np.linspace(0.0, 1.0, len(profile), dtype=np.float64)
        split = (len(profile) + 1) // 2
        inner_slope = _linear_slope(profile[:split], positions[:split])
        outer_slope = _linear_slope(profile[split:], positions[split:])
        gaps = np.diff(original) / (radius + epsilon)
        try:
            gap_quantiles = np.quantile(
                gaps, self.spec.gap_quantiles, method="linear"
            )
        except TypeError:  # NumPy < 1.22 compatibility.
            gap_quantiles = np.quantile(
                gaps, self.spec.gap_quantiles, interpolation="linear"
            )
        distortion = np.log(
            (projected + epsilon) / (original + epsilon)
        )
        values = (
            float(np.log(observation.pilot_lid)),
            float(np.log(radius + epsilon)),
            float(np.mean(profile)),
            float(np.std(profile, ddof=0)),
            inner_slope,
            outer_slope,
            outer_slope - inner_slope,
            *(float(value) for value in gap_quantiles),
            float(np.mean(distortion)),
            float(np.std(distortion, ddof=0)),
            1.0,
            float(np.clip(count / self.spec.minimum_count, 0.0, 1.0)),
        )
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            return self._invalid("nonfinite_derived_feature", valid_distance_count=count)
        canonical_values = tuple(
            _canonical_feature_float(value, self.spec.output_decimals)
            for value in values
        )
        return PilotFeatureVector(
            FEATURE_SCHEMA,
            self.spec.fingerprint,
            self.spec.feature_names,
            canonical_values,
            True,
            None,
        )
