from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class LIDEstimate:
    valid: bool
    raw: Optional[float]
    clipped: float
    valid_distance_count: int
    reason: Optional[str]

    def serialize(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_lid_from_squared_distances(
    squared_distances: np.ndarray,
    *,
    s_lid: int,
    min_neighbors: int,
    clip_min: float,
    clip_max: float,
    duplicate_tolerance: float,
    fallback: float,
) -> LIDEstimate:
    """Hill/MLE LID over Euclidean distances converted from squared L2."""
    values = np.asarray(squared_distances, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if np.any(finite <= 0.0):
        return LIDEstimate(False, None, float(fallback), len(finite), "nonpositive_distance")
    distances = np.sqrt(np.sort(finite))[:s_lid]
    if len(distances) < min_neighbors:
        return LIDEstimate(False, None, float(fallback), len(distances), "insufficient_distances")
    if len(distances) < 2:
        return LIDEstimate(False, None, float(fallback), len(distances), "insufficient_distances")
    if np.any(np.diff(distances) <= duplicate_tolerance):
        return LIDEstimate(False, None, float(fallback), len(distances), "duplicate_distances")
    boundary = distances[-1]
    log_ratios = np.log(distances[:-1] / boundary)
    denominator = float(np.mean(log_ratios))
    if not np.isfinite(denominator) or denominator >= 0.0:
        return LIDEstimate(False, None, float(fallback), len(distances), "invalid_log_ratio")
    raw = -1.0 / denominator
    if not np.isfinite(raw) or raw <= 0.0:
        return LIDEstimate(False, None, float(fallback), len(distances), "invalid_estimate")
    return LIDEstimate(
        True,
        float(raw),
        float(np.clip(raw, clip_min, clip_max)),
        len(distances),
        None,
    )
