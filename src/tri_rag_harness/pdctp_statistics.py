from __future__ import annotations

from math import log, sqrt
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from .certification import plan_sample_size
from .utils import fingerprint


class PairedBoundError(ValueError):
    pass


def bonferroni_allocation(
    hypothesis_names: Sequence[str], total_alpha: float
) -> Dict[str, float]:
    names = tuple(str(value) for value in hypothesis_names)
    if not names or any(not value for value in names) or len(set(names)) != len(names):
        raise PairedBoundError("hypothesis names must be nonempty and unique")
    if not np.isfinite(total_alpha) or not 0.0 < total_alpha < 1.0:
        raise PairedBoundError("family-wise alpha must lie in (0,1)")
    per_hypothesis = float(total_alpha / len(names))
    return {name: per_hypothesis for name in names}


def _paired_terms(
    differences: np.ndarray,
    *,
    lower: float,
    upper: float,
    alpha: float,
) -> Dict[str, float]:
    if len(differences) < 2:
        raise PairedBoundError("paired bounds require at least two queries")
    if not np.all(np.isfinite(differences)):
        raise PairedBoundError("paired differences must be finite")
    tolerance = 1e-12 * max(1.0, abs(lower), abs(upper))
    if np.any(differences < lower - tolerance) or np.any(
        differences > upper + tolerance
    ):
        raise PairedBoundError("paired differences violate their frozen range")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise PairedBoundError("paired difference range must be finite and increasing")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise PairedBoundError("paired alpha must lie in (0,1)")
    width = upper - lower
    normalized = (differences - lower) / width
    n = len(normalized)
    variance = float(np.var(normalized, ddof=1))
    log_term = log(2.0 / alpha)
    variance_term = sqrt(2.0 * variance * log_term / n)
    range_term = 7.0 * log_term / (3.0 * (n - 1))
    radius_normalized = variance_term + range_term
    mean_normalized = float(np.mean(normalized))
    lower_normalized = max(0.0, mean_normalized - radius_normalized)
    upper_normalized = min(1.0, mean_normalized + radius_normalized)
    return {
        "n": n,
        "mean_difference": float(np.mean(differences)),
        "normalized_unbiased_variance": variance,
        "radius_normalized_variance_term": variance_term,
        "radius_normalized_range_term": range_term,
        "radius_normalized_total": radius_normalized,
        "radius_original_scale": radius_normalized * width,
        "lower_bound": lower + width * lower_normalized,
        "upper_bound": lower + width * upper_normalized,
    }


def make_paired_bound(
    query_ids: Sequence[str],
    left_values: Iterable[float],
    right_values: Iterable[float],
    *,
    hypothesis: str,
    metric: str,
    alpha: float,
    difference_bounds: tuple[float, float],
    side: str,
    margin: float,
    left_policy_fingerprint: str,
    right_policy_fingerprint: str,
) -> Dict[str, Any]:
    ids = tuple(str(value) for value in query_ids)
    left = np.asarray(list(left_values), dtype=np.float64)
    right = np.asarray(list(right_values), dtype=np.float64)
    if not ids or len(set(ids)) != len(ids):
        raise PairedBoundError("paired query IDs must be nonempty and unique")
    if left.ndim != 1 or right.ndim != 1 or len(left) != len(ids) or len(right) != len(ids):
        raise PairedBoundError("paired values must align with query IDs")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise PairedBoundError("paired values must be finite")
    if side not in {"lower", "upper"}:
        raise PairedBoundError("paired bound side must be lower or upper")
    if not hypothesis or not metric or not left_policy_fingerprint or not right_policy_fingerprint:
        raise PairedBoundError("paired bound identities must be nonempty")
    lower, upper = (float(difference_bounds[0]), float(difference_bounds[1]))
    differences = left - right
    terms = _paired_terms(differences, lower=lower, upper=upper, alpha=alpha)
    if side == "lower":
        passed = bool(terms["lower_bound"] >= margin)
        decision_rule = "lower_bound_greater_or_equal_margin"
    else:
        passed = bool(terms["upper_bound"] < margin)
        decision_rule = "upper_bound_strictly_less_than_margin"
    body: Dict[str, Any] = {
        "name": "paired_empirical_bernstein_bound",
        "schema_version": 1,
        "hypothesis": hypothesis,
        "metric": metric,
        "left_policy_fingerprint": left_policy_fingerprint,
        "right_policy_fingerprint": right_policy_fingerprint,
        "query_order_hash": fingerprint(list(ids)),
        "alpha": float(alpha),
        "difference_definition": "left_minus_right",
        "difference_bounds": [lower, upper],
        "side": side,
        "margin": float(margin),
        "decision_rule": decision_rule,
        **terms,
        "passed": passed,
        "per_query": [
            {
                "query_id": query_id,
                "left": float(left_value),
                "right": float(right_value),
                "difference": float(difference),
                "normalized_difference": float((difference - lower) / (upper - lower)),
            }
            for query_id, left_value, right_value, difference in zip(
                ids, left, right, differences
            )
        ],
    }
    body["fingerprint"] = fingerprint(body)
    return body


def validate_paired_bound(artifact: Mapping[str, Any]) -> None:
    raw = dict(artifact)
    stored = raw.pop("fingerprint", None)
    if not isinstance(stored, str) or fingerprint(raw) != stored:
        raise PairedBoundError("paired bound fingerprint mismatch")
    per_query = raw.get("per_query")
    bounds = raw.get("difference_bounds")
    if not isinstance(per_query, list) or not isinstance(bounds, list) or len(bounds) != 2:
        raise PairedBoundError("paired bound reconstruction data are incomplete")
    ids = [str(row["query_id"]) for row in per_query]
    left = [float(row["left"]) for row in per_query]
    right = [float(row["right"]) for row in per_query]
    rebuilt = make_paired_bound(
        ids,
        left,
        right,
        hypothesis=str(raw["hypothesis"]),
        metric=str(raw["metric"]),
        alpha=float(raw["alpha"]),
        difference_bounds=(float(bounds[0]), float(bounds[1])),
        side=str(raw["side"]),
        margin=float(raw["margin"]),
        left_policy_fingerprint=str(raw["left_policy_fingerprint"]),
        right_policy_fingerprint=str(raw["right_policy_fingerprint"]),
    )
    if rebuilt != artifact:
        raise PairedBoundError("paired bound cannot be reconstructed query by query")


def plan_paired_sample_size(
    *, alpha: float, desired_radius: float, difference_bounds: tuple[float, float]
) -> int:
    lower, upper = map(float, difference_bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise PairedBoundError("power-plan difference range must be increasing")
    if not np.isfinite(desired_radius) or desired_radius <= 0.0:
        raise PairedBoundError("desired paired radius must be positive")
    normalized_radius = desired_radius / (upper - lower)
    if normalized_radius > 1.0:
        raise PairedBoundError("desired radius exceeds the paired outcome range")
    return plan_sample_size(alpha=alpha, desired_radius=normalized_radius)


def make_power_plan(
    hypotheses: Sequence[Mapping[str, Any]], *, total_alpha: float
) -> Dict[str, Any]:
    names = [str(row["name"]) for row in hypotheses]
    allocation = bonferroni_allocation(names, total_alpha)
    planned = []
    for row in hypotheses:
        name = str(row["name"])
        bounds_raw = row["difference_bounds"]
        bounds = (float(bounds_raw[0]), float(bounds_raw[1]))
        desired_radius = float(row["desired_radius"])
        planned_n = plan_paired_sample_size(
            alpha=allocation[name],
            desired_radius=desired_radius,
            difference_bounds=bounds,
        )
        planned.append(
            {
                "name": name,
                "metric": str(row["metric"]),
                "comparison": str(row["comparison"]),
                "side": str(row["side"]),
                "margin": float(row["margin"]),
                "difference_bounds": list(bounds),
                "desired_radius": desired_radius,
                "alpha": allocation[name],
                "planned_n_worst_case_empirical_bernstein": planned_n,
            }
        )
    body: Dict[str, Any] = {
        "name": "pdctp_sample_size_and_power_plan",
        "schema_version": 1,
        "planning_method": "worst_case_empirical_bernstein_paired_difference",
        "variance_ceiling": "n/(4*(n-1))_after_range_normalization",
        "family_wise_method": "bonferroni",
        "family_wise_alpha": float(total_alpha),
        "hypotheses": planned,
        "required_role_size": max(
            row["planned_n_worst_case_empirical_bernstein"] for row in planned
        ),
        "availability_gate": "stop_before_method_evaluation_if_fresh_roles_are_too_small",
    }
    body["fingerprint"] = fingerprint(body)
    return body
