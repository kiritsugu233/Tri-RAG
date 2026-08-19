from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log, sqrt
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class BernsteinResult:
    n: int
    mean: float
    unbiased_variance: float
    alpha: float
    radius_variance_term: float
    radius_range_term: float
    radius_total: float
    lower_bound_unclipped: float
    lower_bound: float

    def serialize(self) -> Dict[str, Any]:
        return asdict(self)


def empirical_bernstein(values: Iterable[float], alpha: float) -> BernsteinResult:
    observations = np.asarray(list(values), dtype=np.float64)
    if observations.ndim != 1 or len(observations) < 2:
        raise ValueError("empirical Bernstein bound requires at least two observations")
    if not np.all(np.isfinite(observations)) or np.any((observations < 0) | (observations > 1)):
        raise ValueError("observations must be finite and lie in [0,1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    n = len(observations)
    mean = float(np.mean(observations))
    variance = float(np.var(observations, ddof=1))
    log_term = log(2.0 / alpha)
    variance_term = sqrt(2.0 * variance * log_term / n)
    range_term = 7.0 * log_term / (3.0 * (n - 1))
    radius = variance_term + range_term
    lower_unclipped = mean - radius
    return BernsteinResult(
        n=n,
        mean=mean,
        unbiased_variance=variance,
        alpha=float(alpha),
        radius_variance_term=variance_term,
        radius_range_term=range_term,
        radius_total=radius,
        lower_bound_unclipped=lower_unclipped,
        lower_bound=max(0.0, lower_unclipped),
    )


def plan_sample_size(alpha: float, desired_radius: float, maximum: int = 1_000_000) -> int:
    """Use the sharp finite-n sample-variance ceiling n/(4(n-1)) for [0,1]."""
    if not 0 < desired_radius <= 1:
        raise ValueError("desired_radius must lie in (0,1]")
    for n in range(2, maximum + 1):
        worst_variance = n / (4.0 * (n - 1))
        log_term = log(2.0 / alpha)
        radius = sqrt(2.0 * worst_variance * log_term / n) + 7.0 * log_term / (
            3.0 * (n - 1)
        )
        if radius <= desired_radius:
            return n
    raise ValueError("sample-size plan exceeds configured search maximum")


def make_certificate(
    values: Iterable[float],
    *,
    alpha: float,
    target: float,
    policy_fingerprint: str,
    split_hash: str,
    metric: str = "embedding_retention",
    planned_n: Optional[int] = None,
) -> Dict[str, Any]:
    result = empirical_bernstein(values, alpha)
    artifact: Dict[str, Any] = result.serialize()
    artifact.update(
        {
            "schema_version": 1,
            "policy_fingerprint": policy_fingerprint,
            "split_hash": split_hash,
            "metric": metric,
            "target": float(target),
            "passed": bool(result.lower_bound >= target),
            "planned_n": planned_n,
            "sample_size_sufficient": planned_n is None or result.n >= planned_n,
        }
    )
    return artifact


def per_bin_certificates(
    records: List[Mapping[str, Any]],
    *,
    bin_count: int,
    alpha_total: float,
    target: float,
    policy_fingerprint: str,
    split_hash: str,
    min_bin_size: int,
) -> List[Dict[str, Any]]:
    results = []
    alpha_bin = alpha_total / bin_count
    for bin_index in range(bin_count):
        values = [
            float(record["embedding_retention"])
            for record in records
            if int(record["lid_bin"]) == bin_index
        ]
        if len(values) < max(2, min_bin_size):
            results.append(
                {
                    "bin_index": bin_index,
                    "n": len(values),
                    "alpha": alpha_bin,
                    "target": target,
                    "passed": False,
                    "status": "insufficient_sample",
                }
            )
            continue
        certificate = make_certificate(
            values,
            alpha=alpha_bin,
            target=target,
            policy_fingerprint=policy_fingerprint,
            split_hash=split_hash,
        )
        certificate["bin_index"] = bin_index
        certificate["status"] = "evaluated"
        results.append(certificate)
    return results


def validate_certificate_identity(
    artifact: Mapping[str, Any], *, policy_fingerprint: str, split_hash: str
) -> None:
    if artifact.get("policy_fingerprint") != policy_fingerprint:
        raise ValueError("certificate policy fingerprint mismatch")
    if artifact.get("split_hash") != split_hash:
        raise ValueError("certificate split fingerprint mismatch")
