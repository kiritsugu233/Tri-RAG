from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

from .utils import fingerprint


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SeedsConfig:
    data: int
    projection: int


@dataclass(frozen=True)
class SyntheticConfig:
    n_clusters: int
    docs_per_cluster: int
    query_tune: int
    query_cert: int
    query_test: int
    dimension: int
    cluster_noise: float
    query_noise_min: float
    query_noise_max: float


@dataclass(frozen=True)
class RetrievalConfig:
    m_prime: int
    k_gt: int
    k_ctx: int
    m_pilot: int
    s_lid: int
    min_lid_neighbors: int
    m_grid: List[int]
    batch_size: int


@dataclass(frozen=True)
class LIDConfig:
    clip_min: float
    clip_max: float
    duplicate_tolerance: float
    fallback: float


@dataclass(frozen=True)
class PolicyConfig:
    n_bins: int
    tune_target: float
    safety_margin: float
    fallback_budget: int


@dataclass(frozen=True)
class CertificationConfig:
    alpha: float
    target: float
    desired_radius: float
    per_bin: bool
    min_bin_size: int


@dataclass(frozen=True)
class HarnessConfig:
    schema_version: int
    run_name: str
    seeds: SeedsConfig
    synthetic: SyntheticConfig
    retrieval: RetrievalConfig
    lid: LIDConfig
    policy: PolicyConfig
    certification: CertificationConfig
    raw: Dict[str, Any]
    config_fingerprint: str


def _section(raw: Dict[str, Any], key: str, expected: set[str]) -> Dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be an object")
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise ConfigError(
            f"invalid {key} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def load_config(path: Union[str, Path]) -> HarnessConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be an object")
    required_root = {
        "schema_version",
        "run_name",
        "seeds",
        "synthetic",
        "retrieval",
        "lid",
        "policy",
        "certification",
    }
    if set(raw) != required_root:
        raise ConfigError(
            f"invalid root keys; missing={sorted(required_root-set(raw))}, "
            f"unknown={sorted(set(raw)-required_root)}"
        )

    seeds = _section(raw, "seeds", {"data", "projection"})
    synthetic = _section(
        raw,
        "synthetic",
        {
            "n_clusters",
            "docs_per_cluster",
            "query_tune",
            "query_cert",
            "query_test",
            "dimension",
            "cluster_noise",
            "query_noise_min",
            "query_noise_max",
        },
    )
    retrieval = _section(
        raw,
        "retrieval",
        {
            "m_prime",
            "k_gt",
            "k_ctx",
            "m_pilot",
            "s_lid",
            "min_lid_neighbors",
            "m_grid",
            "batch_size",
        },
    )
    lid = _section(
        raw, "lid", {"clip_min", "clip_max", "duplicate_tolerance", "fallback"}
    )
    policy = _section(
        raw,
        "policy",
        {"n_bins", "tune_target", "safety_margin", "fallback_budget"},
    )
    certification = _section(
        raw,
        "certification",
        {"alpha", "target", "desired_radius", "per_bin", "min_bin_size"},
    )

    if raw["schema_version"] != 1:
        raise ConfigError("schema_version must be 1")
    if not isinstance(raw["run_name"], str) or not raw["run_name"]:
        raise ConfigError("run_name must be a nonempty string")

    for key in ("data", "projection"):
        if isinstance(seeds[key], bool) or not isinstance(seeds[key], int) or seeds[key] < 0:
            raise ConfigError(f"seeds.{key} must be a nonnegative integer")
    for key in (
        "n_clusters",
        "docs_per_cluster",
        "query_tune",
        "query_cert",
        "query_test",
        "dimension",
    ):
        _positive_int(synthetic[key], f"synthetic.{key}")
    for key in ("cluster_noise", "query_noise_min", "query_noise_max"):
        if not isinstance(synthetic[key], (int, float)) or synthetic[key] <= 0:
            raise ConfigError(f"synthetic.{key} must be positive")
    if synthetic["query_noise_min"] > synthetic["query_noise_max"]:
        raise ConfigError("query_noise_min cannot exceed query_noise_max")

    for key in (
        "m_prime",
        "k_gt",
        "k_ctx",
        "m_pilot",
        "s_lid",
        "min_lid_neighbors",
        "batch_size",
    ):
        _positive_int(retrieval[key], f"retrieval.{key}")
    grid = retrieval["m_grid"]
    if not isinstance(grid, list) or not grid:
        raise ConfigError("retrieval.m_grid must be a nonempty list")
    for value in grid:
        _positive_int(value, "retrieval.m_grid item")
    if grid != sorted(set(grid)):
        raise ConfigError("retrieval.m_grid must be strictly increasing")
    minimum_budget = max(retrieval["k_gt"], retrieval["k_ctx"], retrieval["m_pilot"])
    if grid[0] < minimum_budget:
        raise ConfigError(f"all budgets must be >= {minimum_budget}")
    corpus_size = synthetic["n_clusters"] * synthetic["docs_per_cluster"]
    if grid[-1] > corpus_size:
        raise ConfigError("maximum budget cannot exceed synthetic corpus size")
    if retrieval["s_lid"] > retrieval["m_pilot"]:
        raise ConfigError("s_lid cannot exceed m_pilot")
    if retrieval["min_lid_neighbors"] > retrieval["s_lid"]:
        raise ConfigError("min_lid_neighbors cannot exceed s_lid")

    for key in ("clip_min", "clip_max", "duplicate_tolerance", "fallback"):
        if not isinstance(lid[key], (int, float)) or lid[key] <= 0:
            raise ConfigError(f"lid.{key} must be positive")
    if lid["clip_min"] >= lid["clip_max"]:
        raise ConfigError("lid.clip_min must be below clip_max")
    _positive_int(policy["n_bins"], "policy.n_bins")
    if policy["fallback_budget"] not in grid:
        raise ConfigError("policy.fallback_budget must be in m_grid")
    if not 0 <= policy["tune_target"] <= 1:
        raise ConfigError("policy.tune_target must lie in [0,1]")
    if not 0 <= policy["safety_margin"] <= 1:
        raise ConfigError("policy.safety_margin must lie in [0,1]")
    if policy["tune_target"] + policy["safety_margin"] > 1:
        raise ConfigError("tune_target + safety_margin cannot exceed 1")
    if not 0 < certification["alpha"] < 1:
        raise ConfigError("certification.alpha must lie in (0,1)")
    for key in ("target", "desired_radius"):
        if not 0 < certification[key] <= 1:
            raise ConfigError(f"certification.{key} must lie in (0,1]")
    if not isinstance(certification["per_bin"], bool):
        raise ConfigError("certification.per_bin must be Boolean")
    _positive_int(certification["min_bin_size"], "certification.min_bin_size")

    return HarnessConfig(
        schema_version=1,
        run_name=raw["run_name"],
        seeds=SeedsConfig(**seeds),
        synthetic=SyntheticConfig(**synthetic),
        retrieval=RetrievalConfig(**retrieval),
        lid=LIDConfig(**lid),
        policy=PolicyConfig(**policy),
        certification=CertificationConfig(**certification),
        raw=raw,
        config_fingerprint=fingerprint(raw),
    )
