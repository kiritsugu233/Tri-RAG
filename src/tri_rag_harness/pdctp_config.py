from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from .pdctp_features import PilotDistanceFeatureSpec
from .pdctp_protocol import FIVE_ROLES
from .utils import fingerprint


class PDCTPConfigError(ValueError):
    pass


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PDCTPConfigError(f"{context} keys mismatch: missing={missing}, extra={extra}")


def _int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PDCTPConfigError(f"{name} must be an integer at least {minimum}")
    return value


def _float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PDCTPConfigError(f"{name} must be numeric")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise PDCTPConfigError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class PDCTPSyntheticConfig:
    corpus_size: int
    dimension: int
    role_counts: Mapping[str, int]
    cluster_count: int
    corpus_noise: float
    query_noise_min: float
    query_noise_max: float


@dataclass(frozen=True)
class PDCTPRetrievalConfig:
    m_prime: int
    k_gt: int
    k_ctx: int
    m_pilot: int
    s_lid: int
    min_lid_neighbors: int
    m_grid: Tuple[int, ...]
    max_rank_samples: int


@dataclass(frozen=True)
class PDCTPCalibrationCandidatesConfig:
    raw_tri_threshold_grid: Tuple[float, ...]
    lid_regularization_grid: Tuple[float, ...]
    lid_output_domain: Tuple[float, float]
    lid_fallback: float
    residual_training_levels: Tuple[float, ...]
    residual_quantiles: Tuple[float, ...]
    residual_regularization_grid: Tuple[float, ...]
    safety_offsets: Tuple[float, ...]


@dataclass(frozen=True)
class PDCTPSelectionConfig:
    retention_lower_bound_target: float
    candidate_evidence_noninferiority: float
    final_evidence_noninferiority: float
    objective: str
    tie_breaks: Tuple[str, ...]
    shuffled_profile_seed: int


@dataclass(frozen=True)
class PDCTPCertificationConfig:
    family_wise_alpha: float
    retention_target: float
    hypotheses: Tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class PDCTPLatencyConfig:
    backends: Tuple[str, ...]
    cpu_class: str
    gpu_class: str
    gpu_device_count: int
    required_packages: Mapping[str, str]
    warmups: int
    repetitions: int
    method_order_seed: int
    threads: int
    boundary_tie_overfetch: int
    paired_family_wise_alpha: float
    batching: str
    cache_state: str


@dataclass(frozen=True)
class PDCTPFoundationConfig:
    raw: Mapping[str, Any]
    config_fingerprint: str
    run_name: str
    data_seed: int
    projection_seed: int
    synthetic: PDCTPSyntheticConfig
    retrieval: PDCTPRetrievalConfig
    feature_spec: PilotDistanceFeatureSpec
    calibration: PDCTPCalibrationCandidatesConfig
    selection: PDCTPSelectionConfig
    certification: PDCTPCertificationConfig
    latency: PDCTPLatencyConfig


def load_pdctp_foundation_config(path: Path) -> PDCTPFoundationConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PDCTPConfigError("PDCTP config root must be an object")
    _exact_keys(
        raw,
        {
            "schema",
            "version",
            "run_name",
            "seeds",
            "synthetic",
            "retrieval",
            "features",
            "calibration_candidates",
            "roles",
            "selection",
            "certification",
            "latency",
        },
        "root",
    )
    if raw["schema"] != "pdctp_network_free_foundation_config_v1" or raw["version"] != 1:
        raise PDCTPConfigError("unsupported PDCTP foundation config schema")
    if not isinstance(raw["run_name"], str) or not raw["run_name"]:
        raise PDCTPConfigError("run_name must be a nonempty string")

    seeds = raw["seeds"]
    _exact_keys(seeds, {"data", "projection"}, "seeds")
    data_seed = _int(seeds["data"], "seeds.data")
    projection_seed = _int(seeds["projection"], "seeds.projection")

    synthetic_raw = raw["synthetic"]
    _exact_keys(
        synthetic_raw,
        {
            "corpus_size",
            "dimension",
            "cluster_count",
            "corpus_noise",
            "query_noise_min",
            "query_noise_max",
        },
        "synthetic",
    )
    roles_raw = raw["roles"]
    _exact_keys(roles_raw, {"schema", "counts"}, "roles")
    if roles_raw["schema"] != "pdctp_five_roles_v1":
        raise PDCTPConfigError("unsupported five-role schema")
    _exact_keys(roles_raw["counts"], set(FIVE_ROLES), "roles.counts")
    role_counts = {
        role: _int(roles_raw["counts"][role], f"roles.counts.{role}", 2)
        for role in FIVE_ROLES
    }
    synthetic = PDCTPSyntheticConfig(
        corpus_size=_int(synthetic_raw["corpus_size"], "synthetic.corpus_size", 16),
        dimension=_int(synthetic_raw["dimension"], "synthetic.dimension", 4),
        role_counts=role_counts,
        cluster_count=_int(synthetic_raw["cluster_count"], "synthetic.cluster_count", 2),
        corpus_noise=_float(synthetic_raw["corpus_noise"], "synthetic.corpus_noise"),
        query_noise_min=_float(
            synthetic_raw["query_noise_min"], "synthetic.query_noise_min"
        ),
        query_noise_max=_float(
            synthetic_raw["query_noise_max"], "synthetic.query_noise_max"
        ),
    )
    if synthetic.corpus_size % synthetic.cluster_count:
        raise PDCTPConfigError("synthetic corpus_size must divide by cluster_count")
    if not 0.0 < synthetic.corpus_noise:
        raise PDCTPConfigError("synthetic corpus noise must be positive")
    if not 0.0 < synthetic.query_noise_min <= synthetic.query_noise_max:
        raise PDCTPConfigError("synthetic query noise range is invalid")

    retrieval_raw = raw["retrieval"]
    _exact_keys(
        retrieval_raw,
        {
            "schema",
            "m_prime",
            "k_gt",
            "k_ctx",
            "m_pilot",
            "s_lid",
            "min_lid_neighbors",
            "m_grid",
            "max_rank_samples",
        },
        "retrieval",
    )
    if retrieval_raw["schema"] != "pdctp_exact_retrieval_v1":
        raise PDCTPConfigError("unsupported PDCTP retrieval schema")
    grid = tuple(_int(value, "retrieval.m_grid", 1) for value in retrieval_raw["m_grid"])
    if not grid or list(grid) != sorted(set(grid)):
        raise PDCTPConfigError("retrieval budget grid must be strictly increasing")
    retrieval = PDCTPRetrievalConfig(
        m_prime=_int(retrieval_raw["m_prime"], "retrieval.m_prime", 1),
        k_gt=_int(retrieval_raw["k_gt"], "retrieval.k_gt", 1),
        k_ctx=_int(retrieval_raw["k_ctx"], "retrieval.k_ctx", 1),
        m_pilot=_int(retrieval_raw["m_pilot"], "retrieval.m_pilot", 1),
        s_lid=_int(retrieval_raw["s_lid"], "retrieval.s_lid", 5),
        min_lid_neighbors=_int(
            retrieval_raw["min_lid_neighbors"], "retrieval.min_lid_neighbors", 2
        ),
        m_grid=grid,
        max_rank_samples=_int(
            retrieval_raw["max_rank_samples"], "retrieval.max_rank_samples", 1
        ),
    )
    minimum_budget = max(retrieval.k_gt, retrieval.m_pilot)
    if grid[0] < minimum_budget or grid[-1] != synthetic.corpus_size:
        raise PDCTPConfigError("budget grid must cover the safe lower bound and full corpus")
    if retrieval.s_lid > retrieval.m_pilot:
        raise PDCTPConfigError("s_lid cannot exceed the pilot budget")

    feature_raw = raw["features"]
    _exact_keys(
        feature_raw,
        {
            "schema",
            "lid_boundary",
            "minimum_count",
            "gap_quantiles",
            "epsilon",
            "duplicate_tolerance",
            "invalid_fill",
            "output_decimals",
        },
        "features",
    )
    try:
        feature_spec = PilotDistanceFeatureSpec(
            schema=feature_raw["schema"],
            lid_boundary=feature_raw["lid_boundary"],
            minimum_count=feature_raw["minimum_count"],
            gap_quantiles=tuple(feature_raw["gap_quantiles"]),
            epsilon=feature_raw["epsilon"],
            duplicate_tolerance=feature_raw["duplicate_tolerance"],
            invalid_fill=feature_raw["invalid_fill"],
            output_decimals=feature_raw["output_decimals"],
        )
    except ValueError as exc:
        raise PDCTPConfigError(str(exc)) from exc
    if feature_spec.lid_boundary != retrieval.s_lid:
        raise PDCTPConfigError("feature LID boundary must equal retrieval s_lid")
    if feature_spec.minimum_count > retrieval.m_pilot:
        raise PDCTPConfigError("feature minimum_count cannot exceed M_pilot")

    calibration_raw = raw["calibration_candidates"]
    _exact_keys(
        calibration_raw,
        {
            "schema",
            "raw_tri_threshold_grid",
            "lid_regularization_grid",
            "lid_output_domain",
            "lid_fallback",
            "residual_training_levels",
            "residual_quantiles",
            "residual_regularization_grid",
            "safety_offsets",
        },
        "calibration_candidates",
    )
    if calibration_raw["schema"] != "pdctp_calibration_candidates_v1":
        raise PDCTPConfigError("unsupported calibration-candidate schema")
    domain = tuple(
        _float(value, "calibration_candidates.lid_output_domain")
        for value in calibration_raw["lid_output_domain"]
    )
    if len(domain) != 2 or not 0.0 < domain[0] < domain[1]:
        raise PDCTPConfigError("LID output domain must be positive and increasing")
    calibration = PDCTPCalibrationCandidatesConfig(
        raw_tri_threshold_grid=tuple(
            _float(value, "raw_tri_threshold_grid")
            for value in calibration_raw["raw_tri_threshold_grid"]
        ),
        lid_regularization_grid=tuple(
            _float(value, "lid_regularization_grid")
            for value in calibration_raw["lid_regularization_grid"]
        ),
        lid_output_domain=(domain[0], domain[1]),
        lid_fallback=_float(calibration_raw["lid_fallback"], "lid_fallback"),
        residual_training_levels=tuple(
            _float(value, "residual_training_levels")
            for value in calibration_raw["residual_training_levels"]
        ),
        residual_quantiles=tuple(
            _float(value, "residual_quantiles")
            for value in calibration_raw["residual_quantiles"]
        ),
        residual_regularization_grid=tuple(
            _float(value, "residual_regularization_grid")
            for value in calibration_raw["residual_regularization_grid"]
        ),
        safety_offsets=tuple(
            _float(value, "safety_offsets")
            for value in calibration_raw["safety_offsets"]
        ),
    )
    candidate_sequences = {
        "raw_tri_threshold_grid": calibration.raw_tri_threshold_grid,
        "lid_regularization_grid": calibration.lid_regularization_grid,
        "residual_training_levels": calibration.residual_training_levels,
        "residual_quantiles": calibration.residual_quantiles,
        "residual_regularization_grid": calibration.residual_regularization_grid,
        "safety_offsets": calibration.safety_offsets,
    }
    for name, values in candidate_sequences.items():
        if not values or len(set(values)) != len(values):
            raise PDCTPConfigError(f"{name} must be nonempty and unique")
    if any(value < 0.0 for value in calibration.lid_regularization_grid):
        raise PDCTPConfigError("LID regularization candidates must be nonnegative")
    if any(value < 0.0 for value in calibration.residual_regularization_grid):
        raise PDCTPConfigError("residual regularization candidates must be nonnegative")
    if any(not 0.0 < value <= 1.0 for value in calibration.residual_training_levels):
        raise PDCTPConfigError("residual training levels must lie in (0,1]")
    if any(not 0.0 < value < 1.0 for value in calibration.residual_quantiles):
        raise PDCTPConfigError("residual quantiles must lie in (0,1)")
    if any(not 0.0 < value <= 1.0 for value in calibration.raw_tri_threshold_grid):
        raise PDCTPConfigError("Raw Tri threshold candidates must lie in (0,1]")
    if not domain[0] <= calibration.lid_fallback <= domain[1]:
        raise PDCTPConfigError("LID fallback must lie inside the output domain")

    selection_raw = raw["selection"]
    _exact_keys(
        selection_raw,
        {
            "schema",
            "retention_lower_bound_target",
            "candidate_evidence_noninferiority",
            "final_evidence_noninferiority",
            "objective",
            "tie_breaks",
            "shuffled_profile_seed",
        },
        "selection",
    )
    if selection_raw["schema"] != "pdctp_selection_v1":
        raise PDCTPConfigError("unsupported selection schema")
    selection = PDCTPSelectionConfig(
        retention_lower_bound_target=_float(
            selection_raw["retention_lower_bound_target"],
            "selection.retention_lower_bound_target",
        ),
        candidate_evidence_noninferiority=_float(
            selection_raw["candidate_evidence_noninferiority"],
            "selection.candidate_evidence_noninferiority",
        ),
        final_evidence_noninferiority=_float(
            selection_raw["final_evidence_noninferiority"],
            "selection.final_evidence_noninferiority",
        ),
        objective=str(selection_raw["objective"]),
        tie_breaks=tuple(str(value) for value in selection_raw["tie_breaks"]),
        shuffled_profile_seed=_int(
            selection_raw["shuffled_profile_seed"],
            "selection.shuffled_profile_seed",
            0,
        ),
    )
    if selection.objective != "common_coordinate_work" or selection.tie_breaks != (
        "lower_mean_budget",
        "canonical_fingerprint",
    ):
        raise PDCTPConfigError("selection objective or deterministic tie breaks changed")
    if not 0.0 <= selection.retention_lower_bound_target <= 1.0:
        raise PDCTPConfigError("selection retention target must lie in [0,1]")
    if not 0.0 <= selection.candidate_evidence_noninferiority <= 1.0:
        raise PDCTPConfigError("candidate evidence tolerance must lie in [0,1]")
    if not 0.0 <= selection.final_evidence_noninferiority <= 1.0:
        raise PDCTPConfigError("final evidence tolerance must lie in [0,1]")

    certification_raw = raw["certification"]
    _exact_keys(
        certification_raw,
        {"schema", "family_wise_method", "family_wise_alpha", "retention_target", "hypotheses"},
        "certification",
    )
    if (
        certification_raw["schema"] != "pdctp_certification_v1"
        or certification_raw["family_wise_method"] != "bonferroni"
    ):
        raise PDCTPConfigError("unsupported certification schema or alpha method")
    hypotheses = tuple(certification_raw["hypotheses"])
    if not hypotheses:
        raise PDCTPConfigError("certification needs predeclared hypotheses")
    hypothesis_keys = {
        "name",
        "metric",
        "comparison",
        "side",
        "margin",
        "difference_bounds",
        "desired_radius",
    }
    hypothesis_names = []
    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, Mapping):
            raise PDCTPConfigError("certification hypothesis must be an object")
        _exact_keys(hypothesis, hypothesis_keys, f"certification.hypotheses[{index}]")
        name = hypothesis["name"]
        if not isinstance(name, str) or not name:
            raise PDCTPConfigError("certification hypothesis name must be nonempty")
        hypothesis_names.append(name)
        if hypothesis["side"] not in {"lower", "upper"}:
            raise PDCTPConfigError("certification hypothesis side is invalid")
        bounds = hypothesis["difference_bounds"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise PDCTPConfigError("hypothesis difference_bounds need two values")
        lower = _float(bounds[0], "hypothesis difference lower")
        upper = _float(bounds[1], "hypothesis difference upper")
        radius = _float(hypothesis["desired_radius"], "hypothesis desired_radius")
        margin = _float(hypothesis["margin"], "hypothesis margin")
        if lower >= upper or not 0.0 < radius <= upper - lower:
            raise PDCTPConfigError("hypothesis range or desired radius is invalid")
        if not lower <= margin <= upper:
            raise PDCTPConfigError("hypothesis margin lies outside its difference range")
    if len(set(hypothesis_names)) != len(hypothesis_names):
        raise PDCTPConfigError("certification hypothesis names must be unique")
    expected_hypotheses = {
        "pdctp_absolute_retention",
        "candidate_evidence_noninferiority_fixed",
        "final_evidence_noninferiority_fixed",
        "normalized_budget_superiority_fixed",
        "normalized_budget_superiority_monotone",
        "normalized_budget_superiority_raw_tri",
    }
    if set(hypothesis_names) != expected_hypotheses:
        raise PDCTPConfigError("certification primary family is incomplete")
    certification = PDCTPCertificationConfig(
        family_wise_alpha=_float(
            certification_raw["family_wise_alpha"], "certification.family_wise_alpha"
        ),
        retention_target=_float(
            certification_raw["retention_target"], "certification.retention_target"
        ),
        hypotheses=hypotheses,
    )
    if not 0.0 < certification.family_wise_alpha < 1.0:
        raise PDCTPConfigError("family-wise alpha must lie in (0,1)")
    if not 0.0 < certification.retention_target <= 1.0:
        raise PDCTPConfigError("certification retention target must lie in (0,1]")
    retention_hypothesis = next(
        row for row in hypotheses if row["name"] == "pdctp_absolute_retention"
    )
    if (
        retention_hypothesis["comparison"] != "zero_anchor"
        or retention_hypothesis["side"] != "lower"
        or float(retention_hypothesis["margin"]) != certification.retention_target
        or list(retention_hypothesis["difference_bounds"]) != [0.0, 1.0]
    ):
        raise PDCTPConfigError("absolute retention hypothesis changed")

    latency_raw = raw["latency"]
    _exact_keys(
        latency_raw,
        {
            "schema",
            "backends",
            "hardware",
            "required_packages",
            "warmups",
            "repetitions",
            "method_order_seed",
            "threads",
            "boundary_tie_overfetch",
            "paired_family_wise_alpha",
            "batching",
            "cache_state",
        },
        "latency",
    )
    if latency_raw["schema"] != "pdctp_latency_v1":
        raise PDCTPConfigError("unsupported latency schema")
    hardware = latency_raw["hardware"]
    _exact_keys(
        hardware,
        {"cpu_class", "gpu_class", "gpu_device_count"},
        "latency.hardware",
    )
    required_packages = latency_raw["required_packages"]
    _exact_keys(
        required_packages, {"numpy", "scipy", "faiss"}, "latency.required_packages"
    )
    latency = PDCTPLatencyConfig(
        backends=tuple(str(value) for value in latency_raw["backends"]),
        cpu_class=str(hardware["cpu_class"]),
        gpu_class=str(hardware["gpu_class"]),
        gpu_device_count=_int(
            hardware["gpu_device_count"], "latency.hardware.gpu_device_count", 1
        ),
        required_packages={key: str(value) for key, value in required_packages.items()},
        warmups=_int(latency_raw["warmups"], "latency.warmups", 0),
        repetitions=_int(latency_raw["repetitions"], "latency.repetitions", 1),
        method_order_seed=_int(
            latency_raw["method_order_seed"], "latency.method_order_seed", 0
        ),
        threads=_int(latency_raw["threads"], "latency.threads", 1),
        boundary_tie_overfetch=_int(
            latency_raw["boundary_tie_overfetch"],
            "latency.boundary_tie_overfetch",
            1,
        ),
        paired_family_wise_alpha=_float(
            latency_raw["paired_family_wise_alpha"],
            "latency.paired_family_wise_alpha",
        ),
        batching=str(latency_raw["batching"]),
        cache_state=str(latency_raw["cache_state"]),
    )
    if latency.backends != ("faiss_cpu_exact", "faiss_gpu_exact"):
        raise PDCTPConfigError("latency backends must freeze exact FAISS CPU then GPU")
    if latency.batching != "single_query" or latency.cache_state != "warm_index":
        raise PDCTPConfigError("latency batching/cache protocol changed")
    if latency.gpu_device_count != 1 or latency.threads != 1:
        raise PDCTPConfigError("latency device/thread count changed")
    if not latency.cpu_class or not latency.gpu_class:
        raise PDCTPConfigError("latency hardware classes must be frozen")
    if latency.required_packages != {
        "numpy": "1.26.4",
        "scipy": "1.13.0",
        "faiss": "1.10.0",
    }:
        raise PDCTPConfigError("latency package versions changed")
    if not 0.0 < latency.paired_family_wise_alpha < 1.0:
        raise PDCTPConfigError("latency family-wise alpha must lie in (0,1)")

    return PDCTPFoundationConfig(
        raw=raw,
        config_fingerprint=fingerprint(raw),
        run_name=raw["run_name"],
        data_seed=data_seed,
        projection_seed=projection_seed,
        synthetic=synthetic,
        retrieval=retrieval,
        feature_spec=feature_spec,
        calibration=calibration,
        selection=selection,
        certification=certification,
        latency=latency,
    )
