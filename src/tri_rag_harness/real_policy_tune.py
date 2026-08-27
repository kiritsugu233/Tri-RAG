"""Fit and freeze real SciFact retrieval policies on query_tune only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from .certification import empirical_bernstein
from .embeddings import load_embedding_array
from .indexes import ExactSquaredL2Index
from .lid import estimate_lid_from_squared_distances
from .policies import (
    CompiledTriPredictPolicy,
    MonotoneBinnedPolicy,
    PolicyDecision,
    TriPredictPolicy,
)
from .projection import dense_gaussian_projection, project_rows, projection_metadata
from .real_dimension_sweep import _exact_projected_rankings, _ranking_hash
from .text_embeddings import load_text_embedding_config, validate_text_embedding_cache
from .utils import fingerprint, stable_id_hash, write_json


class RealPolicyTuneError(ValueError):
    pass


@dataclass(frozen=True)
class RealPolicyTuneConfig:
    raw: Dict[str, Any]
    config_fingerprint: str
    dataset_manifest_fingerprint: str
    embedding_config_fingerprint: str
    embedding_request_fingerprint: str
    embedding_manifest_fingerprint: str
    original_baseline_result_fingerprint: str
    dimension_selection_result_fingerprint: str
    dimension_selection_fingerprint: str
    frozen_projection_fingerprint: str
    projection_fingerprint: str
    evaluation_split: str
    lid_decimal_places: int
    feature_version: str
    compiled_policy_role: str
    projection_seed: int
    m_prime: int
    query_batch_size: int
    k_ctx: int
    k_gt: int
    m_pilot: int
    s_lid: int
    min_lid_neighbors: int
    m_grid: list[int]
    lid_clip_min: float
    lid_clip_max: float
    duplicate_tolerance: float
    lid_fallback: float
    selection_alpha: float
    selection_target: float
    binned_n_bins: int
    binned_target_grid: list[float]
    fallback_budget: int
    tri_target_grid: list[float]
    max_rank_samples: int
    safety_quantiles: list[Optional[float]]
    compiled_lid_min: float
    compiled_lid_max: float
    compile_validation_samples: int


_SHA256_LENGTH = 64
_POLICY_FLOAT_DECIMALS = 12
_SUPPORTED_LID_DECIMAL_PLACES = 9
_SUPPORTED_FEATURE_VERSION = "pilot_rerank_lid_rounded_9_v2"
_COMPILED_POLICY_ROLE = (
    "platform_deployment_artifact_excluded_from_scientific_identity"
)
_SCIENTIFIC_RESULT_NAMES = (
    "per_query.jsonl",
    "selection.json",
    "fixed_policies.json",
    "monotone_binned_policy.json",
    "tri_predict_policy.json",
    "summary.json",
    "report.md",
)
_SELECTION_RULE = {
    "common_eligibility": (
        "query_tune empirical-Bernstein embedding-retention lower bound must "
        "reach the predeclared target; this is a tune score, not certification"
    ),
    "monotone_binned": (
        "fit four tune-only LID quantile bins at each predeclared bin-mean target; "
        "among eligible policies minimize mean budget, then maximize lower bound, "
        "then choose the smaller bin-mean target"
    ),
    "tri_predict": (
        "freeze rank quadrature; cross the predeclared analytic prediction-target "
        "grid with residual safety quantiles on tune; among eligible policies "
        "minimize mean budget, then maximize lower bound, then choose the smaller "
        "effective prediction requirement"
    ),
    "fixed": "evaluate every budget in the already frozen common M_grid",
    "protected_data": (
        "query_cert and query_test vectors are not indexed, searched, or scored; "
        "all selected policy artifacts are written before certification"
    ),
    "labels": "evidence labels and evidence metrics do not enter policy selection",
}


def _exact_keys(value: Any, expected: set[str], name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RealPolicyTuneError(f"{name} must be an object")
    if set(value) != expected:
        raise RealPolicyTuneError(
            f"invalid {name} keys; missing={sorted(expected-set(value))}, "
            f"unknown={sorted(set(value)-expected)}"
        )
    return dict(value)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RealPolicyTuneError(f"{name} must be a positive integer")
    return value


def _strict_integer_grid(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise RealPolicyTuneError(f"{name} must be a nonempty list")
    result = [_positive_integer(item, f"{name} item") for item in value]
    if result != sorted(set(result)):
        raise RealPolicyTuneError(f"{name} must be strictly increasing")
    return result


def _fingerprint_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise RealPolicyTuneError(f"{name} must be a SHA-256 fingerprint")
    result = value.strip().lower()
    if len(result) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise RealPolicyTuneError(f"{name} must be a SHA-256 fingerprint")
    return result


def _unit_interval(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RealPolicyTuneError(f"{name} must be numeric")
    result = float(value)
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not lower_ok or result > 1.0:
        boundary = "[0,1]" if allow_zero else "(0,1]"
        raise RealPolicyTuneError(f"{name} must lie in {boundary}")
    return result


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RealPolicyTuneError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise RealPolicyTuneError(f"{name} must be finite and positive")
    return result


def load_real_policy_tune_config(path: Union[str, Path]) -> RealPolicyTuneConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealPolicyTuneError(f"cannot load policy config {config_path}: {exc}") from exc
    root = _exact_keys(
        raw,
        {
            "schema_version",
            "benchmark",
            "dataset_manifest_fingerprint",
            "embedding_config_fingerprint",
            "embedding_request_fingerprint",
            "embedding_manifest_fingerprint",
            "original_baseline_result_fingerprint",
            "dimension_selection_result_fingerprint",
            "dimension_selection_fingerprint",
            "frozen_projection_fingerprint",
            "projection_fingerprint",
            "evaluation_split",
            "determinism",
            "projection",
            "search",
            "lid",
            "selection",
            "monotone_binned",
            "tri_predict",
        },
        "root",
    )
    if root["schema_version"] != 2 or root["benchmark"] != "real_tune_only_policy_fit_v2":
        raise RealPolicyTuneError("unsupported policy config schema/benchmark")
    if root["evaluation_split"] != "query_tune":
        raise RealPolicyTuneError("real policy fitting accepts query_tune only")
    determinism = _exact_keys(
        root["determinism"],
        {"lid_decimal_places", "feature_version", "compiled_policy_role"},
        "determinism",
    )
    lid_decimal_places = _positive_integer(
        determinism["lid_decimal_places"], "determinism.lid_decimal_places"
    )
    if (
        lid_decimal_places != _SUPPORTED_LID_DECIMAL_PLACES
        or determinism["feature_version"] != _SUPPORTED_FEATURE_VERSION
        or determinism["compiled_policy_role"] != _COMPILED_POLICY_ROLE
    ):
        raise RealPolicyTuneError("unsupported cross-platform determinism contract")
    projection = _exact_keys(
        root["projection"],
        {"family", "seed", "m_prime", "post_projection_normalize"},
        "projection",
    )
    if projection["family"] != "dense_gaussian_n0_variance_1_over_m_prime":
        raise RealPolicyTuneError("projection family is not frozen")
    if projection["post_projection_normalize"] is not False:
        raise RealPolicyTuneError("projected vectors must not be renormalized")
    search = _exact_keys(
        root["search"],
        {
            "normalized_inputs",
            "distance",
            "arithmetic",
            "stable_tie_break",
            "query_batch_size",
            "k_ctx",
            "k_gt",
            "m_pilot",
            "s_lid",
            "min_lid_neighbors",
            "m_grid",
        },
        "search",
    )
    if (
        search["normalized_inputs"] is not True
        or search["distance"] != "squared_l2"
        or search["arithmetic"] != "numpy_float64"
        or search["stable_tie_break"] != "lexicographic_doc_id"
    ):
        raise RealPolicyTuneError("search geometry/arithmetic contract is not frozen")
    k_ctx = _positive_integer(search["k_ctx"], "search.k_ctx")
    k_gt = _positive_integer(search["k_gt"], "search.k_gt")
    m_pilot = _positive_integer(search["m_pilot"], "search.m_pilot")
    s_lid = _positive_integer(search["s_lid"], "search.s_lid")
    min_neighbors = _positive_integer(
        search["min_lid_neighbors"], "search.min_lid_neighbors"
    )
    m_grid = _strict_integer_grid(search["m_grid"], "search.m_grid")
    if not k_ctx <= k_gt <= s_lid <= m_pilot or m_grid[0] != m_pilot:
        raise RealPolicyTuneError(
            "require k_ctx <= k_gt <= s_lid <= m_pilot == first M_grid value"
        )
    if min_neighbors > s_lid:
        raise RealPolicyTuneError("min_lid_neighbors cannot exceed s_lid")
    lid = _exact_keys(
        root["lid"],
        {"clip_min", "clip_max", "duplicate_tolerance", "fallback"},
        "lid",
    )
    clip_min = _positive_float(lid["clip_min"], "lid.clip_min")
    clip_max = _positive_float(lid["clip_max"], "lid.clip_max")
    fallback = _positive_float(lid["fallback"], "lid.fallback")
    duplicate_tolerance = _positive_float(
        lid["duplicate_tolerance"], "lid.duplicate_tolerance"
    )
    if not clip_min < clip_max or not clip_min <= fallback <= clip_max:
        raise RealPolicyTuneError("invalid LID clipping/fallback interval")
    selection = _exact_keys(
        root["selection"],
        {
            "metric",
            "alpha",
            "target",
            "statistic_role",
            "objective",
            "cost_formula",
            "tie_break",
            "evidence_labels_used",
        },
        "selection",
    )
    expected_formula = (
        "(corpus_size + embedding_dimension) * m_prime + "
        "embedding_dimension * mean_budget"
    )
    if (
        selection["metric"] != "embedding_neighbor_retention_at_k_gt"
        or selection["statistic_role"] != "tune_selection_score_not_certificate"
        or selection["objective"] != "coordinate_multiply_adds_per_query"
        or selection["cost_formula"] != expected_formula
        or selection["evidence_labels_used"] is not False
        or selection["tie_break"]
        != [
            "smaller_mean_budget",
            "higher_lower_bound",
            "smaller_policy_hyperparameter",
        ]
    ):
        raise RealPolicyTuneError("policy selection contract is not frozen")
    alpha = _unit_interval(selection["alpha"], "selection.alpha")
    if alpha >= 1.0:
        raise RealPolicyTuneError("selection.alpha must lie in (0,1)")
    target = _unit_interval(selection["target"], "selection.target")
    binned = _exact_keys(
        root["monotone_binned"],
        {"n_bins", "bin_mean_target_grid", "fallback_budget"},
        "monotone_binned",
    )
    raw_targets = binned["bin_mean_target_grid"]
    if not isinstance(raw_targets, list) or not raw_targets:
        raise RealPolicyTuneError("bin_mean_target_grid must be nonempty")
    target_grid = [
        _unit_interval(value, "bin_mean_target_grid item") for value in raw_targets
    ]
    if target_grid != sorted(set(target_grid)):
        raise RealPolicyTuneError("bin_mean_target_grid must be strictly increasing")
    if target_grid[-1] != 1.0:
        raise RealPolicyTuneError("bin_mean_target_grid must end at 1")
    fallback_budget = _positive_integer(
        binned["fallback_budget"], "monotone_binned.fallback_budget"
    )
    if fallback_budget != m_grid[-1]:
        raise RealPolicyTuneError("fallback budget must equal terminal M_grid value")
    tri = _exact_keys(
        root["tri_predict"],
        {
            "target_grid",
            "max_rank_samples",
            "safety_quantiles",
            "compiled_lid_min",
            "compiled_lid_max",
            "compile_validation_samples",
        },
        "tri_predict",
    )
    quantiles = tri["safety_quantiles"]
    if not isinstance(quantiles, list) or not quantiles or quantiles[0] is not None:
        raise RealPolicyTuneError("safety_quantiles must begin with null")
    parsed_quantiles: list[Optional[float]] = [None]
    parsed_quantiles.extend(
        _unit_interval(value, "safety_quantiles item") for value in quantiles[1:]
    )
    if parsed_quantiles[1:] != sorted(set(parsed_quantiles[1:])):
        raise RealPolicyTuneError("numeric safety quantiles must be strictly increasing")
    raw_tri_targets = tri["target_grid"]
    if not isinstance(raw_tri_targets, list) or not raw_tri_targets:
        raise RealPolicyTuneError("tri_predict.target_grid must be nonempty")
    tri_target_grid = [
        _unit_interval(value, "tri_predict.target_grid item")
        for value in raw_tri_targets
    ]
    if tri_target_grid != sorted(set(tri_target_grid)):
        raise RealPolicyTuneError("tri_predict.target_grid must be strictly increasing")
    if tri_target_grid[0] != target or tri_target_grid[-1] != 1.0:
        raise RealPolicyTuneError(
            "Tri-Predict target grid must start at selection target and end at 1"
        )
    compiled_min = _positive_float(tri["compiled_lid_min"], "compiled_lid_min")
    compiled_max = _positive_float(tri["compiled_lid_max"], "compiled_lid_max")
    if compiled_min != clip_min or compiled_max != clip_max:
        raise RealPolicyTuneError("compiled LID domain must equal clipping interval")
    return RealPolicyTuneConfig(
        raw=root,
        config_fingerprint=fingerprint(root),
        dataset_manifest_fingerprint=_fingerprint_string(
            root["dataset_manifest_fingerprint"], "dataset_manifest_fingerprint"
        ),
        embedding_config_fingerprint=_fingerprint_string(
            root["embedding_config_fingerprint"], "embedding_config_fingerprint"
        ),
        embedding_request_fingerprint=_fingerprint_string(
            root["embedding_request_fingerprint"], "embedding_request_fingerprint"
        ),
        embedding_manifest_fingerprint=_fingerprint_string(
            root["embedding_manifest_fingerprint"], "embedding_manifest_fingerprint"
        ),
        original_baseline_result_fingerprint=_fingerprint_string(
            root["original_baseline_result_fingerprint"],
            "original_baseline_result_fingerprint",
        ),
        dimension_selection_result_fingerprint=_fingerprint_string(
            root["dimension_selection_result_fingerprint"],
            "dimension_selection_result_fingerprint",
        ),
        dimension_selection_fingerprint=_fingerprint_string(
            root["dimension_selection_fingerprint"],
            "dimension_selection_fingerprint",
        ),
        frozen_projection_fingerprint=_fingerprint_string(
            root["frozen_projection_fingerprint"], "frozen_projection_fingerprint"
        ),
        projection_fingerprint=_fingerprint_string(
            root["projection_fingerprint"], "projection_fingerprint"
        ),
        evaluation_split="query_tune",
        lid_decimal_places=lid_decimal_places,
        feature_version=_SUPPORTED_FEATURE_VERSION,
        compiled_policy_role=_COMPILED_POLICY_ROLE,
        projection_seed=_positive_integer(projection["seed"], "projection.seed"),
        m_prime=_positive_integer(projection["m_prime"], "projection.m_prime"),
        query_batch_size=_positive_integer(
            search["query_batch_size"], "search.query_batch_size"
        ),
        k_ctx=k_ctx,
        k_gt=k_gt,
        m_pilot=m_pilot,
        s_lid=s_lid,
        min_lid_neighbors=min_neighbors,
        m_grid=m_grid,
        lid_clip_min=clip_min,
        lid_clip_max=clip_max,
        duplicate_tolerance=duplicate_tolerance,
        lid_fallback=fallback,
        selection_alpha=alpha,
        selection_target=target,
        binned_n_bins=_positive_integer(binned["n_bins"], "monotone_binned.n_bins"),
        binned_target_grid=target_grid,
        fallback_budget=fallback_budget,
        tri_target_grid=tri_target_grid,
        max_rank_samples=_positive_integer(
            tri["max_rank_samples"], "tri_predict.max_rank_samples"
        ),
        safety_quantiles=parsed_quantiles,
        compiled_lid_min=compiled_min,
        compiled_lid_max=compiled_max,
        compile_validation_samples=_positive_integer(
            tri["compile_validation_samples"], "compile_validation_samples"
        ),
    )


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealPolicyTuneError(f"cannot load {description} {path}: {exc}") from exc


def _load_jsonl(path: Path, description: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RealPolicyTuneError(
                        f"non-object {description} at {path}:{line_number}"
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise RealPolicyTuneError(f"cannot load {description} {path}: {exc}") from exc
    if not rows:
        raise RealPolicyTuneError(f"{description} cannot be empty")
    return rows


def _file_identity(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def _canonical_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    result = float(np.round(float(value), decimals=_POLICY_FLOAT_DECIMALS))
    if not np.isfinite(result):
        raise RealPolicyTuneError("nonfinite deterministic policy value")
    return result


def _canonical_lid_float(value: Optional[float], decimal_places: int) -> Optional[float]:
    if value is None:
        return None
    if decimal_places != _SUPPORTED_LID_DECIMAL_PLACES:
        raise RealPolicyTuneError("unsupported LID canonicalization precision")
    result = float(np.round(float(value), decimals=decimal_places))
    if not np.isfinite(result):
        raise RealPolicyTuneError("nonfinite deterministic LID value")
    return result


def _validate_tune_only(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) < 2:
        raise RealPolicyTuneError("policy fitting requires at least two tune records")
    unexpected = sorted({str(record.get("split")) for record in records} - {"query_tune"})
    if unexpected:
        raise RealPolicyTuneError(
            f"policy fitting accepts query_tune only; found {unexpected}"
        )


def _validate_dimension_selection(
    run_dir: Path,
    config: RealPolicyTuneConfig,
    dataset_manifest: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], list[Dict[str, Any]]]:
    manifest = _load_json(run_dir / "manifest.json", "dimension manifest")
    if (
        manifest.get("kind") != "real_fixed_dimension_tune_manifest_v1"
        or manifest.get("data_scope") != "query_tune_only"
    ):
        raise RealPolicyTuneError("dimension-selection run is not tune-only")
    if (
        manifest.get("result_fingerprint")
        != config.dimension_selection_result_fingerprint
        or manifest.get("selection_fingerprint")
        != config.dimension_selection_fingerprint
        or manifest.get("frozen_projection_fingerprint")
        != config.frozen_projection_fingerprint
        or manifest.get("original_baseline_result_fingerprint")
        != config.original_baseline_result_fingerprint
        or manifest.get("dataset_manifest_fingerprint")
        != config.dataset_manifest_fingerprint
        or manifest.get("embedding_manifest_fingerprint")
        != config.embedding_manifest_fingerprint
        or manifest.get("query_tune_id_hash")
        != dataset_manifest["splits"]["query_tune"]["id_hash"]
    ):
        raise RealPolicyTuneError("dimension-selection input identity mismatch")
    artifacts = manifest.get("result_artifacts")
    required = (
        "per_query.jsonl",
        "selection.json",
        "selected_projection.json",
        "summary.json",
        "report.md",
    )
    if not isinstance(artifacts, dict):
        raise RealPolicyTuneError("dimension result artifact identities are missing")
    observed = {name: _file_identity(run_dir / name) for name in required}
    if any(artifacts.get(name) != observed[name] for name in required):
        raise RealPolicyTuneError("dimension result artifact identity mismatch")
    result_identity = {
        "config_fingerprint": manifest.get("config_fingerprint"),
        "dataset_manifest_fingerprint": manifest.get("dataset_manifest_fingerprint"),
        "embedding_manifest_fingerprint": manifest.get("embedding_manifest_fingerprint"),
        "original_baseline_result_fingerprint": manifest.get(
            "original_baseline_result_fingerprint"
        ),
        "query_tune_id_hash": manifest.get("query_tune_id_hash"),
        "artifacts": observed,
    }
    if fingerprint(result_identity) != manifest.get("result_fingerprint"):
        raise RealPolicyTuneError("dimension result fingerprint is invalid")
    selection = _load_json(run_dir / "selection.json", "dimension selection")
    frozen = _load_json(run_dir / "selected_projection.json", "frozen projection")
    if (
        selection.get("selection_fingerprint")
        != config.dimension_selection_fingerprint
        or selection.get("selected", {}).get("m_prime") != config.m_prime
        or frozen.get("frozen_fingerprint") != config.frozen_projection_fingerprint
        or frozen.get("fingerprint") != config.projection_fingerprint
        or frozen.get("m_prime") != config.m_prime
        or frozen.get("seed") != config.projection_seed
        or frozen.get("post_projection_normalized") is not False
        or frozen.get("m_grid") != config.m_grid
        or frozen.get("m_pilot") != config.m_pilot
        or frozen.get("k_gt") != config.k_gt
        or frozen.get("k_ctx") != config.k_ctx
    ):
        raise RealPolicyTuneError("frozen projection/search contract mismatch")
    all_records = _load_jsonl(
        run_dir / "per_query.jsonl", "dimension per-query records"
    )
    selected_records = [
        record for record in all_records if record.get("m_prime") == config.m_prime
    ]
    _validate_tune_only(selected_records)
    if (
        len(selected_records) != dataset_manifest["splits"]["query_tune"]["n"]
        or stable_id_hash([str(record.get("query_id")) for record in selected_records])
        != dataset_manifest["splits"]["query_tune"]["id_hash"]
    ):
        raise RealPolicyTuneError("selected dimension records do not match tune IDs")
    return selection, frozen, selected_records


def _coordinate_work(
    *, corpus_size: int, dimension: int, m_prime: int, mean_budget: float
) -> Dict[str, float]:
    projection = float(dimension * m_prime)
    projected_scan = float(corpus_size * m_prime)
    rerank = float(dimension * mean_budget)
    total = projection + projected_scan + rerank
    original = float(corpus_size * dimension)
    return {
        "query_projection": projection,
        "projected_full_scan": projected_scan,
        "mean_original_rerank": rerank,
        "total": total,
        "original_full_scan_reference": original,
        "reduction_fraction_vs_original_full_scan": 1.0 - total / original,
    }


def _evaluate_decisions(
    records: Sequence[Mapping[str, Any]],
    decisions: Sequence[PolicyDecision],
    *,
    config: RealPolicyTuneConfig,
    corpus_size: int,
    dimension: int,
    fixed_reference_budget: int,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    _validate_tune_only(records)
    if len(records) != len(decisions):
        raise RealPolicyTuneError("policy decisions are not row-aligned")
    budgets: list[int] = []
    retentions: list[float] = []
    query_values: list[Dict[str, Any]] = []
    for record, decision in zip(records, decisions):
        if decision.budget not in config.m_grid or decision.budget < config.m_pilot:
            raise RealPolicyTuneError("policy emitted an unsafe budget")
        retention = float(record["retention_by_budget"][str(decision.budget)])
        budgets.append(decision.budget)
        retentions.append(retention)
        query_values.append(
            {
                "chosen_m": decision.budget,
                "embedding_retention": retention,
                "lid_bin": decision.bin_index,
                "used_lid_fallback": decision.used_fallback,
                "policy_saturated": decision.saturated,
                "predicted_retention": _canonical_float(
                    decision.predicted_retention
                ),
                "raw_predicted_retention": _canonical_float(
                    decision.raw_predicted_retention
                ),
            }
        )
    bound = empirical_bernstein(retentions, config.selection_alpha).serialize()
    mean_budget = float(np.mean(budgets))
    evaluation = {
        "n": len(records),
        "tune_retention_bound": bound,
        "eligible": bool(bound["lower_bound"] >= config.selection_target),
        "retention_distribution": {
            "mean": float(np.mean(retentions)),
            "median": float(np.median(retentions)),
            "p05": float(np.quantile(retentions, 0.05)),
            "minimum": float(np.min(retentions)),
            "below_target_n": int(
                sum(value < config.selection_target for value in retentions)
            ),
        },
        "budget": {
            "mean": mean_budget,
            "median": float(np.median(budgets)),
            "p95": float(np.quantile(budgets, 0.95)),
            "p99": float(np.quantile(budgets, 0.99)),
            "distribution": {
                str(budget): budgets.count(budget) for budget in config.m_grid
            },
        },
        "fallback_n": int(sum(decision.used_fallback for decision in decisions)),
        "saturated_n": int(sum(decision.saturated for decision in decisions)),
        "candidate_saving_vs_tune_fixed_reference": 1.0
        - mean_budget / fixed_reference_budget,
        "coordinate_work": _coordinate_work(
            corpus_size=corpus_size,
            dimension=dimension,
            m_prime=config.m_prime,
            mean_budget=mean_budget,
        ),
    }
    fixed_work = _coordinate_work(
        corpus_size=corpus_size,
        dimension=dimension,
        m_prime=config.m_prime,
        mean_budget=float(fixed_reference_budget),
    )["total"]
    evaluation["coordinate_work_reduction_vs_tune_fixed_reference"] = (
        1.0 - evaluation["coordinate_work"]["total"] / fixed_work
    )
    return evaluation, query_values


def _choose_tri_from_raw(
    raw: Optional[Mapping[int, float]], policy: TriPredictPolicy
) -> PolicyDecision:
    if raw is None:
        return PolicyDecision(policy.grid[-1], -1, True, False, None, None)
    for budget in policy.grid:
        if budget == policy.corpus_size:
            corrected = 1.0
        else:
            corrected = max(0.0, float(raw[budget]) - policy.safety_correction)
            if policy.target == 1.0:
                continue
        if corrected >= policy.target:
            return PolicyDecision(
                budget,
                -1,
                False,
                False,
                corrected,
                float(raw[budget]),
            )
    maximum = policy.grid[-1]
    corrected = (
        1.0
        if maximum == policy.corpus_size
        else max(0.0, float(raw[maximum]) - policy.safety_correction)
    )
    return PolicyDecision(
        maximum,
        -1,
        False,
        True,
        corrected,
        float(raw[maximum]),
    )


def _select_candidate(
    candidates: Sequence[Mapping[str, Any]], *, hyperparameter: str
) -> Mapping[str, Any]:
    eligible = [candidate for candidate in candidates if candidate["evaluation"]["eligible"]]
    if not eligible:
        raise RuntimeError("no predeclared policy candidate reached the tune target")
    return min(
        eligible,
        key=lambda candidate: (
            candidate["evaluation"]["budget"]["mean"],
            -candidate["evaluation"]["tune_retention_bound"]["lower_bound"],
            candidate[hyperparameter],
        ),
    )


def _serialize_lid(estimate: Any, decimal_places: int) -> Dict[str, Any]:
    return {
        "valid": bool(estimate.valid),
        "raw": _canonical_lid_float(estimate.raw, decimal_places),
        "clipped": _canonical_lid_float(estimate.clipped, decimal_places),
        "valid_distance_count": int(estimate.valid_distance_count),
        "reason": estimate.reason,
    }


def _report(summary: Mapping[str, Any]) -> str:
    fixed = summary["fixed_reference"]
    binned = summary["monotone_binned"]
    tri = summary["tri_predict"]
    lines = [
        "# SciFact tune-only policy fitting",
        "",
        "All LID features, retention outcomes, safety corrections, and policy "
        "choices use `query_tune` only. These empirical-Bernstein values are tune "
        "selection scores, not certificates. Evidence labels do not enter fitting.",
        "",
        f"- frozen projection: `m_prime={summary['m_prime']}`, seed "
        f"`{summary['projection_seed']}`",
        f"- pilot/LID: `M_pilot={summary['m_pilot']}`, "
        f"`s_lid={summary['s_lid']}`",
        f"- fixed tune reference: `M={fixed['budget']}`",
        "",
        "| policy | mean M | p95 M | mean retention | tune lower bound | candidate saving | coordinate saving | saturation |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, value in (
        ("fixed", fixed),
        ("monotone binned", binned),
        ("Tri-Predict", tri),
    ):
        evaluation = value["evaluation"]
        lines.append(
            f"| {name} | {evaluation['budget']['mean']:.4f} | "
            f"{evaluation['budget']['p95']:.4f} | "
            f"{evaluation['tune_retention_bound']['mean']:.6f} | "
            f"{evaluation['tune_retention_bound']['lower_bound']:.6f} | "
            f"{evaluation['candidate_saving_vs_tune_fixed_reference']:.2%} | "
            f"{evaluation['coordinate_work_reduction_vs_tune_fixed_reference']:.2%} | "
            f"{evaluation['saturated_n']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen policies",
            "",
            f"- monotone bins: `{binned['policy']['budgets']}`",
            "- monotone policy fingerprint: "
            f"`{binned['policy']['fingerprint']}`",
            f"- Tri-Predict prediction target: `{tri['prediction_target']}`",
            f"- Tri-Predict safety correction: "
            f"`{tri['policy']['safety_correction']}`",
            "- analytic Tri-Predict fingerprint: "
            f"`{tri['policy']['fingerprint']}`",
            f"- policy-selection fingerprint: `{summary['selection_fingerprint']}`",
            "",
            "The coordinate metric includes one query projection, one projected "
            "full-corpus scan, and original reranking. It is not a latency claim. "
            "The compiled lookup table is a platform deployment artifact and is "
            "excluded from scientific selection/result identity. No policy has "
            "yet been evaluated on `query_cert` or `query_test`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_real_policy_tune(
    config: RealPolicyTuneConfig,
    prepared_dir: Union[str, Path],
    embedding_config_path: Union[str, Path],
    embedding_cache_dir: Union[str, Path],
    dimension_selection_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    if config.evaluation_split != "query_tune":
        raise RealPolicyTuneError("real policy fitting accepts query_tune only")
    prepared = Path(prepared_dir)
    embedding_cache = Path(embedding_cache_dir)
    dimension_run = Path(dimension_selection_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite policy output: {output}")

    embedding_config = load_text_embedding_config(embedding_config_path)
    if embedding_config.config_fingerprint != config.embedding_config_fingerprint:
        raise RealPolicyTuneError("embedding config fingerprint mismatch")
    validation = validate_text_embedding_cache(
        embedding_config, prepared, embedding_cache
    )
    dataset_manifest = validation["dataset_manifest"]
    embedding_manifest = validation["embedding_manifest"]
    if (
        dataset_manifest["fingerprint"] != config.dataset_manifest_fingerprint
        or validation["request_fingerprint"]
        != config.embedding_request_fingerprint
        or embedding_manifest["fingerprint"]
        != config.embedding_manifest_fingerprint
    ):
        raise RealPolicyTuneError("dataset/embedding input identity mismatch")
    selection_input, _, dimension_records = _validate_dimension_selection(
        dimension_run, config, dataset_manifest
    )
    fixed_reference_budget = int(selection_input["selected"]["fixed_budget"])
    if fixed_reference_budget not in config.m_grid:
        raise RealPolicyTuneError("dimension-selected fixed reference is not in M_grid")

    corpus_table = load_embedding_array(
        embedding_cache / "corpus_embeddings.f32.npy",
        embedding_cache / "corpus_ids.json",
    )
    query_rows = _load_jsonl(prepared / "queries.jsonl", "prepared queries")
    query_ids = _load_json(embedding_cache / "query_ids.json", "query IDs")
    if (
        not isinstance(query_ids, list)
        or not all(isinstance(value, str) for value in query_ids)
        or query_ids != [row.get("query_id") for row in query_rows]
    ):
        raise RealPolicyTuneError("query embedding IDs do not match prepared order")
    query_vectors = np.load(
        embedding_cache / "query_embeddings.f32.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    selected_rows = [
        index for index, row in enumerate(query_rows) if row.get("split") == "query_tune"
    ]
    selected_query_ids = [query_ids[index] for index in selected_rows]
    if selected_query_ids != [record["query_id"] for record in dimension_records]:
        raise RealPolicyTuneError("dimension records do not align with tune query order")
    tune_queries = np.asarray(
        query_vectors[np.asarray(selected_rows, dtype=np.int64)], dtype=np.float32
    )
    corpus = np.asarray(corpus_table.vectors, dtype=np.float32)
    corpus_ids = corpus_table.ids.tolist()
    corpus_size, dimension = corpus.shape
    if (
        dimension != embedding_config.model.embedding_dimension
        or config.m_grid[-1] != corpus_size
        or config.m_prime > dimension
    ):
        raise RealPolicyTuneError("corpus shape is incompatible with frozen policy config")
    corpus_id_to_row = {doc_id: row for row, doc_id in enumerate(corpus_ids)}

    projection_started = perf_counter()
    matrix = dense_gaussian_projection(
        config.m_prime, dimension, config.projection_seed
    )
    projected_corpus = project_rows(corpus, matrix)
    projected_queries = project_rows(tune_queries, matrix)
    projection_ms = (perf_counter() - projection_started) * 1000.0
    metadata = projection_metadata(
        dimension=dimension,
        m_prime=config.m_prime,
        seed=config.projection_seed,
        normalization=True,
        embedding_model=(
            f"{embedding_config.model.name}@{embedding_config.model.revision}"
        ),
        corpus_hash=embedding_manifest["arrays"]["corpus"]["array_fingerprint"],
    )
    if metadata["fingerprint"] != config.projection_fingerprint:
        raise RealPolicyTuneError("reconstructed projection fingerprint mismatch")
    rankings, projected_search_ms = _exact_projected_rankings(
        projected_corpus,
        projected_queries,
        corpus_ids,
        k=corpus_size,
        batch_size=config.query_batch_size,
    )

    oracle_started = perf_counter()
    original_index = ExactSquaredL2Index(
        corpus_ids, corpus, batch_size=config.query_batch_size
    )
    oracle = original_index.search(tune_queries, max(config.k_gt, config.s_lid))
    oracle_search_ms = (perf_counter() - oracle_started) * 1000.0
    lid_kwargs = {
        "s_lid": config.s_lid,
        "min_neighbors": config.min_lid_neighbors,
        "clip_min": config.lid_clip_min,
        "clip_max": config.lid_clip_max,
        "duplicate_tolerance": config.duplicate_tolerance,
        "fallback": config.lid_fallback,
    }
    base_records: list[Dict[str, Any]] = []
    lid_started = perf_counter()
    for query_index, (ranking, stage_record) in enumerate(
        zip(rankings, dimension_records)
    ):
        if _ranking_hash(ranking) != stage_record["projected_ranking_rows_sha256"]:
            raise RealPolicyTuneError("reconstructed projected ranking hash mismatch")
        exact_ids = stage_record["exact_top_k_ids"]
        expected_original = oracle.ids[query_index, : config.k_gt].tolist()
        if exact_ids != expected_original:
            raise RealPolicyTuneError("original exact top-k disagrees with frozen baseline")
        exact_rows = {corpus_id_to_row[doc_id] for doc_id in exact_ids}
        retention_by_budget: Dict[str, float] = {}
        for budget in config.m_grid:
            overlap = len(exact_rows.intersection(ranking[:budget].tolist()))
            retention = overlap / config.k_gt
            if (
                retention
                != stage_record["embedding_retention_by_budget"][str(budget)]
            ):
                raise RealPolicyTuneError("reconstructed retention disagrees with dimension run")
            retention_by_budget[str(budget)] = retention
        pilot_rows = ranking[: config.m_pilot]
        difference = (
            np.asarray(corpus[pilot_rows], dtype=np.float64)
            - np.asarray(tune_queries[query_index], dtype=np.float64)
        )
        pilot_squared = np.einsum("ij,ij->i", difference, difference)
        pilot_lid = estimate_lid_from_squared_distances(pilot_squared, **lid_kwargs)
        oracle_lid = estimate_lid_from_squared_distances(
            oracle.squared_distances[query_index, : config.s_lid], **lid_kwargs
        )
        pilot_serialized = _serialize_lid(
            pilot_lid, config.lid_decimal_places
        )
        oracle_serialized = _serialize_lid(
            oracle_lid, config.lid_decimal_places
        )
        base_records.append(
            {
                "query_index": query_index,
                "query_id": selected_query_ids[query_index],
                "split": "query_tune",
                "lid": pilot_serialized["clipped"],
                "lid_valid": pilot_serialized["valid"],
                "pilot_lid": pilot_serialized,
                "oracle_lid": oracle_serialized,
                "pilot_candidate_ids": [corpus_ids[row] for row in pilot_rows],
                "exact_top_k_ids": exact_ids,
                "projected_ranking_rows_sha256": stage_record[
                    "projected_ranking_rows_sha256"
                ],
                "retention_by_budget": retention_by_budget,
            }
        )
    lid_ms = (perf_counter() - lid_started) * 1000.0
    _validate_tune_only(base_records)

    fixed_candidates: list[Dict[str, Any]] = []
    for budget in config.m_grid:
        policy_value = {
            "name": "fixed",
            "version": 1,
            "budget": budget,
            "grid": config.m_grid,
        }
        policy_value["fingerprint"] = fingerprint(policy_value)
        decisions = [PolicyDecision(budget, -1, False) for _ in base_records]
        evaluation, _ = _evaluate_decisions(
            base_records,
            decisions,
            config=config,
            corpus_size=corpus_size,
            dimension=dimension,
            fixed_reference_budget=fixed_reference_budget,
        )
        fixed_candidates.append({"policy": policy_value, "evaluation": evaluation})
    fixed_eligible = [
        candidate
        for candidate in fixed_candidates
        if candidate["evaluation"]["eligible"]
    ]
    if not fixed_eligible:
        raise RuntimeError("no fixed budget reached the tune target")
    fixed_selected = min(fixed_eligible, key=lambda value: value["policy"]["budget"])
    if fixed_selected["policy"]["budget"] != fixed_reference_budget:
        raise RealPolicyTuneError(
            "fixed-policy tune reference disagrees with dimension selection"
        )

    binned_candidates: list[Dict[str, Any]] = []
    binned_objects: Dict[str, MonotoneBinnedPolicy] = {}
    binned_query_values: Dict[str, list[Dict[str, Any]]] = {}
    for bin_target in config.binned_target_grid:
        policy = MonotoneBinnedPolicy.fit(
            base_records,
            grid=config.m_grid,
            n_bins=config.binned_n_bins,
            target=bin_target,
            safety_margin=0.0,
            fallback_budget=config.fallback_budget,
            feature_version=config.feature_version,
        )
        artifact = policy.serialize()
        decisions = [
            policy.choose(float(record["lid"]), bool(record["lid_valid"]))
            for record in base_records
        ]
        evaluation, query_values = _evaluate_decisions(
            base_records,
            decisions,
            config=config,
            corpus_size=corpus_size,
            dimension=dimension,
            fixed_reference_budget=fixed_reference_budget,
        )
        candidate = {
            "bin_mean_target": bin_target,
            "policy": artifact,
            "evaluation": evaluation,
        }
        binned_candidates.append(candidate)
        binned_objects[artifact["fingerprint"]] = policy
        binned_query_values[artifact["fingerprint"]] = query_values
    selected_binned_candidate = _select_candidate(
        binned_candidates, hyperparameter="bin_mean_target"
    )
    selected_binned_policy = binned_objects[
        selected_binned_candidate["policy"]["fingerprint"]
    ]
    selected_binned_values = binned_query_values[
        selected_binned_candidate["policy"]["fingerprint"]
    ]

    raw_started = perf_counter()
    provisional = TriPredictPolicy(
        corpus_size=corpus_size,
        m_prime=config.m_prime,
        k_gt=config.k_gt,
        grid=config.m_grid,
        target=config.tri_target_grid[0],
        max_rank_samples=config.max_rank_samples,
        feature_version=config.feature_version,
    )
    raw_predictions: list[Optional[Dict[int, float]]] = []
    residuals: list[float] = []
    for record in base_records:
        if not bool(record["lid_valid"]):
            raw_predictions.append(None)
            continue
        raw = provisional.raw_predictions(float(record["lid"]))
        raw_predictions.append(raw)
        for budget in config.m_grid:
            residuals.append(
                raw[budget] - float(record["retention_by_budget"][str(budget)])
            )
    if not residuals:
        raise RealPolicyTuneError("no valid tune LID values for Tri-Predict")
    raw_prediction_ms = (perf_counter() - raw_started) * 1000.0
    tri_candidates: list[Dict[str, Any]] = []
    tri_objects: Dict[str, TriPredictPolicy] = {}
    tri_query_values: Dict[str, list[Dict[str, Any]]] = {}
    corrections: list[tuple[Optional[float], float, int]] = []
    for quantile in config.safety_quantiles:
        if quantile is None:
            corrections.append((None, 0.0, 0))
            continue
        try:
            quantile_value = float(np.quantile(residuals, quantile, method="linear"))
        except TypeError:
            quantile_value = float(
                np.quantile(residuals, quantile, interpolation="linear")
            )
        corrections.append((quantile, max(0.0, quantile_value), len(residuals)))
    for prediction_target in config.tri_target_grid:
        for quantile, correction, observations in corrections:
            policy = TriPredictPolicy(
                corpus_size=corpus_size,
                m_prime=config.m_prime,
                k_gt=config.k_gt,
                grid=config.m_grid,
                target=prediction_target,
                max_rank_samples=config.max_rank_samples,
                safety_correction=correction,
                safety_quantile=quantile,
                correction_fit_observations=observations,
                feature_version=config.feature_version,
            )
            artifact = policy.serialize()
            decisions = [
                _choose_tri_from_raw(raw, policy) for raw in raw_predictions
            ]
            evaluation, query_values = _evaluate_decisions(
                base_records,
                decisions,
                config=config,
                corpus_size=corpus_size,
                dimension=dimension,
                fixed_reference_budget=fixed_reference_budget,
            )
            candidate = {
                "prediction_target": prediction_target,
                "safety_quantile": quantile,
                "safety_correction": artifact["safety_correction"],
                "selection_hyperparameter": prediction_target
                + artifact["safety_correction"],
                "policy": artifact,
                "evaluation": evaluation,
            }
            tri_candidates.append(candidate)
            tri_objects[artifact["fingerprint"]] = policy
            tri_query_values[artifact["fingerprint"]] = query_values
    selected_tri_candidate = _select_candidate(
        tri_candidates, hyperparameter="selection_hyperparameter"
    )
    selected_tri_policy = tri_objects[selected_tri_candidate["policy"]["fingerprint"]]
    selected_tri_values = tri_query_values[
        selected_tri_candidate["policy"]["fingerprint"]
    ]
    for record, raw, expected in zip(
        base_records, raw_predictions, selected_tri_values
    ):
        analytic = selected_tri_policy.choose(
            float(record["lid"]), bool(record["lid_valid"])
        )
        cached = _choose_tri_from_raw(raw, selected_tri_policy)
        if (analytic.budget, analytic.saturated, analytic.used_fallback) != (
            cached.budget,
            cached.saturated,
            cached.used_fallback,
        ) or analytic.budget != expected["chosen_m"]:
            raise AssertionError("cached Tri-Predict decision disagrees with analytic policy")

    compile_started = perf_counter()
    compiled_policy = CompiledTriPredictPolicy.compile(
        selected_tri_policy,
        lid_min=config.compiled_lid_min,
        lid_max=config.compiled_lid_max,
        validation_samples=config.compile_validation_samples,
    )
    compiled_artifact = compiled_policy.serialize()
    compile_ms = (perf_counter() - compile_started) * 1000.0
    for record, expected in zip(base_records, selected_tri_values):
        decision = compiled_policy.choose(
            float(record["lid"]), bool(record["lid_valid"])
        )
        if (
            decision.budget != expected["chosen_m"]
            or decision.saturated != expected["policy_saturated"]
            or decision.used_fallback != expected["used_lid_fallback"]
        ):
            raise AssertionError("compiled Tri-Predict disagrees on a tune query")

    fixed_artifact = {
        "schema_version": 1,
        "kind": "real_fixed_policy_grid_v1",
        "data_scope": "query_tune_only",
        "candidates": fixed_candidates,
        "selected_reference_budget": fixed_reference_budget,
    }
    fixed_artifact["fingerprint"] = fingerprint(fixed_artifact)
    selection_artifact = {
        "schema_version": 2,
        "kind": "real_tune_policy_selection_v2",
        "data_scope": "query_tune_only",
        "config_fingerprint": config.config_fingerprint,
        "query_tune_n": len(base_records),
        "query_tune_id_hash": stable_id_hash(selected_query_ids),
        "dimension_selection_fingerprint": config.dimension_selection_fingerprint,
        "frozen_projection_fingerprint": config.frozen_projection_fingerprint,
        "selection_alpha": config.selection_alpha,
        "selection_target": config.selection_target,
        "selection_rule": _SELECTION_RULE,
        "fixed": fixed_artifact,
        "monotone_binned_candidates": binned_candidates,
        "tri_predict_candidates": tri_candidates,
        "selected": {
            "fixed_reference_budget": fixed_reference_budget,
            "monotone_binned_policy_fingerprint": selected_binned_candidate[
                "policy"
            ]["fingerprint"],
            "tri_predict_policy_fingerprint": selected_tri_candidate["policy"][
                "fingerprint"
            ],
        },
    }
    selection_artifact["selection_fingerprint"] = fingerprint(selection_artifact)

    per_query: list[Dict[str, Any]] = []
    for record, binned_value, tri_value in zip(
        base_records, selected_binned_values, selected_tri_values
    ):
        per_query.append(
            {
                "query_index": record["query_index"],
                "query_id": record["query_id"],
                "split": "query_tune",
                "pilot_lid": record["pilot_lid"],
                "oracle_lid": record["oracle_lid"],
                "pilot_candidate_ids": record["pilot_candidate_ids"],
                "exact_top_k_ids": record["exact_top_k_ids"],
                "projected_ranking_rows_sha256": record[
                    "projected_ranking_rows_sha256"
                ],
                "fixed_retention_by_budget": record["retention_by_budget"],
                "monotone_binned": binned_value,
                "tri_predict": dict(
                    tri_value,
                    compiled_decision_match=True,
                ),
            }
        )

    paired_gaps = [
        abs(float(record["pilot_lid"]["clipped"]) - float(record["oracle_lid"]["clipped"]))
        for record in base_records
        if record["pilot_lid"]["valid"] and record["oracle_lid"]["valid"]
    ]
    fixed_summary = {
        "budget": fixed_reference_budget,
        "policy": fixed_selected["policy"],
        "evaluation": fixed_selected["evaluation"],
    }
    binned_summary = {
        "bin_mean_target": selected_binned_candidate["bin_mean_target"],
        "policy": selected_binned_candidate["policy"],
        "evaluation": selected_binned_candidate["evaluation"],
    }
    tri_summary = {
        "prediction_target": selected_tri_candidate["prediction_target"],
        "safety_quantile": selected_tri_candidate["safety_quantile"],
        "policy": selected_tri_candidate["policy"],
        "evaluation": selected_tri_candidate["evaluation"],
    }
    summary = {
        "schema_version": 2,
        "kind": "real_tune_policy_summary_v2",
        "data_scope": "query_tune_only",
        "n_queries": len(base_records),
        "corpus_size": corpus_size,
        "embedding_dimension": dimension,
        "m_prime": config.m_prime,
        "projection_seed": config.projection_seed,
        "m_pilot": config.m_pilot,
        "s_lid": config.s_lid,
        "selection_target": config.selection_target,
        "selection_fingerprint": selection_artifact["selection_fingerprint"],
        "fixed_reference": fixed_summary,
        "monotone_binned": binned_summary,
        "tri_predict": tri_summary,
        "lid_diagnostic": {
            "pilot_valid_n": int(
                sum(record["pilot_lid"]["valid"] for record in base_records)
            ),
            "oracle_valid_n": int(
                sum(record["oracle_lid"]["valid"] for record in base_records)
            ),
            "paired_valid_n": len(paired_gaps),
            "mean_absolute_clipped_gap": (
                0.0
                if not paired_gaps
                else _canonical_lid_float(
                    float(np.mean(paired_gaps)), config.lid_decimal_places
                )
            ),
            "oracle_role": "diagnostic_only_not_used_for_policy_selection",
        },
    }
    summary["fingerprint"] = fingerprint(summary)
    timings = {
        "role": "systems_diagnostic_excluded_from_result_identity",
        "projection_and_materialization_ms": projection_ms,
        "exact_projected_full_ranking_ms": projected_search_ms,
        "oracle_original_top_s_lid_ms": oracle_search_ms,
        "lid_materialization_ms": lid_ms,
        "tri_predict_raw_grid_ms": raw_prediction_ms,
        "tri_predict_compile_ms": compile_ms,
        "projected_distance_evaluations": len(base_records) * corpus_size,
        "pilot_original_distance_evaluations": len(base_records) * config.m_pilot,
        "oracle_distance_evaluations_diagnostic": len(base_records) * corpus_size,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        paths = {
            "manifest.json": temporary / "manifest.json",
            "per_query.jsonl": temporary / "per_query.jsonl",
            "selection.json": temporary / "selection.json",
            "fixed_policies.json": temporary / "fixed_policies.json",
            "monotone_binned_policy.json": temporary / "monotone_binned_policy.json",
            "tri_predict_policy.json": temporary / "tri_predict_policy.json",
            "compiled_tri_predict_policy.json": temporary
            / "compiled_tri_predict_policy.json",
            "summary.json": temporary / "summary.json",
            "timings.json": temporary / "timings.json",
            "report.md": temporary / "report.md",
        }
        _write_jsonl(paths["per_query.jsonl"], per_query)
        write_json(paths["selection.json"], selection_artifact)
        write_json(paths["fixed_policies.json"], fixed_artifact)
        write_json(
            paths["monotone_binned_policy.json"],
            selected_binned_policy.serialize(),
        )
        write_json(paths["tri_predict_policy.json"], selected_tri_policy.serialize())
        write_json(paths["compiled_tri_predict_policy.json"], compiled_artifact)
        write_json(paths["summary.json"], summary)
        write_json(paths["timings.json"], timings)
        paths["report.md"].write_text(_report(summary), encoding="utf-8")
        result_artifacts = {
            name: _file_identity(paths[name]) for name in _SCIENTIFIC_RESULT_NAMES
        }
        deployment_artifacts = {
            "compiled_tri_predict_policy.json": _file_identity(
                paths["compiled_tri_predict_policy.json"]
            )
        }
        result_identity = {
            "config_fingerprint": config.config_fingerprint,
            "dataset_manifest_fingerprint": dataset_manifest["fingerprint"],
            "embedding_manifest_fingerprint": embedding_manifest["fingerprint"],
            "dimension_selection_result_fingerprint": config.dimension_selection_result_fingerprint,
            "query_tune_id_hash": stable_id_hash(selected_query_ids),
            "selection_fingerprint": selection_artifact["selection_fingerprint"],
            "artifacts": result_artifacts,
        }
        manifest = {
            "schema_version": 2,
            "kind": "real_tune_policy_manifest_v2",
            "data_scope": "query_tune_only",
            "config_fingerprint": config.config_fingerprint,
            "dataset_manifest_fingerprint": dataset_manifest["fingerprint"],
            "embedding_manifest_fingerprint": embedding_manifest["fingerprint"],
            "original_baseline_result_fingerprint": config.original_baseline_result_fingerprint,
            "dimension_selection_result_fingerprint": config.dimension_selection_result_fingerprint,
            "dimension_selection_fingerprint": config.dimension_selection_fingerprint,
            "frozen_projection_fingerprint": config.frozen_projection_fingerprint,
            "projection_fingerprint": config.projection_fingerprint,
            "query_tune_n": len(base_records),
            "query_tune_id_hash": stable_id_hash(selected_query_ids),
            "selection_fingerprint": selection_artifact["selection_fingerprint"],
            "policy_fingerprints": {
                "fixed_grid": fixed_artifact["fingerprint"],
                "monotone_binned": selected_binned_policy.serialize()["fingerprint"],
                "tri_predict": selected_tri_policy.serialize()["fingerprint"],
            },
            "result_artifacts": result_artifacts,
            "result_fingerprint": fingerprint(result_identity),
            "deployment": {
                "role": config.compiled_policy_role,
                "reference_policy_fingerprint": selected_tri_policy.serialize()[
                    "fingerprint"
                ],
                "policy_fingerprint": compiled_artifact["fingerprint"],
                "artifacts": deployment_artifacts,
            },
            "timings_artifact": "timings.json",
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
        }
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(paths["manifest.json"], manifest)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {name: output / name for name in paths}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--embedding-config", required=True, type=Path)
    parser.add_argument("--embedding-cache", required=True, type=Path)
    parser.add_argument("--dimension-selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_real_policy_tune_config(args.config)
    paths = run_real_policy_tune(
        config,
        args.dataset,
        args.embedding_config,
        args.embedding_cache,
        args.dimension_selection,
        args.output,
    )
    print(f"completed tune-only policy fitting: {paths['report.md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
