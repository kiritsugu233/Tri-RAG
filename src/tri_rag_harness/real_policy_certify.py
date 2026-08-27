"""Certify frozen real-data retrieval policies on query_cert exactly once."""

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
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from .certification import make_certificate
from .embeddings import load_embedding_array
from .indexes import ExactSquaredL2Index
from .lid import estimate_lid_from_squared_distances
from .policies import (
    CompiledTriPredictPolicy,
    FixedBudgetPolicy,
    MonotoneBinnedPolicy,
    PolicyDecision,
    TriPredictPolicy,
)
from .projection import dense_gaussian_projection, project_rows, projection_metadata
from .real_dimension_sweep import _exact_projected_rankings, _ranking_hash
from .text_embeddings import load_text_embedding_config, validate_text_embedding_cache
from .utils import fingerprint, stable_id_hash, write_json


class RealPolicyCertificationError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenPolicySource:
    manifest_fingerprint: str
    result_fingerprint: str
    selection_fingerprint: str
    fixed_grid_fingerprint: str
    fixed_reference_budget: int
    fixed_reference_policy_fingerprint: str
    monotone_binned_fingerprint: str
    analytic_tri_predict_fingerprint: str
    compiled_tri_predict_fingerprint: str
    compiled_artifact_sha256: str
    compiled_policy_role: str


@dataclass(frozen=True)
class RealPolicyCertificationConfig:
    raw: Dict[str, Any]
    config_fingerprint: str
    dataset_manifest_fingerprint: str
    embedding_config_fingerprint: str
    embedding_request_fingerprint: str
    embedding_manifest_fingerprint: str
    evaluation_split: str
    query_split_n: int
    query_split_id_hash: str
    policy_source: FrozenPolicySource
    frozen_projection_fingerprint: str
    projection_fingerprint: str
    projection_seed: int
    m_prime: int
    query_batch_size: int
    k_ctx: int
    k_gt: int
    m_pilot: int
    s_lid: int
    min_lid_neighbors: int
    m_grid: list[int]
    fixed_reference_budget: int
    lid_decimal_places: int
    feature_version: str
    lid_clip_min: float
    lid_clip_max: float
    duplicate_tolerance: float
    lid_fallback: float
    certification_alpha: float
    certification_target: float
    planned_n: int


@dataclass(frozen=True)
class FrozenPolicyBundle:
    manifest: Dict[str, Any]
    fixed: FixedBudgetPolicy
    fixed_fingerprint: str
    monotone: MonotoneBinnedPolicy
    analytic_tri: TriPredictPolicy
    compiled_tri: CompiledTriPredictPolicy
    input_artifacts: Dict[str, Dict[str, Any]]


_SHA256_LENGTH = 64
_FEATURE_VERSION = "pilot_rerank_lid_rounded_9_v2"
_LID_DECIMAL_PLACES = 9
_COMPILED_POLICY_ROLE = (
    "platform_deployment_artifact_excluded_from_scientific_identity"
)
_POLICY_RESULT_NAMES = (
    "per_query.jsonl",
    "selection.json",
    "fixed_policies.json",
    "monotone_binned_policy.json",
    "tri_predict_policy.json",
    "summary.json",
    "report.md",
)
_CERT_RESULT_NAMES = (
    "per_query.jsonl",
    "certifications.json",
    "summary.json",
    "report.md",
)
_POLICY_NAMES = ("fixed_reference", "monotone_binned", "tri_predict")
_FAMILYWISE_ROLE = "none_three_predeclared_standalone_certificates_no_selection"
_FAILURE_BEHAVIOR = "terminal_no_retuning_no_budget_expansion"


def _exact_keys(value: Any, expected: set[str], name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RealPolicyCertificationError(f"{name} must be an object")
    if set(value) != expected:
        raise RealPolicyCertificationError(
            f"invalid {name} keys; missing={sorted(expected-set(value))}, "
            f"unknown={sorted(set(value)-expected)}"
        )
    return dict(value)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RealPolicyCertificationError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RealPolicyCertificationError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise RealPolicyCertificationError(f"{name} must be finite and positive")
    return result


def _probability(value: Any, name: str) -> float:
    result = _positive_float(value, name)
    if result > 1.0:
        raise RealPolicyCertificationError(f"{name} must lie in (0,1]")
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise RealPolicyCertificationError(f"{name} must be a SHA-256 value")
    result = value.strip().lower()
    if len(result) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise RealPolicyCertificationError(f"{name} must be a SHA-256 value")
    return result


def _integer_grid(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise RealPolicyCertificationError(f"{name} must be a nonempty list")
    result = [_positive_integer(item, f"{name} item") for item in value]
    if result != sorted(set(result)):
        raise RealPolicyCertificationError(f"{name} must be strictly increasing")
    return result


def load_real_policy_certification_config(
    path: Union[str, Path],
) -> RealPolicyCertificationConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealPolicyCertificationError(
            f"cannot load certification config {config_path}: {exc}"
        ) from exc
    root = _exact_keys(
        raw,
        {
            "schema_version",
            "benchmark",
            "dataset_manifest_fingerprint",
            "embedding_config_fingerprint",
            "embedding_request_fingerprint",
            "embedding_manifest_fingerprint",
            "evaluation_split",
            "query_split",
            "policy_source",
            "projection",
            "search",
            "lid",
            "certification",
        },
        "root",
    )
    if (
        root["schema_version"] != 1
        or root["benchmark"] != "real_frozen_policy_certification_v1"
    ):
        raise RealPolicyCertificationError(
            "unsupported certification config schema/benchmark"
        )
    if root["evaluation_split"] != "query_cert":
        raise RealPolicyCertificationError("certification accepts query_cert only")

    query_split = _exact_keys(root["query_split"], {"n", "id_hash"}, "query_split")
    policy_raw = _exact_keys(
        root["policy_source"],
        {
            "manifest_fingerprint",
            "result_fingerprint",
            "selection_fingerprint",
            "fixed_grid_fingerprint",
            "fixed_reference_budget",
            "fixed_reference_policy_fingerprint",
            "monotone_binned_fingerprint",
            "analytic_tri_predict_fingerprint",
            "compiled_tri_predict_fingerprint",
            "compiled_artifact_sha256",
            "compiled_policy_role",
        },
        "policy_source",
    )
    if policy_raw["compiled_policy_role"] != _COMPILED_POLICY_ROLE:
        raise RealPolicyCertificationError("compiled deployment role is not frozen")
    policy_source = FrozenPolicySource(
        manifest_fingerprint=_sha256(
            policy_raw["manifest_fingerprint"], "policy_source.manifest_fingerprint"
        ),
        result_fingerprint=_sha256(
            policy_raw["result_fingerprint"], "policy_source.result_fingerprint"
        ),
        selection_fingerprint=_sha256(
            policy_raw["selection_fingerprint"],
            "policy_source.selection_fingerprint",
        ),
        fixed_grid_fingerprint=_sha256(
            policy_raw["fixed_grid_fingerprint"],
            "policy_source.fixed_grid_fingerprint",
        ),
        fixed_reference_budget=_positive_integer(
            policy_raw["fixed_reference_budget"],
            "policy_source.fixed_reference_budget",
        ),
        fixed_reference_policy_fingerprint=_sha256(
            policy_raw["fixed_reference_policy_fingerprint"],
            "policy_source.fixed_reference_policy_fingerprint",
        ),
        monotone_binned_fingerprint=_sha256(
            policy_raw["monotone_binned_fingerprint"],
            "policy_source.monotone_binned_fingerprint",
        ),
        analytic_tri_predict_fingerprint=_sha256(
            policy_raw["analytic_tri_predict_fingerprint"],
            "policy_source.analytic_tri_predict_fingerprint",
        ),
        compiled_tri_predict_fingerprint=_sha256(
            policy_raw["compiled_tri_predict_fingerprint"],
            "policy_source.compiled_tri_predict_fingerprint",
        ),
        compiled_artifact_sha256=_sha256(
            policy_raw["compiled_artifact_sha256"],
            "policy_source.compiled_artifact_sha256",
        ),
        compiled_policy_role=_COMPILED_POLICY_ROLE,
    )

    projection = _exact_keys(
        root["projection"],
        {
            "family",
            "seed",
            "m_prime",
            "post_projection_normalize",
            "frozen_projection_fingerprint",
            "projection_fingerprint",
        },
        "projection",
    )
    if (
        projection["family"] != "dense_gaussian_n0_variance_1_over_m_prime"
        or projection["post_projection_normalize"] is not False
    ):
        raise RealPolicyCertificationError("projection contract is not frozen")

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
            "fixed_reference_budget",
            "pilot_expansion_reuse",
        },
        "search",
    )
    if (
        search["normalized_inputs"] is not True
        or search["distance"] != "squared_l2"
        or search["arithmetic"] != "numpy_float64"
        or search["stable_tie_break"] != "lexicographic_doc_id"
        or search["pilot_expansion_reuse"]
        != "one_projected_scan_and_reuse_pilot_original_distances"
    ):
        raise RealPolicyCertificationError("search contract is not frozen")
    k_ctx = _positive_integer(search["k_ctx"], "search.k_ctx")
    k_gt = _positive_integer(search["k_gt"], "search.k_gt")
    m_pilot = _positive_integer(search["m_pilot"], "search.m_pilot")
    s_lid = _positive_integer(search["s_lid"], "search.s_lid")
    minimum_lid = _positive_integer(
        search["min_lid_neighbors"], "search.min_lid_neighbors"
    )
    grid = _integer_grid(search["m_grid"], "search.m_grid")
    fixed_budget = _positive_integer(
        search["fixed_reference_budget"], "search.fixed_reference_budget"
    )
    if (
        not k_ctx <= k_gt <= s_lid <= m_pilot
        or minimum_lid > s_lid
        or grid[0] != m_pilot
        or fixed_budget not in grid
        or fixed_budget != policy_source.fixed_reference_budget
    ):
        raise RealPolicyCertificationError("unsafe or inconsistent search budgets")

    lid = _exact_keys(
        root["lid"],
        {
            "decimal_places",
            "feature_version",
            "clip_min",
            "clip_max",
            "duplicate_tolerance",
            "fallback",
        },
        "lid",
    )
    if (
        lid["decimal_places"] != _LID_DECIMAL_PLACES
        or lid["feature_version"] != _FEATURE_VERSION
    ):
        raise RealPolicyCertificationError("LID determinism contract is not frozen")
    clip_min = _positive_float(lid["clip_min"], "lid.clip_min")
    clip_max = _positive_float(lid["clip_max"], "lid.clip_max")
    fallback = _positive_float(lid["fallback"], "lid.fallback")
    if not clip_min < clip_max or not clip_min <= fallback <= clip_max:
        raise RealPolicyCertificationError("invalid LID clipping/fallback interval")

    certification = _exact_keys(
        root["certification"],
        {
            "metric",
            "alpha",
            "target",
            "planned_n",
            "statistic_role",
            "policies",
            "familywise_adjustment",
            "failure_behavior",
            "evidence_labels_used",
        },
        "certification",
    )
    if (
        certification["metric"] != "embedding_neighbor_retention_at_k_gt"
        or certification["statistic_role"]
        != "independent_empirical_bernstein_certificate"
        or certification["policies"] != list(_POLICY_NAMES)
        or certification["familywise_adjustment"] != _FAMILYWISE_ROLE
        or certification["failure_behavior"] != _FAILURE_BEHAVIOR
        or certification["evidence_labels_used"] is not False
    ):
        raise RealPolicyCertificationError("certification protocol is not frozen")
    alpha = _probability(certification["alpha"], "certification.alpha")
    if alpha >= 1.0:
        raise RealPolicyCertificationError("certification.alpha must lie in (0,1)")
    query_n = _positive_integer(query_split["n"], "query_split.n")
    planned_n = _positive_integer(certification["planned_n"], "planned_n")
    if planned_n != query_n:
        raise RealPolicyCertificationError("planned_n must equal the frozen cert split")

    return RealPolicyCertificationConfig(
        raw=root,
        config_fingerprint=fingerprint(root),
        dataset_manifest_fingerprint=_sha256(
            root["dataset_manifest_fingerprint"], "dataset_manifest_fingerprint"
        ),
        embedding_config_fingerprint=_sha256(
            root["embedding_config_fingerprint"], "embedding_config_fingerprint"
        ),
        embedding_request_fingerprint=_sha256(
            root["embedding_request_fingerprint"], "embedding_request_fingerprint"
        ),
        embedding_manifest_fingerprint=_sha256(
            root["embedding_manifest_fingerprint"], "embedding_manifest_fingerprint"
        ),
        evaluation_split="query_cert",
        query_split_n=query_n,
        query_split_id_hash=_sha256(query_split["id_hash"], "query_split.id_hash"),
        policy_source=policy_source,
        frozen_projection_fingerprint=_sha256(
            projection["frozen_projection_fingerprint"],
            "projection.frozen_projection_fingerprint",
        ),
        projection_fingerprint=_sha256(
            projection["projection_fingerprint"], "projection.projection_fingerprint"
        ),
        projection_seed=_positive_integer(projection["seed"], "projection.seed"),
        m_prime=_positive_integer(projection["m_prime"], "projection.m_prime"),
        query_batch_size=_positive_integer(
            search["query_batch_size"], "search.query_batch_size"
        ),
        k_ctx=k_ctx,
        k_gt=k_gt,
        m_pilot=m_pilot,
        s_lid=s_lid,
        min_lid_neighbors=minimum_lid,
        m_grid=grid,
        fixed_reference_budget=fixed_budget,
        lid_decimal_places=_LID_DECIMAL_PLACES,
        feature_version=_FEATURE_VERSION,
        lid_clip_min=clip_min,
        lid_clip_max=clip_max,
        duplicate_tolerance=_positive_float(
            lid["duplicate_tolerance"], "lid.duplicate_tolerance"
        ),
        lid_fallback=fallback,
        certification_alpha=alpha,
        certification_target=_probability(
            certification["target"], "certification.target"
        ),
        planned_n=planned_n,
    )


def _load_json(path: Path, name: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealPolicyCertificationError(f"cannot load {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise RealPolicyCertificationError(f"{name} must be an object")
    return value


def _load_jsonl(path: Path, name: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RealPolicyCertificationError(
                        f"{name} line {line_number} must be an object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RealPolicyCertificationError(f"cannot load {name}: {exc}") from exc
    return rows


def _file_identity(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                size += len(block)
    except OSError as exc:
        raise RealPolicyCertificationError(f"cannot hash artifact {path}: {exc}") from exc
    return {"bytes": size, "sha256": digest.hexdigest()}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )


def _verify_embedded_fingerprint(
    artifact: Mapping[str, Any], *, field: str, name: str
) -> str:
    raw = dict(artifact)
    stored = raw.pop(field, None)
    if not isinstance(stored, str) or fingerprint(raw) != stored:
        raise RealPolicyCertificationError(f"{name} fingerprint is invalid")
    return stored


def _validate_policy_bundle(
    directory: Path, config: RealPolicyCertificationConfig
) -> FrozenPolicyBundle:
    manifest = _load_json(directory / "manifest.json", "policy manifest")
    manifest_fingerprint = _verify_embedded_fingerprint(
        manifest, field="fingerprint", name="policy manifest"
    )
    source = config.policy_source
    if manifest_fingerprint != source.manifest_fingerprint:
        raise RealPolicyCertificationError("policy manifest fingerprint mismatch")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") != "real_tune_policy_manifest_v2"
        or manifest.get("data_scope") != "query_tune_only"
        or manifest.get("dataset_manifest_fingerprint")
        != config.dataset_manifest_fingerprint
        or manifest.get("embedding_manifest_fingerprint")
        != config.embedding_manifest_fingerprint
        or manifest.get("result_fingerprint") != source.result_fingerprint
        or manifest.get("selection_fingerprint") != source.selection_fingerprint
        or manifest.get("frozen_projection_fingerprint")
        != config.frozen_projection_fingerprint
        or manifest.get("projection_fingerprint") != config.projection_fingerprint
    ):
        raise RealPolicyCertificationError("policy manifest identity mismatch")

    artifact_metadata = manifest.get("result_artifacts")
    if not isinstance(artifact_metadata, dict) or set(artifact_metadata) != set(
        _POLICY_RESULT_NAMES
    ):
        raise RealPolicyCertificationError("policy result artifact set is invalid")
    input_artifacts: Dict[str, Dict[str, Any]] = {}
    for name in _POLICY_RESULT_NAMES:
        observed = _file_identity(directory / name)
        if observed != artifact_metadata.get(name):
            raise RealPolicyCertificationError(
                f"policy result artifact identity mismatch: {name}"
            )
        input_artifacts[name] = observed
    result_identity = {
        "config_fingerprint": manifest.get("config_fingerprint"),
        "dataset_manifest_fingerprint": manifest.get(
            "dataset_manifest_fingerprint"
        ),
        "embedding_manifest_fingerprint": manifest.get(
            "embedding_manifest_fingerprint"
        ),
        "dimension_selection_result_fingerprint": manifest.get(
            "dimension_selection_result_fingerprint"
        ),
        "query_tune_id_hash": manifest.get("query_tune_id_hash"),
        "selection_fingerprint": manifest.get("selection_fingerprint"),
        "artifacts": input_artifacts,
    }
    if fingerprint(result_identity) != source.result_fingerprint:
        raise RealPolicyCertificationError("policy scientific result identity is invalid")

    selection = _load_json(directory / "selection.json", "policy selection")
    if (
        _verify_embedded_fingerprint(
            selection, field="selection_fingerprint", name="policy selection"
        )
        != source.selection_fingerprint
    ):
        raise RealPolicyCertificationError("policy selection fingerprint mismatch")

    fixed_artifact = _load_json(directory / "fixed_policies.json", "fixed policies")
    fixed_grid_fingerprint = _verify_embedded_fingerprint(
        fixed_artifact, field="fingerprint", name="fixed policies"
    )
    if (
        fixed_grid_fingerprint != source.fixed_grid_fingerprint
        or fixed_artifact.get("selected_reference_budget")
        != source.fixed_reference_budget
    ):
        raise RealPolicyCertificationError("fixed policy grid identity mismatch")
    candidates = fixed_artifact.get("candidates")
    if not isinstance(candidates, list):
        raise RealPolicyCertificationError("fixed policy candidates are invalid")
    selected_fixed = [
        candidate.get("policy")
        for candidate in candidates
        if isinstance(candidate, dict)
        and isinstance(candidate.get("policy"), dict)
        and candidate["policy"].get("budget") == source.fixed_reference_budget
    ]
    if len(selected_fixed) != 1:
        raise RealPolicyCertificationError("frozen fixed reference is ambiguous")
    fixed_value = dict(selected_fixed[0])
    fixed_fingerprint = fixed_value.pop("fingerprint", None)
    if (
        fixed_fingerprint != source.fixed_reference_policy_fingerprint
        or fingerprint(fixed_value) != fixed_fingerprint
    ):
        raise RealPolicyCertificationError("fixed reference policy fingerprint mismatch")
    try:
        fixed = FixedBudgetPolicy(
            fixed_value["budget"], fixed_value["grid"], config.m_pilot
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RealPolicyCertificationError("invalid fixed reference policy") from exc
    if fixed.serialize() != fixed_value:
        raise RealPolicyCertificationError("fixed reference policy schema mismatch")

    monotone_artifact = _load_json(
        directory / "monotone_binned_policy.json", "monotone policy"
    )
    tri_artifact = _load_json(
        directory / "tri_predict_policy.json", "analytic Tri-Predict policy"
    )
    compiled_artifact = _load_json(
        directory / "compiled_tri_predict_policy.json", "compiled Tri-Predict policy"
    )
    try:
        monotone = MonotoneBinnedPolicy.from_serialized(monotone_artifact)
        analytic_tri = TriPredictPolicy.from_serialized(tri_artifact)
        compiled_tri = CompiledTriPredictPolicy.from_serialized(
            compiled_artifact,
            expected_reference_policy_fingerprint=source.analytic_tri_predict_fingerprint,
        )
    except ValueError as exc:
        raise RealPolicyCertificationError(str(exc)) from exc
    if (
        monotone.serialize()["fingerprint"] != source.monotone_binned_fingerprint
        or analytic_tri.serialize()["fingerprint"]
        != source.analytic_tri_predict_fingerprint
        or compiled_tri.serialize()["fingerprint"]
        != source.compiled_tri_predict_fingerprint
    ):
        raise RealPolicyCertificationError("frozen policy fingerprint mismatch")

    deployment = manifest.get("deployment")
    policy_fingerprints = manifest.get("policy_fingerprints")
    if not isinstance(deployment, dict) or not isinstance(policy_fingerprints, dict):
        raise RealPolicyCertificationError("policy manifest bindings are missing")
    compiled_identity = _file_identity(directory / "compiled_tri_predict_policy.json")
    if (
        policy_fingerprints
        != {
            "fixed_grid": source.fixed_grid_fingerprint,
            "monotone_binned": source.monotone_binned_fingerprint,
            "tri_predict": source.analytic_tri_predict_fingerprint,
        }
        or deployment.get("role") != source.compiled_policy_role
        or deployment.get("policy_fingerprint")
        != source.compiled_tri_predict_fingerprint
        or deployment.get("reference_policy_fingerprint")
        != source.analytic_tri_predict_fingerprint
        or deployment.get("artifacts", {}).get("compiled_tri_predict_policy.json")
        != compiled_identity
        or compiled_identity["sha256"] != source.compiled_artifact_sha256
    ):
        raise RealPolicyCertificationError("compiled deployment binding mismatch")
    input_artifacts["compiled_tri_predict_policy.json"] = compiled_identity
    input_artifacts["manifest.json"] = _file_identity(directory / "manifest.json")

    if (
        tuple(fixed.grid) != tuple(config.m_grid)
        or tuple(monotone.grid) != tuple(config.m_grid)
        or tuple(analytic_tri.grid) != tuple(config.m_grid)
        or tuple(compiled_tri.grid) != tuple(config.m_grid)
        or monotone.feature_version != config.feature_version
        or analytic_tri.feature_version != config.feature_version
        or compiled_tri.feature_version != config.feature_version
        or analytic_tri.m_prime != config.m_prime
        or analytic_tri.k_gt != config.k_gt
        or compiled_tri.lid_min != config.lid_clip_min
        or compiled_tri.lid_max != config.lid_clip_max
    ):
        raise RealPolicyCertificationError("frozen policies disagree with cert config")
    return FrozenPolicyBundle(
        manifest=manifest,
        fixed=fixed,
        fixed_fingerprint=str(fixed_fingerprint),
        monotone=monotone,
        analytic_tri=analytic_tri,
        compiled_tri=compiled_tri,
        input_artifacts=input_artifacts,
    )


def _canonical_float(value: Optional[float], decimals: int = 12) -> Optional[float]:
    if value is None:
        return None
    result = float(np.round(float(value), decimals=decimals))
    if not np.isfinite(result):
        raise RealPolicyCertificationError("non-finite policy diagnostic")
    return result


def _serialize_lid(estimate: Any, decimal_places: int) -> Dict[str, Any]:
    return {
        "valid": bool(estimate.valid),
        "raw": _canonical_float(estimate.raw, decimal_places),
        "clipped": _canonical_float(estimate.clipped, decimal_places),
        "valid_distance_count": int(estimate.valid_distance_count),
        "reason": estimate.reason,
    }


def _prefix_hash(rows: np.ndarray) -> str:
    values = np.asarray(rows, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _rerank_record(
    *,
    decision: PolicyDecision,
    candidate_rows: np.ndarray,
    candidate_distances: np.ndarray,
    tie_rank: np.ndarray,
    corpus_ids: Sequence[str],
    exact_rows: set[int],
    exact_ids: Sequence[str],
    k_gt: int,
) -> Dict[str, Any]:
    budget = int(decision.budget)
    rows = np.asarray(candidate_rows[:budget], dtype=np.int64)
    distances = np.asarray(candidate_distances[:budget], dtype=np.float64)
    order = np.lexsort((tie_rank[rows], distances))[:k_gt]
    reranked_rows = rows[order]
    candidate_overlap = len(exact_rows.intersection(rows.tolist()))
    reranked_overlap = len(set(exact_ids).intersection(corpus_ids[row] for row in reranked_rows))
    if candidate_overlap != reranked_overlap:
        raise RealPolicyCertificationError(
            "candidate overlap disagrees with exact original rerank overlap"
        )
    return {
        "chosen_m": budget,
        "embedding_retention": candidate_overlap / k_gt,
        "candidate_overlap": candidate_overlap,
        "reranked_overlap": reranked_overlap,
        "reranked_top_k_ids": [corpus_ids[row] for row in reranked_rows],
        "candidate_prefix_rows_sha256": _prefix_hash(rows),
        "lid_bin": int(decision.bin_index),
        "used_lid_fallback": bool(decision.used_fallback),
        "policy_saturated": bool(decision.saturated),
        "predicted_retention": _canonical_float(decision.predicted_retention),
        "raw_predicted_retention": _canonical_float(
            decision.raw_predicted_retention
        ),
    }


def _coordinate_work(
    *, corpus_size: int, dimension: int, m_prime: int, mean_budget: float
) -> Dict[str, float]:
    query_projection = float(dimension * m_prime)
    projected_scan = float(corpus_size * m_prime)
    original_rerank = float(dimension * mean_budget)
    total = query_projection + projected_scan + original_rerank
    return {
        "query_projection": query_projection,
        "projected_full_scan": projected_scan,
        "mean_original_rerank": original_rerank,
        "total": total,
        "original_full_scan_reference": float(corpus_size * dimension),
        "reduction_fraction_vs_original_full_scan": 1.0
        - total / (corpus_size * dimension),
    }


def _policy_evaluation(
    *,
    name: str,
    policy_fingerprint: str,
    records: Sequence[Mapping[str, Any]],
    config: RealPolicyCertificationConfig,
    corpus_size: int,
    dimension: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    values = [record["policies"][name] for record in records]
    retentions = [float(value["embedding_retention"]) for value in values]
    budgets = [int(value["chosen_m"]) for value in values]
    certificate = make_certificate(
        retentions,
        alpha=config.certification_alpha,
        target=config.certification_target,
        policy_fingerprint=policy_fingerprint,
        split_hash=config.query_split_id_hash,
        metric="embedding_neighbor_retention_at_k_gt",
        planned_n=config.planned_n,
    )
    mean_budget = float(np.mean(budgets))
    work = _coordinate_work(
        corpus_size=corpus_size,
        dimension=dimension,
        m_prime=config.m_prime,
        mean_budget=mean_budget,
    )
    fixed_work = _coordinate_work(
        corpus_size=corpus_size,
        dimension=dimension,
        m_prime=config.m_prime,
        mean_budget=float(config.fixed_reference_budget),
    )["total"]
    evaluation = {
        "policy_fingerprint": policy_fingerprint,
        "certificate": certificate,
        "decision": "PASS" if certificate["passed"] else "FAIL",
        "budget": {
            "mean": mean_budget,
            "median": float(np.median(budgets)),
            "p95": float(np.quantile(budgets, 0.95)),
            "p99": float(np.quantile(budgets, 0.99)),
            "distribution": {
                str(budget): budgets.count(budget) for budget in config.m_grid
            },
        },
        "retention_distribution": {
            "mean": float(np.mean(retentions)),
            "median": float(np.median(retentions)),
            "p05": float(np.quantile(retentions, 0.05)),
            "minimum": float(np.min(retentions)),
            "below_target_n": int(
                sum(value < config.certification_target for value in retentions)
            ),
        },
        "fallback_n": int(sum(value["used_lid_fallback"] for value in values)),
        "saturated_n": int(sum(value["policy_saturated"] for value in values)),
        "candidate_saving_vs_frozen_fixed_reference": 1.0
        - mean_budget / config.fixed_reference_budget,
        "coordinate_work": work,
        "coordinate_work_reduction_vs_frozen_fixed_reference": 1.0
        - work["total"] / fixed_work,
        "work_per_query": {
            "query_projection_coordinates": dimension * config.m_prime,
            "projected_distance_evaluations": corpus_size,
            "projected_scan_count": 1,
            "original_rerank_distance_evaluations": mean_budget,
            "pilot_original_distances_reused_in_rerank": config.m_pilot,
            "original_ground_truth_distance_evaluations_diagnostic": corpus_size,
        },
    }
    return evaluation, certificate


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# SciFact frozen-policy certification",
        "",
        "This run evaluates three policies frozen on `query_tune` exactly once on "
        "untouched `query_cert`. Each PASS/FAIL is terminal for this protocol; no "
        "certification outcome may trigger retuning or budget expansion. Evidence "
        "labels and `query_test` are not evaluated.",
        "",
        f"- certification queries: {summary['n_queries']}",
        f"- empirical-Bernstein alpha per predeclared policy: "
        f"{summary['certification']['alpha']}",
        f"- embedding-retention target: {summary['certification']['target']}",
        f"- fixed reference: `M={summary['fixed_reference_budget']}`",
        "",
        "| policy | decision | mean M | mean retention | lower bound | candidate saving | coordinate saving | fallback | saturation |",
        "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, name in (
        ("fixed", "fixed_reference"),
        ("monotone binned", "monotone_binned"),
        ("Tri-Predict", "tri_predict"),
    ):
        value = summary["policies"][name]
        lines.append(
            f"| {label} | {value['decision']} | {value['budget']['mean']:.4f} | "
            f"{value['certificate']['mean']:.6f} | "
            f"{value['certificate']['lower_bound']:.6f} | "
            f"{value['candidate_saving_vs_frozen_fixed_reference']:.2%} | "
            f"{value['coordinate_work_reduction_vs_frozen_fixed_reference']:.2%} | "
            f"{value['fallback_n']} | {value['saturated_n']} |"
        )
    lines.extend(
        [
            "",
            "Candidate and coordinate savings are deterministic work proxies, not "
            "measured latency claims. The Tri-Predict certificate names the portable "
            "analytic policy while the manifest separately binds the exact Genoa "
            "compiled deployment artifact used for decisions.",
            "",
        ]
    )
    return "\n".join(lines)


def run_real_policy_certification(
    config: RealPolicyCertificationConfig,
    prepared_dir: Union[str, Path],
    embedding_config_path: Union[str, Path],
    embedding_cache_dir: Union[str, Path],
    policy_run_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    if config.evaluation_split != "query_cert":
        raise RealPolicyCertificationError("certification accepts query_cert only")
    prepared = Path(prepared_dir)
    embedding_cache = Path(embedding_cache_dir)
    policy_run = Path(policy_run_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite certification output: {output}")

    # Validate every frozen scientific and deployment policy identity before the
    # protected query split is selected or scored.
    bundle = _validate_policy_bundle(policy_run, config)

    embedding_config = load_text_embedding_config(embedding_config_path)
    if embedding_config.config_fingerprint != config.embedding_config_fingerprint:
        raise RealPolicyCertificationError("embedding config fingerprint mismatch")
    validation = validate_text_embedding_cache(
        embedding_config, prepared, embedding_cache
    )
    dataset_manifest = validation["dataset_manifest"]
    embedding_manifest = validation["embedding_manifest"]
    if (
        dataset_manifest.get("fingerprint")
        != config.dataset_manifest_fingerprint
        or validation.get("request_fingerprint")
        != config.embedding_request_fingerprint
        or embedding_manifest.get("fingerprint")
        != config.embedding_manifest_fingerprint
    ):
        raise RealPolicyCertificationError("dataset/embedding input identity mismatch")
    expected_split = dataset_manifest.get("splits", {}).get("query_cert")
    if not isinstance(expected_split, dict) or (
        expected_split.get("n") != config.query_split_n
        or expected_split.get("id_hash") != config.query_split_id_hash
    ):
        raise RealPolicyCertificationError("dataset cert split identity mismatch")

    corpus_table = load_embedding_array(
        embedding_cache / "corpus_embeddings.f32.npy",
        embedding_cache / "corpus_ids.json",
    )
    query_table = load_embedding_array(
        embedding_cache / "query_embeddings.f32.npy",
        embedding_cache / "query_ids.json",
    )
    query_rows = _load_jsonl(prepared / "queries.jsonl", "prepared queries")
    prepared_query_ids = [row.get("query_id") for row in query_rows]
    if query_table.ids.tolist() != prepared_query_ids:
        raise RealPolicyCertificationError(
            "query embedding rows do not match prepared query order"
        )
    selected_rows = [
        index
        for index, row in enumerate(query_rows)
        if row.get("split") == config.evaluation_split
    ]
    selected_query_ids = [prepared_query_ids[index] for index in selected_rows]
    if (
        len(selected_rows) != config.query_split_n
        or not all(isinstance(value, str) for value in selected_query_ids)
        or stable_id_hash(selected_query_ids) != config.query_split_id_hash
    ):
        raise RealPolicyCertificationError("selected queries are not the frozen cert IDs")

    corpus = np.asarray(corpus_table.vectors, dtype=np.float32)
    cert_queries = np.asarray(
        query_table.vectors[np.asarray(selected_rows, dtype=np.int64)],
        dtype=np.float32,
    )
    corpus_ids = corpus_table.ids.tolist()
    corpus_size, dimension = corpus.shape
    if (
        len(cert_queries) != config.query_split_n
        or cert_queries.shape[1] != dimension
        or dimension != embedding_config.model.embedding_dimension
        or config.m_grid[-1] != corpus_size
        or bundle.analytic_tri.corpus_size != corpus_size
        or config.m_prime > dimension
    ):
        raise RealPolicyCertificationError("embedding shape is incompatible with protocol")

    projection_started = perf_counter()
    matrix = dense_gaussian_projection(
        config.m_prime, dimension, config.projection_seed
    )
    projected_corpus = project_rows(corpus, matrix)
    projected_queries = project_rows(cert_queries, matrix)
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
        raise RealPolicyCertificationError("reconstructed projection fingerprint mismatch")

    rankings, projected_search_ms = _exact_projected_rankings(
        projected_corpus,
        projected_queries,
        corpus_ids,
        k=corpus_size,
        batch_size=config.query_batch_size,
    )
    ground_truth_started = perf_counter()
    original_index = ExactSquaredL2Index(
        corpus_ids, corpus, batch_size=config.query_batch_size
    )
    original = original_index.search(
        cert_queries, max(config.k_gt, config.s_lid)
    )
    ground_truth_ms = (perf_counter() - ground_truth_started) * 1000.0

    tie_rank = np.argsort(
        np.argsort(np.asarray(corpus_ids, dtype=str), kind="stable"), kind="stable"
    )
    lid_kwargs = {
        "s_lid": config.s_lid,
        "min_neighbors": config.min_lid_neighbors,
        "clip_min": config.lid_clip_min,
        "clip_max": config.lid_clip_max,
        "duplicate_tolerance": config.duplicate_tolerance,
        "fallback": config.lid_fallback,
    }
    records: list[Dict[str, Any]] = []
    pilot_ms = 0.0
    lid_ms = 0.0
    policy_ms = 0.0
    expansion_ms = 0.0
    rerank_ms = 0.0
    combined_original_evaluations = 0
    for query_index, ranking in enumerate(rankings):
        query = np.asarray(cert_queries[query_index], dtype=np.float64)
        pilot_rows = ranking[: config.m_pilot]
        started = perf_counter()
        pilot_difference = np.asarray(corpus[pilot_rows], dtype=np.float64) - query
        pilot_squared = np.einsum("ij,ij->i", pilot_difference, pilot_difference)
        pilot_ms += (perf_counter() - started) * 1000.0

        started = perf_counter()
        pilot_lid = estimate_lid_from_squared_distances(pilot_squared, **lid_kwargs)
        oracle_lid = estimate_lid_from_squared_distances(
            original.squared_distances[query_index, : config.s_lid], **lid_kwargs
        )
        pilot_serialized = _serialize_lid(
            pilot_lid, config.lid_decimal_places
        )
        oracle_serialized = _serialize_lid(
            oracle_lid, config.lid_decimal_places
        )
        lid_ms += (perf_counter() - started) * 1000.0

        lid_value = float(pilot_serialized["clipped"])
        lid_valid = bool(pilot_serialized["valid"])
        started = perf_counter()
        fixed_decision = bundle.fixed.choose(lid_value, lid_valid)
        monotone_decision = bundle.monotone.choose(lid_value, lid_valid)
        analytic_decision = bundle.analytic_tri.choose(lid_value, lid_valid)
        compiled_decision = bundle.compiled_tri.choose(lid_value, lid_valid)
        policy_ms += (perf_counter() - started) * 1000.0
        if (
            analytic_decision.budget,
            analytic_decision.saturated,
            analytic_decision.used_fallback,
        ) != (
            compiled_decision.budget,
            compiled_decision.saturated,
            compiled_decision.used_fallback,
        ):
            raise RealPolicyCertificationError(
                "compiled Tri-Predict disagrees with analytic policy on query_cert"
            )
        decisions = {
            "fixed_reference": fixed_decision,
            "monotone_binned": monotone_decision,
            "tri_predict": PolicyDecision(
                compiled_decision.budget,
                compiled_decision.bin_index,
                compiled_decision.used_fallback,
                compiled_decision.saturated,
                analytic_decision.predicted_retention,
                analytic_decision.raw_predicted_retention,
            ),
        }
        if any(
            decision.budget not in config.m_grid
            or decision.budget < config.m_pilot
            for decision in decisions.values()
        ):
            raise RealPolicyCertificationError("frozen policy emitted an unsafe budget")

        maximum_budget = max(decision.budget for decision in decisions.values())
        started = perf_counter()
        if maximum_budget > config.m_pilot:
            tail_rows = ranking[config.m_pilot : maximum_budget]
            tail_difference = np.asarray(corpus[tail_rows], dtype=np.float64) - query
            tail_squared = np.einsum("ij,ij->i", tail_difference, tail_difference)
            candidate_squared = np.concatenate((pilot_squared, tail_squared))
        else:
            candidate_squared = pilot_squared
        expansion_ms += (perf_counter() - started) * 1000.0
        combined_original_evaluations += maximum_budget

        exact_rows = set(original.rows[query_index, : config.k_gt].tolist())
        exact_ids = original.ids[query_index, : config.k_gt].tolist()
        started = perf_counter()
        policy_records = {
            name: _rerank_record(
                decision=decision,
                candidate_rows=ranking,
                candidate_distances=candidate_squared,
                tie_rank=tie_rank,
                corpus_ids=corpus_ids,
                exact_rows=exact_rows,
                exact_ids=exact_ids,
                k_gt=config.k_gt,
            )
            for name, decision in decisions.items()
        }
        rerank_ms += (perf_counter() - started) * 1000.0
        policy_records["tri_predict"]["compiled_decision_match"] = True
        records.append(
            {
                "query_index": query_index,
                "query_id": selected_query_ids[query_index],
                "split": "query_cert",
                "pilot_lid": pilot_serialized,
                "oracle_lid": oracle_serialized,
                "oracle_lid_role": "diagnostic_only_not_used_for_policy_decisions",
                "pilot_candidate_ids": [corpus_ids[row] for row in pilot_rows],
                "exact_top_k_ids": exact_ids,
                "projected_ranking_rows_sha256": _ranking_hash(ranking),
                "policies": policy_records,
            }
        )

    if (
        len(records) != config.query_split_n
        or {record["split"] for record in records} != {"query_cert"}
        or stable_id_hash([record["query_id"] for record in records])
        != config.query_split_id_hash
    ):
        raise RealPolicyCertificationError("certification records lost split identity")

    policy_fingerprints = {
        "fixed_reference": bundle.fixed_fingerprint,
        "monotone_binned": bundle.monotone.serialize()["fingerprint"],
        "tri_predict": bundle.analytic_tri.serialize()["fingerprint"],
    }
    evaluations: Dict[str, Any] = {}
    certificates: Dict[str, Any] = {}
    for name in _POLICY_NAMES:
        evaluation, certificate = _policy_evaluation(
            name=name,
            policy_fingerprint=policy_fingerprints[name],
            records=records,
            config=config,
            corpus_size=corpus_size,
            dimension=dimension,
        )
        evaluations[name] = evaluation
        certificates[name] = certificate

    certification_artifact = {
        "schema_version": 1,
        "kind": "real_frozen_policy_certificates_v1",
        "data_scope": "query_cert_only",
        "split_hash": config.query_split_id_hash,
        "n": len(records),
        "metric": "embedding_neighbor_retention_at_k_gt",
        "alpha_per_policy": config.certification_alpha,
        "target": config.certification_target,
        "familywise_adjustment": _FAMILYWISE_ROLE,
        "failure_behavior": _FAILURE_BEHAVIOR,
        "certificates": certificates,
        "all_passed": bool(all(value["passed"] for value in certificates.values())),
        "terminal": True,
    }
    certification_artifact["fingerprint"] = fingerprint(certification_artifact)
    paired_lid_gaps = [
        abs(
            float(record["pilot_lid"]["clipped"])
            - float(record["oracle_lid"]["clipped"])
        )
        for record in records
        if record["pilot_lid"]["valid"] and record["oracle_lid"]["valid"]
    ]
    summary = {
        "schema_version": 1,
        "kind": "real_frozen_policy_certification_summary_v1",
        "data_scope": "query_cert_only",
        "n_queries": len(records),
        "corpus_size": corpus_size,
        "embedding_dimension": dimension,
        "m_prime": config.m_prime,
        "projection_seed": config.projection_seed,
        "m_pilot": config.m_pilot,
        "s_lid": config.s_lid,
        "fixed_reference_budget": config.fixed_reference_budget,
        "certification": {
            "alpha": config.certification_alpha,
            "target": config.certification_target,
            "planned_n": config.planned_n,
            "familywise_adjustment": _FAMILYWISE_ROLE,
            "failure_behavior": _FAILURE_BEHAVIOR,
        },
        "policy_source_result_fingerprint": config.policy_source.result_fingerprint,
        "policy_fingerprints": policy_fingerprints,
        "compiled_deployment_fingerprint": config.policy_source.compiled_tri_predict_fingerprint,
        "compiled_reference_match_n": len(records),
        "policies": evaluations,
        "lid_diagnostic": {
            "pilot_valid_n": int(sum(record["pilot_lid"]["valid"] for record in records)),
            "oracle_valid_n": int(sum(record["oracle_lid"]["valid"] for record in records)),
            "paired_valid_n": len(paired_lid_gaps),
            "mean_absolute_clipped_gap": (
                0.0
                if not paired_lid_gaps
                else _canonical_float(
                    float(np.mean(paired_lid_gaps)), config.lid_decimal_places
                )
            ),
            "oracle_role": "diagnostic_only_not_used_for_policy_decisions",
        },
    }
    summary["fingerprint"] = fingerprint(summary)
    timings = {
        "role": "systems_diagnostic_excluded_from_result_identity",
        "projection_and_materialization_ms": projection_ms,
        "exact_projected_full_ranking_ms": projected_search_ms,
        "original_ground_truth_top_s_lid_ms": ground_truth_ms,
        "pilot_original_distance_ms": pilot_ms,
        "lid_ms": lid_ms,
        "all_policy_decisions_including_analytic_validation_ms": policy_ms,
        "shared_original_expansion_ms": expansion_ms,
        "all_policy_original_rerank_ms": rerank_ms,
        "projected_scan_count_per_query": 1,
        "projected_distance_evaluations": len(records) * corpus_size,
        "original_ground_truth_distance_evaluations_diagnostic": len(records)
        * corpus_size,
        "combined_shared_original_distance_evaluations": combined_original_evaluations,
        "counterfactual_original_rerank_distance_evaluations": {
            name: int(
                sum(record["policies"][name]["chosen_m"] for record in records)
            )
            for name in _POLICY_NAMES
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        paths = {
            "manifest.json": temporary / "manifest.json",
            "per_query.jsonl": temporary / "per_query.jsonl",
            "certifications.json": temporary / "certifications.json",
            "summary.json": temporary / "summary.json",
            "timings.json": temporary / "timings.json",
            "report.md": temporary / "report.md",
        }
        _write_jsonl(paths["per_query.jsonl"], records)
        write_json(paths["certifications.json"], certification_artifact)
        write_json(paths["summary.json"], summary)
        write_json(paths["timings.json"], timings)
        paths["report.md"].write_text(_report(summary), encoding="utf-8")
        result_artifacts = {
            name: _file_identity(paths[name]) for name in _CERT_RESULT_NAMES
        }
        result_identity = {
            "config_fingerprint": config.config_fingerprint,
            "dataset_manifest_fingerprint": config.dataset_manifest_fingerprint,
            "embedding_manifest_fingerprint": config.embedding_manifest_fingerprint,
            "policy_source_result_fingerprint": config.policy_source.result_fingerprint,
            "compiled_deployment_fingerprint": config.policy_source.compiled_tri_predict_fingerprint,
            "query_cert_id_hash": config.query_split_id_hash,
            "artifacts": result_artifacts,
        }
        manifest = {
            "schema_version": 1,
            "kind": "real_frozen_policy_certification_manifest_v1",
            "data_scope": "query_cert_only",
            "config_fingerprint": config.config_fingerprint,
            "dataset_manifest_fingerprint": config.dataset_manifest_fingerprint,
            "embedding_manifest_fingerprint": config.embedding_manifest_fingerprint,
            "query_cert_n": len(records),
            "query_cert_id_hash": config.query_split_id_hash,
            "projection_fingerprint": config.projection_fingerprint,
            "frozen_projection_fingerprint": config.frozen_projection_fingerprint,
            "policy_source": {
                "manifest_fingerprint": config.policy_source.manifest_fingerprint,
                "result_fingerprint": config.policy_source.result_fingerprint,
                "selection_fingerprint": config.policy_source.selection_fingerprint,
                "input_artifacts": bundle.input_artifacts,
            },
            "policy_fingerprints": policy_fingerprints,
            "deployment": {
                "role": config.policy_source.compiled_policy_role,
                "policy_fingerprint": config.policy_source.compiled_tri_predict_fingerprint,
                "reference_policy_fingerprint": config.policy_source.analytic_tri_predict_fingerprint,
                "artifact_sha256": config.policy_source.compiled_artifact_sha256,
                "reference_match_n": len(records),
            },
            "result_artifacts": result_artifacts,
            "result_fingerprint": fingerprint(result_identity),
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
    parser.add_argument("--policy-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_real_policy_certification_config(args.config)
    artifacts = run_real_policy_certification(
        config,
        args.dataset,
        args.embedding_config,
        args.embedding_cache,
        args.policy_run,
        args.output,
    )
    print(f"completed frozen-policy certification: {artifacts['report.md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
