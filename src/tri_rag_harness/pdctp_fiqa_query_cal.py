"""One-time, fingerprint-gated FiQA ``query_cal`` calibration runner.

This module opens only the complete frozen calibration role.  It never reads
qrels, never selects an operating point, and leaves tune/cert/latency/test
closed.  Exact original-space neighbors may supervise pilot-LID calibration
and exact top-k retention may supervise budget-residual calibration here only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .lid import estimate_lid_from_squared_distances
from .pdctp_calibration import (
    BudgetResidualRecord,
    LIDCalibrationRecord,
    PilotLIDCalibrator,
    TriBudgetResidualCalibrator,
)
from .pdctp_features import (
    PilotDistanceFeatureExtractor,
    PilotDistanceFeatureSpec,
    PilotDistanceObservation,
    PilotFeatureVector,
    stable_sort_pilot_distances,
)
from .pdctp_protocol import FIVE_ROLES, FiveRoleAssignments, FiveRoleProtocolGuard
from .pdctp_real_protocol import (
    PDCTPRealProtocolConfig,
    load_pdctp_real_protocol_config,
)
from .policies import TriPredictPolicy
from .projection import dense_gaussian_projection, projection_metadata
from .text_embeddings import load_text_embedding_config, validate_text_embedding_cache
from .tri_predict import tri_predict_retention_grid
from .utils import array_fingerprint, fingerprint, write_json


class PDCTPQueryCalError(ValueError):
    """Raised before protected access or when calibration cannot be frozen."""


@dataclass(frozen=True)
class PDCTPQueryCalConfig:
    raw: Mapping[str, Any]
    config_fingerprint: str
    run_name: str
    query_batch_size: int
    record_distance_decimals: int


def _exact(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise PDCTPQueryCalError(
            f"{context} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _hex(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PDCTPQueryCalError(f"{name} must be a lowercase SHA-256 value")
    return value


def load_pdctp_query_cal_config(
    path: Union[str, Path],
) -> PDCTPQueryCalConfig:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PDCTPQueryCalError(f"cannot load query_cal config: {exc}") from exc
    _exact(raw, {"schema", "version", "run_name", "bindings", "access", "execution"}, "root")
    if raw["schema"] != "pdctp_fiqa_query_cal_gate_v1" or raw["version"] != 1:
        raise PDCTPQueryCalError("unsupported FiQA query_cal gate schema")
    if not isinstance(raw["run_name"], str) or not raw["run_name"]:
        raise PDCTPQueryCalError("query_cal run name must be nonempty")
    bindings = raw["bindings"]
    binding_keys = {
        "protocol_freeze_fingerprint",
        "protocol_state_fingerprint",
        "role_assignments_fingerprint",
        "dataset_manifest_fingerprint",
        "embedding_config_fingerprint",
        "embedding_manifest_fingerprint",
        "embedding_audit_fingerprint",
        "embedding_audit_sha256",
        "query_cal_ordered_id_hash",
    }
    _exact(bindings, binding_keys, "bindings")
    for key, value in bindings.items():
        _hex(value, f"bindings.{key}")
    access = raw["access"]
    _exact(
        access,
        {
            "role",
            "require_complete_frozen_order",
            "allowed_supervision",
            "qrels_or_relevance_allowed",
            "blocked_roles",
            "selection_allowed",
        },
        "access",
    )
    if access != {
        "role": "query_cal",
        "require_complete_frozen_order": True,
        "allowed_supervision": [
            "oracle_exact_lid",
            "exact_original_top_k_identities",
            "realized_embedding_retention",
        ],
        "qrels_or_relevance_allowed": False,
        "blocked_roles": ["query_tune", "query_cert", "query_latency", "query_test"],
        "selection_allowed": False,
    }:
        raise PDCTPQueryCalError("query_cal access scope changed")
    execution = raw["execution"]
    _exact(
        execution,
        {
            "backend",
            "query_batch_size",
            "projection_dtype",
            "post_projection_normalize",
            "distance",
            "stable_tie_break",
            "pilot_expansion_reuse",
            "record_distance_decimals",
            "residual_solver",
            "candidate_storage",
        },
        "execution",
    )
    fixed_execution = {
        "backend": "numpy_exact_float64_batched_v1",
        "projection_dtype": "float64",
        "post_projection_normalize": False,
        "distance": "squared_l2",
        "stable_tie_break": "lexicographic_doc_id",
        "pilot_expansion_reuse": "one_projected_scan",
        "residual_solver": TriBudgetResidualCalibrator.COMPACT_SOLVER,
        "candidate_storage": "shared_fit_ids_reconstructable_v1",
    }
    if any(execution.get(key) != value for key, value in fixed_execution.items()):
        raise PDCTPQueryCalError("query_cal numerical or scope contract changed")
    batch_size = execution["query_batch_size"]
    decimals = execution["record_distance_decimals"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise PDCTPQueryCalError("query batch size must be a positive integer")
    if isinstance(decimals, bool) or not isinstance(decimals, int) or not 6 <= decimals <= 15:
        raise PDCTPQueryCalError("record distance decimals must be from 6 to 15")
    return PDCTPQueryCalConfig(
        raw=raw,
        config_fingerprint=fingerprint(raw),
        run_name=raw["run_name"],
        query_batch_size=batch_size,
        record_distance_decimals=decimals,
    )


def _file_identity(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _load_fingerprinted(path: Path, name: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PDCTPQueryCalError(f"cannot load {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise PDCTPQueryCalError(f"{name} must be an object")
    body = dict(value)
    stored = body.pop("fingerprint", None)
    if not isinstance(stored, str) or fingerprint(body) != stored:
        raise PDCTPQueryCalError(f"{name} fingerprint mismatch")
    return value


def _role_assignments(roles: Mapping[str, Any]) -> FiveRoleAssignments:
    role_rows = roles.get("roles")
    if not isinstance(role_rows, Mapping) or set(role_rows) != set(FIVE_ROLES):
        raise PDCTPQueryCalError("role artifact does not contain exactly five roles")
    ids_by_role: Dict[str, Tuple[str, ...]] = {}
    for role in FIVE_ROLES:
        row = role_rows[role]
        if not isinstance(row, Mapping):
            raise PDCTPQueryCalError(f"invalid role row: {role}")
        ids = tuple(str(value) for value in row.get("ordered_ids", ()))
        if (
            row.get("n") != len(ids)
            or row.get("ordered_id_hash") != fingerprint(list(ids))
        ):
            raise PDCTPQueryCalError(f"role identity mismatch: {role}")
        ids_by_role[role] = ids
    all_ids = [query_id for role in FIVE_ROLES for query_id in ids_by_role[role]]
    return FiveRoleAssignments(
        ids_by_role=ids_by_role,
        normalized_text_group_by_id={query_id: query_id for query_id in all_ids},
    )


def validate_query_cal_documents(
    config: PDCTPQueryCalConfig,
    protocol: Mapping[str, Any],
    state: Mapping[str, Any],
    roles: Mapping[str, Any],
    embedding_audit: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    embedding_manifest: Mapping[str, Any],
    *,
    embedding_audit_sha256: str,
) -> FiveRoleAssignments:
    """Validate all portable upstream identities before protected outcomes open."""
    bindings = config.raw["bindings"]
    observed = {
        "protocol_freeze_fingerprint": protocol.get("fingerprint"),
        "protocol_state_fingerprint": state.get("fingerprint"),
        "role_assignments_fingerprint": roles.get("fingerprint"),
        "dataset_manifest_fingerprint": dataset_manifest.get("fingerprint"),
        "embedding_config_fingerprint": embedding_audit.get(
            "embedding_config_fingerprint"
        ),
        "embedding_manifest_fingerprint": embedding_manifest.get("fingerprint"),
        "embedding_audit_fingerprint": embedding_audit.get("fingerprint"),
        "embedding_audit_sha256": embedding_audit_sha256,
    }
    for key, expected in bindings.items():
        if key == "query_cal_ordered_id_hash":
            continue
        if observed.get(key) != expected:
            raise PDCTPQueryCalError(f"frozen upstream binding changed: {key}")
    if (
        protocol.get("decision") != "READY_FOR_DATASET_AND_EMBEDDING_AUDIT_ONLY"
        or protocol.get("authorizes_method_evaluation") is not False
        or protocol.get("authorizes_protected_outcome_access") is not False
        or protocol.get("initial_guard_state_fingerprint") != state.get("fingerprint")
        or protocol.get("config_fingerprint") != state.get("config_fingerprint")
    ):
        raise PDCTPQueryCalError("protocol is not at the closed post-embedding gate")
    if (
        roles.get("all_roles_initially_closed") is not True
        or roles.get("authorizes_outcome_access") is not False
        or roles.get("fingerprint")
        != protocol.get("resolved_roles", {}).get("assignment_fingerprint")
    ):
        raise PDCTPQueryCalError("role freeze is not closed or protocol-bound")
    assignments = _role_assignments(roles)
    reconstructed = FiveRoleProtocolGuard(assignments, str(protocol["config_fingerprint"]))
    if reconstructed.serialize() != state:
        raise PDCTPQueryCalError("initial five-role guard state is not reconstructable")
    cal_ids = assignments.ids_by_role["query_cal"]
    if fingerprint(list(cal_ids)) != bindings["query_cal_ordered_id_hash"]:
        raise PDCTPQueryCalError("query_cal ordered identity changed")
    if (
        embedding_audit.get("decision") != "READY_TO_OPEN_QUERY_CAL"
        or embedding_audit.get("protocol_freeze_fingerprint") != protocol.get("fingerprint")
        or embedding_audit.get("protocol_state_fingerprint") != state.get("fingerprint")
        or embedding_audit.get("role_assignments_fingerprint") != roles.get("fingerprint")
        or embedding_audit.get("dataset_manifest_fingerprint")
        != dataset_manifest.get("fingerprint")
        or embedding_audit.get("embedding_manifest_fingerprint")
        != embedding_manifest.get("fingerprint")
        or embedding_audit.get("checks", {}).get("all_roles_remained_closed") is not True
        or embedding_audit.get("checks", {}).get("qrels_or_relevance_opened") is not False
    ):
        raise PDCTPQueryCalError("accepted embedding audit does not authorize query_cal")
    if embedding_audit.get("scope_guards") != {
        "contains_qrels_or_relevance": False,
        "contains_retrieval_or_policy_outcomes": False,
        "fits_or_selects_a_method": False,
        "runs_an_llm": False,
        "uses_an_approximate_index": False,
    }:
        raise PDCTPQueryCalError("embedding audit scope guards changed")
    if (
        dataset_manifest.get("protocol_freeze_fingerprint") != protocol.get("fingerprint")
        or dataset_manifest.get("role_assignments_fingerprint") != roles.get("fingerprint")
        or embedding_manifest.get("dataset", {}).get("manifest_fingerprint")
        != dataset_manifest.get("fingerprint")
        or embedding_audit.get("arrays") != embedding_manifest.get("arrays")
    ):
        raise PDCTPQueryCalError("dataset/cache/audit identities are not transitive")
    return assignments


def _stable_top_k_rows(
    distances: np.ndarray, tie_rank: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(tie_rank) or not 1 <= k <= len(values):
        raise PDCTPQueryCalError("invalid exact-search row")
    if k == len(values):
        selected = np.arange(len(values), dtype=np.int64)
    else:
        boundary = np.partition(values, k - 1)[k - 1]
        lower = np.flatnonzero(values < boundary)
        equal = np.flatnonzero(values == boundary)
        needed = k - len(lower)
        selected = np.concatenate(
            [lower, equal[np.argsort(tie_rank[equal], kind="stable")[:needed]]]
        )
    order = np.lexsort((tie_rank[selected], values[selected]))
    selected = selected[order]
    return selected, values[selected]


def _squared_l2_batch(
    queries: np.ndarray, corpus: np.ndarray, corpus_norms: np.ndarray
) -> np.ndarray:
    query_norms = np.einsum("ij,ij->i", queries, queries)
    distances = query_norms[:, None] + corpus_norms[None, :] - 2.0 * (queries @ corpus.T)
    np.maximum(distances, 0.0, out=distances)
    return np.asarray(distances, dtype=np.float64)


def _projected_ranks(
    distances: np.ndarray, target_rows: np.ndarray, tie_rank: np.ndarray
) -> np.ndarray:
    ranks = []
    for row in target_rows:
        value = distances[int(row)]
        ranks.append(
            1
            + int(np.count_nonzero(distances < value))
            + int(
                np.count_nonzero(
                    (distances == value) & (tie_rank < tie_rank[int(row)])
                )
            )
        )
    return np.asarray(ranks, dtype=np.int64)


def _round_values(values: Iterable[float], decimals: int) -> List[float]:
    return [
        0.0 if (rounded := float(np.round(float(value), decimals))) == 0.0 else rounded
        for value in values
    ]


def _level_key(value: float) -> str:
    return format(float(value), ".12g")


def build_query_cal_records(
    protocol: Mapping[str, Any],
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    corpus_embeddings: np.ndarray,
    query_cal_embeddings: np.ndarray,
    *,
    batch_size: int,
    record_distance_decimals: int,
    progress: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compute query_cal supervision with one exact projected scan per query."""
    retrieval = protocol["retrieval"]
    candidate_suite = protocol["candidate_suite"]
    feature_raw = protocol["features"]
    corpus = np.asarray(corpus_embeddings, dtype=np.float64)
    queries = np.asarray(query_cal_embeddings, dtype=np.float64)
    ids = np.asarray(corpus_ids, dtype=str)
    if (
        corpus.ndim != 2
        or queries.ndim != 2
        or corpus.shape[1] != queries.shape[1]
        or len(corpus) != len(ids)
        or len(queries) != len(query_ids)
        or len(set(ids.tolist())) != len(ids)
    ):
        raise PDCTPQueryCalError("query_cal arrays and stable IDs are not aligned")
    if not np.all(np.isfinite(corpus)) or not np.all(np.isfinite(queries)):
        raise PDCTPQueryCalError("query_cal arrays must be finite")
    if retrieval["corpus_size"] != len(corpus) or retrieval["embedding_dimension"] != corpus.shape[1]:
        raise PDCTPQueryCalError("retrieval dimensions differ from loaded embeddings")
    if retrieval["query_batch_size"] != batch_size:
        raise PDCTPQueryCalError("query_cal batch size differs from frozen protocol")
    feature_spec = PilotDistanceFeatureSpec(
        lid_boundary=feature_raw["lid_boundary"],
        minimum_count=feature_raw["minimum_count"],
        gap_quantiles=tuple(feature_raw["gap_quantiles"]),
        epsilon=feature_raw["epsilon"],
        duplicate_tolerance=feature_raw["duplicate_tolerance"],
        invalid_fill=feature_raw["invalid_fill"],
        output_decimals=feature_raw["output_decimals"],
        schema=feature_raw["schema"],
    )
    extractor = PilotDistanceFeatureExtractor(feature_spec)
    projection_raw = retrieval["projection"]
    matrix = dense_gaussian_projection(
        projection_raw["m_prime"], corpus.shape[1], projection_raw["seed"]
    )
    projected_corpus = corpus @ matrix.T
    projected_queries = queries @ matrix.T
    if projection_raw["post_projection_normalize"] is not False:
        raise PDCTPQueryCalError("post-projection normalization is forbidden")
    corpus_norms = np.einsum("ij,ij->i", corpus, corpus)
    projected_norms = np.einsum("ij,ij->i", projected_corpus, projected_corpus)
    lexical_order = np.argsort(ids, kind="stable")
    tie_rank = np.empty(len(ids), dtype=np.int64)
    tie_rank[lexical_order] = np.arange(len(ids), dtype=np.int64)
    oracle_k = max(retrieval["k_gt"], retrieval["s_lid"])
    grid = tuple(int(value) for value in retrieval["m_grid"])
    levels = tuple(float(value) for value in candidate_suite["residual_training_levels"])
    records: List[Dict[str, Any]] = []

    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        original_block = _squared_l2_batch(queries[start:stop], corpus, corpus_norms)
        projected_block = _squared_l2_batch(
            projected_queries[start:stop], projected_corpus, projected_norms
        )
        for offset, query_id in enumerate(query_ids[start:stop]):
            query = queries[start + offset]
            original_distances = original_block[offset]
            projected_distances = projected_block[offset]
            oracle_rows, oracle_sq = _stable_top_k_rows(
                original_distances, tie_rank, oracle_k
            )
            pilot_rows, pilot_projected_sq = _stable_top_k_rows(
                projected_distances, tie_rank, retrieval["m_pilot"]
            )
            pilot_diff = corpus[pilot_rows] - query
            pilot_original_sq = np.einsum("ij,ij->i", pilot_diff, pilot_diff)
            sorted_ids, sorted_original_sq, sorted_projected_sq = stable_sort_pilot_distances(
                ids[pilot_rows], pilot_original_sq, pilot_projected_sq
            )
            pilot_lid = estimate_lid_from_squared_distances(
                sorted_original_sq,
                s_lid=retrieval["s_lid"],
                min_neighbors=retrieval["min_lid_neighbors"],
                clip_min=candidate_suite["lid_output_domain"][0],
                clip_max=candidate_suite["lid_output_domain"][1],
                duplicate_tolerance=feature_spec.duplicate_tolerance,
                fallback=candidate_suite["lid_fallback"],
            )
            observation = PilotDistanceObservation.from_arrays(
                sorted_original_sq,
                sorted_projected_sq,
                pilot_lid=pilot_lid.clipped,
                pilot_lid_valid=pilot_lid.valid,
                pilot_lid_failure_reason=pilot_lid.reason,
                valid_distance_count=pilot_lid.valid_distance_count,
            )
            features = extractor.extract(observation)
            oracle_lid = estimate_lid_from_squared_distances(
                oracle_sq[: retrieval["s_lid"]],
                s_lid=retrieval["s_lid"],
                min_neighbors=retrieval["min_lid_neighbors"],
                clip_min=candidate_suite["lid_output_domain"][0],
                clip_max=candidate_suite["lid_output_domain"][1],
                duplicate_tolerance=feature_spec.duplicate_tolerance,
                fallback=candidate_suite["lid_fallback"],
            )
            gt_rows = oracle_rows[: retrieval["k_gt"]]
            gt_projected_ranks = _projected_ranks(
                projected_distances, gt_rows, tie_rank
            )
            retention_by_budget = {
                str(budget): float(np.mean(gt_projected_ranks <= budget))
                for budget in grid
            }
            required_by_level: Dict[str, int] = {}
            for level in levels:
                matches = [
                    budget
                    for budget in grid
                    if retention_by_budget[str(budget)] >= level
                ]
                if not matches:
                    raise AssertionError("full corpus must retain every exact top-k row")
                required_by_level[_level_key(level)] = int(matches[0])
            body: Dict[str, Any] = {
                "schema": "pdctp_fiqa_query_cal_record_v1",
                "query_id": str(query_id),
                "role": "query_cal",
                "supervision": {
                    "oracle_exact_lid": True,
                    "exact_original_top_k_identities": True,
                    "realized_embedding_retention": True,
                    "qrels_or_relevance": False,
                },
                "pilot": {
                    "candidate_doc_ids_original_distance_order": sorted_ids.tolist(),
                    "original_squared_distances": _round_values(
                        sorted_original_sq, record_distance_decimals
                    ),
                    "projected_squared_distances": _round_values(
                        sorted_projected_sq, record_distance_decimals
                    ),
                    "lid": float(np.round(pilot_lid.clipped, 10)),
                    "lid_valid": pilot_lid.valid,
                    "lid_failure_reason": pilot_lid.reason,
                    "features": features.serialize(),
                },
                "oracle": {
                    "top_k_doc_ids": ids[gt_rows].tolist(),
                    "top_lid_squared_distances": _round_values(
                        oracle_sq[: retrieval["s_lid"]], record_distance_decimals
                    ),
                    "lid": float(np.round(oracle_lid.clipped, 10)),
                    "lid_valid": oracle_lid.valid,
                    "lid_failure_reason": oracle_lid.reason,
                },
                "projected_rank_of_exact_top_k": gt_projected_ranks.tolist(),
                "retention_by_budget": retention_by_budget,
                "required_budget_by_training_level": required_by_level,
                "work": {
                    "projected_scan_count": 1,
                    "projected_distance_count": len(corpus),
                    "original_reference_distance_count": len(corpus),
                    "pilot_original_rerank_distance_count": retrieval["m_pilot"],
                    "pilot_is_prefix_of_same_projected_scan": True,
                },
            }
            body["fingerprint"] = fingerprint(body)
            body["_features_obj"] = features
            records.append(body)
        if progress:
            print(f"query_cal exact retrieval: {stop}/{len(queries)}", flush=True)

    projection = projection_metadata(
        dimension=corpus.shape[1],
        m_prime=projection_raw["m_prime"],
        seed=projection_raw["seed"],
        normalization=True,
        embedding_model=protocol["embedding"]["model"]["name"],
        corpus_hash="supplied_by_embedding_audit",
    )
    projection.update(
        {
            "schema": "pdctp_fiqa_dense_gaussian_projection_v1",
            "matrix": {
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
                "array_fingerprint": array_fingerprint(matrix),
            },
            "projected_corpus_shape": list(projected_corpus.shape),
            "projected_query_role": "query_cal",
            "projected_query_shape": list(projected_queries.shape),
            "projected_vectors_persisted": False,
        }
    )
    projection.pop("fingerprint", None)
    projection["fingerprint"] = fingerprint(projection)
    return records, projection


def _public_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _raw_budget_from_predictions(
    policy: TriPredictPolicy, predictions: Mapping[int, float]
) -> int:
    for budget in policy.grid:
        if budget == policy.corpus_size:
            corrected = 1.0
        else:
            corrected = float(predictions[budget])
            if policy.target == 1.0:
                continue
        if corrected >= policy.target:
            return budget
    return policy.grid[-1]


def _compact_model_artifact(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    compact = json.loads(json.dumps(artifact))
    compact["fit"].pop("ordered_ids")
    return compact


def _restore_model_artifact(
    compact: Mapping[str, Any], ordered_ids: Sequence[str]
) -> Dict[str, Any]:
    restored = json.loads(json.dumps(compact))
    restored["fit"]["ordered_ids"] = list(ordered_ids)
    return restored


def reconstruct_residual_candidate(
    bundle: Mapping[str, Any], candidate_fingerprint: str
) -> TriBudgetResidualCalibrator:
    """Reconstruct and validate one compactly stored residual operating point."""
    body = dict(bundle)
    stored = body.pop("fingerprint", None)
    if not isinstance(stored, str) or fingerprint(body) != stored:
        raise PDCTPQueryCalError("residual candidate bundle fingerprint mismatch")
    shared_fit = bundle.get("shared_fit")
    if not isinstance(shared_fit, Mapping):
        raise PDCTPQueryCalError("residual candidate bundle has no shared fit")
    ordered_ids = shared_fit.get("ordered_ids")
    if (
        not isinstance(ordered_ids, list)
        or shared_fit.get("ordered_id_hash") != fingerprint(ordered_ids)
        or shared_fit.get("n") != len(ordered_ids)
    ):
        raise PDCTPQueryCalError("residual shared fit identity is invalid")
    candidates = list(bundle.get("full_operating_points", ())) + list(
        bundle.get("residual_only_operating_points", ())
    )
    matches = [row for row in candidates if row.get("fingerprint") == candidate_fingerprint]
    if len(matches) != 1:
        raise PDCTPQueryCalError("residual operating point is missing or ambiguous")
    candidate = matches[0]
    candidate_body = dict(candidate)
    claimed = candidate_body.pop("fingerprint")
    if fingerprint(candidate_body) != claimed:
        raise PDCTPQueryCalError("residual operating-point fingerprint mismatch")
    models = [
        row
        for row in bundle.get("base_models", ())
        if row.get("storage_key") == candidate.get("base_model_storage_key")
    ]
    if len(models) != 1:
        raise PDCTPQueryCalError("residual base model is missing or ambiguous")
    model_row = models[0]
    model_body = dict(model_row)
    model_stored = model_body.pop("fingerprint", None)
    if not isinstance(model_stored, str) or fingerprint(model_body) != model_stored:
        raise PDCTPQueryCalError("residual base-model storage fingerprint mismatch")
    base = TriBudgetResidualCalibrator.from_serialized(
        _restore_model_artifact(model_row["artifact"], ordered_ids)
    )
    effective = base.with_operating_point(safety_offset=candidate["safety_offset"])
    if effective.fingerprint != candidate.get("effective_calibrator_fingerprint"):
        raise PDCTPQueryCalError("residual operating point does not reconstruct")
    return effective


def fit_query_cal_candidates(
    protocol: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *, progress: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fit all preregistered candidates without selecting among them."""
    suite = protocol["candidate_suite"]
    retrieval = protocol["retrieval"]
    cal_ids = [str(row["query_id"]) for row in records]
    lid_rows = [
        LIDCalibrationRecord(
            query_id=str(row["query_id"]),
            role="query_cal",
            features=row["_features_obj"],
            oracle_lid=float(row["oracle"]["lid"]),
        )
        for row in records
        if row["_features_obj"].valid and row["oracle"]["lid_valid"]
    ]
    if not lid_rows:
        raise PDCTPQueryCalError("no valid query_cal rows for LID fitting")
    lid_candidates = [
        PilotLIDCalibrator.fit(
            lid_rows,
            regularization=regularization,
            output_min=suite["lid_output_domain"][0],
            output_max=suite["lid_output_domain"][1],
            fallback=suite["lid_fallback"],
        )
        for regularization in suite["lid_regularization_grid"]
    ]
    lid_artifacts = [candidate.serialize() for candidate in lid_candidates]
    lid_bundle: Dict[str, Any] = {
        "name": "pdctp_fiqa_lid_calibrator_candidate_bundle",
        "schema": "pdctp_fiqa_lid_calibrator_candidates_v1",
        "version": 1,
        "fit_role": "query_cal",
        "query_cal_access": {
            "ordered_ids": cal_ids,
            "ordered_id_hash": fingerprint(cal_ids),
            "n": len(cal_ids),
        },
        "fit_eligibility": {
            "ordered_ids": [row.query_id for row in lid_rows],
            "ordered_id_hash": fingerprint([row.query_id for row in lid_rows]),
            "n": len(lid_rows),
            "excluded_n": len(records) - len(lid_rows),
            "rule": "valid_pilot_features_and_valid_oracle_lid",
        },
        "candidates": lid_artifacts,
        "candidate_count": len(lid_artifacts),
        "selection_performed": False,
    }
    lid_bundle["fingerprint"] = fingerprint(lid_bundle)

    residual_records = [row for row in records if row["_features_obj"].valid]
    if not residual_records:
        raise PDCTPQueryCalError("no valid query_cal rows for residual fitting")
    residual_ids = [str(row["query_id"]) for row in residual_records]
    raw_policies = [
        TriPredictPolicy(
            corpus_size=retrieval["corpus_size"],
            m_prime=retrieval["projection"]["m_prime"],
            k_gt=retrieval["k_gt"],
            grid=retrieval["m_grid"],
            target=target,
            max_rank_samples=retrieval["max_rank_samples"],
        )
        for target in suite["raw_tri_threshold_grid"]
    ]
    raw_artifacts = [policy.serialize() for policy in raw_policies]
    lid_sources: List[Tuple[str, Optional[PilotLIDCalibrator]]] = [
        ("raw_pilot_lid", None)
    ] + [(candidate.fingerprint, candidate) for candidate in lid_candidates]
    prediction_cache: Dict[float, Dict[int, float]] = {}
    base_models: List[TriBudgetResidualCalibrator] = []
    base_model_lid_keys: List[Optional[str]] = []
    full_points: List[Dict[str, Any]] = []
    raw_points: List[Dict[str, Any]] = []
    completed_anchors = 0
    anchor_count = len(lid_sources) * len(raw_policies)

    for lid_key, lid_calibrator in lid_sources:
        raw_budgets_by_policy: Dict[str, List[int]] = {
            artifact["fingerprint"]: [] for artifact in raw_artifacts
        }
        for row in residual_records:
            if lid_calibrator is None:
                lid_value = float(row["pilot"]["lid"])
            else:
                prediction = lid_calibrator.predict(row["_features_obj"])
                if not prediction.valid:
                    raise AssertionError("valid fit features must produce calibrated LID")
                lid_value = prediction.value
            cache_key = float(lid_value)
            predictions = prediction_cache.get(cache_key)
            if predictions is None:
                predictions = tri_predict_retention_grid(
                    lid=lid_value,
                    m_prime=retrieval["projection"]["m_prime"],
                    k_gt=retrieval["k_gt"],
                    budgets=retrieval["m_grid"],
                    corpus_size=retrieval["corpus_size"],
                    max_rank_samples=retrieval["max_rank_samples"],
                )
                prediction_cache[cache_key] = predictions
            for policy, artifact in zip(raw_policies, raw_artifacts):
                raw_budgets_by_policy[artifact["fingerprint"]].append(
                    _raw_budget_from_predictions(policy, predictions)
                )

        for policy, raw_artifact in zip(raw_policies, raw_artifacts):
            raw_fingerprint = raw_artifact["fingerprint"]
            raw_budgets = raw_budgets_by_policy[raw_fingerprint]
            required_vectors = {
                float(level): tuple(
                    int(row["required_budget_by_training_level"][_level_key(level)])
                    for row in residual_records
                )
                for level in suite["residual_training_levels"]
            }
            fitted_by_required: Dict[Tuple[int, ...], Dict[Tuple[float, float], TriBudgetResidualCalibrator]] = {}
            for level in suite["residual_training_levels"]:
                required = required_vectors[float(level)]
                fitted_grid = fitted_by_required.setdefault(required, {})
                fit_rows = [
                    BudgetResidualRecord(
                        query_id=str(row["query_id"]),
                        role="query_cal",
                        features=row["_features_obj"],
                        raw_budget=raw_budget,
                        required_budget=required_budget,
                        training_level=float(level),
                    )
                    for row, raw_budget, required_budget in zip(
                        residual_records, raw_budgets, required
                    )
                ]
                for quantile in suite["residual_quantiles"]:
                    for regularization in suite["residual_regularization_grid"]:
                        model_key = (float(quantile), float(regularization))
                        if model_key in fitted_grid:
                            fitted = fitted_grid[model_key].with_operating_point(
                                safety_offset=0.0, training_level=float(level)
                            )
                        else:
                            fitted = TriBudgetResidualCalibrator.fit_compact(
                                fit_rows,
                                quantile=quantile,
                                regularization=regularization,
                                safety_offset=0.0,
                                grid=retrieval["m_grid"],
                                minimum_budget=max(
                                    retrieval["k_gt"], retrieval["m_pilot"]
                                ),
                                fallback_budget=retrieval["m_grid"][-1],
                                raw_policy_fingerprint=raw_fingerprint,
                                anchor_lid_source=(
                                    "raw_pilot_lid"
                                    if lid_calibrator is None
                                    else "calibrated_pilot_lid"
                                ),
                            )
                            fitted_grid[model_key] = fitted
                        base_models.append(fitted)
                        base_model_lid_keys.append(
                            None if lid_calibrator is None else str(lid_key)
                        )
                        storage_key = fingerprint(
                            {
                                "calibrator_fingerprint": fitted.fingerprint,
                                "lid_calibrator_fingerprint": (
                                    None if lid_calibrator is None else lid_key
                                ),
                            }
                        )
                        for safety_offset in suite["safety_offsets"]:
                            effective = fitted.with_operating_point(
                                safety_offset=safety_offset
                            )
                            point: Dict[str, Any] = {
                                "raw_policy_fingerprint": raw_fingerprint,
                                "raw_tri_threshold": policy.target,
                                "lid_source": (
                                    "raw_pilot_lid"
                                    if lid_calibrator is None
                                    else "calibrated_pilot_lid"
                                ),
                                "lid_calibrator_fingerprint": (
                                    None if lid_calibrator is None else lid_key
                                ),
                                "training_level": float(level),
                                "quantile": float(quantile),
                                "regularization": float(regularization),
                                "safety_offset": float(safety_offset),
                                "base_model_fingerprint": fitted.fingerprint,
                                "base_model_storage_key": storage_key,
                                "effective_calibrator_fingerprint": effective.fingerprint,
                            }
                            point["fingerprint"] = fingerprint(point)
                            if lid_calibrator is None:
                                raw_points.append(point)
                            else:
                                full_points.append(point)
            completed_anchors += 1
            if progress:
                print(
                    f"query_cal residual anchors: {completed_anchors}/{anchor_count}",
                    flush=True,
                )

    base_artifacts = [model.serialize() for model in base_models]
    base_storage_rows = []
    for model_artifact, lid_key in zip(base_artifacts, base_model_lid_keys):
        storage_body: Dict[str, Any] = {
            "storage_key": fingerprint(
                {
                    "calibrator_fingerprint": model_artifact["fingerprint"],
                    "lid_calibrator_fingerprint": lid_key,
                }
            ),
            "lid_calibrator_fingerprint": lid_key,
            "artifact": _compact_model_artifact(model_artifact),
        }
        storage_body["fingerprint"] = fingerprint(storage_body)
        base_storage_rows.append(storage_body)
    if len({row["storage_key"] for row in base_storage_rows}) != len(base_storage_rows):
        raise AssertionError("residual base-model storage keys must be unique")
    expected_full = (
        len(raw_policies)
        * len(lid_candidates)
        * len(suite["residual_training_levels"])
        * len(suite["residual_quantiles"])
        * len(suite["residual_regularization_grid"])
        * len(suite["safety_offsets"])
    )
    expected_raw = (
        len(raw_policies)
        * len(suite["residual_training_levels"])
        * len(suite["residual_quantiles"])
        * len(suite["residual_regularization_grid"])
        * len(suite["safety_offsets"])
    )
    if expected_full != suite["expected_full_pdctp_tuples"]:
        raise PDCTPQueryCalError("full PDCTP tuple count differs from protocol")
    if len(full_points) != expected_full or len(raw_points) != expected_raw:
        raise AssertionError("residual operating-point enumeration is incomplete")
    residual_bundle: Dict[str, Any] = {
        "name": "pdctp_fiqa_budget_residual_candidate_bundle",
        "schema": "pdctp_fiqa_budget_residual_candidates_v1",
        "version": 1,
        "fit_role": "query_cal",
        "query_cal_access": {
            "ordered_ids": cal_ids,
            "ordered_id_hash": fingerprint(cal_ids),
            "n": len(cal_ids),
        },
        "shared_fit": {
            "role": "query_cal",
            "ordered_ids": residual_ids,
            "ordered_id_hash": fingerprint(residual_ids),
            "n": len(residual_ids),
            "excluded_n": len(records) - len(residual_records),
            "eligibility_rule": "valid_deployable_pilot_features",
        },
        "storage": {
            "name": "shared_fit_ids_reconstructable_v1",
            "base_model_order": "lid_source_then_raw_threshold_then_training_level_then_quantile_then_regularization",
            "operating_point_order": "base_model_then_safety_offset",
            "base_models_store_safety_offset": 0.0,
            "every_operating_point_stores_effective_calibrator_fingerprint": True,
        },
        "raw_policies": raw_artifacts,
        "base_models": base_storage_rows,
        "full_operating_points": full_points,
        "residual_only_operating_points": raw_points,
        "counts": {
            "base_models": len(base_models),
            "full_pdctp_operating_points": len(full_points),
            "residual_only_operating_points": len(raw_points),
        },
        "selection_performed": False,
    }
    residual_bundle["fingerprint"] = fingerprint(residual_bundle)
    # Exercise reconstruction at both ends before allowing the bundle to freeze.
    for point in (full_points[0], full_points[-1], raw_points[0], raw_points[-1]):
        reconstruct_residual_candidate(residual_bundle, point["fingerprint"])
    return lid_bundle, residual_bundle


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )


def run_pdctp_fiqa_query_cal(
    config: PDCTPQueryCalConfig,
    real_protocol_config: PDCTPRealProtocolConfig,
    protocol_freeze_path: Union[str, Path],
    protocol_state_path: Union[str, Path],
    role_assignments_path: Union[str, Path],
    embedding_audit_path: Union[str, Path],
    embedding_config_path: Union[str, Path],
    prepared_dir: Union[str, Path],
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    progress: bool = True,
) -> Dict[str, Path]:
    """Validate, open query_cal once, fit every candidate, and freeze the fits."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite query_cal run: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = _load_fingerprinted(Path(protocol_freeze_path), "protocol freeze")
    state = _load_fingerprinted(Path(protocol_state_path), "protocol state")
    roles = _load_fingerprinted(Path(role_assignments_path), "role assignments")
    audit_path = Path(embedding_audit_path)
    embedding_audit = _load_fingerprinted(audit_path, "embedding audit")
    embedding_config = load_text_embedding_config(Path(embedding_config_path))
    validated_cache = validate_text_embedding_cache(
        embedding_config, Path(prepared_dir), Path(cache_dir)
    )
    dataset_manifest = validated_cache["dataset_manifest"]
    embedding_manifest = validated_cache["embedding_manifest"]
    if real_protocol_config.config_fingerprint != protocol.get("config_fingerprint"):
        raise PDCTPQueryCalError("real protocol config differs from protocol freeze")
    if real_protocol_config.raw != protocol.get("protocol"):
        raise PDCTPQueryCalError("embedded real protocol differs from checked config")
    if embedding_config.config_fingerprint != config.raw["bindings"]["embedding_config_fingerprint"]:
        raise PDCTPQueryCalError("embedding config fingerprint changed")
    assignments = validate_query_cal_documents(
        config,
        protocol,
        state,
        roles,
        embedding_audit,
        dataset_manifest,
        embedding_manifest,
        embedding_audit_sha256=_file_identity(audit_path)["sha256"],
    )
    # No query outcome or query-vector slice is opened before every check above.
    guard = FiveRoleProtocolGuard(assignments, real_protocol_config.config_fingerprint)
    cal_ids = assignments.ids_by_role["query_cal"]
    token = guard.open_calibration(cal_ids)

    cache = Path(cache_dir)
    corpus_ids = json.loads((cache / "corpus_ids.json").read_text(encoding="utf-8"))
    all_query_ids = json.loads((cache / "query_ids.json").read_text(encoding="utf-8"))
    if not isinstance(corpus_ids, list) or not isinstance(all_query_ids, list):
        raise PDCTPQueryCalError("embedding stable-ID artifacts are invalid")
    query_row = {query_id: index for index, query_id in enumerate(all_query_ids)}
    if len(query_row) != len(all_query_ids) or any(query_id not in query_row for query_id in cal_ids):
        raise PDCTPQueryCalError("query_cal IDs do not map uniquely into the cache")
    cal_rows = np.asarray([query_row[query_id] for query_id in cal_ids], dtype=np.int64)
    corpus_embeddings = np.load(
        cache / embedding_manifest["arrays"]["corpus"]["file"],
        mmap_mode="r",
        allow_pickle=False,
    )
    query_embeddings = np.load(
        cache / embedding_manifest["arrays"]["queries"]["file"],
        mmap_mode="r",
        allow_pickle=False,
    )
    # Materialize only query_cal rows; other protected role vectors remain unused.
    query_cal_embeddings = np.asarray(query_embeddings[cal_rows], dtype=np.float64)
    records, projection = build_query_cal_records(
        real_protocol_config.raw,
        cal_ids,
        corpus_ids,
        corpus_embeddings,
        query_cal_embeddings,
        batch_size=config.query_batch_size,
        record_distance_decimals=config.record_distance_decimals,
        progress=progress,
    )
    lid_bundle, residual_bundle = fit_query_cal_candidates(
        real_protocol_config.raw, records, progress=progress
    )
    guard.register_fit(
        "lid_calibrator",
        role="query_cal",
        ids=cal_ids,
        artifact_fingerprint=lid_bundle["fingerprint"],
    )
    guard.register_fit(
        "residual_calibrator",
        role="query_cal",
        ids=cal_ids,
        artifact_fingerprint=residual_bundle["fingerprint"],
    )
    access: Dict[str, Any] = {
        "name": "pdctp_fiqa_query_cal_access",
        "schema": "pdctp_fiqa_query_cal_access_v1",
        "version": 1,
        "token": token.serialize(),
        "allowed_supervision": config.raw["access"]["allowed_supervision"],
        "qrels_or_relevance_accessed": False,
        "roles_opened": ["query_cal"],
        "roles_remaining_closed": list(config.raw["access"]["blocked_roles"]),
        "selection_performed": False,
    }
    access["fingerprint"] = fingerprint(access)

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        write_json(temporary / "query_cal_access.json", access)
        write_json(temporary / "projection.json", projection)
        _write_jsonl(
            temporary / "query_cal_records.jsonl",
            (_public_record(record) for record in records),
        )
        write_json(temporary / "lid_calibrator_candidates.json", lid_bundle)
        write_json(temporary / "residual_calibrator_candidates.json", residual_bundle)
        write_json(temporary / "protocol_state_after_query_cal.json", guard.serialize())
        artifact_names = [
            "query_cal_access.json",
            "projection.json",
            "query_cal_records.jsonl",
            "lid_calibrator_candidates.json",
            "residual_calibrator_candidates.json",
            "protocol_state_after_query_cal.json",
        ]
        valid_features = sum(bool(row["_features_obj"].valid) for row in records)
        valid_oracle = sum(bool(row["oracle"]["lid_valid"]) for row in records)
        level_vectors = {
            tuple(row["required_budget_by_training_level"].values()) for row in records
        }
        manifest: Dict[str, Any] = {
            "name": "pdctp_fiqa_query_cal_fit_manifest",
            "schema": "pdctp_fiqa_query_cal_fit_manifest_v1",
            "version": 1,
            "config_fingerprint": config.config_fingerprint,
            "upstream": {
                key: value for key, value in config.raw["bindings"].items()
            },
            "query_cal_access_fingerprint": access["fingerprint"],
            "projection_fingerprint": projection["fingerprint"],
            "lid_candidate_bundle_fingerprint": lid_bundle["fingerprint"],
            "residual_candidate_bundle_fingerprint": residual_bundle["fingerprint"],
            "protocol_state_fingerprint": guard.state_fingerprint,
            "counts": {
                "query_cal": len(records),
                "valid_pilot_features": valid_features,
                "valid_oracle_lid": valid_oracle,
                "lid_calibrator_candidates": lid_bundle["candidate_count"],
                **residual_bundle["counts"],
            },
            "diagnostics": {
                "all_training_levels_require_same_budget_per_query": all(
                    len(set(row["required_budget_by_training_level"].values())) == 1
                    for row in records
                ),
                "distinct_required_budget_level_vectors": len(level_vectors),
                "discrete_retention_increment": 1.0 / real_protocol_config.raw["retrieval"]["k_gt"],
            },
            "checks": {
                "all_upstream_fingerprints_validated_before_query_cal": True,
                "only_query_cal_opened": True,
                "all_query_cal_ids_used_in_frozen_order": True,
                "qrels_or_relevance_accessed": False,
                "query_tune_accessed": False,
                "query_cert_accessed": False,
                "query_latency_accessed": False,
                "query_test_accessed": False,
                "selection_performed": False,
                "one_projected_scan_per_query": all(
                    row["work"]["projected_scan_count"] == 1 for row in records
                ),
                "projected_vectors_renormalized": False,
                "llm_run": False,
                "approximate_index_used": False,
                "raw_tri_predict_v1_behavior_modified": False,
            },
            "artifacts": {
                name: _file_identity(temporary / name) for name in artifact_names
            },
            "decision": "QUERY_CAL_FITS_FROZEN_READY_FOR_QUERY_TUNE",
        }
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(temporary / "manifest.json", manifest)
        report = (
            "# PDCTP FiQA query_cal fit gate\n\n"
            f"Decision: `{manifest['decision']}`.\n\n"
            f"Opened role: `query_cal` ({len(records):,} frozen IDs). All other roles remain closed.\n\n"
            f"Valid pilot features: {valid_features:,}; valid oracle LID targets: {valid_oracle:,}.\n\n"
            f"Fit {lid_bundle['candidate_count']} LID candidates, "
            f"{residual_bundle['counts']['base_models']} residual base models, "
            f"{residual_bundle['counts']['full_pdctp_operating_points']} full-PDCTP "
            f"and {residual_bundle['counts']['residual_only_operating_points']} residual-only operating points.\n\n"
            "No candidate was selected. No qrel/relevance value, query_tune, query_cert, "
            "query_latency, query_test, LLM, or approximate index was accessed.\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        name: output / name
        for name in (
            "query_cal_access.json",
            "projection.json",
            "query_cal_records.jsonl",
            "lid_calibrator_candidates.json",
            "residual_calibrator_candidates.json",
            "protocol_state_after_query_cal.json",
            "manifest.json",
            "report.md",
        )
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--real-protocol-config", required=True, type=Path)
    parser.add_argument("--protocol-freeze", required=True, type=Path)
    parser.add_argument("--protocol-state", required=True, type=Path)
    parser.add_argument("--role-assignments", required=True, type=Path)
    parser.add_argument("--embedding-audit", required=True, type=Path)
    parser.add_argument("--embedding-config", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--embedding-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = run_pdctp_fiqa_query_cal(
        load_pdctp_query_cal_config(args.config),
        load_pdctp_real_protocol_config(args.real_protocol_config),
        args.protocol_freeze,
        args.protocol_state,
        args.role_assignments,
        args.embedding_audit,
        args.embedding_config,
        args.prepared,
        args.embedding_cache,
        args.output,
    )
    print(
        f"PDCTP FiQA query_cal gate wrote {len(paths)} artifacts to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
