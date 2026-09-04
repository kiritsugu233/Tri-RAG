"""Fingerprint-gated FiQA ``query_tune`` policy-selection runner.

The runner reconstructs and freezes the accepted ``query_cal`` fits before it
opens the complete tune role.  Tune labels select one independently optimized
member of each preregistered method family.  Certification, latency, and test
roles remain closed, and no calibrator is refit here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .certification import empirical_bernstein
from .lid import estimate_lid_from_squared_distances
from .pdctp_calibration import PilotLIDCalibrator, TriBudgetResidualCalibrator
from .pdctp_features import (
    PilotDistanceFeatureExtractor,
    PilotDistanceFeatureSpec,
    PilotDistanceObservation,
    PilotFeatureVector,
    stable_sort_pilot_distances,
)
from .pdctp_fiqa_query_cal import (
    _file_identity,
    _load_fingerprinted,
    _public_record,
    _restore_model_artifact,
    _round_values,
    _squared_l2_batch,
    _stable_top_k_rows,
)
from .pdctp_policies import (
    CalibratedTriPredictPolicy,
    FixedPDCTPPolicy,
    MonotonePDCTPPolicy,
    PDCTPDecisionInput,
    PDCTPDecisionPolicy,
    RawTriPredictPDCTPPolicy,
    validate_policy_suite,
)
from .pdctp_protocol import FIVE_ROLES, FiveRoleAssignments, FiveRoleProtocolGuard
from .pdctp_real_protocol import PDCTPRealProtocolConfig, load_pdctp_real_protocol_config
from .pdctp_statistics import bonferroni_allocation, make_power_plan
from .policies import MonotoneBinnedPolicy, TriPredictPolicy
from .projection import dense_gaussian_projection, projection_metadata
from .text_embeddings import load_text_embedding_config, validate_text_embedding_cache
from .utils import array_fingerprint, fingerprint, write_json


class PDCTPQueryTuneError(ValueError):
    """Raised before tune access or when the frozen selection contract fails."""


@dataclass(frozen=True)
class PDCTPQueryTuneConfig:
    raw: Mapping[str, Any]
    config_fingerprint: str
    run_name: str
    query_batch_size: int
    record_distance_decimals: int
    selection_float_decimals: int


_METHOD_ORDER = (
    "fixed",
    "monotone_binned",
    "raw_tri_predict",
    "lid_calibration_only",
    "budget_residual_only",
    "pdctp",
)
_POLICY_KEYS = {
    "fixed": "fixed",
    "monotone_binned": "monotone",
    "raw_tri_predict": "raw_tri",
    "lid_calibration_only": "lid_only",
    "budget_residual_only": "residual_only",
    "pdctp": "full_pdctp",
}
_EXPECTED_CANDIDATE_COUNTS = {
    "fixed": 21,
    "monotone_binned": 15,
    "raw_tri_predict": 5,
    "lid_calibration_only": 20,
    "budget_residual_only": 405,
    "pdctp": 1620,
}
_QUERY_PREFIX = "pdctp-beir-fiqa:query:"
_DOC_PREFIX = "pdctp-beir-fiqa:doc:"


def _exact(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise PDCTPQueryTuneError(
            f"{context} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _hex(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PDCTPQueryTuneError(f"{name} must be a lowercase SHA-256 value")
    return value


def load_pdctp_query_tune_config(
    path: Union[str, Path],
) -> PDCTPQueryTuneConfig:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PDCTPQueryTuneError(f"cannot load query_tune config: {exc}") from exc
    _exact(
        raw,
        {
            "schema",
            "version",
            "run_name",
            "bindings",
            "access",
            "selection_contract",
            "execution",
        },
        "root",
    )
    if raw["schema"] != "pdctp_fiqa_query_tune_gate_v1" or raw["version"] != 1:
        raise PDCTPQueryTuneError("unsupported FiQA query_tune gate schema")
    if not isinstance(raw["run_name"], str) or not raw["run_name"]:
        raise PDCTPQueryTuneError("query_tune run name must be nonempty")
    binding_keys = {
        "protocol_freeze_fingerprint",
        "role_assignments_fingerprint",
        "source_audit_fingerprint",
        "source_audit_sha256",
        "source_archive_sha256",
        "qrels_train_member_sha256",
        "dataset_manifest_fingerprint",
        "embedding_config_fingerprint",
        "embedding_manifest_fingerprint",
        "embedding_audit_fingerprint",
        "query_cal_audit_fingerprint",
        "query_cal_audit_sha256",
        "query_cal_manifest_fingerprint",
        "query_cal_protocol_state_fingerprint",
        "lid_candidate_bundle_fingerprint",
        "residual_candidate_bundle_fingerprint",
        "query_tune_ordered_id_hash",
        "power_plan_fingerprint",
    }
    _exact(raw["bindings"], binding_keys, "bindings")
    for key, value in raw["bindings"].items():
        _hex(value, f"bindings.{key}")

    access = raw["access"]
    _exact(
        access,
        {
            "role",
            "require_complete_frozen_order",
            "allowed_supervision",
            "qrel_member",
            "non_tune_qrel_outcomes_must_not_be_parsed",
            "calibrator_fit_allowed",
            "selection_allowed",
            "blocked_roles",
        },
        "access",
    )
    if access != {
        "role": "query_tune",
        "require_complete_frozen_order": True,
        "allowed_supervision": [
            "exact_original_top_k_identities",
            "realized_embedding_retention",
            "query_tune_positive_qrels",
            "candidate_evidence_recall",
            "final_reranked_evidence_recall",
        ],
        "qrel_member": "qrels_train",
        "non_tune_qrel_outcomes_must_not_be_parsed": True,
        "calibrator_fit_allowed": False,
        "selection_allowed": True,
        "blocked_roles": ["query_cert", "query_latency", "query_test"],
    }:
        raise PDCTPQueryTuneError("query_tune access scope changed")

    contract = raw["selection_contract"]
    _exact(
        contract,
        {
            "fixed_reference",
            "family_selection",
            "eligibility",
            "objective",
            "tie_breaks",
            "method_order",
            "expected_candidate_counts",
            "failure_behavior",
            "freeze_certification_hypotheses_after_success",
            "shuffled_profile_used_for_selection",
        },
        "selection_contract",
    )
    if contract != {
        "fixed_reference": "smallest_budget_meeting_retention_lcb",
        "family_selection": "independent_minimum_work_eligible_candidate_per_method",
        "eligibility": [
            "retention_lower_bound_at_least_target",
            "candidate_evidence_noninferior_to_fixed",
            "final_evidence_noninferior_to_fixed",
        ],
        "objective": "common_coordinate_work",
        "tie_breaks": [
            "lower_mean_budget",
            "canonical_candidate_fingerprint",
        ],
        "method_order": list(_METHOD_ORDER),
        "expected_candidate_counts": _EXPECTED_CANDIDATE_COUNTS,
        "failure_behavior": "terminal_no_retuning_no_budget_expansion",
        "freeze_certification_hypotheses_after_success": True,
        "shuffled_profile_used_for_selection": False,
    }:
        raise PDCTPQueryTuneError("query_tune selection contract changed")

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
            "selection_float_decimals",
            "candidate_budget_matrix_dtype",
            "qrel_filter",
            "approximate_index",
            "llm",
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
        "candidate_budget_matrix_dtype": "int32",
        "qrel_filter": "first_column_role_filter_before_outcome_parse_v1",
        "approximate_index": False,
        "llm": False,
    }
    if any(execution.get(key) != value for key, value in fixed_execution.items()):
        raise PDCTPQueryTuneError("query_tune numerical or scope contract changed")
    batch = execution["query_batch_size"]
    record_decimals = execution["record_distance_decimals"]
    selection_decimals = execution["selection_float_decimals"]
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise PDCTPQueryTuneError("query batch size must be a positive integer")
    if (
        isinstance(record_decimals, bool)
        or not isinstance(record_decimals, int)
        or not 6 <= record_decimals <= 15
    ):
        raise PDCTPQueryTuneError("record distance decimals must be from 6 to 15")
    if (
        isinstance(selection_decimals, bool)
        or not isinstance(selection_decimals, int)
        or not 8 <= selection_decimals <= 15
    ):
        raise PDCTPQueryTuneError("selection float decimals must be from 8 to 15")
    return PDCTPQueryTuneConfig(
        raw=raw,
        config_fingerprint=fingerprint(raw),
        run_name=raw["run_name"],
        query_batch_size=batch,
        record_distance_decimals=record_decimals,
        selection_float_decimals=selection_decimals,
    )


def _role_assignments(roles: Mapping[str, Any]) -> FiveRoleAssignments:
    role_rows = roles.get("roles")
    if not isinstance(role_rows, Mapping) or set(role_rows) != set(FIVE_ROLES):
        raise PDCTPQueryTuneError("role artifact does not contain exactly five roles")
    ids_by_role: Dict[str, Tuple[str, ...]] = {}
    for role in FIVE_ROLES:
        row = role_rows[role]
        if not isinstance(row, Mapping):
            raise PDCTPQueryTuneError(f"invalid role row: {role}")
        ids = tuple(str(value) for value in row.get("ordered_ids", ()))
        if row.get("n") != len(ids) or row.get("ordered_id_hash") != fingerprint(list(ids)):
            raise PDCTPQueryTuneError(f"role identity mismatch: {role}")
        ids_by_role[role] = ids
    all_ids = [query_id for role in FIVE_ROLES for query_id in ids_by_role[role]]
    return FiveRoleAssignments(
        ids_by_role=ids_by_role,
        normalized_text_group_by_id={query_id: query_id for query_id in all_ids},
    )


def _verify_fingerprinted(value: Mapping[str, Any], name: str) -> None:
    body = dict(value)
    stored = body.pop("fingerprint", None)
    if not isinstance(stored, str) or fingerprint(body) != stored:
        raise PDCTPQueryTuneError(f"{name} fingerprint mismatch")


def reconstruct_all_residual_candidates(
    bundle: Mapping[str, Any],
) -> Dict[str, TriBudgetResidualCalibrator]:
    """Validate a compact bundle once and reconstruct every operating point."""
    _verify_fingerprinted(bundle, "residual candidate bundle")
    shared = bundle.get("shared_fit")
    if not isinstance(shared, Mapping):
        raise PDCTPQueryTuneError("residual candidate bundle has no shared fit")
    ordered_ids = shared.get("ordered_ids")
    if (
        not isinstance(ordered_ids, list)
        or shared.get("n") != len(ordered_ids)
        or shared.get("ordered_id_hash") != fingerprint(ordered_ids)
    ):
        raise PDCTPQueryTuneError("residual shared fit identity changed")
    base_by_key: Dict[str, Tuple[Optional[str], TriBudgetResidualCalibrator]] = {}
    for row in bundle.get("base_models", ()):
        _verify_fingerprinted(row, "residual base-model storage row")
        storage_key = str(row.get("storage_key"))
        if storage_key in base_by_key:
            raise PDCTPQueryTuneError("duplicate residual base-model storage key")
        artifact = row.get("artifact")
        if not isinstance(artifact, Mapping):
            raise PDCTPQueryTuneError("residual base-model artifact is missing")
        calibrator = TriBudgetResidualCalibrator.from_serialized(
            _restore_model_artifact(artifact, ordered_ids)
        )
        expected_storage_key = fingerprint(
            {
                "calibrator_fingerprint": calibrator.fingerprint,
                "lid_calibrator_fingerprint": row.get(
                    "lid_calibrator_fingerprint"
                ),
            }
        )
        if storage_key != expected_storage_key:
            raise PDCTPQueryTuneError("residual base-model storage identity changed")
        base_by_key[storage_key] = (
            row.get("lid_calibrator_fingerprint"),
            calibrator,
        )
    result: Dict[str, TriBudgetResidualCalibrator] = {}
    points = list(bundle.get("full_operating_points", ())) + list(
        bundle.get("residual_only_operating_points", ())
    )
    for point in points:
        _verify_fingerprinted(point, "residual operating point")
        point_fp = str(point.get("fingerprint"))
        stored = base_by_key.get(str(point.get("base_model_storage_key")))
        if stored is None or point_fp in result:
            raise PDCTPQueryTuneError("residual operating-point storage changed")
        lid_fp, base = stored
        if (
            base.fingerprint != point.get("base_model_fingerprint")
            or lid_fp != point.get("lid_calibrator_fingerprint")
            or base.raw_policy_fingerprint != point.get("raw_policy_fingerprint")
            or base.training_level != float(point.get("training_level"))
            or base.quantile != float(point.get("quantile"))
            or base.regularization != float(point.get("regularization"))
        ):
            raise PDCTPQueryTuneError("residual operating-point anchor changed")
        effective = base.with_operating_point(
            safety_offset=float(point.get("safety_offset"))
        )
        if effective.fingerprint != point.get("effective_calibrator_fingerprint"):
            raise PDCTPQueryTuneError("residual operating point does not reconstruct")
        result[point_fp] = effective
    counts = bundle.get("counts", {})
    if (
        counts.get("base_models") != len(base_by_key)
        or counts.get("full_pdctp_operating_points")
        != len(bundle.get("full_operating_points", ()))
        or counts.get("residual_only_operating_points")
        != len(bundle.get("residual_only_operating_points", ()))
        or len(result) != len(points)
    ):
        raise PDCTPQueryTuneError("residual candidate counts changed")
    return result


def validate_query_tune_documents(
    config: PDCTPQueryTuneConfig,
    protocol: Mapping[str, Any],
    roles: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    embedding_audit: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    embedding_manifest: Mapping[str, Any],
    query_cal_audit: Mapping[str, Any],
    query_cal_manifest: Mapping[str, Any],
    query_cal_state: Mapping[str, Any],
    query_cal_access: Mapping[str, Any],
    query_cal_projection: Mapping[str, Any],
    lid_bundle: Mapping[str, Any],
    residual_bundle: Mapping[str, Any],
    *,
    source_audit_sha256: str,
    query_cal_audit_sha256: str,
    query_cal_file_identities: Mapping[str, Mapping[str, Any]],
) -> Tuple[FiveRoleAssignments, FiveRoleProtocolGuard]:
    """Validate and replay every upstream artifact before tune outcomes open."""
    for name, value in (
        ("protocol freeze", protocol),
        ("role assignments", roles),
        ("source audit", source_audit),
        ("embedding audit", embedding_audit),
        ("dataset manifest", dataset_manifest),
        ("embedding manifest", embedding_manifest),
        ("query_cal audit", query_cal_audit),
        ("query_cal manifest", query_cal_manifest),
        ("query_cal state", query_cal_state),
        ("query_cal access", query_cal_access),
        ("query_cal projection", query_cal_projection),
        ("LID candidate bundle", lid_bundle),
        ("residual candidate bundle", residual_bundle),
    ):
        _verify_fingerprinted(value, name)

    bindings = config.raw["bindings"]
    observed = {
        "protocol_freeze_fingerprint": protocol.get("fingerprint"),
        "role_assignments_fingerprint": roles.get("fingerprint"),
        "source_audit_fingerprint": source_audit.get("fingerprint"),
        "source_audit_sha256": source_audit_sha256,
        "source_archive_sha256": source_audit.get("source", {})
        .get("archive", {})
        .get("sha256"),
        "qrels_train_member_sha256": source_audit.get("source", {})
        .get("members", {})
        .get("qrels_train", {})
        .get("sha256"),
        "dataset_manifest_fingerprint": dataset_manifest.get("fingerprint"),
        "embedding_config_fingerprint": embedding_audit.get(
            "embedding_config_fingerprint"
        ),
        "embedding_manifest_fingerprint": embedding_manifest.get("fingerprint"),
        "embedding_audit_fingerprint": embedding_audit.get("fingerprint"),
        "query_cal_audit_fingerprint": query_cal_audit.get("fingerprint"),
        "query_cal_audit_sha256": query_cal_audit_sha256,
        "query_cal_manifest_fingerprint": query_cal_manifest.get("fingerprint"),
        "query_cal_protocol_state_fingerprint": query_cal_state.get("fingerprint"),
        "lid_candidate_bundle_fingerprint": lid_bundle.get("fingerprint"),
        "residual_candidate_bundle_fingerprint": residual_bundle.get("fingerprint"),
    }
    for key, expected in bindings.items():
        if key in {"query_tune_ordered_id_hash", "power_plan_fingerprint"}:
            continue
        if observed.get(key) != expected:
            raise PDCTPQueryTuneError(f"frozen upstream binding changed: {key}")

    if (
        protocol.get("decision") != "READY_FOR_DATASET_AND_EMBEDDING_AUDIT_ONLY"
        or protocol.get("config_fingerprint") != query_cal_state.get("config_fingerprint")
        or protocol.get("resolved_roles", {}).get("assignment_fingerprint")
        != roles.get("fingerprint")
        or protocol.get("resolved_inputs", {}).get("source_audit", {}).get(
            "fingerprint"
        )
        != source_audit.get("fingerprint")
    ):
        raise PDCTPQueryTuneError("protocol, role, and query_cal state are not bound")
    if (
        roles.get("all_roles_initially_closed") is not True
        or roles.get("authorizes_outcome_access") is not False
        or source_audit.get("decision") != "GO_TO_PROTOCOL_FREEZE"
    ):
        raise PDCTPQueryTuneError("source or role freeze is not an accepted closed input")
    assignments = _role_assignments(roles)
    tune_ids = assignments.ids_by_role["query_tune"]
    if fingerprint(list(tune_ids)) != bindings["query_tune_ordered_id_hash"]:
        raise PDCTPQueryTuneError("query_tune ordered identity changed")
    if (
        dataset_manifest.get("protocol_freeze_fingerprint") != protocol.get("fingerprint")
        or dataset_manifest.get("role_assignments_fingerprint") != roles.get("fingerprint")
        or embedding_manifest.get("dataset", {}).get("manifest_fingerprint")
        != dataset_manifest.get("fingerprint")
        or embedding_audit.get("embedding_manifest_fingerprint")
        != embedding_manifest.get("fingerprint")
        or embedding_audit.get("decision") != "READY_TO_OPEN_QUERY_CAL"
        or embedding_audit.get("checks", {}).get("all_roles_remained_closed")
        is not True
        or embedding_audit.get("checks", {}).get("qrels_or_relevance_opened")
        is not False
        or dataset_manifest.get("scope_guards", {}).get("contains_qrels") is not False
        or dataset_manifest.get("scope_guards", {}).get("contains_relevance_values")
        is not False
    ):
        raise PDCTPQueryTuneError("dataset and embedding identities are not transitive")

    audit_checks = query_cal_audit.get("checks", {})
    if (
        query_cal_audit.get("decision")
        != "ACCEPT_QUERY_CAL_FITS_READY_TO_IMPLEMENT_QUERY_TUNE"
        or audit_checks.get("candidate_refit_from_records_exact") is not True
        or audit_checks.get("only_query_cal_opened") is not True
        or audit_checks.get("query_tune_accessed") is not False
        or audit_checks.get("query_cert_accessed") is not False
        or audit_checks.get("query_latency_accessed") is not False
        or audit_checks.get("query_test_accessed") is not False
        or audit_checks.get("qrels_or_relevance_accessed") is not False
        or audit_checks.get("no_policy_selected") is not True
    ):
        raise PDCTPQueryTuneError("accepted query_cal audit does not authorize tune")
    audit_file_ids = query_cal_audit.get("file_identities")
    if not isinstance(audit_file_ids, Mapping) or set(audit_file_ids) != set(
        query_cal_file_identities
    ):
        raise PDCTPQueryTuneError("query_cal audit file inventory changed")
    for name, actual in query_cal_file_identities.items():
        if audit_file_ids.get(name) != actual.get("sha256"):
            raise PDCTPQueryTuneError(f"query_cal file identity changed: {name}")

    manifest_checks = query_cal_manifest.get("checks", {})
    if (
        query_cal_manifest.get("decision")
        != "QUERY_CAL_FITS_FROZEN_READY_FOR_QUERY_TUNE"
        or query_cal_manifest.get("protocol_state_fingerprint")
        != query_cal_state.get("fingerprint")
        or query_cal_manifest.get("query_cal_access_fingerprint")
        != query_cal_access.get("fingerprint")
        or query_cal_manifest.get("projection_fingerprint")
        != query_cal_projection.get("fingerprint")
        or query_cal_manifest.get("lid_candidate_bundle_fingerprint")
        != lid_bundle.get("fingerprint")
        or query_cal_manifest.get("residual_candidate_bundle_fingerprint")
        != residual_bundle.get("fingerprint")
        or manifest_checks.get("only_query_cal_opened") is not True
        or manifest_checks.get("query_tune_accessed") is not False
        or manifest_checks.get("selection_performed") is not False
        or query_cal_access.get("roles_opened") != ["query_cal"]
        or query_cal_access.get("qrels_or_relevance_accessed") is not False
        or query_cal_access.get("selection_performed") is not False
    ):
        raise PDCTPQueryTuneError("query_cal manifest or state transition changed")
    for name, identity in query_cal_manifest.get("artifacts", {}).items():
        actual = query_cal_file_identities.get(name)
        if actual != identity:
            raise PDCTPQueryTuneError(f"query_cal manifest artifact changed: {name}")

    lid_artifacts = lid_bundle.get("candidates")
    if (
        lid_bundle.get("fit_role") != "query_cal"
        or lid_bundle.get("selection_performed") is not False
        or not isinstance(lid_artifacts, list)
        or lid_bundle.get("candidate_count") != len(lid_artifacts)
    ):
        raise PDCTPQueryTuneError("LID candidate bundle scope changed")
    for artifact in lid_artifacts:
        PilotLIDCalibrator.from_serialized(artifact)
    counts = residual_bundle.get("counts", {})
    if (
        residual_bundle.get("fit_role") != "query_cal"
        or residual_bundle.get("selection_performed") is not False
        or counts.get("full_pdctp_operating_points")
        != len(residual_bundle.get("full_operating_points", ()))
        or counts.get("residual_only_operating_points")
        != len(residual_bundle.get("residual_only_operating_points", ()))
    ):
        raise PDCTPQueryTuneError("residual candidate bundle scope changed")
    reconstruct_all_residual_candidates(residual_bundle)

    guard = FiveRoleProtocolGuard(assignments, str(protocol["config_fingerprint"]))
    cal_ids = assignments.ids_by_role["query_cal"]
    guard.open_calibration(cal_ids)
    guard.register_fit(
        "lid_calibrator",
        role="query_cal",
        ids=cal_ids,
        artifact_fingerprint=str(lid_bundle["fingerprint"]),
    )
    guard.register_fit(
        "residual_calibrator",
        role="query_cal",
        ids=cal_ids,
        artifact_fingerprint=str(residual_bundle["fingerprint"]),
    )
    if guard.serialize() != query_cal_state:
        raise PDCTPQueryTuneError("post-query_cal guard state is not reconstructable")
    return assignments, guard


def _archive_identity(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _zip_member_identity(archive: zipfile.ZipFile, member: str) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(member, "r") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"path": member, "bytes": size, "sha256": digest.hexdigest()}


def load_query_tune_qrels(
    archive_path: Union[str, Path],
    member_metadata: Mapping[str, Any],
    query_tune_ids: Sequence[str],
    *,
    minimum_relevance: int,
) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, Any]]:
    """Parse outcomes only after a row's first column is known to be tune."""
    path = Path(archive_path)
    tune_ids = tuple(str(value) for value in query_tune_ids)
    tune_set = set(tune_ids)
    if not tune_ids or len(tune_set) != len(tune_ids):
        raise PDCTPQueryTuneError("query_tune qrel IDs must be nonempty and unique")
    member = str(member_metadata.get("path"))
    positives: Dict[str, set[str]] = {query_id: set() for query_id in tune_ids}
    scoped_rows = 0
    scoped_positive_rows = 0
    skipped_before_outcome_parse = 0
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or member not in names:
                raise PDCTPQueryTuneError("FiQA qrel member inventory changed")
            if _zip_member_identity(archive, member) != dict(member_metadata):
                raise PDCTPQueryTuneError("FiQA train qrel member identity changed")
            with io.TextIOWrapper(
                archive.open(member, "r"), encoding="utf-8", newline=""
            ) as source:
                header = source.readline().rstrip("\r\n").split("\t")
                if header[:3] != ["query-id", "corpus-id", "score"]:
                    raise PDCTPQueryTuneError("unexpected FiQA qrel header")
                for line_number, line in enumerate(source, start=2):
                    source_query_id, separator, remainder = line.partition("\t")
                    source_query_id = source_query_id.strip()
                    if not separator or not source_query_id:
                        raise PDCTPQueryTuneError(
                            f"invalid qrel query field at {member}:{line_number}"
                        )
                    query_id = _QUERY_PREFIX + source_query_id
                    if query_id not in tune_set:
                        skipped_before_outcome_parse += 1
                        continue
                    outcome_fields = remainder.rstrip("\r\n").split("\t")
                    if len(outcome_fields) < 2:
                        raise PDCTPQueryTuneError(
                            f"invalid tune qrel outcome at {member}:{line_number}"
                        )
                    doc_id = outcome_fields[0].strip()
                    try:
                        relevance = int(outcome_fields[1])
                    except ValueError as exc:
                        raise PDCTPQueryTuneError(
                            f"invalid tune relevance at {member}:{line_number}"
                        ) from exc
                    if not doc_id:
                        raise PDCTPQueryTuneError(
                            f"empty tune qrel document at {member}:{line_number}"
                        )
                    scoped_rows += 1
                    if relevance >= minimum_relevance:
                        stable_doc_id = _DOC_PREFIX + doc_id
                        if stable_doc_id in positives[query_id]:
                            raise PDCTPQueryTuneError("duplicate positive tune qrel pair")
                        positives[query_id].add(stable_doc_id)
                        scoped_positive_rows += 1
    except zipfile.BadZipFile as exc:
        raise PDCTPQueryTuneError(f"invalid FiQA archive: {path}") from exc
    missing = [query_id for query_id in tune_ids if not positives[query_id]]
    if missing:
        raise PDCTPQueryTuneError(
            f"query_tune contains queries without positive qrels: {missing[:3]}"
        )
    result = {query_id: tuple(sorted(positives[query_id])) for query_id in tune_ids}
    audit: Dict[str, Any] = {
        "name": "pdctp_fiqa_query_tune_qrel_access",
        "schema": "pdctp_fiqa_query_tune_qrel_access_v1",
        "version": 1,
        "role": "query_tune",
        "member": dict(member_metadata),
        "filter": "first_column_role_filter_before_outcome_parse_v1",
        "query_order_hash": fingerprint(list(tune_ids)),
        "queries": len(tune_ids),
        "scoped_rows": scoped_rows,
        "scoped_positive_rows": scoped_positive_rows,
        "non_tune_rows_skipped_before_outcome_parse": skipped_before_outcome_parse,
        "non_tune_qrel_outcomes_parsed": False,
        "minimum_relevance": int(minimum_relevance),
    }
    audit["fingerprint"] = fingerprint(audit)
    return result, audit


def _feature_spec(protocol: Mapping[str, Any]) -> PilotDistanceFeatureSpec:
    raw = protocol["features"]
    return PilotDistanceFeatureSpec(
        lid_boundary=raw["lid_boundary"],
        minimum_count=raw["minimum_count"],
        gap_quantiles=tuple(raw["gap_quantiles"]),
        epsilon=raw["epsilon"],
        duplicate_tolerance=raw["duplicate_tolerance"],
        invalid_fill=raw["invalid_fill"],
        output_decimals=raw["output_decimals"],
        schema=raw["schema"],
    )


def build_query_tune_records(
    protocol: Mapping[str, Any],
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    corpus_embeddings: np.ndarray,
    query_tune_embeddings: np.ndarray,
    qrels: Mapping[str, Sequence[str]],
    *,
    batch_size: int,
    record_distance_decimals: int,
    progress: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compute tune labels and evidence curves with one projected scan/query."""
    retrieval = protocol["retrieval"]
    suite = protocol["candidate_suite"]
    corpus = np.asarray(corpus_embeddings, dtype=np.float64)
    queries = np.asarray(query_tune_embeddings, dtype=np.float64)
    ids = np.asarray(corpus_ids, dtype=str)
    query_ids = tuple(str(value) for value in query_ids)
    if (
        corpus.ndim != 2
        or queries.ndim != 2
        or corpus.shape[1] != queries.shape[1]
        or len(corpus) != len(ids)
        or len(queries) != len(query_ids)
        or len(set(ids.tolist())) != len(ids)
        or set(qrels) != set(query_ids)
    ):
        raise PDCTPQueryTuneError("query_tune arrays, qrels, and stable IDs are not aligned")
    if not np.all(np.isfinite(corpus)) or not np.all(np.isfinite(queries)):
        raise PDCTPQueryTuneError("query_tune arrays must be finite")
    if (
        retrieval["corpus_size"] != len(corpus)
        or retrieval["embedding_dimension"] != corpus.shape[1]
        or retrieval["query_batch_size"] != batch_size
    ):
        raise PDCTPQueryTuneError("query_tune retrieval dimensions changed")
    id_to_row = {doc_id: index for index, doc_id in enumerate(ids.tolist())}
    if any(
        not relevant or any(doc_id not in id_to_row for doc_id in relevant)
        for relevant in qrels.values()
    ):
        raise PDCTPQueryTuneError("query_tune qrels reference invalid corpus IDs")

    spec = _feature_spec(protocol)
    extractor = PilotDistanceFeatureExtractor(spec)
    projection_raw = retrieval["projection"]
    if projection_raw["post_projection_normalize"] is not False:
        raise PDCTPQueryTuneError("post-projection normalization is forbidden")
    matrix = dense_gaussian_projection(
        projection_raw["m_prime"], corpus.shape[1], projection_raw["seed"]
    )
    projected_corpus = corpus @ matrix.T
    projected_queries = queries @ matrix.T
    corpus_norms = np.einsum("ij,ij->i", corpus, corpus)
    projected_norms = np.einsum("ij,ij->i", projected_corpus, projected_corpus)
    lexical_order = np.argsort(ids, kind="stable")
    tie_rank = np.empty(len(ids), dtype=np.int64)
    tie_rank[lexical_order] = np.arange(len(ids), dtype=np.int64)
    oracle_k = max(retrieval["k_gt"], retrieval["s_lid"])
    grid = tuple(int(value) for value in retrieval["m_grid"])
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
            oracle_rows, _ = _stable_top_k_rows(original_distances, tie_rank, oracle_k)
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
                clip_min=suite["lid_output_domain"][0],
                clip_max=suite["lid_output_domain"][1],
                duplicate_tolerance=spec.duplicate_tolerance,
                fallback=suite["lid_fallback"],
            )
            features = extractor.extract(
                PilotDistanceObservation.from_arrays(
                    sorted_original_sq,
                    sorted_projected_sq,
                    pilot_lid=pilot_lid.clipped,
                    pilot_lid_valid=pilot_lid.valid,
                    pilot_lid_failure_reason=pilot_lid.reason,
                    valid_distance_count=pilot_lid.valid_distance_count,
                )
            )
            projected_order = np.lexsort((tie_rank, projected_distances))
            inverse_projected_rank = np.empty(len(ids), dtype=np.int64)
            inverse_projected_rank[projected_order] = np.arange(1, len(ids) + 1)
            gt_rows = oracle_rows[: retrieval["k_gt"]]
            gt_ranks = inverse_projected_rank[gt_rows]
            relevant_ids = tuple(sorted(str(value) for value in qrels[query_id]))
            relevant_rows = np.asarray(
                [id_to_row[doc_id] for doc_id in relevant_ids], dtype=np.int64
            )
            relevant_ranks = inverse_projected_rank[relevant_rows]
            retention_by_budget: Dict[str, float] = {}
            candidate_evidence_by_budget: Dict[str, float] = {}
            final_evidence_by_budget: Dict[str, float] = {}
            final_ids_by_budget: Dict[str, List[str]] = {}
            relevant_set = set(relevant_rows.tolist())
            for budget in grid:
                key = str(budget)
                candidates = projected_order[:budget]
                final_local_rows, _ = _stable_top_k_rows(
                    original_distances[candidates],
                    tie_rank[candidates],
                    min(retrieval["k_ctx"], len(candidates)),
                )
                final_rows = candidates[final_local_rows]
                retention_by_budget[key] = float(np.mean(gt_ranks <= budget))
                candidate_evidence_by_budget[key] = float(
                    np.mean(relevant_ranks <= budget)
                )
                final_evidence_by_budget[key] = float(
                    len(relevant_set.intersection(final_rows.tolist()))
                    / len(relevant_set)
                )
                final_ids_by_budget[key] = ids[final_rows].tolist()
            body: Dict[str, Any] = {
                "schema": "pdctp_fiqa_query_tune_record_v1",
                "query_id": query_id,
                "role": "query_tune",
                "supervision": {
                    "exact_original_top_k_identities": True,
                    "realized_embedding_retention": True,
                    "query_tune_positive_qrels": True,
                    "candidate_evidence_recall": True,
                    "final_reranked_evidence_recall": True,
                    "oracle_lid": False,
                    "non_tune_qrel_outcomes": False,
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
                "exact_original_top_k_doc_ids": ids[gt_rows].tolist(),
                "projected_rank_of_exact_top_k": gt_ranks.tolist(),
                "positive_qrel_doc_ids": list(relevant_ids),
                "projected_rank_of_positive_qrels": relevant_ranks.tolist(),
                "retention_by_budget": retention_by_budget,
                "candidate_evidence_by_budget": candidate_evidence_by_budget,
                "final_evidence_by_budget": final_evidence_by_budget,
                "final_top_k_doc_ids_by_budget": final_ids_by_budget,
                "work": {
                    "projected_scan_count": 1,
                    "projected_distance_count": len(corpus),
                    "original_reference_distance_count": len(corpus),
                    "pilot_original_rerank_distance_count": retrieval["m_pilot"],
                    "pilot_is_prefix_of_same_projected_scan": True,
                    "exact_rerank_uses_same_original_distance_vector": True,
                },
            }
            body["fingerprint"] = fingerprint(body)
            body["_features_obj"] = features
            records.append(body)
        if progress:
            print(f"query_tune exact retrieval: {stop}/{len(queries)}", flush=True)

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
            "projected_query_role": "query_tune",
            "projected_query_shape": list(projected_queries.shape),
            "projected_vectors_persisted": False,
        }
    )
    projection.pop("fingerprint", None)
    projection["fingerprint"] = fingerprint(projection)
    return records, projection


def _canonical(value: float, decimals: int) -> float:
    result = float(np.round(float(value), decimals=decimals))
    if not np.isfinite(result):
        raise PDCTPQueryTuneError("selection statistic must be finite")
    return 0.0 if result == 0.0 else result


def _observation(record: Mapping[str, Any]) -> PDCTPDecisionInput:
    feature = record.get("_features_obj")
    if not isinstance(feature, PilotFeatureVector):
        raise PDCTPQueryTuneError("tune record lacks a reconstructed feature vector")
    pilot = record["pilot"]
    return PDCTPDecisionInput(
        features=feature,
        pilot_lid=float(pilot["lid"]),
        pilot_lid_valid=bool(pilot["lid_valid"]),
    )


def _candidate_evaluation(
    records: Sequence[Mapping[str, Any]],
    budgets: np.ndarray,
    *,
    fixed_candidate_evidence_mean: float,
    fixed_final_evidence_mean: float,
    protocol: Mapping[str, Any],
    decimals: int,
) -> Dict[str, Any]:
    if budgets.shape != (len(records),) or budgets.dtype != np.int32:
        raise PDCTPQueryTuneError("candidate budget vector has the wrong shape or dtype")
    grid = set(int(value) for value in protocol["retrieval"]["m_grid"])
    if any(int(value) not in grid for value in budgets.tolist()):
        raise PDCTPQueryTuneError("candidate emitted a budget outside the frozen grid")
    retentions = np.asarray(
        [
            float(row["retention_by_budget"][str(int(budget))])
            for row, budget in zip(records, budgets)
        ],
        dtype=np.float64,
    )
    candidate_evidence = np.asarray(
        [
            float(row["candidate_evidence_by_budget"][str(int(budget))])
            for row, budget in zip(records, budgets)
        ],
        dtype=np.float64,
    )
    final_evidence = np.asarray(
        [
            float(row["final_evidence_by_budget"][str(int(budget))])
            for row, budget in zip(records, budgets)
        ],
        dtype=np.float64,
    )
    bound = {
        key: (
            value
            if isinstance(value, int)
            else _canonical(float(value), decimals)
        )
        for key, value in empirical_bernstein(
            retentions, float(protocol["selection"]["tune_bound_alpha"])
        ).serialize().items()
    }
    mean_budget = _canonical(float(np.mean(budgets, dtype=np.float64)), decimals)
    candidate_mean = _canonical(float(np.mean(candidate_evidence)), decimals)
    final_mean = _canonical(float(np.mean(final_evidence)), decimals)
    candidate_gap = _canonical(
        candidate_mean - fixed_candidate_evidence_mean, decimals
    )
    final_gap = _canonical(final_mean - fixed_final_evidence_mean, decimals)
    selection = protocol["selection"]
    eligible = bool(
        bound["lower_bound"] >= selection["retention_lower_bound_target"]
        and candidate_gap >= -selection["candidate_evidence_noninferiority"]
        and final_gap >= -selection["final_evidence_noninferiority"]
    )
    retrieval = protocol["retrieval"]
    work = _canonical(
        (retrieval["corpus_size"] + retrieval["embedding_dimension"])
        * retrieval["projection"]["m_prime"]
        + retrieval["embedding_dimension"] * mean_budget,
        decimals,
    )
    try:
        p95 = float(np.quantile(budgets, 0.95, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        p95 = float(np.quantile(budgets, 0.95, interpolation="higher"))
    return {
        "eligible": eligible,
        "retention_tune_bound": bound,
        "candidate_evidence_mean": candidate_mean,
        "candidate_evidence_difference_vs_fixed": candidate_gap,
        "final_evidence_mean": final_mean,
        "final_evidence_difference_vs_fixed": final_gap,
        "budget": {
            "mean": mean_budget,
            "minimum": int(np.min(budgets)),
            "maximum": int(np.max(budgets)),
            "p95_higher": int(p95),
            "terminal_fraction": _canonical(
                float(np.mean(budgets == retrieval["m_grid"][-1])), decimals
            ),
        },
        "common_coordinate_work": work,
    }


def _candidate_identity(
    family: str, policy: PDCTPDecisionPolicy, metadata: Mapping[str, Any]
) -> str:
    return fingerprint(
        {
            "family": family,
            "policy_fingerprint": policy.serialize()["fingerprint"],
            "metadata": dict(metadata),
        }
    )


def _policy_budgets(
    records: Sequence[Mapping[str, Any]], policy: PDCTPDecisionPolicy
) -> np.ndarray:
    return np.asarray(
        [policy.choose(_observation(row)).budget for row in records], dtype=np.int32
    )


def _validate_candidate_inputs(
    protocol: Mapping[str, Any],
    lid_bundle: Mapping[str, Any],
    residual_bundle: Mapping[str, Any],
) -> Tuple[
    Dict[str, PilotLIDCalibrator],
    Dict[str, TriPredictPolicy],
    Dict[str, TriBudgetResidualCalibrator],
]:
    _verify_fingerprinted(lid_bundle, "LID candidate bundle")
    _verify_fingerprinted(residual_bundle, "residual candidate bundle")
    lid_calibrators = [
        PilotLIDCalibrator.from_serialized(value)
        for value in lid_bundle.get("candidates", ())
    ]
    if len(lid_calibrators) != len(protocol["candidate_suite"]["lid_regularization_grid"]):
        raise PDCTPQueryTuneError("LID candidate count differs from frozen protocol")
    if [value.regularization for value in lid_calibrators] != [
        float(value) for value in protocol["candidate_suite"]["lid_regularization_grid"]
    ]:
        raise PDCTPQueryTuneError("LID candidate order or regularization changed")
    raw_policies = [
        TriPredictPolicy.from_serialized(value)
        for value in residual_bundle.get("raw_policies", ())
    ]
    if [value.target for value in raw_policies] != [
        float(value) for value in protocol["candidate_suite"]["raw_tri_threshold_grid"]
    ]:
        raise PDCTPQueryTuneError("Raw Tri candidate order or threshold changed")
    return (
        {value.fingerprint: value for value in lid_calibrators},
        {value.serialize()["fingerprint"]: value for value in raw_policies},
        reconstruct_all_residual_candidates(residual_bundle),
    )


def select_query_tune_policies(
    protocol: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    lid_bundle: Mapping[str, Any],
    residual_bundle: Mapping[str, Any],
    *,
    selection_float_decimals: int,
    expected_candidate_counts: Optional[Mapping[str, int]] = None,
    progress: bool = False,
) -> Dict[str, Any]:
    """Evaluate every frozen candidate and select independently by family."""
    if not records or any(row.get("role") != "query_tune" for row in records):
        raise PDCTPQueryTuneError("selection accepts complete query_tune records only")
    query_ids = [str(row["query_id"]) for row in records]
    if len(set(query_ids)) != len(query_ids):
        raise PDCTPQueryTuneError("query_tune records must have unique IDs")
    lid_by_fp, raw_by_fp, residual_by_point = _validate_candidate_inputs(
        protocol, lid_bundle, residual_bundle
    )
    suite = protocol["candidate_suite"]
    retrieval = protocol["retrieval"]
    minimum_budget = max(retrieval["k_gt"], retrieval["m_pilot"])
    candidates: List[Dict[str, Any]] = []

    def add_candidate(
        family: str,
        policy: PDCTPDecisionPolicy,
        metadata: Mapping[str, Any],
        *,
        budgets: Optional[np.ndarray] = None,
        components: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        vector = _policy_budgets(records, policy) if budgets is None else budgets
        if vector.dtype != np.int32 or vector.shape != (len(records),):
            raise PDCTPQueryTuneError("candidate budget vector is malformed")
        candidates.append(
            {
                "family": family,
                "candidate_fingerprint": _candidate_identity(family, policy, metadata),
                "policy_fingerprint": policy.serialize()["fingerprint"],
                "metadata": dict(metadata),
                "_policy": policy,
                "_components": dict(components or {}),
                "_budgets": vector,
            }
        )

    for budget in suite["fixed_budgets"]:
        policy = FixedPDCTPPolicy(budget, retrieval["m_grid"], minimum_budget)
        add_candidate("fixed", policy, {"budget": int(budget)})

    monotone_records = [
        {
            "lid": float(row["pilot"]["lid"]),
            "lid_valid": bool(row["pilot"]["lid_valid"]),
            "retention_by_budget": row["retention_by_budget"],
        }
        for row in records
    ]
    monotone_candidates: Dict[str, MonotoneBinnedPolicy] = {}
    for n_bins in suite["monotone_binned"]["n_bins_grid"]:
        for target in suite["monotone_binned"]["bin_target_grid"]:
            reference = MonotoneBinnedPolicy.fit(
                monotone_records,
                grid=retrieval["m_grid"],
                n_bins=int(n_bins),
                target=float(target),
                safety_margin=0.0,
                fallback_budget=suite["monotone_binned"]["fallback_budget"],
                feature_version=protocol["features"]["schema"],
            )
            policy = MonotonePDCTPPolicy(reference, minimum_budget=minimum_budget)
            reference_artifact = reference.serialize()
            monotone_candidates[reference_artifact["fingerprint"]] = reference
            add_candidate(
                "monotone_binned",
                policy,
                {"n_bins": int(n_bins), "bin_target": float(target)},
                components={"monotone_reference": reference_artifact},
            )

    raw_baseline_budgets: Dict[str, np.ndarray] = {}
    raw_residual_budgets: Dict[str, np.ndarray] = {}
    for raw_fp, reference in raw_by_fp.items():
        raw_policy = RawTriPredictPDCTPPolicy(reference, minimum_budget=minimum_budget)
        raw_vector = _policy_budgets(records, raw_policy)
        raw_baseline_budgets[raw_fp] = raw_vector
        residual_vector = np.asarray(
            [
                reference.choose(
                    observation.pilot_lid,
                    observation.pilot_lid_valid and observation.features.valid,
                ).budget
                for observation in (_observation(row) for row in records)
            ],
            dtype=np.int32,
        )
        raw_residual_budgets[raw_fp] = residual_vector
        add_candidate(
            "raw_tri_predict",
            raw_policy,
            {"raw_tri_threshold": reference.target},
            budgets=raw_vector,
            components={"raw_reference": reference.serialize()},
        )

    calibrated_anchor_budgets: Dict[Tuple[str, str], np.ndarray] = {}
    for raw_fp, reference in raw_by_fp.items():
        for lid_fp, lid_calibrator in lid_by_fp.items():
            policy = CalibratedTriPredictPolicy(
                mode="lid_only",
                raw_reference=reference,
                minimum_budget=minimum_budget,
                lid_calibrator=lid_calibrator,
            )
            vector = _policy_budgets(records, policy)
            calibrated_anchor_budgets[(lid_fp, raw_fp)] = vector
            add_candidate(
                "lid_calibration_only",
                policy,
                {
                    "raw_tri_threshold": reference.target,
                    "raw_policy_fingerprint": raw_fp,
                    "lid_regularization": lid_calibrator.regularization,
                    "lid_calibrator_fingerprint": lid_fp,
                },
                budgets=vector,
                components={
                    "raw_reference": reference.serialize(),
                    "lid_calibrator": lid_calibrator.serialize(),
                },
            )

    def residual_vector(
        anchor: np.ndarray, calibrator: TriBudgetResidualCalibrator
    ) -> np.ndarray:
        output = np.empty(len(records), dtype=np.int32)
        terminal = retrieval["m_grid"][-1]
        for index, (row, raw_budget) in enumerate(zip(records, anchor)):
            observation = _observation(row)
            if int(raw_budget) == terminal and (
                not observation.features.valid
                or not observation.pilot_lid_valid
            ):
                output[index] = terminal
            else:
                output[index] = calibrator.choose_budget(
                    int(raw_budget), observation.features
                ).budget
        return output

    residual_points = residual_bundle.get("residual_only_operating_points", ())
    for index, point in enumerate(residual_points, start=1):
        raw_fp = str(point["raw_policy_fingerprint"])
        reference = raw_by_fp.get(raw_fp)
        if reference is None or point.get("lid_source") != "raw_pilot_lid":
            raise PDCTPQueryTuneError("residual-only point has an invalid Raw Tri anchor")
        residual = residual_by_point[str(point["fingerprint"])]
        policy = CalibratedTriPredictPolicy(
            mode="residual_only",
            raw_reference=reference,
            minimum_budget=minimum_budget,
            residual_calibrator=residual,
        )
        add_candidate(
            "budget_residual_only",
            policy,
            {
                "query_cal_operating_point_fingerprint": point["fingerprint"],
                **{key: value for key, value in point.items() if key != "fingerprint"},
            },
            budgets=residual_vector(raw_residual_budgets[raw_fp], residual),
            components={
                "raw_reference": reference.serialize(),
                "residual_calibrator": residual.serialize(),
            },
        )
        if progress and (index == len(residual_points) or index % 100 == 0):
            print(
                f"query_tune residual-only candidates: {index}/{len(residual_points)}",
                flush=True,
            )

    full_points = residual_bundle.get("full_operating_points", ())
    for index, point in enumerate(full_points, start=1):
        raw_fp = str(point["raw_policy_fingerprint"])
        lid_fp = str(point["lid_calibrator_fingerprint"])
        reference = raw_by_fp.get(raw_fp)
        lid_calibrator = lid_by_fp.get(lid_fp)
        if (
            reference is None
            or lid_calibrator is None
            or point.get("lid_source") != "calibrated_pilot_lid"
        ):
            raise PDCTPQueryTuneError("full PDCTP point has an invalid calibration anchor")
        residual = residual_by_point[str(point["fingerprint"])]
        policy = CalibratedTriPredictPolicy(
            mode="full",
            raw_reference=reference,
            minimum_budget=minimum_budget,
            lid_calibrator=lid_calibrator,
            residual_calibrator=residual,
        )
        add_candidate(
            "pdctp",
            policy,
            {
                "query_cal_operating_point_fingerprint": point["fingerprint"],
                **{key: value for key, value in point.items() if key != "fingerprint"},
            },
            budgets=residual_vector(
                calibrated_anchor_budgets[(lid_fp, raw_fp)], residual
            ),
            components={
                "raw_reference": reference.serialize(),
                "lid_calibrator": lid_calibrator.serialize(),
                "residual_calibrator": residual.serialize(),
            },
        )
        if progress and (index == len(full_points) or index % 200 == 0):
            print(
                f"query_tune full PDCTP candidates: {index}/{len(full_points)}",
                flush=True,
            )

    actual_counts = {
        family: sum(row["family"] == family for row in candidates)
        for family in _METHOD_ORDER
    }
    expected = (
        dict(_EXPECTED_CANDIDATE_COUNTS)
        if expected_candidate_counts is None
        else {str(key): int(value) for key, value in expected_candidate_counts.items()}
    )
    if actual_counts != expected:
        raise PDCTPQueryTuneError(
            f"candidate enumeration changed: expected={expected}, actual={actual_counts}"
        )
    if len({row["candidate_fingerprint"] for row in candidates}) != len(candidates):
        raise PDCTPQueryTuneError("candidate fingerprints are not unique")
    budget_matrix = np.vstack([row["_budgets"] for row in candidates]).astype(
        np.int32, copy=False
    )

    fixed_rows = [row for row in candidates if row["family"] == "fixed"]
    fixed_reference: Optional[Dict[str, Any]] = None
    for row in fixed_rows:
        values = [
            float(record["retention_by_budget"][str(int(budget))])
            for record, budget in zip(records, row["_budgets"])
        ]
        lower = _canonical(
            empirical_bernstein(
                values, float(protocol["selection"]["tune_bound_alpha"])
            ).lower_bound,
            selection_float_decimals,
        )
        if lower >= protocol["selection"]["retention_lower_bound_target"]:
            fixed_reference = row
            break
    if fixed_reference is None:
        raise PDCTPQueryTuneError(
            "full-corpus fixed reference unexpectedly missed the tune retention target"
        )
    fixed_budgets = fixed_reference["_budgets"]
    fixed_candidate_evidence_mean = _canonical(
        float(
            np.mean(
                [
                    record["candidate_evidence_by_budget"][str(int(budget))]
                    for record, budget in zip(records, fixed_budgets)
                ]
            )
        ),
        selection_float_decimals,
    )
    fixed_final_evidence_mean = _canonical(
        float(
            np.mean(
                [
                    record["final_evidence_by_budget"][str(int(budget))]
                    for record, budget in zip(records, fixed_budgets)
                ]
            )
        ),
        selection_float_decimals,
    )

    public_candidates: List[Dict[str, Any]] = []
    for index, row in enumerate(candidates):
        evaluation = _candidate_evaluation(
            records,
            row["_budgets"],
            fixed_candidate_evidence_mean=fixed_candidate_evidence_mean,
            fixed_final_evidence_mean=fixed_final_evidence_mean,
            protocol=protocol,
            decimals=selection_float_decimals,
        )
        row["_evaluation"] = evaluation
        public: Dict[str, Any] = {
            "family": row["family"],
            "candidate_fingerprint": row["candidate_fingerprint"],
            "policy_fingerprint": row["policy_fingerprint"],
            "metadata": row["metadata"],
            "budget_matrix_row": index,
            "budget_vector_fingerprint": array_fingerprint(row["_budgets"]),
            "evaluation": evaluation,
        }
        public["fingerprint"] = fingerprint(public)
        public_candidates.append(public)

    selected: Dict[str, Dict[str, Any]] = {"fixed": fixed_reference}
    missing_families: List[str] = []
    for family in _METHOD_ORDER[1:]:
        eligible = [
            row
            for row in candidates
            if row["family"] == family and row["_evaluation"]["eligible"]
        ]
        if not eligible:
            missing_families.append(family)
            continue
        selected[family] = min(
            eligible,
            key=lambda row: (
                row["_evaluation"]["common_coordinate_work"],
                row["_evaluation"]["budget"]["mean"],
                row["candidate_fingerprint"],
            ),
        )

    outcomes: Dict[str, Any] = {
        "name": "pdctp_fiqa_query_tune_candidate_outcomes",
        "schema": "pdctp_fiqa_query_tune_candidate_outcomes_v1",
        "version": 1,
        "role": "query_tune",
        "query_order_hash": fingerprint(query_ids),
        "query_count": len(query_ids),
        "candidate_order": "method_order_then_frozen_hyperparameter_order",
        "method_order": list(_METHOD_ORDER),
        "counts": actual_counts,
        "budget_matrix": {
            "shape": list(budget_matrix.shape),
            "dtype": str(budget_matrix.dtype),
            "array_fingerprint": array_fingerprint(budget_matrix),
        },
        "fixed_reference": {
            "candidate_fingerprint": fixed_reference["candidate_fingerprint"],
            "budget": fixed_reference["metadata"]["budget"],
            "candidate_evidence_mean": fixed_candidate_evidence_mean,
            "final_evidence_mean": fixed_final_evidence_mean,
        },
        "candidates": public_candidates,
    }
    outcomes["fingerprint"] = fingerprint(outcomes)

    if missing_families:
        selection: Dict[str, Any] = {
            "name": "pdctp_fiqa_query_tune_selection",
            "schema": "pdctp_fiqa_query_tune_selection_v1",
            "version": 1,
            "role": "query_tune",
            "query_order_hash": fingerprint(query_ids),
            "candidate_outcomes_fingerprint": outcomes["fingerprint"],
            "selection_contract": protocol["selection"],
            "family_selection_contract": {
                "fixed_reference": "smallest_budget_meeting_retention_lcb",
                "family_selection": "independent_minimum_work_eligible_candidate_per_method",
                "method_order": list(_METHOD_ORDER),
                "tie_breaks": [
                    "lower_mean_budget",
                    "canonical_candidate_fingerprint",
                ],
                "failure_behavior": "terminal_no_retuning_no_budget_expansion",
            },
            "selected_by_method": None,
            "missing_eligible_method_families": missing_families,
            "calibrator_refit": False,
            "certification_used": False,
            "decision": "TERMINAL_QUERY_TUNE_FAILURE_NO_RETUNING",
        }
        selection["fingerprint"] = fingerprint(selection)
        return {
            "success": False,
            "budget_matrix": budget_matrix,
            "candidate_outcomes": outcomes,
            "selection": selection,
            "selected": {},
        }

    policies = {
        _POLICY_KEYS[family]: selected[family]["_policy"] for family in _METHOD_ORDER
    }
    validate_policy_suite(policies)
    policy_artifacts = {name: value.serialize() for name, value in policies.items()}
    policies_artifact: Dict[str, Any] = {
        "name": "pdctp_fiqa_frozen_policies",
        "schema": "pdctp_fiqa_frozen_policies_v1",
        "version": 1,
        "policies": policy_artifacts,
    }
    policies_artifact["fingerprint"] = fingerprint(policies_artifact)

    components_by_fp: Dict[str, Dict[str, Any]] = {}
    method_components: Dict[str, Dict[str, str]] = {}
    for family in _METHOD_ORDER:
        references: Dict[str, str] = {}
        for kind, artifact in selected[family]["_components"].items():
            artifact_fp = str(artifact["fingerprint"])
            existing = components_by_fp.get(artifact_fp)
            component = {"kind": kind, "artifact": artifact}
            if existing is not None and existing["artifact"] != artifact:
                raise PDCTPQueryTuneError("selected component fingerprint collision")
            if existing is None:
                components_by_fp[artifact_fp] = component
            references[kind] = artifact_fp
        method_components[_POLICY_KEYS[family]] = references
    registry: Dict[str, Any] = {
        "name": "pdctp_fiqa_selected_policy_component_registry",
        "schema": "pdctp_fiqa_selected_policy_components_v1",
        "version": 1,
        "components": [
            {"fingerprint": component_fp, **components_by_fp[component_fp]}
            for component_fp in sorted(components_by_fp)
        ],
    }
    registry["fingerprint"] = fingerprint(registry)
    suite_artifact: Dict[str, Any] = {
        "name": "pdctp_fiqa_frozen_policy_suite",
        "schema": "pdctp_fiqa_frozen_policy_suite_v1",
        "version": 1,
        "method_order": list(_METHOD_ORDER),
        "policy_key_mapping": dict(_POLICY_KEYS),
        "policies_fingerprint": policies_artifact["fingerprint"],
        "component_registry_fingerprint": registry["fingerprint"],
        "selected_methods": {
            _POLICY_KEYS[family]: {
                "family": family,
                "candidate_fingerprint": selected[family]["candidate_fingerprint"],
                "policy_fingerprint": selected[family]["policy_fingerprint"],
                "component_fingerprints": method_components[_POLICY_KEYS[family]],
            }
            for family in _METHOD_ORDER
        },
    }
    suite_artifact["fingerprint"] = fingerprint(suite_artifact)
    selection = {
        "name": "pdctp_fiqa_query_tune_selection",
        "schema": "pdctp_fiqa_query_tune_selection_v1",
        "version": 1,
        "role": "query_tune",
        "query_order_hash": fingerprint(query_ids),
        "candidate_outcomes_fingerprint": outcomes["fingerprint"],
        "selection_contract": protocol["selection"],
        "family_selection_contract": {
            "fixed_reference": "smallest_budget_meeting_retention_lcb",
            "family_selection": "independent_minimum_work_eligible_candidate_per_method",
            "method_order": list(_METHOD_ORDER),
            "tie_breaks": [
                "lower_mean_budget",
                "canonical_candidate_fingerprint",
            ],
            "failure_behavior": "terminal_no_retuning_no_budget_expansion",
        },
        "selected_by_method": {
            family: {
                "candidate_fingerprint": selected[family]["candidate_fingerprint"],
                "policy_fingerprint": selected[family]["policy_fingerprint"],
                "metadata": selected[family]["metadata"],
                "evaluation": selected[family]["_evaluation"],
            }
            for family in _METHOD_ORDER
        },
        "missing_eligible_method_families": [],
        "frozen_policy_suite_fingerprint": suite_artifact["fingerprint"],
        "calibrator_refit": False,
        "certification_used": False,
        "decision": "QUERY_TUNE_SELECTION_FROZEN_READY_FOR_CERT_IMPLEMENTATION",
    }
    selection["fingerprint"] = fingerprint(selection)
    return {
        "success": True,
        "budget_matrix": budget_matrix,
        "candidate_outcomes": outcomes,
        "selection": selection,
        "policies": policies_artifact,
        "component_registry": registry,
        "policy_suite": suite_artifact,
        "selected": selected,
    }


def reconstruct_frozen_policy_suite(
    policies_artifact: Mapping[str, Any],
    registry: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> Dict[str, PDCTPDecisionPolicy]:
    """Rebuild every frozen method from the compact selected component registry."""
    for name, value in (
        ("policies", policies_artifact),
        ("component registry", registry),
        ("policy suite", suite),
    ):
        _verify_fingerprinted(value, name)
    if (
        suite.get("policies_fingerprint") != policies_artifact.get("fingerprint")
        or suite.get("component_registry_fingerprint") != registry.get("fingerprint")
        or suite.get("policy_key_mapping") != _POLICY_KEYS
        or suite.get("method_order") != list(_METHOD_ORDER)
    ):
        raise PDCTPQueryTuneError("frozen policy suite bindings changed")
    components: Dict[str, Mapping[str, Any]] = {}
    for row in registry.get("components", ()):
        artifact = row.get("artifact")
        component_fp = row.get("fingerprint")
        if not isinstance(artifact, Mapping) or artifact.get("fingerprint") != component_fp:
            raise PDCTPQueryTuneError("selected policy component identity changed")
        _verify_fingerprinted(artifact, "selected policy component")
        components[str(component_fp)] = artifact
    serialized = policies_artifact.get("policies")
    selected_methods = suite.get("selected_methods")
    if not isinstance(serialized, Mapping) or not isinstance(selected_methods, Mapping):
        raise PDCTPQueryTuneError("frozen policy suite is incomplete")

    def component(method: str, kind: str) -> Mapping[str, Any]:
        fingerprint_value = selected_methods[method]["component_fingerprints"].get(kind)
        if fingerprint_value not in components:
            raise PDCTPQueryTuneError(f"missing {kind} component for {method}")
        return components[str(fingerprint_value)]

    fixed_raw = serialized["fixed"]
    fixed = FixedPDCTPPolicy(
        fixed_raw["budget"], fixed_raw["grid"], fixed_raw["minimum_budget"]
    )
    monotone_reference = MonotoneBinnedPolicy.from_serialized(
        component("monotone", "monotone_reference")
    )
    monotone = MonotonePDCTPPolicy(
        monotone_reference, minimum_budget=serialized["monotone"]["minimum_budget"]
    )
    raw_reference = TriPredictPolicy.from_serialized(
        component("raw_tri", "raw_reference")
    )
    raw = RawTriPredictPDCTPPolicy(
        raw_reference, minimum_budget=serialized["raw_tri"]["minimum_budget"]
    )

    def calibrated(method: str, mode: str) -> CalibratedTriPredictPolicy:
        raw_component = TriPredictPolicy.from_serialized(
            component(method, "raw_reference")
        )
        lid_component = (
            PilotLIDCalibrator.from_serialized(component(method, "lid_calibrator"))
            if mode in {"lid_only", "full"}
            else None
        )
        residual_component = (
            TriBudgetResidualCalibrator.from_serialized(
                component(method, "residual_calibrator")
            )
            if mode in {"residual_only", "full"}
            else None
        )
        return CalibratedTriPredictPolicy(
            mode=mode,
            raw_reference=raw_component,
            minimum_budget=serialized[method]["minimum_budget"],
            lid_calibrator=lid_component,
            residual_calibrator=residual_component,
        )

    result: Dict[str, PDCTPDecisionPolicy] = {
        "fixed": fixed,
        "monotone": monotone,
        "raw_tri": raw,
        "lid_only": calibrated("lid_only", "lid_only"),
        "residual_only": calibrated("residual_only", "residual_only"),
        "full_pdctp": calibrated("full_pdctp", "full"),
    }
    validate_policy_suite(result)
    if {name: value.serialize() for name, value in result.items()} != serialized:
        raise PDCTPQueryTuneError("frozen policy suite does not reconstruct exactly")
    return result


def _selected_policy_records(
    records: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    outcomes = result["candidate_outcomes"]
    candidate_rows = {
        row["candidate_fingerprint"]: row for row in outcomes["candidates"]
    }
    matrix = result["budget_matrix"]
    output: List[Dict[str, Any]] = []
    for family in _METHOD_ORDER:
        selected = result["selection"]["selected_by_method"][family]
        candidate = candidate_rows[selected["candidate_fingerprint"]]
        budgets = matrix[int(candidate["budget_matrix_row"])]
        for record, budget in zip(records, budgets):
            key = str(int(budget))
            row: Dict[str, Any] = {
                "schema": "pdctp_fiqa_query_tune_selected_policy_record_v1",
                "query_id": record["query_id"],
                "role": "query_tune",
                "method": family,
                "policy_fingerprint": selected["policy_fingerprint"],
                "chosen_m": int(budget),
                "embedding_retention": record["retention_by_budget"][key],
                "candidate_evidence_recall": record["candidate_evidence_by_budget"][key],
                "final_evidence_recall": record["final_evidence_by_budget"][key],
                "used_for_selection": True,
                "used_for_certification": False,
            }
            row["fingerprint"] = fingerprint(row)
            output.append(row)
    return output


def _shuffled_tune_diagnostic(
    records: Sequence[Mapping[str, Any]],
    selected_budgets: np.ndarray,
    *,
    seed: int,
    policy_fingerprint: str,
    decimals: int,
) -> Dict[str, Any]:
    donor_indices = np.random.default_rng(seed).permutation(len(records))
    shuffled_rows = []
    for target, donor_index in zip(records, donor_indices):
        donor_budget = int(selected_budgets[int(donor_index)])
        key = str(donor_budget)
        shuffled_rows.append(
            {
                "query_id": target["query_id"],
                "donor_query_id": records[int(donor_index)]["query_id"],
                "role": "query_tune",
                "chosen_m": donor_budget,
                "embedding_retention": target["retention_by_budget"][key],
                "candidate_evidence_recall": target["candidate_evidence_by_budget"][key],
                "final_evidence_recall": target["final_evidence_by_budget"][key],
            }
        )

    def metrics(budgets: np.ndarray) -> Dict[str, float]:
        values: Dict[str, List[float]] = {
            "embedding_retention": [],
            "candidate_evidence_recall": [],
            "final_evidence_recall": [],
        }
        for record, budget in zip(records, budgets):
            key = str(int(budget))
            values["embedding_retention"].append(record["retention_by_budget"][key])
            values["candidate_evidence_recall"].append(
                record["candidate_evidence_by_budget"][key]
            )
            values["final_evidence_recall"].append(
                record["final_evidence_by_budget"][key]
            )
        return {
            "mean_budget": _canonical(float(np.mean(budgets)), decimals),
            **{
                f"mean_{name}": _canonical(float(np.mean(vector)), decimals)
                for name, vector in values.items()
            },
        }

    shuffled_budgets = selected_budgets[donor_indices]
    observed = metrics(selected_budgets)
    shuffled = metrics(shuffled_budgets)
    body: Dict[str, Any] = {
        "name": "pdctp_fiqa_shuffled_pilot_profile_tune_diagnostic",
        "schema": "pdctp_fiqa_shuffled_pilot_profile_tune_diagnostic_v1",
        "version": 1,
        "role": "query_tune",
        "seed": int(seed),
        "policy_fingerprint": policy_fingerprint,
        "used_for_fit": False,
        "used_for_selection": False,
        "used_for_certification": False,
        "n": len(records),
        "observed": observed,
        "shuffled": shuffled,
        "observed_minus_shuffled": {
            metric: _canonical(
                observed[f"mean_{metric}"] - shuffled[f"mean_{metric}"], decimals
            )
            for metric in (
                "embedding_retention",
                "candidate_evidence_recall",
                "final_evidence_recall",
            )
        },
        "records": shuffled_rows,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def _hypotheses_artifact(
    protocol: Mapping[str, Any], power_plan: Mapping[str, Any]
) -> Dict[str, Any]:
    certification = protocol["certification"]
    names = [str(row["name"]) for row in certification["hypotheses"]]
    body: Dict[str, Any] = {
        "name": "pdctp_fiqa_frozen_certification_hypotheses",
        "schema": "pdctp_fiqa_frozen_certification_hypotheses_v1",
        "version": 1,
        "family_wise_method": certification["family_wise_method"],
        "family_wise_alpha": certification["family_wise_alpha"],
        "alpha_allocation": bonferroni_allocation(
            names, certification["family_wise_alpha"]
        ),
        "hypotheses": certification["hypotheses"],
        "required_query_count": certification["required_query_count"],
        "power_plan_fingerprint": power_plan["fingerprint"],
        "frozen_before_query_cert": True,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
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


def run_pdctp_fiqa_query_tune(
    config: PDCTPQueryTuneConfig,
    real_protocol_config: PDCTPRealProtocolConfig,
    protocol_freeze_path: Union[str, Path],
    role_assignments_path: Union[str, Path],
    source_audit_path: Union[str, Path],
    source_archive_path: Union[str, Path],
    embedding_audit_path: Union[str, Path],
    embedding_config_path: Union[str, Path],
    prepared_dir: Union[str, Path],
    cache_dir: Union[str, Path],
    query_cal_audit_path: Union[str, Path],
    query_cal_run_dir: Union[str, Path],
    power_plan_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    progress: bool = True,
) -> Dict[str, Path]:
    """Validate fits, open tune once, select/freeze, and keep later roles closed."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite query_tune run: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = _load_fingerprinted(Path(protocol_freeze_path), "protocol freeze")
    roles = _load_fingerprinted(Path(role_assignments_path), "role assignments")
    source_path = Path(source_audit_path)
    source_audit = _load_fingerprinted(source_path, "source audit")
    embedding_audit = _load_fingerprinted(
        Path(embedding_audit_path), "embedding audit"
    )
    embedding_config = load_text_embedding_config(Path(embedding_config_path))
    validated_cache = validate_text_embedding_cache(
        embedding_config, Path(prepared_dir), Path(cache_dir)
    )
    dataset_manifest = validated_cache["dataset_manifest"]
    embedding_manifest = validated_cache["embedding_manifest"]
    cal_audit_path = Path(query_cal_audit_path)
    cal_audit = _load_fingerprinted(cal_audit_path, "query_cal independent audit")
    cal_run = Path(query_cal_run_dir)
    cal_files = {
        name: cal_run / name
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
    missing_cal = [name for name, path in cal_files.items() if not path.is_file()]
    if missing_cal:
        raise PDCTPQueryTuneError(
            f"query_cal run is incomplete: missing={missing_cal}"
        )
    cal_identities = {name: _file_identity(path) for name, path in cal_files.items()}
    cal_manifest = _load_fingerprinted(cal_files["manifest.json"], "query_cal manifest")
    cal_state = _load_fingerprinted(
        cal_files["protocol_state_after_query_cal.json"], "query_cal state"
    )
    cal_access = _load_fingerprinted(
        cal_files["query_cal_access.json"], "query_cal access"
    )
    cal_projection = _load_fingerprinted(
        cal_files["projection.json"], "query_cal projection"
    )
    lid_bundle = _load_fingerprinted(
        cal_files["lid_calibrator_candidates.json"], "LID candidate bundle"
    )
    residual_bundle = _load_fingerprinted(
        cal_files["residual_calibrator_candidates.json"],
        "residual candidate bundle",
    )
    power_plan = _load_fingerprinted(Path(power_plan_path), "power plan")

    if real_protocol_config.config_fingerprint != protocol.get("config_fingerprint"):
        raise PDCTPQueryTuneError("real protocol config differs from protocol freeze")
    if real_protocol_config.raw != protocol.get("protocol"):
        raise PDCTPQueryTuneError("embedded real protocol differs from checked config")
    if (
        embedding_config.config_fingerprint
        != config.raw["bindings"]["embedding_config_fingerprint"]
    ):
        raise PDCTPQueryTuneError("embedding config fingerprint changed")
    if power_plan.get("fingerprint") != config.raw["bindings"]["power_plan_fingerprint"]:
        raise PDCTPQueryTuneError("power plan fingerprint changed")
    expected_power = make_power_plan(
        real_protocol_config.raw["certification"]["hypotheses"],
        total_alpha=real_protocol_config.raw["certification"]["family_wise_alpha"],
    )
    if expected_power != power_plan:
        raise PDCTPQueryTuneError("power plan no longer matches frozen hypotheses")
    assignments, guard = validate_query_tune_documents(
        config,
        protocol,
        roles,
        source_audit,
        embedding_audit,
        dataset_manifest,
        embedding_manifest,
        cal_audit,
        cal_manifest,
        cal_state,
        cal_access,
        cal_projection,
        lid_bundle,
        residual_bundle,
        source_audit_sha256=_file_identity(source_path)["sha256"],
        query_cal_audit_sha256=_file_identity(cal_audit_path)["sha256"],
        query_cal_file_identities=cal_identities,
    )

    source_archive = Path(source_archive_path)
    if not source_archive.is_file():
        raise FileNotFoundError(f"FiQA archive does not exist: {source_archive}")
    actual_archive = _archive_identity(source_archive)
    expected_archive = source_audit.get("source", {}).get("archive", {})
    if actual_archive != {
        "bytes": expected_archive.get("bytes"),
        "sha256": expected_archive.get("sha256"),
    }:
        raise PDCTPQueryTuneError("FiQA source archive identity changed")

    # Only after every portable upstream identity and the complete archive hash
    # pass do we mint the tune token and parse tune-scoped outcomes.
    tune_ids = assignments.ids_by_role["query_tune"]
    tune_token = guard.open_tune_selection(tune_ids)
    member_metadata = source_audit["source"]["members"][
        config.raw["access"]["qrel_member"]
    ]
    qrels, qrel_access = load_query_tune_qrels(
        source_archive,
        member_metadata,
        tune_ids,
        minimum_relevance=real_protocol_config.raw["dataset"]["minimum_relevance"],
    )

    cache = Path(cache_dir)
    corpus_ids = json.loads((cache / "corpus_ids.json").read_text(encoding="utf-8"))
    all_query_ids = json.loads(
        (cache / "query_ids.json").read_text(encoding="utf-8")
    )
    if not isinstance(corpus_ids, list) or not isinstance(all_query_ids, list):
        raise PDCTPQueryTuneError("embedding stable-ID artifacts are invalid")
    query_row = {query_id: index for index, query_id in enumerate(all_query_ids)}
    if len(query_row) != len(all_query_ids) or any(
        query_id not in query_row for query_id in tune_ids
    ):
        raise PDCTPQueryTuneError("query_tune IDs do not map uniquely into the cache")
    tune_rows = np.asarray([query_row[query_id] for query_id in tune_ids], dtype=np.int64)
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
    query_tune_embeddings = np.asarray(query_embeddings[tune_rows], dtype=np.float64)
    records, projection = build_query_tune_records(
        real_protocol_config.raw,
        tune_ids,
        corpus_ids,
        corpus_embeddings,
        query_tune_embeddings,
        qrels,
        batch_size=config.query_batch_size,
        record_distance_decimals=config.record_distance_decimals,
        progress=progress,
    )
    projection_shared_keys = (
        "schema",
        "family",
        "dimension",
        "m_prime",
        "seed",
        "entry_mean",
        "entry_variance",
        "numpy_scale",
        "input_l2_normalized",
        "post_projection_normalized",
        "embedding_model",
        "corpus_hash",
        "matrix",
        "projected_corpus_shape",
        "projected_vectors_persisted",
    )
    if any(projection.get(key) != cal_projection.get(key) for key in projection_shared_keys):
        raise PDCTPQueryTuneError("query_tune did not reuse the frozen query_cal projection")
    selection_result = select_query_tune_policies(
        real_protocol_config.raw,
        records,
        lid_bundle,
        residual_bundle,
        selection_float_decimals=config.selection_float_decimals,
        expected_candidate_counts=config.raw["selection_contract"][
            "expected_candidate_counts"
        ],
        progress=progress,
    )

    success = bool(selection_result["success"])
    optional_artifacts: Dict[str, Any] = {}
    selected_records: List[Dict[str, Any]] = []
    # Success and failure are both terminal selection outcomes. Freezing the
    # result fingerprint prevents a second tune opening; only success also
    # freezes hypotheses and can become a certification-runner prerequisite.
    guard.freeze_selection(tune_token, selection_result["selection"]["fingerprint"])
    if success:
        reconstructed = reconstruct_frozen_policy_suite(
            selection_result["policies"],
            selection_result["component_registry"],
            selection_result["policy_suite"],
        )
        if set(reconstructed) != set(_POLICY_KEYS.values()):
            raise AssertionError("frozen policy suite reconstruction is incomplete")
        hypotheses = _hypotheses_artifact(real_protocol_config.raw, power_plan)
        guard.freeze_hypotheses(hypotheses["fingerprint"])
        selected_records = _selected_policy_records(records, selection_result)
        pdctp_selected = selection_result["selection"]["selected_by_method"]["pdctp"]
        candidate_by_fp = {
            row["candidate_fingerprint"]: row
            for row in selection_result["candidate_outcomes"]["candidates"]
        }
        pdctp_candidate = candidate_by_fp[pdctp_selected["candidate_fingerprint"]]
        pdctp_budgets = selection_result["budget_matrix"][
            int(pdctp_candidate["budget_matrix_row"])
        ]
        shuffled = _shuffled_tune_diagnostic(
            records,
            pdctp_budgets,
            seed=real_protocol_config.raw["seeds"]["shuffled_profile"],
            policy_fingerprint=pdctp_selected["policy_fingerprint"],
            decimals=config.selection_float_decimals,
        )
        optional_artifacts = {
            "policies.json": selection_result["policies"],
            "selected_policy_components.json": selection_result[
                "component_registry"
            ],
            "frozen_policy_suite.json": selection_result["policy_suite"],
            "shuffled_tune_diagnostic.json": shuffled,
            "hypotheses.json": hypotheses,
        }

    access: Dict[str, Any] = {
        "name": "pdctp_fiqa_query_tune_access",
        "schema": "pdctp_fiqa_query_tune_access_v1",
        "version": 1,
        "token": tune_token.serialize(),
        "allowed_supervision": config.raw["access"]["allowed_supervision"],
        "roles_with_outcomes_opened": ["query_cal", "query_tune"],
        "roles_remaining_closed": config.raw["access"]["blocked_roles"],
        "query_tune_qrels_accessed": True,
        "non_tune_qrel_outcomes_parsed": False,
        "calibrator_refit": False,
        "selection_terminal": True,
        "policy_suite_frozen": success,
    }
    access["fingerprint"] = fingerprint(access)

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        write_json(temporary / "query_tune_access.json", access)
        write_json(temporary / "query_tune_qrel_access.json", qrel_access)
        write_json(temporary / "projection.json", projection)
        _write_jsonl(
            temporary / "query_tune_records.jsonl",
            (_public_record(record) for record in records),
        )
        np.save(
            temporary / "candidate_budgets.npy",
            selection_result["budget_matrix"],
            allow_pickle=False,
        )
        write_json(
            temporary / "candidate_outcomes.json",
            selection_result["candidate_outcomes"],
        )
        write_json(temporary / "selection.json", selection_result["selection"])
        _write_jsonl(
            temporary / "selected_policy_records.jsonl", selected_records
        )
        for name, artifact in optional_artifacts.items():
            write_json(temporary / name, artifact)
        write_json(
            temporary / "protocol_state_after_query_tune.json", guard.serialize()
        )
        artifact_names = [
            "query_tune_access.json",
            "query_tune_qrel_access.json",
            "projection.json",
            "query_tune_records.jsonl",
            "candidate_budgets.npy",
            "candidate_outcomes.json",
            "selection.json",
            "selected_policy_records.jsonl",
            *optional_artifacts.keys(),
            "protocol_state_after_query_tune.json",
        ]
        decision = selection_result["selection"]["decision"]
        manifest: Dict[str, Any] = {
            "name": "pdctp_fiqa_query_tune_manifest",
            "schema": "pdctp_fiqa_query_tune_manifest_v1",
            "version": 1,
            "config_fingerprint": config.config_fingerprint,
            "upstream": dict(config.raw["bindings"]),
            "query_tune_access_fingerprint": access["fingerprint"],
            "qrel_access_fingerprint": qrel_access["fingerprint"],
            "projection_fingerprint": projection["fingerprint"],
            "candidate_outcomes_fingerprint": selection_result[
                "candidate_outcomes"
            ]["fingerprint"],
            "selection_fingerprint": selection_result["selection"]["fingerprint"],
            "frozen_policy_suite_fingerprint": (
                selection_result["policy_suite"]["fingerprint"] if success else None
            ),
            "hypotheses_fingerprint": (
                optional_artifacts["hypotheses.json"]["fingerprint"]
                if success
                else None
            ),
            "protocol_state_fingerprint": guard.state_fingerprint,
            "counts": {
                "query_tune": len(records),
                "positive_qrel_rows": qrel_access["scoped_positive_rows"],
                "valid_pilot_lid": sum(row["pilot"]["lid_valid"] for row in records),
                "valid_pilot_features": sum(
                    row["_features_obj"].valid for row in records
                ),
                "candidates": selection_result["candidate_outcomes"]["counts"],
                "selected_methods": len(_METHOD_ORDER) if success else 0,
            },
            "checks": {
                "all_upstream_fingerprints_validated_before_query_tune": True,
                "query_cal_fits_reconstructed_before_query_tune": True,
                "only_query_tune_newly_opened": True,
                "all_query_tune_ids_used_in_frozen_order": True,
                "query_tune_qrels_accessed": True,
                "non_tune_qrel_outcomes_parsed": False,
                "query_cert_accessed": False,
                "query_latency_accessed": False,
                "query_test_accessed": False,
                "calibrator_refit": False,
                "all_candidate_families_enumerated": True,
                "one_projected_scan_per_query": all(
                    row["work"]["projected_scan_count"] == 1 for row in records
                ),
                "projected_vectors_renormalized": False,
                "llm_run": False,
                "approximate_index_used": False,
                "raw_tri_predict_v1_behavior_modified": False,
                "certification_run": False,
                "latency_measured": False,
            },
            "artifacts": {
                name: _file_identity(temporary / name) for name in artifact_names
            },
            "decision": decision,
        }
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(temporary / "manifest.json", manifest)
        report = (
            "# PDCTP FiQA query_tune selection gate\n\n"
            f"Decision: `{decision}`.\n\n"
            f"Opened the complete `query_tune` role ({len(records):,} IDs) after "
            "reconstructing the accepted query_cal fits.\n\n"
            f"Evaluated {sum(selection_result['candidate_outcomes']['counts'].values()):,} "
            "predeclared candidates without refitting a calibrator.\n\n"
            + (
                "One independently optimized candidate per method family and the "
                "certification hypotheses are frozen. "
                if success
                else "At least one method family had no eligible candidate; this is a "
                "terminal tune failure with no retuning. "
            )
            + "query_cert, query_latency, and query_test remain closed. No LLM or "
            "approximate index was used.\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        name: output / name
        for name in (
            *artifact_names,
            "manifest.json",
            "report.md",
        )
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--real-protocol-config", required=True, type=Path)
    parser.add_argument("--protocol-freeze", required=True, type=Path)
    parser.add_argument("--role-assignments", required=True, type=Path)
    parser.add_argument("--source-audit", required=True, type=Path)
    parser.add_argument("--fiqa-archive", required=True, type=Path)
    parser.add_argument("--embedding-audit", required=True, type=Path)
    parser.add_argument("--embedding-config", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--embedding-cache", required=True, type=Path)
    parser.add_argument("--query-cal-audit", required=True, type=Path)
    parser.add_argument("--query-cal-run", required=True, type=Path)
    parser.add_argument("--power-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = run_pdctp_fiqa_query_tune(
        load_pdctp_query_tune_config(args.config),
        load_pdctp_real_protocol_config(args.real_protocol_config),
        args.protocol_freeze,
        args.role_assignments,
        args.source_audit,
        args.fiqa_archive,
        args.embedding_audit,
        args.embedding_config,
        args.prepared,
        args.embedding_cache,
        args.query_cal_audit,
        args.query_cal_run,
        args.power_plan,
        args.output,
    )
    print(f"PDCTP FiQA query_tune gate wrote {len(paths)} artifacts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
