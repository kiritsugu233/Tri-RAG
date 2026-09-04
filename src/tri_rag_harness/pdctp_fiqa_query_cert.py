"""Fingerprint-gated one-time FiQA ``query_cert`` certification runner.

The runner reconstructs the accepted six-policy tune suite and its frozen
hypotheses before it opens the complete certification role.  It makes no fit
or selection, evaluates every frozen policy from deployable pilot inputs, and
closes certification with a terminal pass or failure artifact.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .lid import estimate_lid_from_squared_distances
from .pdctp_features import (
    PilotDistanceFeatureExtractor,
    PilotDistanceObservation,
    stable_sort_pilot_distances,
)
from .pdctp_fiqa_query_cal import (
    _file_identity,
    _load_fingerprinted,
    _public_record,
    _round_values,
    _squared_l2_batch,
    _stable_top_k_rows,
)
from .pdctp_fiqa_query_tune import (
    _archive_identity,
    _feature_spec,
    _role_assignments,
    _write_jsonl,
    _zip_member_identity,
    reconstruct_frozen_policy_suite,
)
from .pdctp_policies import (
    PDCTPDecisionInput,
    PDCTPDecisionPolicy,
    validate_policy_suite,
)
from .pdctp_protocol import FiveRoleAssignments, FiveRoleProtocolGuard
from .pdctp_real_protocol import (
    PDCTPRealProtocolConfig,
    load_pdctp_real_protocol_config,
)
from .pdctp_statistics import make_paired_bound, make_power_plan, validate_paired_bound
from .projection import dense_gaussian_projection, projection_metadata
from .text_embeddings import load_text_embedding_config, validate_text_embedding_cache
from .utils import array_fingerprint, fingerprint, write_json


class PDCTPQueryCertError(ValueError):
    """Raised before cert access or when the frozen certification contract fails."""


@dataclass(frozen=True)
class PDCTPQueryCertConfig:
    raw: Mapping[str, Any]
    config_fingerprint: str
    run_name: str
    query_batch_size: int
    record_distance_decimals: int


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
_QUERY_PREFIX = "pdctp-beir-fiqa:query:"
_DOC_PREFIX = "pdctp-beir-fiqa:doc:"
_TUNE_FILES = (
    "query_tune_access.json",
    "query_tune_qrel_access.json",
    "projection.json",
    "query_tune_records.jsonl",
    "candidate_budgets.npy",
    "candidate_outcomes.json",
    "selection.json",
    "selected_policy_records.jsonl",
    "policies.json",
    "selected_policy_components.json",
    "frozen_policy_suite.json",
    "shuffled_tune_diagnostic.json",
    "hypotheses.json",
    "protocol_state_after_query_tune.json",
    "manifest.json",
    "report.md",
)


def _exact(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise PDCTPQueryCertError(
            f"{context} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _hex(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PDCTPQueryCertError(f"{name} must be a lowercase SHA-256 value")
    return value


def _verify_fingerprinted(value: Mapping[str, Any], name: str) -> None:
    body = dict(value)
    stored = body.pop("fingerprint", None)
    if not isinstance(stored, str) or fingerprint(body) != stored:
        raise PDCTPQueryCertError(f"{name} fingerprint mismatch")


def load_pdctp_query_cert_config(
    path: Union[str, Path],
) -> PDCTPQueryCertConfig:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PDCTPQueryCertError(f"cannot load query_cert config: {exc}") from exc
    _exact(
        raw,
        {
            "schema",
            "version",
            "run_name",
            "bindings",
            "access",
            "certification_contract",
            "execution",
        },
        "root",
    )
    if raw["schema"] != "pdctp_fiqa_query_cert_gate_v1" or raw["version"] != 1:
        raise PDCTPQueryCertError("unsupported FiQA query_cert gate schema")
    if not isinstance(raw["run_name"], str) or not raw["run_name"]:
        raise PDCTPQueryCertError("query_cert run name must be nonempty")
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
        "query_tune_audit_fingerprint",
        "query_tune_audit_sha256",
        "query_tune_manifest_fingerprint",
        "query_tune_protocol_state_fingerprint",
        "query_tune_selection_fingerprint",
        "frozen_policies_fingerprint",
        "selected_component_registry_fingerprint",
        "frozen_policy_suite_fingerprint",
        "hypotheses_fingerprint",
        "query_cert_ordered_id_hash",
        "power_plan_fingerprint",
    }
    _exact(raw["bindings"], binding_keys, "bindings")
    for key, value in raw["bindings"].items():
        _hex(value, f"bindings.{key}")

    access = raw["access"]
    expected_access = {
        "role": "query_cert",
        "require_complete_frozen_order": True,
        "allowed_supervision": [
            "exact_original_top_k_identities",
            "realized_embedding_retention",
            "query_cert_positive_qrels",
            "candidate_evidence_recall",
            "final_reranked_evidence_recall",
        ],
        "qrel_member": "qrels_train",
        "non_cert_qrel_outcomes_must_not_be_parsed": True,
        "calibrator_fit_allowed": False,
        "selection_allowed": False,
        "blocked_roles": ["query_latency", "query_test"],
    }
    _exact(access, set(expected_access), "access")
    if access != expected_access:
        raise PDCTPQueryCertError("query_cert access scope changed")

    contract = raw["certification_contract"]
    expected_contract = {
        "primary_policy": "pdctp",
        "method_order": list(_METHOD_ORDER),
        "family_wise_method": "bonferroni",
        "family_wise_alpha": 0.05,
        "required_query_count": 1567,
        "bounds": "paired_empirical_bernstein",
        "difference_definition": "pdctp_minus_comparator",
        "upper_bound_is_strict": True,
        "all_hypotheses_must_pass": True,
        "failure_behavior": "terminal_no_retuning_no_budget_expansion",
        "policy_or_hypothesis_mutation_after_open": False,
    }
    _exact(contract, set(expected_contract), "certification_contract")
    if contract != expected_contract:
        raise PDCTPQueryCertError("query_cert certification contract changed")

    execution = raw["execution"]
    fixed_execution = {
        "backend": "numpy_exact_float64_batched_v1",
        "projection_dtype": "float64",
        "post_projection_normalize": False,
        "distance": "squared_l2",
        "stable_tie_break": "lexicographic_doc_id",
        "pilot_expansion_reuse": "one_projected_scan",
        "policy_decision_engine": "frozen_policy_choose_without_mutation",
        "qrel_filter": "first_column_role_filter_before_outcome_parse_v1",
        "approximate_index": False,
        "llm": False,
    }
    expected_execution_keys = set(fixed_execution) | {
        "query_batch_size",
        "record_distance_decimals",
    }
    _exact(execution, expected_execution_keys, "execution")
    if any(execution.get(key) != value for key, value in fixed_execution.items()):
        raise PDCTPQueryCertError("query_cert numerical or scope contract changed")
    batch = execution["query_batch_size"]
    decimals = execution["record_distance_decimals"]
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise PDCTPQueryCertError("query batch size must be a positive integer")
    if (
        isinstance(decimals, bool)
        or not isinstance(decimals, int)
        or not 6 <= decimals <= 15
    ):
        raise PDCTPQueryCertError("record distance decimals must be from 6 to 15")
    return PDCTPQueryCertConfig(
        raw=raw,
        config_fingerprint=fingerprint(raw),
        run_name=raw["run_name"],
        query_batch_size=batch,
        record_distance_decimals=decimals,
    )


def load_query_cert_qrels(
    archive_path: Union[str, Path],
    member_metadata: Mapping[str, Any],
    query_cert_ids: Sequence[str],
    *,
    minimum_relevance: int,
) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, Any]]:
    """Parse outcome columns only for rows whose first field is a cert ID."""
    path = Path(archive_path)
    cert_ids = tuple(str(value) for value in query_cert_ids)
    cert_set = set(cert_ids)
    if not cert_ids or len(cert_set) != len(cert_ids):
        raise PDCTPQueryCertError("query_cert qrel IDs must be nonempty and unique")
    member = str(member_metadata.get("path"))
    positives: Dict[str, set[str]] = {query_id: set() for query_id in cert_ids}
    scoped_rows = 0
    scoped_positive_rows = 0
    skipped_before_outcome_parse = 0
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or member not in names:
                raise PDCTPQueryCertError("FiQA qrel member inventory changed")
            if _zip_member_identity(archive, member) != dict(member_metadata):
                raise PDCTPQueryCertError("FiQA train qrel member identity changed")
            with io.TextIOWrapper(
                archive.open(member, "r"), encoding="utf-8", newline=""
            ) as source:
                header = source.readline().rstrip("\r\n").split("\t")
                if header[:3] != ["query-id", "corpus-id", "score"]:
                    raise PDCTPQueryCertError("unexpected FiQA qrel header")
                for line_number, line in enumerate(source, start=2):
                    source_query_id, separator, remainder = line.partition("\t")
                    source_query_id = source_query_id.strip()
                    if not separator or not source_query_id:
                        raise PDCTPQueryCertError(
                            f"invalid qrel query field at {member}:{line_number}"
                        )
                    query_id = _QUERY_PREFIX + source_query_id
                    if query_id not in cert_set:
                        skipped_before_outcome_parse += 1
                        continue
                    outcome_fields = remainder.rstrip("\r\n").split("\t")
                    if len(outcome_fields) < 2:
                        raise PDCTPQueryCertError(
                            f"invalid cert qrel outcome at {member}:{line_number}"
                        )
                    doc_id = outcome_fields[0].strip()
                    try:
                        relevance = int(outcome_fields[1])
                    except ValueError as exc:
                        raise PDCTPQueryCertError(
                            f"invalid cert relevance at {member}:{line_number}"
                        ) from exc
                    if not doc_id:
                        raise PDCTPQueryCertError(
                            f"empty cert qrel document at {member}:{line_number}"
                        )
                    scoped_rows += 1
                    if relevance >= minimum_relevance:
                        stable_doc_id = _DOC_PREFIX + doc_id
                        if stable_doc_id in positives[query_id]:
                            raise PDCTPQueryCertError(
                                "duplicate positive cert qrel pair"
                            )
                        positives[query_id].add(stable_doc_id)
                        scoped_positive_rows += 1
    except zipfile.BadZipFile as exc:
        raise PDCTPQueryCertError(f"invalid FiQA archive: {path}") from exc
    missing = [query_id for query_id in cert_ids if not positives[query_id]]
    if missing:
        raise PDCTPQueryCertError(
            f"query_cert contains queries without positive qrels: {missing[:3]}"
        )
    result = {query_id: tuple(sorted(positives[query_id])) for query_id in cert_ids}
    audit: Dict[str, Any] = {
        "name": "pdctp_fiqa_query_cert_qrel_access",
        "schema": "pdctp_fiqa_query_cert_qrel_access_v1",
        "version": 1,
        "role": "query_cert",
        "member": dict(member_metadata),
        "filter": "first_column_role_filter_before_outcome_parse_v1",
        "query_order_hash": fingerprint(list(cert_ids)),
        "queries": len(cert_ids),
        "scoped_rows": scoped_rows,
        "scoped_positive_rows": scoped_positive_rows,
        "non_cert_rows_skipped_before_outcome_parse": skipped_before_outcome_parse,
        "non_cert_qrel_outcomes_parsed": False,
        "minimum_relevance": int(minimum_relevance),
    }
    audit["fingerprint"] = fingerprint(audit)
    return result, audit


def _validate_query_tune_file_inventory(
    tune_audit: Mapping[str, Any],
    query_tune_file_identities: Mapping[str, Mapping[str, Any]],
) -> None:
    audit_file_ids = tune_audit.get("file_identities")
    if not isinstance(audit_file_ids, Mapping) or set(audit_file_ids) != set(
        _TUNE_FILES
    ):
        raise PDCTPQueryCertError("query_tune audit file inventory changed")
    if set(query_tune_file_identities) != set(_TUNE_FILES):
        raise PDCTPQueryCertError("query_tune run file inventory is incomplete")
    for name, actual in query_tune_file_identities.items():
        if audit_file_ids.get(name) != actual.get("sha256"):
            raise PDCTPQueryCertError(f"query_tune file identity changed: {name}")


def validate_query_cert_documents(
    config: PDCTPQueryCertConfig,
    protocol: Mapping[str, Any],
    roles: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    embedding_audit: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    embedding_manifest: Mapping[str, Any],
    tune_audit: Mapping[str, Any],
    tune_manifest: Mapping[str, Any],
    tune_state: Mapping[str, Any],
    tune_access: Mapping[str, Any],
    tune_projection: Mapping[str, Any],
    selection: Mapping[str, Any],
    policies_artifact: Mapping[str, Any],
    component_registry: Mapping[str, Any],
    policy_suite: Mapping[str, Any],
    hypotheses: Mapping[str, Any],
    *,
    source_audit_sha256: str,
    query_tune_audit_sha256: str,
    query_tune_file_identities: Mapping[str, Mapping[str, Any]],
) -> Tuple[FiveRoleAssignments, FiveRoleProtocolGuard, Dict[str, PDCTPDecisionPolicy]]:
    """Validate every portable identity and replay the post-tune guard state."""
    for name, value in (
        ("protocol freeze", protocol),
        ("role assignments", roles),
        ("source audit", source_audit),
        ("embedding audit", embedding_audit),
        ("dataset manifest", dataset_manifest),
        ("embedding manifest", embedding_manifest),
        ("query_tune audit", tune_audit),
        ("query_tune manifest", tune_manifest),
        ("query_tune state", tune_state),
        ("query_tune access", tune_access),
        ("query_tune projection", tune_projection),
        ("selection", selection),
        ("policies", policies_artifact),
        ("component registry", component_registry),
        ("policy suite", policy_suite),
        ("hypotheses", hypotheses),
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
        "query_tune_audit_fingerprint": tune_audit.get("fingerprint"),
        "query_tune_audit_sha256": query_tune_audit_sha256,
        "query_tune_manifest_fingerprint": tune_manifest.get("fingerprint"),
        "query_tune_protocol_state_fingerprint": tune_state.get("fingerprint"),
        "query_tune_selection_fingerprint": selection.get("fingerprint"),
        "frozen_policies_fingerprint": policies_artifact.get("fingerprint"),
        "selected_component_registry_fingerprint": component_registry.get(
            "fingerprint"
        ),
        "frozen_policy_suite_fingerprint": policy_suite.get("fingerprint"),
        "hypotheses_fingerprint": hypotheses.get("fingerprint"),
        "power_plan_fingerprint": hypotheses.get("power_plan_fingerprint"),
    }
    for key, expected in bindings.items():
        if key == "query_cert_ordered_id_hash":
            continue
        if observed.get(key) != expected:
            raise PDCTPQueryCertError(f"frozen upstream binding changed: {key}")

    if (
        protocol.get("decision") != "READY_FOR_DATASET_AND_EMBEDDING_AUDIT_ONLY"
        or protocol.get("resolved_roles", {}).get("assignment_fingerprint")
        != roles.get("fingerprint")
        or protocol.get("resolved_inputs", {})
        .get("source_audit", {})
        .get("fingerprint")
        != source_audit.get("fingerprint")
        or roles.get("all_roles_initially_closed") is not True
        or roles.get("authorizes_outcome_access") is not False
        or source_audit.get("decision") != "GO_TO_PROTOCOL_FREEZE"
    ):
        raise PDCTPQueryCertError("protocol, source, and role freeze are not bound")
    assignments = _role_assignments(roles)
    cert_ids = assignments.ids_by_role["query_cert"]
    if fingerprint(list(cert_ids)) != bindings["query_cert_ordered_id_hash"]:
        raise PDCTPQueryCertError("query_cert ordered identity changed")
    if len(cert_ids) != config.raw["certification_contract"]["required_query_count"]:
        raise PDCTPQueryCertError("query_cert size differs from the frozen power gate")

    if (
        dataset_manifest.get("protocol_freeze_fingerprint")
        != protocol.get("fingerprint")
        or dataset_manifest.get("role_assignments_fingerprint")
        != roles.get("fingerprint")
        or dataset_manifest.get("scope_guards", {}).get("contains_qrels") is not False
        or dataset_manifest.get("scope_guards", {}).get("contains_relevance_values")
        is not False
        or embedding_manifest.get("dataset", {}).get("manifest_fingerprint")
        != dataset_manifest.get("fingerprint")
        or embedding_audit.get("embedding_manifest_fingerprint")
        != embedding_manifest.get("fingerprint")
        or embedding_audit.get("decision") != "READY_TO_OPEN_QUERY_CAL"
        or embedding_audit.get("checks", {}).get("all_roles_remained_closed")
        is not True
        or embedding_audit.get("checks", {}).get("qrels_or_relevance_opened")
        is not False
    ):
        raise PDCTPQueryCertError("dataset and embedding identities are not transitive")

    audit_checks = tune_audit.get("checks", {})
    required_audit_checks = (
        "all_16_run_artifact_hashes_valid",
        "all_candidate_budget_vectors_recomputed_exact",
        "all_candidate_evaluations_recomputed_exact",
        "all_query_tune_record_fingerprints_valid",
        "all_query_tune_records_in_frozen_order",
        "all_selected_policy_records_recomputed_exact",
        "certification_hypotheses_reconstructed_exact",
        "full_candidate_selection_replayed_exact",
        "only_query_tune_newly_opened",
        "projection_matrix_regenerated_exact",
        "protocol_state_reconstructed_exact",
        "query_tune_qrels_reloaded_exact",
        "selected_policy_suite_reconstructed_exact",
    )
    if (
        tune_audit.get("decision")
        != "ACCEPT_QUERY_TUNE_SELECTION_READY_TO_IMPLEMENT_QUERY_CERT"
        or any(audit_checks.get(key) is not True for key in required_audit_checks)
        or audit_checks.get("query_cert_accessed") is not False
        or audit_checks.get("query_latency_accessed") is not False
        or audit_checks.get("query_test_accessed") is not False
        or audit_checks.get("calibrator_refit") is not False
        or audit_checks.get("approximate_index_used") is not False
        or audit_checks.get("llm_run") is not False
        or audit_checks.get("raw_tri_predict_v1_behavior_modified") is not False
    ):
        raise PDCTPQueryCertError("accepted query_tune audit does not authorize cert")
    _validate_query_tune_file_inventory(tune_audit, query_tune_file_identities)

    manifest_checks = tune_manifest.get("checks", {})
    if (
        tune_manifest.get("decision")
        != "QUERY_TUNE_SELECTION_FROZEN_READY_FOR_CERT_IMPLEMENTATION"
        or tune_manifest.get("protocol_state_fingerprint")
        != tune_state.get("fingerprint")
        or tune_manifest.get("query_tune_access_fingerprint")
        != tune_access.get("fingerprint")
        or tune_manifest.get("projection_fingerprint")
        != tune_projection.get("fingerprint")
        or tune_manifest.get("selection_fingerprint") != selection.get("fingerprint")
        or tune_manifest.get("frozen_policy_suite_fingerprint")
        != policy_suite.get("fingerprint")
        or tune_manifest.get("hypotheses_fingerprint") != hypotheses.get("fingerprint")
        or manifest_checks.get("only_query_tune_newly_opened") is not True
        or manifest_checks.get("query_cert_accessed") is not False
        or manifest_checks.get("query_latency_accessed") is not False
        or manifest_checks.get("query_test_accessed") is not False
        or manifest_checks.get("calibrator_refit") is not False
        or manifest_checks.get("certification_run") is not False
        or manifest_checks.get("latency_measured") is not False
        or manifest_checks.get("approximate_index_used") is not False
        or manifest_checks.get("llm_run") is not False
    ):
        raise PDCTPQueryCertError("query_tune manifest or transition changed")
    for name, identity in tune_manifest.get("artifacts", {}).items():
        if query_tune_file_identities.get(name) != identity:
            raise PDCTPQueryCertError(f"query_tune manifest artifact changed: {name}")
    if (
        tune_access.get("roles_with_outcomes_opened") != ["query_cal", "query_tune"]
        or tune_access.get("roles_remaining_closed")
        != ["query_cert", "query_latency", "query_test"]
        or tune_access.get("selection_terminal") is not True
        or tune_access.get("policy_suite_frozen") is not True
        or tune_access.get("calibrator_refit") is not False
        or tune_access.get("non_tune_qrel_outcomes_parsed") is not False
    ):
        raise PDCTPQueryCertError("query_tune access artifact changed")

    if (
        selection.get("decision")
        != "QUERY_TUNE_SELECTION_FROZEN_READY_FOR_CERT_IMPLEMENTATION"
        or selection.get("role") != "query_tune"
        or selection.get("frozen_policy_suite_fingerprint")
        != policy_suite.get("fingerprint")
        or selection.get("calibrator_refit") is not False
        or selection.get("certification_used") is not False
        or set(selection.get("selected_by_method", {})) != set(_METHOD_ORDER)
        or policy_suite.get("method_order") != list(_METHOD_ORDER)
        or policy_suite.get("policy_key_mapping") != _POLICY_KEYS
        or policy_suite.get("policies_fingerprint")
        != policies_artifact.get("fingerprint")
        or policy_suite.get("component_registry_fingerprint")
        != component_registry.get("fingerprint")
    ):
        raise PDCTPQueryCertError("frozen tune selection or suite changed")
    certification = protocol.get("protocol", {}).get("certification", {})
    if (
        hypotheses.get("frozen_before_query_cert") is not True
        or hypotheses.get("family_wise_method")
        != certification.get("family_wise_method")
        or hypotheses.get("family_wise_alpha") != certification.get("family_wise_alpha")
        or hypotheses.get("hypotheses") != certification.get("hypotheses")
        or hypotheses.get("required_query_count")
        != certification.get("required_query_count")
    ):
        raise PDCTPQueryCertError("frozen certification hypotheses changed")

    try:
        frozen_policies = reconstruct_frozen_policy_suite(
            policies_artifact, component_registry, policy_suite
        )
        validate_policy_suite(frozen_policies)
    except ValueError as exc:
        raise PDCTPQueryCertError(f"frozen policy suite is invalid: {exc}") from exc
    serialized = {name: policy.serialize() for name, policy in frozen_policies.items()}
    if serialized != policies_artifact.get("policies"):
        raise PDCTPQueryCertError("frozen policies did not reconstruct exactly")

    fit_artifacts = tune_state.get("fit_artifacts")
    if not isinstance(fit_artifacts, Mapping) or set(fit_artifacts) != {
        "lid_calibrator",
        "residual_calibrator",
    }:
        raise PDCTPQueryCertError("post-tune fit identities changed")
    guard = FiveRoleProtocolGuard(assignments, str(protocol["config_fingerprint"]))
    cal_ids = assignments.ids_by_role["query_cal"]
    guard.open_calibration(cal_ids)
    for fit_name in ("lid_calibrator", "residual_calibrator"):
        guard.register_fit(
            fit_name,
            role="query_cal",
            ids=cal_ids,
            artifact_fingerprint=str(fit_artifacts[fit_name]),
        )
    tune_token = guard.open_tune_selection(assignments.ids_by_role["query_tune"])
    guard.freeze_selection(tune_token, str(selection["fingerprint"]))
    guard.freeze_hypotheses(str(hypotheses["fingerprint"]))
    if guard.serialize() != tune_state:
        raise PDCTPQueryCertError("post-query_tune guard state is not reconstructable")
    return assignments, guard, frozen_policies


def build_query_cert_records(
    protocol: Mapping[str, Any],
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    corpus_embeddings: np.ndarray,
    query_cert_embeddings: np.ndarray,
    qrels: Mapping[str, Sequence[str]],
    policies: Mapping[str, PDCTPDecisionPolicy],
    *,
    batch_size: int,
    record_distance_decimals: int,
    progress: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate all frozen policies once using deployable pilot inputs only."""
    retrieval = protocol["retrieval"]
    suite = protocol["candidate_suite"]
    corpus = np.asarray(corpus_embeddings, dtype=np.float64)
    queries = np.asarray(query_cert_embeddings, dtype=np.float64)
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
        raise PDCTPQueryCertError("query_cert arrays, qrels, and IDs are not aligned")
    if not np.all(np.isfinite(corpus)) or not np.all(np.isfinite(queries)):
        raise PDCTPQueryCertError("query_cert arrays must be finite")
    if (
        retrieval["corpus_size"] != len(corpus)
        or retrieval["embedding_dimension"] != corpus.shape[1]
        or retrieval["query_batch_size"] != batch_size
    ):
        raise PDCTPQueryCertError("query_cert retrieval dimensions changed")
    if set(policies) != set(_POLICY_KEYS.values()):
        raise PDCTPQueryCertError(
            "query_cert requires the complete frozen policy suite"
        )
    validate_policy_suite(policies)
    serialized_policies = {
        name: policy.serialize() for name, policy in policies.items()
    }
    policy_fingerprints = {
        name: str(artifact["fingerprint"])
        for name, artifact in serialized_policies.items()
    }
    frozen_grid = tuple(int(value) for value in retrieval["m_grid"])
    if any(tuple(policy.grid) != frozen_grid for policy in policies.values()):
        raise PDCTPQueryCertError("a policy grid differs from the retrieval freeze")

    id_to_row = {doc_id: index for index, doc_id in enumerate(ids.tolist())}
    if any(
        not relevant or any(doc_id not in id_to_row for doc_id in relevant)
        for relevant in qrels.values()
    ):
        raise PDCTPQueryCertError("query_cert qrels reference invalid corpus IDs")
    spec = _feature_spec(protocol)
    extractor = PilotDistanceFeatureExtractor(spec)
    projection_raw = retrieval["projection"]
    if projection_raw["post_projection_normalize"] is not False:
        raise PDCTPQueryCertError("post-projection normalization is forbidden")
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
            pilot_rows, pilot_projected_sq = _stable_top_k_rows(
                projected_distances, tie_rank, retrieval["m_pilot"]
            )
            pilot_diff = corpus[pilot_rows] - query
            pilot_original_sq = np.einsum("ij,ij->i", pilot_diff, pilot_diff)
            (
                sorted_ids,
                sorted_original_sq,
                sorted_projected_sq,
            ) = stable_sort_pilot_distances(
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
            canonical_pilot_lid = float(np.round(pilot_lid.clipped, 10))
            observation = PDCTPDecisionInput(
                features=features,
                pilot_lid=canonical_pilot_lid,
                pilot_lid_valid=bool(pilot_lid.valid),
            )

            # This is deliberately before exact-top-k and relevance processing.
            # The policy API exposes only deployable pilot inputs.
            decisions = {
                family: policies[policy_key].choose(observation)
                for family, policy_key in _POLICY_KEYS.items()
            }
            if any(
                decision.budget not in frozen_grid for decision in decisions.values()
            ):
                raise PDCTPQueryCertError("a frozen policy emitted an unfrozen budget")
            if any(
                decision.budget < max(retrieval["k_gt"], retrieval["m_pilot"])
                for decision in decisions.values()
            ):
                raise PDCTPQueryCertError(
                    "a frozen policy violated the safe lower bound"
                )

            oracle_rows, _ = _stable_top_k_rows(original_distances, tie_rank, oracle_k)
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
            relevant_set = set(relevant_rows.tolist())
            final_rows_by_budget: Dict[int, np.ndarray] = {}
            method_rows: Dict[str, Any] = {}
            for family in _METHOD_ORDER:
                policy_key = _POLICY_KEYS[family]
                decision = decisions[family]
                budget = int(decision.budget)
                final_rows = final_rows_by_budget.get(budget)
                if final_rows is None:
                    candidates = projected_order[:budget]
                    final_local_rows, _ = _stable_top_k_rows(
                        original_distances[candidates],
                        tie_rank[candidates],
                        min(retrieval["k_ctx"], len(candidates)),
                    )
                    final_rows = candidates[final_local_rows]
                    final_rows_by_budget[budget] = final_rows
                method_rows[family] = {
                    "policy_key": policy_key,
                    "policy_fingerprint": policy_fingerprints[policy_key],
                    "decision": asdict(decision),
                    "chosen_m": budget,
                    "embedding_retention": float(np.mean(gt_ranks <= budget)),
                    "candidate_evidence_recall": float(
                        np.mean(relevant_ranks <= budget)
                    ),
                    "final_evidence_recall": float(
                        len(relevant_set.intersection(final_rows.tolist()))
                        / len(relevant_set)
                    ),
                    "final_top_k_doc_ids": ids[final_rows].tolist(),
                    "work": {
                        "pilot_original_rerank_distance_count": retrieval["m_pilot"],
                        "expansion_candidate_count": budget,
                        "exact_original_rerank_candidate_count": budget,
                        "common_coordinate_work": (
                            (len(corpus) + corpus.shape[1]) * projection_raw["m_prime"]
                            + corpus.shape[1] * budget
                        ),
                    },
                }
            body: Dict[str, Any] = {
                "schema": "pdctp_fiqa_query_cert_record_v1",
                "query_id": query_id,
                "role": "query_cert",
                "supervision": {
                    "exact_original_top_k_identities": True,
                    "realized_embedding_retention": True,
                    "query_cert_positive_qrels": True,
                    "candidate_evidence_recall": True,
                    "final_reranked_evidence_recall": True,
                    "oracle_lid_for_policy_decision": False,
                    "exact_top_k_for_policy_decision": False,
                    "qrels_for_policy_decision": False,
                    "non_cert_qrel_outcomes": False,
                },
                "pilot": {
                    "candidate_doc_ids_original_distance_order": sorted_ids.tolist(),
                    "original_squared_distances": _round_values(
                        sorted_original_sq, record_distance_decimals
                    ),
                    "projected_squared_distances": _round_values(
                        sorted_projected_sq, record_distance_decimals
                    ),
                    "lid": canonical_pilot_lid,
                    "lid_valid": pilot_lid.valid,
                    "lid_failure_reason": pilot_lid.reason,
                    "features": features.serialize(),
                },
                "policy_decisions_completed_before_supervision": True,
                "exact_original_top_k_doc_ids": ids[gt_rows].tolist(),
                "projected_rank_of_exact_top_k": gt_ranks.tolist(),
                "positive_qrel_doc_ids": list(relevant_ids),
                "projected_rank_of_positive_qrels": relevant_ranks.tolist(),
                "methods": method_rows,
                "shared_work": {
                    "projected_scan_count": 1,
                    "projected_distance_count": len(corpus),
                    "original_reference_distance_count": len(corpus),
                    "pilot_is_prefix_of_same_projected_scan": True,
                    "exact_rerank_uses_same_original_distance_vector": True,
                },
            }
            body["fingerprint"] = fingerprint(body)
            body["_features_obj"] = features
            records.append(body)
            completed = start + offset + 1
            if progress and (completed % 8 == 0 or completed == len(queries)):
                print(
                    f"query_cert frozen-policy retrieval: {completed}/{len(queries)}",
                    flush=True,
                )

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
            "projected_query_role": "query_cert",
            "projected_query_shape": list(projected_queries.shape),
            "projected_vectors_persisted": False,
        }
    )
    projection.pop("fingerprint", None)
    projection["fingerprint"] = fingerprint(projection)
    return records, projection


def make_query_certification(
    protocol: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    policies: Mapping[str, PDCTPDecisionPolicy],
    hypotheses: Mapping[str, Any],
    policy_suite_fingerprint: str,
) -> Dict[str, Any]:
    """Build and independently reconstruct all six frozen paired bounds."""
    certification = protocol["certification"]
    query_ids = [str(row["query_id"]) for row in records]
    if (
        len(query_ids) != certification["required_query_count"]
        or len(set(query_ids)) != len(query_ids)
        or any(row.get("role") != "query_cert" for row in records)
    ):
        raise PDCTPQueryCertError("certification records violate the frozen role")
    _verify_fingerprinted(hypotheses, "certification hypotheses")
    if hypotheses.get("hypotheses") != certification["hypotheses"]:
        raise PDCTPQueryCertError("certification hypotheses changed")
    allocation = hypotheses["alpha_allocation"]
    policy_fingerprints = {
        name: str(policy.serialize()["fingerprint"])
        for name, policy in policies.items()
    }
    bounds: Dict[str, Any] = {}
    for specification in certification["hypotheses"]:
        name = str(specification["name"])
        metric = str(specification["metric"])
        comparison = str(specification["comparison"])
        left = []
        right = []
        for row in records:
            methods = row["methods"]
            if metric == "normalized_candidate_budget":
                left.append(
                    float(methods["pdctp"]["chosen_m"])
                    / protocol["retrieval"]["corpus_size"]
                )
            else:
                left.append(float(methods["pdctp"][metric]))
            if comparison == "zero_anchor":
                right.append(0.0)
            else:
                comparator_family = {
                    "fixed_reference": "fixed",
                    "monotone_binned": "monotone_binned",
                    "raw_tri_predict": "raw_tri_predict",
                }.get(comparison)
                if comparator_family is None:
                    raise PDCTPQueryCertError(
                        f"unsupported certification comparison: {comparison}"
                    )
                if metric == "normalized_candidate_budget":
                    right.append(
                        float(methods[comparator_family]["chosen_m"])
                        / protocol["retrieval"]["corpus_size"]
                    )
                else:
                    right.append(float(methods[comparator_family][metric]))
        right_fingerprint = (
            "zero_anchor"
            if comparison == "zero_anchor"
            else policy_fingerprints[
                {
                    "fixed_reference": "fixed",
                    "monotone_binned": "monotone",
                    "raw_tri_predict": "raw_tri",
                }[comparison]
            ]
        )
        bound = make_paired_bound(
            query_ids,
            left,
            right,
            hypothesis=name,
            metric=metric,
            alpha=float(allocation[name]),
            difference_bounds=tuple(specification["difference_bounds"]),
            side=str(specification["side"]),
            margin=float(specification["margin"]),
            left_policy_fingerprint=policy_fingerprints["full_pdctp"],
            right_policy_fingerprint=right_fingerprint,
        )
        validate_paired_bound(bound)
        bounds[name] = bound
    all_passed = all(bound["passed"] for bound in bounds.values())
    decision = (
        "TERMINAL_CERTIFICATION_PASS_READY_FOR_LATENCY_IMPLEMENTATION"
        if all_passed
        else "TERMINAL_CERTIFICATION_FAIL_NO_RETUNING_READY_FOR_LATENCY_IMPLEMENTATION"
    )
    body: Dict[str, Any] = {
        "name": "pdctp_fiqa_certification_family",
        "schema": "pdctp_fiqa_certification_family_v1",
        "version": 1,
        "role": "query_cert",
        "query_order_hash": fingerprint(query_ids),
        "query_count": len(query_ids),
        "query_record_fingerprints_hash": fingerprint(
            [str(row["fingerprint"]) for row in records]
        ),
        "policy_suite_fingerprint": policy_suite_fingerprint,
        "hypotheses_fingerprint": hypotheses["fingerprint"],
        "family_wise_method": certification["family_wise_method"],
        "family_wise_alpha": certification["family_wise_alpha"],
        "bounds": bounds,
        "all_passed": all_passed,
        "terminal": True,
        "calibrator_refit": False,
        "selection_performed": False,
        "budget_expanded": False,
        "failure_behavior": certification["failure_behavior"],
        "decision": decision,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def run_pdctp_fiqa_query_cert(
    config: PDCTPQueryCertConfig,
    real_protocol_config: PDCTPRealProtocolConfig,
    protocol_freeze_path: Union[str, Path],
    role_assignments_path: Union[str, Path],
    source_audit_path: Union[str, Path],
    source_archive_path: Union[str, Path],
    embedding_audit_path: Union[str, Path],
    embedding_config_path: Union[str, Path],
    prepared_dir: Union[str, Path],
    cache_dir: Union[str, Path],
    query_tune_audit_path: Union[str, Path],
    query_tune_run_dir: Union[str, Path],
    power_plan_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    progress: bool = True,
    preflight_only: bool = False,
) -> Dict[str, Path]:
    """Validate the frozen suite, open cert exactly once, and close terminally."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite query_cert run: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = _load_fingerprinted(Path(protocol_freeze_path), "protocol freeze")
    roles = _load_fingerprinted(Path(role_assignments_path), "role assignments")
    source_path = Path(source_audit_path)
    source_audit = _load_fingerprinted(source_path, "source audit")
    embedding_audit = _load_fingerprinted(Path(embedding_audit_path), "embedding audit")
    embedding_config = load_text_embedding_config(Path(embedding_config_path))
    validated_cache = validate_text_embedding_cache(
        embedding_config, Path(prepared_dir), Path(cache_dir)
    )
    dataset_manifest = validated_cache["dataset_manifest"]
    embedding_manifest = validated_cache["embedding_manifest"]
    tune_audit_path = Path(query_tune_audit_path)
    tune_audit = _load_fingerprinted(tune_audit_path, "query_tune audit")
    tune_run = Path(query_tune_run_dir)
    tune_files = {name: tune_run / name for name in _TUNE_FILES}
    missing_tune = [name for name, path in tune_files.items() if not path.is_file()]
    if missing_tune:
        raise PDCTPQueryCertError(
            f"query_tune run is incomplete: missing={missing_tune}"
        )
    tune_identities = {name: _file_identity(path) for name, path in tune_files.items()}
    tune_manifest = _load_fingerprinted(
        tune_files["manifest.json"], "query_tune manifest"
    )
    tune_state = _load_fingerprinted(
        tune_files["protocol_state_after_query_tune.json"], "query_tune state"
    )
    tune_access = _load_fingerprinted(
        tune_files["query_tune_access.json"], "query_tune access"
    )
    tune_projection = _load_fingerprinted(
        tune_files["projection.json"], "query_tune projection"
    )
    selection = _load_fingerprinted(tune_files["selection.json"], "selection")
    policies_artifact = _load_fingerprinted(tune_files["policies.json"], "policies")
    component_registry = _load_fingerprinted(
        tune_files["selected_policy_components.json"], "component registry"
    )
    policy_suite = _load_fingerprinted(
        tune_files["frozen_policy_suite.json"], "policy suite"
    )
    hypotheses = _load_fingerprinted(tune_files["hypotheses.json"], "hypotheses")
    power_plan = _load_fingerprinted(Path(power_plan_path), "power plan")

    if real_protocol_config.config_fingerprint != protocol.get("config_fingerprint"):
        raise PDCTPQueryCertError("real protocol config differs from protocol freeze")
    if real_protocol_config.raw != protocol.get("protocol"):
        raise PDCTPQueryCertError("embedded real protocol differs from checked config")
    if (
        embedding_config.config_fingerprint
        != config.raw["bindings"]["embedding_config_fingerprint"]
    ):
        raise PDCTPQueryCertError("embedding config fingerprint changed")
    if (
        power_plan.get("fingerprint")
        != config.raw["bindings"]["power_plan_fingerprint"]
    ):
        raise PDCTPQueryCertError("power plan fingerprint changed")
    expected_power = make_power_plan(
        real_protocol_config.raw["certification"]["hypotheses"],
        total_alpha=real_protocol_config.raw["certification"]["family_wise_alpha"],
    )
    if expected_power != power_plan:
        raise PDCTPQueryCertError("power plan no longer matches frozen hypotheses")
    assignments, guard, policies = validate_query_cert_documents(
        config,
        protocol,
        roles,
        source_audit,
        embedding_audit,
        dataset_manifest,
        embedding_manifest,
        tune_audit,
        tune_manifest,
        tune_state,
        tune_access,
        tune_projection,
        selection,
        policies_artifact,
        component_registry,
        policy_suite,
        hypotheses,
        source_audit_sha256=_file_identity(source_path)["sha256"],
        query_tune_audit_sha256=_file_identity(tune_audit_path)["sha256"],
        query_tune_file_identities=tune_identities,
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
        raise PDCTPQueryCertError("FiQA source archive identity changed")
    if preflight_only:
        if guard.serialize() != tune_state:
            raise AssertionError("preflight changed the closed post-tune state")
        return {}

    # No certification identity, qrel, or metric is opened before this point.
    cert_ids = assignments.ids_by_role["query_cert"]
    cert_token = guard.open_certification(cert_ids)
    member_metadata = source_audit["source"]["members"][
        config.raw["access"]["qrel_member"]
    ]
    qrels, qrel_access = load_query_cert_qrels(
        source_archive,
        member_metadata,
        cert_ids,
        minimum_relevance=real_protocol_config.raw["dataset"]["minimum_relevance"],
    )

    cache = Path(cache_dir)
    corpus_ids = json.loads((cache / "corpus_ids.json").read_text(encoding="utf-8"))
    all_query_ids = json.loads((cache / "query_ids.json").read_text(encoding="utf-8"))
    if not isinstance(corpus_ids, list) or not isinstance(all_query_ids, list):
        raise PDCTPQueryCertError("embedding stable-ID artifacts are invalid")
    query_row = {query_id: index for index, query_id in enumerate(all_query_ids)}
    if len(query_row) != len(all_query_ids) or any(
        query_id not in query_row for query_id in cert_ids
    ):
        raise PDCTPQueryCertError("query_cert IDs do not map uniquely into the cache")
    cert_rows = np.asarray(
        [query_row[query_id] for query_id in cert_ids], dtype=np.int64
    )
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
    query_cert_embeddings = np.asarray(query_embeddings[cert_rows], dtype=np.float64)
    records, projection = build_query_cert_records(
        real_protocol_config.raw,
        cert_ids,
        corpus_ids,
        corpus_embeddings,
        query_cert_embeddings,
        qrels,
        policies,
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
    if any(
        projection.get(key) != tune_projection.get(key)
        for key in projection_shared_keys
    ):
        raise PDCTPQueryCertError(
            "query_cert did not reuse the frozen query_tune projection"
        )
    certification = make_query_certification(
        real_protocol_config.raw,
        records,
        policies,
        hypotheses,
        str(policy_suite["fingerprint"]),
    )
    guard.close_certification(cert_token, str(certification["fingerprint"]))
    access: Dict[str, Any] = {
        "name": "pdctp_fiqa_query_cert_access",
        "schema": "pdctp_fiqa_query_cert_access_v1",
        "version": 1,
        "token": cert_token.serialize(),
        "allowed_supervision": config.raw["access"]["allowed_supervision"],
        "roles_with_outcomes_opened": ["query_cal", "query_tune", "query_cert"],
        "roles_remaining_closed": config.raw["access"]["blocked_roles"],
        "query_cert_qrels_accessed": True,
        "non_cert_qrel_outcomes_parsed": False,
        "calibrator_refit": False,
        "selection_performed": False,
        "certification_terminal": True,
        "certification_result_fingerprint": certification["fingerprint"],
    }
    access["fingerprint"] = fingerprint(access)

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        write_json(temporary / "query_cert_access.json", access)
        write_json(temporary / "query_cert_qrel_access.json", qrel_access)
        write_json(temporary / "projection.json", projection)
        _write_jsonl(
            temporary / "query_cert_records.jsonl",
            (_public_record(record) for record in records),
        )
        write_json(temporary / "certification.json", certification)
        write_json(
            temporary / "protocol_state_after_query_cert.json", guard.serialize()
        )
        artifact_names = (
            "query_cert_access.json",
            "query_cert_qrel_access.json",
            "projection.json",
            "query_cert_records.jsonl",
            "certification.json",
            "protocol_state_after_query_cert.json",
        )
        means = {
            family: {
                "mean_budget": float(
                    np.mean([row["methods"][family]["chosen_m"] for row in records])
                ),
                "mean_embedding_retention": float(
                    np.mean(
                        [
                            row["methods"][family]["embedding_retention"]
                            for row in records
                        ]
                    )
                ),
                "mean_candidate_evidence_recall": float(
                    np.mean(
                        [
                            row["methods"][family]["candidate_evidence_recall"]
                            for row in records
                        ]
                    )
                ),
                "mean_final_evidence_recall": float(
                    np.mean(
                        [
                            row["methods"][family]["final_evidence_recall"]
                            for row in records
                        ]
                    )
                ),
            }
            for family in _METHOD_ORDER
        }
        manifest: Dict[str, Any] = {
            "name": "pdctp_fiqa_query_cert_manifest",
            "schema": "pdctp_fiqa_query_cert_manifest_v1",
            "version": 1,
            "config_fingerprint": config.config_fingerprint,
            "upstream": dict(config.raw["bindings"]),
            "query_cert_access_fingerprint": access["fingerprint"],
            "qrel_access_fingerprint": qrel_access["fingerprint"],
            "projection_fingerprint": projection["fingerprint"],
            "policy_suite_fingerprint": policy_suite["fingerprint"],
            "hypotheses_fingerprint": hypotheses["fingerprint"],
            "certification_fingerprint": certification["fingerprint"],
            "protocol_state_fingerprint": guard.state_fingerprint,
            "counts": {
                "query_cert": len(records),
                "positive_qrel_rows": qrel_access["scoped_positive_rows"],
                "valid_pilot_lid": sum(row["pilot"]["lid_valid"] for row in records),
                "valid_pilot_features": sum(
                    row["_features_obj"].valid for row in records
                ),
                "frozen_methods": len(_METHOD_ORDER),
                "paired_hypotheses": len(certification["bounds"]),
            },
            "method_means": means,
            "checks": {
                "all_upstream_fingerprints_validated_before_query_cert": True,
                "complete_frozen_policy_suite_reconstructed_before_query_cert": True,
                "frozen_hypotheses_validated_before_query_cert": True,
                "only_query_cert_newly_opened": True,
                "all_query_cert_ids_used_in_frozen_order": True,
                "policy_decisions_used_only_deployable_pilot_inputs": True,
                "policy_decisions_completed_before_query_supervision": all(
                    row["policy_decisions_completed_before_supervision"]
                    for row in records
                ),
                "query_cert_qrels_accessed": True,
                "non_cert_qrel_outcomes_parsed": False,
                "query_latency_accessed": False,
                "query_test_accessed": False,
                "calibrator_refit": False,
                "selection_performed": False,
                "budget_expanded": False,
                "all_six_paired_bounds_reconstructable": len(certification["bounds"])
                == 6,
                "certification_terminal": True,
                "one_projected_scan_per_query": all(
                    row["shared_work"]["projected_scan_count"] == 1 for row in records
                ),
                "projected_vectors_renormalized": False,
                "llm_run": False,
                "approximate_index_used": False,
                "raw_tri_predict_v1_behavior_modified": False,
                "latency_measured": False,
            },
            "artifacts": {
                name: _file_identity(temporary / name) for name in artifact_names
            },
            "decision": certification["decision"],
        }
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(temporary / "manifest.json", manifest)
        failed = [
            name
            for name, bound in certification["bounds"].items()
            if not bound["passed"]
        ]
        report = (
            "# PDCTP FiQA query_cert certification gate\n\n"
            f"Decision: `{certification['decision']}`.\n\n"
            f"Evaluated all six frozen policies on the complete {len(records):,}-ID "
            "`query_cert` role and reconstructed all six Bonferroni-adjusted "
            "paired empirical-Bernstein bounds.\n\n"
            f"Failed hypotheses: `{failed}`.\n\n"
            "Certification is terminal whether it passes or fails: no refit, "
            "retuning, budget expansion, comparator substitution, or repeat cert "
            "is allowed. `query_latency` and `query_test` remain closed. No LLM "
            "or approximate index was used.\n"
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
    parser.add_argument("--query-tune-audit", required=True, type=Path)
    parser.add_argument("--query-tune-run", required=True, type=Path)
    parser.add_argument("--power-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate every upstream input and stop before opening query_cert",
    )
    args = parser.parse_args(argv)
    paths = run_pdctp_fiqa_query_cert(
        load_pdctp_query_cert_config(args.config),
        load_pdctp_real_protocol_config(args.real_protocol_config),
        args.protocol_freeze,
        args.role_assignments,
        args.source_audit,
        args.fiqa_archive,
        args.embedding_audit,
        args.embedding_config,
        args.prepared,
        args.embedding_cache,
        args.query_tune_audit,
        args.query_tune_run,
        args.power_plan,
        args.output,
        preflight_only=args.preflight_only,
    )
    if args.preflight_only:
        print("PDCTP FiQA query_cert preflight passed; query_cert remains closed")
    else:
        print(
            f"PDCTP FiQA query_cert gate wrote {len(paths)} artifacts to {args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
