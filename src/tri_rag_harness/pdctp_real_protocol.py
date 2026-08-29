"""Freeze the FiQA PDCTP protocol before embeddings or role access."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from .pdctp_features import PilotDistanceFeatureSpec
from .pdctp_protocol import (
    FIVE_ROLES,
    FiveRoleAssignments,
    FiveRoleProtocolGuard,
    LeakageError,
)
from .utils import fingerprint, write_json


class PDCTPRealProtocolError(ValueError):
    pass


_EXPECTED_GRID = (
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    57638,
)
_PRIMARY_HYPOTHESES = (
    "pdctp_absolute_retention",
    "candidate_evidence_noninferiority_fixed",
    "final_evidence_noninferiority_fixed",
    "normalized_budget_superiority_fixed",
    "normalized_budget_superiority_monotone",
    "normalized_budget_superiority_raw_tri",
)
_LATENCY_COMPARISONS = (
    "pdctp_vs_fixed_reference",
    "pdctp_vs_monotone_binned",
    "pdctp_vs_raw_tri_predict",
)


@dataclass(frozen=True)
class PDCTPRealProtocolConfig:
    raw: Mapping[str, Any]
    config_fingerprint: str
    run_name: str
    feature_spec: PilotDistanceFeatureSpec
    budget_grid: Tuple[int, ...]


def _exact(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if not isinstance(value, Mapping):
        raise PDCTPRealProtocolError(f"{context} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise PDCTPRealProtocolError(
            f"{context} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PDCTPRealProtocolError(f"{name} must be a nonempty string")
    return value


def _hex(value: Any, name: str, length: int = 64) -> str:
    result = _string(value, name)
    if len(result) != length or any(c not in "0123456789abcdef" for c in result):
        raise PDCTPRealProtocolError(
            f"{name} must contain exactly {length} lowercase hex digits"
        )
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PDCTPRealProtocolError(f"{name} must be an integer at least {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PDCTPRealProtocolError(f"{name} must be numeric")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise PDCTPRealProtocolError(f"{name} must be finite")
    return result


def _unique_numbers(value: Any, name: str) -> Tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise PDCTPRealProtocolError(f"{name} must be a nonempty list")
    result = tuple(_number(item, name) for item in value)
    if len(set(result)) != len(result):
        raise PDCTPRealProtocolError(f"{name} must contain unique values")
    return result


def derive_fiqa_budget_grid(corpus_size: int, minimum_budget: int) -> Tuple[int, ...]:
    if corpus_size < minimum_budget or minimum_budget < 1:
        raise PDCTPRealProtocolError("budget-grid endpoints are invalid")
    values = []
    base = minimum_budget
    while base < corpus_size:
        for candidate in (base, 3 * base // 2):
            if candidate < corpus_size and candidate not in values:
                values.append(candidate)
        base *= 2
    values.append(corpus_size)
    return tuple(values)


def load_pdctp_real_protocol_config(
    path: Union[str, Path],
) -> PDCTPRealProtocolConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PDCTPRealProtocolError(
            f"cannot load real protocol config {config_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise PDCTPRealProtocolError("real protocol config root must be an object")
    _exact(
        raw,
        {
            "schema",
            "version",
            "run_name",
            "source_gate",
            "dataset",
            "roles",
            "embedding",
            "seeds",
            "retrieval",
            "features",
            "candidate_suite",
            "selection",
            "certification",
            "latency",
            "stop_gates",
        },
        "root",
    )
    if raw["schema"] != "pdctp_fiqa_real_protocol_freeze_v1" or raw["version"] != 1:
        raise PDCTPRealProtocolError("unsupported real protocol schema")
    run_name = _string(raw["run_name"], "run_name")

    source = raw["source_gate"]
    _exact(
        source,
        {
            "audit_config_fingerprint",
            "source_audit_fingerprint",
            "source_audit_sha256",
            "role_witness_fingerprint",
            "role_witness_sha256",
            "power_plan_fingerprint",
            "power_plan_sha256",
        },
        "source_gate",
    )
    for key, value in source.items():
        _hex(value, f"source_gate.{key}")

    dataset = raw["dataset"]
    _exact(
        dataset,
        {
            "schema",
            "name",
            "version",
            "id_namespace",
            "archive_sha256",
            "corpus_size",
            "source_query_count",
            "minimum_relevance",
            "external_query_ids",
            "empty_documents",
        },
        "dataset",
    )
    if dataset["schema"] != "pdctp_fiqa_dataset_contract_v1":
        raise PDCTPRealProtocolError("unsupported dataset contract")
    if dataset["name"] != "BEIR FiQA-2018" or dataset["id_namespace"] != "pdctp-beir-fiqa":
        raise PDCTPRealProtocolError("FiQA dataset identity changed")
    _hex(dataset["archive_sha256"], "dataset.archive_sha256")
    if _integer(dataset["corpus_size"], "dataset.corpus_size", 1) != 57638:
        raise PDCTPRealProtocolError("FiQA corpus size changed")
    if _integer(dataset["source_query_count"], "dataset.source_query_count", 1) != 6648:
        raise PDCTPRealProtocolError("FiQA source query count changed")
    if dataset["minimum_relevance"] != 1 or dataset["external_query_ids"] is not True:
        raise PDCTPRealProtocolError("dataset relevance or external-query contract changed")
    empty = dataset["empty_documents"]
    _exact(
        empty,
        {
            "source_count",
            "positive_qrel_references_by_native_split",
            "policy",
            "replacement_text",
            "formatted_embedding_text",
            "silent_deletion_allowed",
        },
        "dataset.empty_documents",
    )
    if _integer(empty["source_count"], "empty source count", 1) != 38:
        raise PDCTPRealProtocolError("empty-document count changed")
    if empty["positive_qrel_references_by_native_split"] != {
        "train": 35,
        "dev": 2,
        "test": 1,
    }:
        raise PDCTPRealProtocolError("empty-document qrel counts changed")
    if (
        empty["policy"] != "replace_empty_title_and_text_with_marker_v1"
        or empty["replacement_text"] != "[EMPTY_DOCUMENT]"
        or empty["formatted_embedding_text"] != "passage: [EMPTY_DOCUMENT]"
        or empty["silent_deletion_allowed"] is not False
    ):
        raise PDCTPRealProtocolError("empty-document representation changed")

    roles = raw["roles"]
    _exact(
        roles,
        {
            "schema",
            "assignment_seed",
            "source_witness_fingerprint",
            "counts",
            "ordered_id_hashes",
            "normalized_text_groups_disjoint",
            "initial_state",
        },
        "roles",
    )
    if roles["schema"] != "pdctp_fiqa_five_roles_freeze_v1":
        raise PDCTPRealProtocolError("unsupported role-freeze schema")
    if roles["assignment_seed"] != 62419:
        raise PDCTPRealProtocolError("role assignment seed changed")
    _hex(roles["source_witness_fingerprint"], "roles.source_witness_fingerprint")
    _exact(roles["counts"], set(FIVE_ROLES), "roles.counts")
    _exact(roles["ordered_id_hashes"], set(FIVE_ROLES), "roles.ordered_id_hashes")
    for role in FIVE_ROLES:
        _integer(roles["counts"][role], f"roles.counts.{role}", 1)
        _hex(roles["ordered_id_hashes"][role], f"roles.ordered_id_hashes.{role}")
    if roles["counts"] != {
        "query_cal": 1966,
        "query_tune": 1967,
        "query_cert": 1567,
        "query_latency": 500,
        "query_test": 648,
    }:
        raise PDCTPRealProtocolError("five-role counts changed")
    if roles["normalized_text_groups_disjoint"] is not True or roles["initial_state"] != "all_roles_closed":
        raise PDCTPRealProtocolError("role isolation or initial state changed")

    embedding = raw["embedding"]
    _exact(
        embedding,
        {
            "schema",
            "continuity_source_config_sha256",
            "model",
            "formatting",
            "encoding",
            "runtime_packages",
        },
        "embedding",
    )
    if embedding["schema"] != "pdctp_fiqa_e5_embedding_request_v1":
        raise PDCTPRealProtocolError("unsupported embedding request schema")
    continuity_hash = _hex(
        embedding["continuity_source_config_sha256"],
        "embedding.continuity_source_config_sha256",
    )
    if continuity_hash != "cc18a2ebfe02a38ddfc859cc2398349e59fbdaca2b38b9880fc41e284981aedd":
        raise PDCTPRealProtocolError("E5 continuity source changed")
    model = embedding["model"]
    _exact(
        model,
        {
            "name",
            "revision",
            "embedding_dimension",
            "max_sequence_length",
            "trust_remote_code",
        },
        "embedding.model",
    )
    if (
        model["name"] != "intfloat/e5-base-v2"
        or model["revision"] != "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
        or model["embedding_dimension"] != 768
        or model["max_sequence_length"] != 512
        or model["trust_remote_code"] is not False
    ):
        raise PDCTPRealProtocolError("E5 model identity changed")
    if embedding["formatting"] != {
        "corpus_prefix": "passage: ",
        "query_prefix": "query: ",
        "title_text_separator": "\n",
        "strip_fields": True,
    }:
        raise PDCTPRealProtocolError("E5 formatting changed")
    encoding = embedding["encoding"]
    _exact(
        encoding,
        {
            "model_dtype",
            "output_dtype",
            "l2_normalize",
            "deterministic_algorithms",
            "allow_tf32",
            "attention_implementation",
            "cublas_workspace_config",
        },
        "embedding.encoding",
    )
    if encoding != {
        "model_dtype": "float32",
        "output_dtype": "float32",
        "l2_normalize": True,
        "deterministic_algorithms": True,
        "allow_tf32": False,
        "attention_implementation": "eager",
        "cublas_workspace_config": ":4096:8",
    }:
        raise PDCTPRealProtocolError("embedding numerical contract changed")
    if embedding["runtime_packages"] != {
        "huggingface-hub": "0.28.1",
        "numpy": "1.26.4",
        "safetensors": "0.5.2",
        "sentence-transformers": "3.4.1",
        "scipy": "1.13.0",
        "tokenizers": "0.21.0",
        "torch": "2.5.1",
        "transformers": "4.48.2",
    }:
        raise PDCTPRealProtocolError("embedding runtime packages changed")

    seeds = raw["seeds"]
    _exact(
        seeds,
        {"projection", "shuffled_profile", "latency_method_order"},
        "seeds",
    )
    for key, value in seeds.items():
        _integer(value, f"seeds.{key}")
    if seeds != {
        "projection": 83047,
        "shuffled_profile": 83059,
        "latency_method_order": 83071,
    } or len(set(seeds.values())) != len(seeds):
        raise PDCTPRealProtocolError("stochastic seeds must be distinct")

    retrieval = raw["retrieval"]
    _exact(
        retrieval,
        {
            "schema",
            "corpus_size",
            "embedding_dimension",
            "input_normalization",
            "projection",
            "distance",
            "stable_tie_break",
            "query_batch_size",
            "k_gt",
            "k_ctx",
            "m_pilot",
            "s_lid",
            "min_lid_neighbors",
            "budget_grid_rule",
            "m_grid",
            "max_rank_samples",
            "pilot_expansion_reuse",
            "cost_objective",
        },
        "retrieval",
    )
    if retrieval["schema"] != "pdctp_fiqa_exact_retrieval_v1":
        raise PDCTPRealProtocolError("unsupported retrieval schema")
    if retrieval["corpus_size"] != 57638 or retrieval["embedding_dimension"] != 768:
        raise PDCTPRealProtocolError("retrieval shape changed")
    if retrieval["input_normalization"] != "l2_before_projection":
        raise PDCTPRealProtocolError("input normalization changed")
    projection = retrieval["projection"]
    _exact(
        projection,
        {"family", "variance", "seed", "m_prime", "post_projection_normalize"},
        "retrieval.projection",
    )
    if projection != {
        "family": "dense_gaussian",
        "variance": "1/m_prime",
        "seed": seeds["projection"],
        "m_prime": 192,
        "post_projection_normalize": False,
    }:
        raise PDCTPRealProtocolError("projection contract changed")
    if retrieval["distance"] != {
        "original": "squared_l2",
        "projected": "squared_l2",
    }:
        raise PDCTPRealProtocolError("distance contract changed")
    if retrieval["stable_tie_break"] != "lexicographic_doc_id":
        raise PDCTPRealProtocolError("tie contract changed")
    for key in (
        "query_batch_size",
        "k_gt",
        "k_ctx",
        "m_pilot",
        "s_lid",
        "min_lid_neighbors",
        "max_rank_samples",
    ):
        _integer(retrieval[key], f"retrieval.{key}", 1)
    if (
        retrieval["k_gt"] != 10
        or retrieval["k_ctx"] != 5
        or retrieval["query_batch_size"] != 32
        or retrieval["m_pilot"] != 64
        or retrieval["s_lid"] != 32
        or retrieval["min_lid_neighbors"] != 16
        or retrieval["max_rank_samples"] != 256
    ):
        raise PDCTPRealProtocolError("pilot or rank-sampling contract changed")
    grid_rule = retrieval["budget_grid_rule"]
    _exact(
        grid_rule,
        {"name", "minimum_budget", "per_octave_multipliers", "terminal_full_corpus"},
        "retrieval.budget_grid_rule",
    )
    if grid_rule != {
        "name": "binary_interleaved_geometric_v1",
        "minimum_budget": 64,
        "per_octave_multipliers": [1.0, 1.5],
        "terminal_full_corpus": True,
    }:
        raise PDCTPRealProtocolError("budget-grid derivation rule changed")
    budget_grid = tuple(
        _integer(value, "retrieval.m_grid", 1) for value in retrieval["m_grid"]
    )
    derived_grid = derive_fiqa_budget_grid(57638, 64)
    if budget_grid != derived_grid or budget_grid != _EXPECTED_GRID:
        raise PDCTPRealProtocolError("budget grid does not match its frozen rule")
    if retrieval["pilot_expansion_reuse"] != "one_projected_scan":
        raise PDCTPRealProtocolError("pilot/expansion scan reuse changed")
    if retrieval["cost_objective"] != {
        "name": "common_coordinate_work",
        "formula": "(corpus_size + embedding_dimension) * m_prime + embedding_dimension * M",
    }:
        raise PDCTPRealProtocolError("common cost objective changed")

    features = raw["features"]
    _exact(
        features,
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
            schema=features["schema"],
            lid_boundary=features["lid_boundary"],
            minimum_count=features["minimum_count"],
            gap_quantiles=tuple(features["gap_quantiles"]),
            epsilon=features["epsilon"],
            duplicate_tolerance=features["duplicate_tolerance"],
            invalid_fill=features["invalid_fill"],
            output_decimals=features["output_decimals"],
        )
    except ValueError as exc:
        raise PDCTPRealProtocolError(str(exc)) from exc
    if feature_spec.lid_boundary != 32 or feature_spec.minimum_count != 32:
        raise PDCTPRealProtocolError("feature/pilot boundary changed")

    candidates = raw["candidate_suite"]
    _exact(
        candidates,
        {
            "schema",
            "methods",
            "fixed_budgets",
            "monotone_binned",
            "raw_tri_threshold_grid",
            "lid_regularization_grid",
            "lid_output_domain",
            "lid_fallback",
            "residual_training_levels",
            "residual_quantiles",
            "residual_regularization_grid",
            "safety_offsets",
            "expected_full_pdctp_tuples",
        },
        "candidate_suite",
    )
    if candidates["schema"] != "pdctp_fiqa_candidate_suite_v1":
        raise PDCTPRealProtocolError("unsupported candidate-suite schema")
    if candidates["methods"] != [
        "fixed",
        "monotone_binned",
        "raw_tri_predict",
        "lid_calibration_only",
        "budget_residual_only",
        "pdctp",
    ]:
        raise PDCTPRealProtocolError("method or ablation suite changed")
    if tuple(candidates["fixed_budgets"]) != budget_grid:
        raise PDCTPRealProtocolError("fixed budgets must equal the common grid")
    monotone = candidates["monotone_binned"]
    _exact(monotone, {"n_bins_grid", "bin_target_grid", "fallback_budget"}, "monotone")
    if monotone != {
        "n_bins_grid": [4, 6, 8],
        "bin_target_grid": [0.95, 0.97, 0.98, 0.99, 1.0],
        "fallback_budget": 57638,
    }:
        raise PDCTPRealProtocolError("monotone candidate grid changed")
    threshold_grid = _unique_numbers(
        candidates["raw_tri_threshold_grid"], "raw_tri_threshold_grid"
    )
    lid_grid = _unique_numbers(
        candidates["lid_regularization_grid"], "lid_regularization_grid"
    )
    training_grid = _unique_numbers(
        candidates["residual_training_levels"], "residual_training_levels"
    )
    quantile_grid = _unique_numbers(
        candidates["residual_quantiles"], "residual_quantiles"
    )
    residual_grid = _unique_numbers(
        candidates["residual_regularization_grid"], "residual_regularization_grid"
    )
    safety_grid = _unique_numbers(candidates["safety_offsets"], "safety_offsets")
    if any(not 0.0 < value <= 1.0 for value in threshold_grid + training_grid):
        raise PDCTPRealProtocolError("threshold/training levels must lie in (0,1]")
    if any(not 0.0 < value < 1.0 for value in quantile_grid):
        raise PDCTPRealProtocolError("residual quantiles must lie in (0,1)")
    if any(value < 0.0 for value in lid_grid + residual_grid + safety_grid):
        raise PDCTPRealProtocolError("regularization and safety grids must be nonnegative")
    if candidates["lid_output_domain"] != [1.0, 100.0] or candidates["lid_fallback"] != 100.0:
        raise PDCTPRealProtocolError("LID domain or fallback changed")
    expected_tuple_count = (
        len(threshold_grid)
        * len(lid_grid)
        * len(training_grid)
        * len(quantile_grid)
        * len(residual_grid)
        * len(safety_grid)
    )
    if candidates["expected_full_pdctp_tuples"] != expected_tuple_count or expected_tuple_count != 1620:
        raise PDCTPRealProtocolError("PDCTP candidate tuple count changed")

    selection = raw["selection"]
    _exact(
        selection,
        {
            "schema",
            "role",
            "retention_lower_bound_target",
            "candidate_evidence_noninferiority",
            "final_evidence_noninferiority",
            "tune_bound_alpha",
            "objective",
            "tie_breaks",
            "comparator_eligibility",
            "shuffled_profile_scope",
        },
        "selection",
    )
    if selection != {
        "schema": "pdctp_fiqa_selection_v1",
        "role": "query_tune",
        "retention_lower_bound_target": 0.95,
        "candidate_evidence_noninferiority": 0.02,
        "final_evidence_noninferiority": 0.02,
        "tune_bound_alpha": 0.05,
        "objective": "common_coordinate_work",
        "tie_breaks": ["lower_mean_budget", "canonical_fingerprint"],
        "comparator_eligibility": "must_meet_same_retention_and_evidence_constraints",
        "shuffled_profile_scope": "query_tune_diagnostic_only",
    }:
        raise PDCTPRealProtocolError("selection rule changed")

    certification = raw["certification"]
    _exact(
        certification,
        {
            "schema",
            "role",
            "required_query_count",
            "family_wise_method",
            "family_wise_alpha",
            "hypotheses",
            "evidence",
            "failure_behavior",
        },
        "certification",
    )
    if (
        certification["schema"] != "pdctp_fiqa_certification_v1"
        or certification["role"] != "query_cert"
        or certification["required_query_count"] != 1567
        or certification["family_wise_method"] != "bonferroni"
        or certification["family_wise_alpha"] != 0.05
        or certification["failure_behavior"] != "terminal_no_retuning_no_budget_expansion"
    ):
        raise PDCTPRealProtocolError("certification protocol changed")
    hypotheses = certification["hypotheses"]
    if not isinstance(hypotheses, list) or [row.get("name") for row in hypotheses] != list(
        _PRIMARY_HYPOTHESES
    ):
        raise PDCTPRealProtocolError("primary hypothesis family changed")
    for row in hypotheses:
        _exact(
            row,
            {"name", "metric", "comparison", "side", "margin", "difference_bounds", "desired_radius"},
            f"hypothesis {row.get('name')}",
        )
    if certification["evidence"] != {
        "minimum_relevance": 1,
        "candidate_metric": "macro_qrel_recall_in_projected_candidates",
        "final_metric": "macro_qrel_recall_after_exact_rerank_at_k_ctx",
        "empty_documents_remain_relevant": True,
    }:
        raise PDCTPRealProtocolError("evidence metric contract changed")

    latency = raw["latency"]
    _exact(
        latency,
        {
            "schema",
            "role",
            "labels_allowed",
            "backends",
            "hardware",
            "required_packages",
            "warmups",
            "repetitions",
            "randomized_paired_blocks",
            "method_order_seed",
            "threads",
            "boundary_tie_overfetch",
            "gpu_k_selection_limit",
            "gpu_max_compatible_budget",
            "gpu_incompatibility_behavior",
            "batching",
            "cache_state",
            "measured_methods",
            "primary_statistic",
            "family_wise_method",
            "family_wise_alpha",
            "primary_comparisons",
            "tail_statistics_role",
            "required_stage_metrics",
        },
        "latency",
    )
    if (
        latency["schema"] != "pdctp_fiqa_paired_latency_v1"
        or latency["role"] != "query_latency"
        or latency["labels_allowed"] is not False
        or latency["backends"] != ["faiss_cpu_exact", "faiss_gpu_exact"]
        or latency["warmups"] != 10
        or latency["repetitions"] != 30
        or latency["randomized_paired_blocks"] is not True
        or latency["method_order_seed"] != seeds["latency_method_order"]
        or latency["threads"] != 1
        or latency["boundary_tie_overfetch"] != 64
        or latency["gpu_k_selection_limit"] != 2048
        or latency["gpu_max_compatible_budget"] != 1984
        or latency["gpu_incompatibility_behavior"] != "terminal_no_smaller_budget_substitution"
        or latency["batching"] != "single_query"
        or latency["cache_state"] != "warm_index"
    ):
        raise PDCTPRealProtocolError("latency execution contract changed")
    if latency["gpu_max_compatible_budget"] + latency["boundary_tie_overfetch"] != latency["gpu_k_selection_limit"]:
        raise PDCTPRealProtocolError("GPU boundary guard arithmetic changed")
    if latency["hardware"] != {
        "cpu_class": "genoa_single_node_exclusive",
        "gpu_class": "nvidia_a100_sxm4_80gb_exclusive",
        "gpu_device_count": 1,
    }:
        raise PDCTPRealProtocolError("latency hardware contract changed")
    if latency["required_packages"] != {
        "numpy": "1.26.4",
        "scipy": "1.13.0",
        "faiss": "1.10.0",
        "torch": "2.5.1",
    }:
        raise PDCTPRealProtocolError("latency package contract changed")
    if latency["measured_methods"] != candidates["methods"]:
        raise PDCTPRealProtocolError("latency must measure the complete policy suite")
    if (
        latency["primary_statistic"] != "paired_student_t_upper_bound_on_query_mean_seconds_v1"
        or latency["family_wise_method"] != "bonferroni"
        or latency["family_wise_alpha"] != 0.05
        or latency["primary_comparisons"] != list(_LATENCY_COMPARISONS)
        or latency["tail_statistics_role"] != "descriptive_only"
    ):
        raise PDCTPRealProtocolError("latency inference contract changed")
    if latency["required_stage_metrics"] != [
        "project_query",
        "projected_scan",
        "pilot_rerank",
        "feature_extract",
        "lid_calibration",
        "raw_tri",
        "residual_calibration",
        "expansion_rerank",
        "total",
        "distance_counts",
        "bytes",
        "cpu_rss",
        "gpu_memory",
        "index_build",
        "setup",
    ]:
        raise PDCTPRealProtocolError("latency stage metrics changed")

    stop_gates = raw["stop_gates"]
    _exact(
        stop_gates,
        {
            "all_roles_initially_closed",
            "next_allowed_action",
            "method_evaluation_authorized",
            "protected_outcome_access_authorized",
            "llm_authorized",
            "approximate_index_authorized",
        },
        "stop_gates",
    )
    if stop_gates != {
        "all_roles_initially_closed": True,
        "next_allowed_action": "prepare_dataset_and_build_independently_audited_embedding_cache",
        "method_evaluation_authorized": False,
        "protected_outcome_access_authorized": False,
        "llm_authorized": False,
        "approximate_index_authorized": False,
    }:
        raise PDCTPRealProtocolError("stop/go boundary changed")

    return PDCTPRealProtocolConfig(
        raw=raw,
        config_fingerprint=fingerprint(raw),
        run_name=run_name,
        feature_spec=feature_spec,
        budget_grid=budget_grid,
    )


def _file_hash(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _load_fingerprinted(path: Path, name: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PDCTPRealProtocolError(f"cannot load {name}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("fingerprint"), str):
        raise PDCTPRealProtocolError(f"{name} is not a fingerprinted object")
    body = dict(value)
    claimed = body.pop("fingerprint")
    if fingerprint(body) != claimed:
        raise PDCTPRealProtocolError(f"{name} fingerprint mismatch")
    return value


def _write_report(path: Path, freeze: Mapping[str, Any]) -> None:
    role_counts = freeze["resolved_roles"]["counts"]
    path.write_text(
        "# PDCTP FiQA real protocol freeze\n\n"
        f"Protocol fingerprint: `{freeze['fingerprint']}`.\n\n"
        "Decision: `READY_FOR_DATASET_AND_EMBEDDING_AUDIT_ONLY`.\n\n"
        f"Role counts: cal={role_counts['query_cal']}, "
        f"tune={role_counts['query_tune']}, cert={role_counts['query_cert']}, "
        f"latency={role_counts['query_latency']}, test={role_counts['query_test']}.\n\n"
        f"Frozen `m_prime=192`; budget grid has {len(freeze['protocol']['retrieval']['m_grid'])} "
        "values and terminates at the full 57,638-item corpus.\n\n"
        "All roles remain closed. No embedding, retrieval, method outcome, latency "
        "measurement, LLM, or approximate index was run by this freeze.\n",
        encoding="utf-8",
    )


def freeze_pdctp_real_protocol(
    config: PDCTPRealProtocolConfig,
    source_audit_path: Union[str, Path],
    role_witness_path: Union[str, Path],
    power_plan_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    source_path = Path(source_audit_path)
    witness_path = Path(role_witness_path)
    power_path = Path(power_plan_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite protocol freeze: {output}")

    source_hash = _file_hash(source_path)
    witness_hash = _file_hash(witness_path)
    power_hash = _file_hash(power_path)
    expected = config.raw["source_gate"]
    if source_hash["sha256"] != expected["source_audit_sha256"]:
        raise PDCTPRealProtocolError("source-audit file SHA-256 mismatch")
    if witness_hash["sha256"] != expected["role_witness_sha256"]:
        raise PDCTPRealProtocolError("role-witness file SHA-256 mismatch")
    if power_hash["sha256"] != expected["power_plan_sha256"]:
        raise PDCTPRealProtocolError("power-plan file SHA-256 mismatch")

    source = _load_fingerprinted(source_path, "source audit")
    witness = _load_fingerprinted(witness_path, "role witness")
    power = _load_fingerprinted(power_path, "power plan")
    if source["fingerprint"] != expected["source_audit_fingerprint"]:
        raise PDCTPRealProtocolError("source-audit identity changed")
    if source["config_fingerprint"] != expected["audit_config_fingerprint"]:
        raise PDCTPRealProtocolError("source-audit config identity changed")
    if witness["fingerprint"] != expected["role_witness_fingerprint"]:
        raise PDCTPRealProtocolError("role-witness identity changed")
    if power["fingerprint"] != expected["power_plan_fingerprint"]:
        raise PDCTPRealProtocolError("power-plan identity changed")
    if (
        source.get("decision") != "GO_TO_PROTOCOL_FREEZE"
        or source.get("scope_guards", {}).get("authorizes_method_evaluation") is not False
        or witness.get("authorizes_method_evaluation") is not False
    ):
        raise PDCTPRealProtocolError("source gate does not authorize this freeze")
    if source["artifacts"]["role_feasibility_witness.json"]["sha256"] != witness_hash["sha256"]:
        raise PDCTPRealProtocolError("source audit does not bind the role witness")

    roles = witness.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(FIVE_ROLES):
        raise PDCTPRealProtocolError("role witness must contain exactly five roles")
    ids_by_role: Dict[str, Tuple[str, ...]] = {}
    seen_ids: set[str] = set()
    for role in FIVE_ROLES:
        values = roles[role]
        if not isinstance(values, list) or not values or values != sorted(values):
            raise PDCTPRealProtocolError(f"{role} IDs must be nonempty and ordered")
        ids = tuple(_string(value, f"{role} ID") for value in values)
        if len(set(ids)) != len(ids) or seen_ids.intersection(ids):
            raise PDCTPRealProtocolError("role witness IDs are duplicated or crossed")
        if any(not value.startswith("pdctp-beir-fiqa:query:") for value in ids):
            raise PDCTPRealProtocolError("role witness ID namespace changed")
        seen_ids.update(ids)
        ids_by_role[role] = ids

    capacity = source["eligibility"]["five_role_capacity"]
    config_roles = config.raw["roles"]
    for role in FIVE_ROLES:
        if len(ids_by_role[role]) != capacity["role_counts"][role] or len(
            ids_by_role[role]
        ) != config_roles["counts"][role]:
            raise PDCTPRealProtocolError("role count differs from frozen identities")
        ordered_hash = fingerprint(list(ids_by_role[role]))
        if (
            ordered_hash != capacity["ordered_id_hashes"][role]
            or ordered_hash != config_roles["ordered_id_hashes"][role]
        ):
            raise PDCTPRealProtocolError("role ordered-ID hash changed")
    if (
        capacity["duplicate_normalized_text_groups"] != 0
        or capacity["normalized_texts_disjoint"] is not True
        or config_roles["source_witness_fingerprint"] != witness["fingerprint"]
    ):
        raise PDCTPRealProtocolError("normalized-text role isolation changed")

    try:
        assignments = FiveRoleAssignments(
            ids_by_role=ids_by_role,
            normalized_text_group_by_id={query_id: query_id for query_id in seen_ids},
        )
    except LeakageError as exc:
        raise PDCTPRealProtocolError(str(exc)) from exc
    assignment_base = assignments.serialize()
    role_artifact: Dict[str, Any] = {
        "schema": "pdctp_fiqa_five_role_assignments_v1",
        "version": 1,
        "source_audit_fingerprint": source["fingerprint"],
        "source_witness_fingerprint": witness["fingerprint"],
        "assignment_seed": config_roles["assignment_seed"],
        "roles": assignment_base["roles"],
        "ids_disjoint": True,
        "normalized_text_groups_disjoint": True,
        "normalized_group_validation": "source_audit_reports_zero_duplicate_groups",
        "all_roles_initially_closed": True,
        "authorizes_outcome_access": False,
    }
    role_artifact["fingerprint"] = fingerprint(role_artifact)

    core_fields = (
        "name",
        "metric",
        "comparison",
        "side",
        "margin",
        "difference_bounds",
        "desired_radius",
    )
    power_core = [
        {key: row[key] for key in core_fields} for row in power["hypotheses"]
    ]
    if power_core != config.raw["certification"]["hypotheses"]:
        raise PDCTPRealProtocolError("certification hypotheses differ from power plan")
    if (
        power["required_role_size"] != 1567
        or len(ids_by_role["query_cert"]) < power["required_role_size"]
        or power["family_wise_alpha"] != config.raw["certification"]["family_wise_alpha"]
    ):
        raise PDCTPRealProtocolError("certification role does not satisfy power plan")

    guard = FiveRoleProtocolGuard(assignments, config.config_fingerprint)
    state = guard.serialize()
    if any(
        state[key]
        for key in (
            "calibration_opened",
            "certification_opened",
            "latency_opened",
            "test_opened",
        )
    ):
        raise AssertionError("protocol freeze must leave every role closed")

    freeze: Dict[str, Any] = {
        "schema": "pdctp_fiqa_real_protocol_freeze_v1",
        "version": 1,
        "config_fingerprint": config.config_fingerprint,
        "protocol": config.raw,
        "resolved_inputs": {
            "source_audit": {**source_hash, "fingerprint": source["fingerprint"]},
            "role_witness": {**witness_hash, "fingerprint": witness["fingerprint"]},
            "power_plan": {**power_hash, "fingerprint": power["fingerprint"]},
        },
        "resolved_roles": {
            "assignment_fingerprint": role_artifact["fingerprint"],
            "counts": {role: len(ids_by_role[role]) for role in FIVE_ROLES},
            "ordered_id_hashes": {
                role: fingerprint(list(ids_by_role[role])) for role in FIVE_ROLES
            },
            "all_roles_closed": True,
        },
        "initial_guard_state_fingerprint": state["fingerprint"],
        "power_gate": {
            "required_query_cert": power["required_role_size"],
            "actual_query_cert": len(ids_by_role["query_cert"]),
            "passed": True,
        },
        "decision": "READY_FOR_DATASET_AND_EMBEDDING_AUDIT_ONLY",
        "authorizes_method_evaluation": False,
        "authorizes_protected_outcome_access": False,
    }
    freeze["fingerprint"] = fingerprint(freeze)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        write_json(temporary / "role_assignments.json", role_artifact)
        write_json(temporary / "protocol_state.json", state)
        write_json(temporary / "protocol_freeze.json", freeze)
        _write_report(temporary / "report.md", freeze)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        name: output / name
        for name in (
            "protocol_freeze.json",
            "role_assignments.json",
            "protocol_state.json",
            "report.md",
        )
    }


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-audit", required=True, type=Path)
    parser.add_argument("--role-witness", required=True, type=Path)
    parser.add_argument("--power-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = freeze_pdctp_real_protocol(
        load_pdctp_real_protocol_config(args.config),
        args.source_audit,
        args.role_witness,
        args.power_plan,
        args.output,
    )
    print(f"PDCTP real protocol freeze wrote {len(paths)} artifacts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
