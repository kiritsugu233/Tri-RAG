"""Describe frozen real-data retrieval policies on query_test exactly once."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from .embeddings import load_embedding_array
from .indexes import ExactSquaredL2Index
from .lid import estimate_lid_from_squared_distances
from .policies import PolicyDecision
from .projection import dense_gaussian_projection, project_rows, projection_metadata
from .real_dimension_sweep import _exact_projected_rankings, _ranking_hash
from .real_policy_certify import (
    RealPolicyCertificationConfig,
    RealPolicyCertificationError,
    _CERT_RESULT_NAMES,
    _POLICY_NAMES,
    _canonical_float,
    _coordinate_work,
    _file_identity,
    _load_json,
    _load_jsonl,
    _rerank_record,
    _serialize_lid,
    _validate_policy_bundle,
    _verify_embedded_fingerprint,
    _write_jsonl,
    load_real_policy_certification_config,
)
from .text_embeddings import load_text_embedding_config, validate_text_embedding_cache
from .utils import fingerprint, stable_id_hash, write_json


class RealPolicyTestError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenCertificationSource:
    manifest_fingerprint: str
    result_fingerprint: str
    certificates_fingerprint: str
    query_cert_n: int
    query_cert_id_hash: str
    terminal: bool
    failure_behavior: str
    decisions: Dict[str, str]


@dataclass(frozen=True)
class RealPolicyTestConfig:
    raw: Dict[str, Any]
    config_fingerprint: str
    certification_config_fingerprint: str
    evaluation_split: str
    query_split_n: int
    query_split_id_hash: str
    certification_source: FrozenCertificationSource
    evidence_cutoffs: list[int]
    include_exact_original_reference: bool
    reporting_role: str


@dataclass(frozen=True)
class FrozenCertificationBundle:
    manifest: Dict[str, Any]
    certifications: Dict[str, Any]
    decisions: Dict[str, str]
    input_artifacts: Dict[str, Dict[str, Any]]


_SHA256_LENGTH = 64
_TEST_ROLE = "descriptive_frozen_policy_test_no_selection_no_certificate"
_POST_TEST_SELECTION = "forbidden"
_NEW_CERTIFICATION = "forbidden"
_RETUNING = "forbidden"
_CERT_FAILURE_BEHAVIOR = "terminal_no_retuning_no_budget_expansion"
_TEST_RESULT_NAMES = ("per_query.jsonl", "summary.json", "report.md")


def _exact_keys(value: Any, expected: set[str], name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RealPolicyTestError(f"{name} must be an object")
    if set(value) != expected:
        raise RealPolicyTestError(
            f"invalid {name} keys; missing={sorted(expected-set(value))}, "
            f"unknown={sorted(set(value)-expected)}"
        )
    return dict(value)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RealPolicyTestError(f"{name} must be a positive integer")
    return int(value)


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise RealPolicyTestError(f"{name} must be a SHA-256 value")
    result = value.strip().lower()
    if len(result) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise RealPolicyTestError(f"{name} must be a SHA-256 value")
    return result


def load_real_policy_test_config(
    path: Union[str, Path],
) -> RealPolicyTestConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealPolicyTestError(f"cannot load test config {config_path}: {exc}") from exc
    root = _exact_keys(
        raw,
        {
            "schema_version",
            "benchmark",
            "certification_config_fingerprint",
            "evaluation_split",
            "query_split",
            "certification_source",
            "evidence",
            "reporting",
        },
        "root",
    )
    if root["schema_version"] != 1 or root["benchmark"] != "real_frozen_policy_test_v1":
        raise RealPolicyTestError("unsupported test config schema/benchmark")
    if root["evaluation_split"] != "query_test":
        raise RealPolicyTestError("descriptive test runner accepts query_test only")

    query_split = _exact_keys(root["query_split"], {"n", "id_hash"}, "query_split")
    source = _exact_keys(
        root["certification_source"],
        {
            "manifest_fingerprint",
            "result_fingerprint",
            "certificates_fingerprint",
            "query_cert_n",
            "query_cert_id_hash",
            "terminal",
            "failure_behavior",
            "decisions",
        },
        "certification_source",
    )
    decisions = _exact_keys(
        source["decisions"], set(_POLICY_NAMES), "certification_source.decisions"
    )
    if any(value not in {"PASS", "FAIL"} for value in decisions.values()):
        raise RealPolicyTestError("certification decisions must be PASS or FAIL")
    if source["terminal"] is not True or source["failure_behavior"] != _CERT_FAILURE_BEHAVIOR:
        raise RealPolicyTestError("certification source must be terminal")

    evidence = _exact_keys(
        root["evidence"],
        {"source", "cutoffs", "include_exact_original_reference"},
        "evidence",
    )
    if evidence["source"] != "prepared_query_test_qrels":
        raise RealPolicyTestError("test evidence source is not frozen")
    cutoffs_raw = evidence["cutoffs"]
    if not isinstance(cutoffs_raw, list) or not cutoffs_raw:
        raise RealPolicyTestError("evidence.cutoffs must be a nonempty list")
    cutoffs = [_positive_integer(value, "evidence cutoff") for value in cutoffs_raw]
    if cutoffs != sorted(set(cutoffs)):
        raise RealPolicyTestError("evidence.cutoffs must be strictly increasing")
    if evidence["include_exact_original_reference"] is not True:
        raise RealPolicyTestError("exact original evidence reference must be included")

    reporting = _exact_keys(
        root["reporting"],
        {"role", "policies", "post_test_selection", "new_certification", "retuning"},
        "reporting",
    )
    if (
        reporting["role"] != _TEST_ROLE
        or reporting["policies"] != list(_POLICY_NAMES)
        or reporting["post_test_selection"] != _POST_TEST_SELECTION
        or reporting["new_certification"] != _NEW_CERTIFICATION
        or reporting["retuning"] != _RETUNING
    ):
        raise RealPolicyTestError("test reporting protocol is not frozen")

    return RealPolicyTestConfig(
        raw=root,
        config_fingerprint=fingerprint(root),
        certification_config_fingerprint=_sha256(
            root["certification_config_fingerprint"],
            "certification_config_fingerprint",
        ),
        evaluation_split="query_test",
        query_split_n=_positive_integer(query_split["n"], "query_split.n"),
        query_split_id_hash=_sha256(query_split["id_hash"], "query_split.id_hash"),
        certification_source=FrozenCertificationSource(
            manifest_fingerprint=_sha256(
                source["manifest_fingerprint"],
                "certification_source.manifest_fingerprint",
            ),
            result_fingerprint=_sha256(
                source["result_fingerprint"],
                "certification_source.result_fingerprint",
            ),
            certificates_fingerprint=_sha256(
                source["certificates_fingerprint"],
                "certification_source.certificates_fingerprint",
            ),
            query_cert_n=_positive_integer(
                source["query_cert_n"], "certification_source.query_cert_n"
            ),
            query_cert_id_hash=_sha256(
                source["query_cert_id_hash"],
                "certification_source.query_cert_id_hash",
            ),
            terminal=True,
            failure_behavior=_CERT_FAILURE_BEHAVIOR,
            decisions={name: str(decisions[name]) for name in _POLICY_NAMES},
        ),
        evidence_cutoffs=cutoffs,
        include_exact_original_reference=True,
        reporting_role=_TEST_ROLE,
    )


def _validate_certification_bundle(
    directory: Path,
    config: RealPolicyTestConfig,
    certification_config: RealPolicyCertificationConfig,
) -> FrozenCertificationBundle:
    try:
        manifest = _load_json(directory / "manifest.json", "certification manifest")
        manifest_fingerprint = _verify_embedded_fingerprint(
            manifest, field="fingerprint", name="certification manifest"
        )
    except RealPolicyCertificationError as exc:
        raise RealPolicyTestError(str(exc)) from exc
    source = config.certification_source
    if manifest_fingerprint != source.manifest_fingerprint:
        raise RealPolicyTestError("certification manifest fingerprint mismatch")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "real_frozen_policy_certification_manifest_v1"
        or manifest.get("data_scope") != "query_cert_only"
        or manifest.get("config_fingerprint") != config.certification_config_fingerprint
        or manifest.get("dataset_manifest_fingerprint")
        != certification_config.dataset_manifest_fingerprint
        or manifest.get("embedding_manifest_fingerprint")
        != certification_config.embedding_manifest_fingerprint
        or manifest.get("query_cert_n") != source.query_cert_n
        or manifest.get("query_cert_id_hash") != source.query_cert_id_hash
        or manifest.get("result_fingerprint") != source.result_fingerprint
        or manifest.get("projection_fingerprint")
        != certification_config.projection_fingerprint
        or manifest.get("frozen_projection_fingerprint")
        != certification_config.frozen_projection_fingerprint
    ):
        raise RealPolicyTestError("certification manifest identity mismatch")

    artifact_metadata = manifest.get("result_artifacts")
    if not isinstance(artifact_metadata, dict) or set(artifact_metadata) != set(
        _CERT_RESULT_NAMES
    ):
        raise RealPolicyTestError("certification result artifact set is invalid")
    input_artifacts: Dict[str, Dict[str, Any]] = {}
    for name in _CERT_RESULT_NAMES:
        try:
            observed = _file_identity(directory / name)
        except OSError as exc:
            raise RealPolicyTestError(f"cannot read certification artifact {name}") from exc
        if observed != artifact_metadata.get(name):
            raise RealPolicyTestError(
                f"certification result artifact identity mismatch: {name}"
            )
        input_artifacts[name] = observed
    result_identity = {
        "config_fingerprint": manifest.get("config_fingerprint"),
        "dataset_manifest_fingerprint": manifest.get("dataset_manifest_fingerprint"),
        "embedding_manifest_fingerprint": manifest.get("embedding_manifest_fingerprint"),
        "policy_source_result_fingerprint": manifest.get("policy_source", {}).get(
            "result_fingerprint"
        ),
        "compiled_deployment_fingerprint": manifest.get("deployment", {}).get(
            "policy_fingerprint"
        ),
        "query_cert_id_hash": manifest.get("query_cert_id_hash"),
        "artifacts": input_artifacts,
    }
    if fingerprint(result_identity) != source.result_fingerprint:
        raise RealPolicyTestError("certification result identity is invalid")

    try:
        certifications = _load_json(
            directory / "certifications.json", "certification decisions"
        )
        certificates_fingerprint = _verify_embedded_fingerprint(
            certifications, field="fingerprint", name="certification decisions"
        )
        summary = _load_json(directory / "summary.json", "certification summary")
        _verify_embedded_fingerprint(
            summary, field="fingerprint", name="certification summary"
        )
    except RealPolicyCertificationError as exc:
        raise RealPolicyTestError(str(exc)) from exc
    if certificates_fingerprint != source.certificates_fingerprint:
        raise RealPolicyTestError("certification decisions fingerprint mismatch")
    policy_fingerprints = {
        "fixed_reference": certification_config.policy_source.fixed_reference_policy_fingerprint,
        "monotone_binned": certification_config.policy_source.monotone_binned_fingerprint,
        "tri_predict": certification_config.policy_source.analytic_tri_predict_fingerprint,
    }
    certificate_values = certifications.get("certificates")
    if not isinstance(certificate_values, dict) or set(certificate_values) != set(
        _POLICY_NAMES
    ):
        raise RealPolicyTestError("certification policy set is invalid")
    if any(
        not isinstance(certificate_values[name].get("passed"), bool)
        for name in _POLICY_NAMES
    ):
        raise RealPolicyTestError("certification decisions must be boolean")
    decisions = {
        name: "PASS" if certificate_values[name]["passed"] else "FAIL"
        for name in _POLICY_NAMES
    }
    for name in _POLICY_NAMES:
        certificate = certificate_values[name]
        if (
            not isinstance(certificate, dict)
            or certificate.get("policy_fingerprint") != policy_fingerprints[name]
            or certificate.get("split_hash") != source.query_cert_id_hash
            or certificate.get("n") != source.query_cert_n
            or certificate.get("planned_n") != source.query_cert_n
            or certificate.get("alpha")
            != certification_config.certification_alpha
            or certificate.get("target")
            != certification_config.certification_target
            or certificate.get("metric")
            != "embedding_neighbor_retention_at_k_gt"
            or summary.get("policies", {}).get(name, {}).get("decision")
            != decisions[name]
        ):
            raise RealPolicyTestError("terminal certification decision is invalid")
    if (
        certifications.get("data_scope") != "query_cert_only"
        or certifications.get("n") != source.query_cert_n
        or certifications.get("split_hash") != source.query_cert_id_hash
        or certifications.get("terminal") is not True
        or certifications.get("failure_behavior") != source.failure_behavior
        or certifications.get("alpha_per_policy")
        != certification_config.certification_alpha
        or certifications.get("target")
        != certification_config.certification_target
        or decisions != source.decisions
        or certifications.get("all_passed") != all(
            value == "PASS" for value in decisions.values()
        )
        or manifest.get("policy_fingerprints") != policy_fingerprints
    ):
        raise RealPolicyTestError("certification source is not the frozen terminal result")
    if (
        manifest.get("policy_source", {}).get("result_fingerprint")
        != certification_config.policy_source.result_fingerprint
        or manifest.get("deployment", {}).get("policy_fingerprint")
        != certification_config.policy_source.compiled_tri_predict_fingerprint
    ):
        raise RealPolicyTestError("certification policy/deployment binding mismatch")
    input_artifacts["manifest.json"] = _file_identity(directory / "manifest.json")
    input_artifacts["timings.json"] = _file_identity(directory / "timings.json")
    return FrozenCertificationBundle(
        manifest=manifest,
        certifications=certifications,
        decisions=decisions,
        input_artifacts=input_artifacts,
    )


def _ndcg_at_k(
    retrieved_ids: Sequence[str], relevance: Mapping[str, int], k: int
) -> float:
    gains = [
        (2.0 ** relevance.get(doc_id, 0) - 1.0) / math.log2(rank + 2.0)
        for rank, doc_id in enumerate(retrieved_ids[:k])
    ]
    ideal_relevance = sorted(relevance.values(), reverse=True)[:k]
    ideal = [
        (2.0 ** value - 1.0) / math.log2(rank + 2.0)
        for rank, value in enumerate(ideal_relevance)
    ]
    denominator = sum(ideal)
    if denominator <= 0.0:
        raise RealPolicyTestError("nDCG requires a positive qrel")
    return float(sum(gains) / denominator)


def _evidence_metrics(
    retrieved_ids: Sequence[str],
    relevance: Mapping[str, int],
    cutoffs: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    relevant_ids = set(relevance)
    if not relevant_ids:
        raise RealPolicyTestError("test query has no positive qrels")
    result: Dict[str, Dict[str, Any]] = {}
    for cutoff in cutoffs:
        retained = relevant_ids.intersection(retrieved_ids[:cutoff])
        result[str(cutoff)] = {
            "evidence_hit": bool(retained),
            "evidence_recall": float(len(retained) / len(relevant_ids)),
            "ndcg": _ndcg_at_k(retrieved_ids, relevance, cutoff),
        }
    return result


def _aggregate_evidence(
    metric_sets: Sequence[Mapping[str, Mapping[str, Any]]],
    cutoffs: Sequence[int],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for cutoff in cutoffs:
        values = [metrics[str(cutoff)] for metrics in metric_sets]
        result[str(cutoff)] = {
            "mean_evidence_hit": float(
                np.mean([value["evidence_hit"] for value in values])
            ),
            "mean_evidence_recall": float(
                np.mean([value["evidence_recall"] for value in values])
            ),
            "mean_ndcg": float(np.mean([value["ndcg"] for value in values])),
            "queries_with_evidence_hit": int(
                sum(bool(value["evidence_hit"]) for value in values)
            ),
        }
    return result


def _descriptive_policy_evaluation(
    *,
    name: str,
    records: Sequence[Mapping[str, Any]],
    certification_decision: str,
    certification_config: RealPolicyCertificationConfig,
    corpus_size: int,
    dimension: int,
    cutoffs: Sequence[int],
) -> Dict[str, Any]:
    values = [record["policies"][name] for record in records]
    retentions = [float(value["embedding_retention"]) for value in values]
    budgets = [int(value["chosen_m"]) for value in values]
    mean_budget = float(np.mean(budgets))
    work = _coordinate_work(
        corpus_size=corpus_size,
        dimension=dimension,
        m_prime=certification_config.m_prime,
        mean_budget=mean_budget,
    )
    fixed_work = _coordinate_work(
        corpus_size=corpus_size,
        dimension=dimension,
        m_prime=certification_config.m_prime,
        mean_budget=float(certification_config.fixed_reference_budget),
    )["total"]
    return {
        "terminal_certification_decision": certification_decision,
        "budget": {
            "mean": mean_budget,
            "median": float(np.median(budgets)),
            "p95": float(np.quantile(budgets, 0.95)),
            "p99": float(np.quantile(budgets, 0.99)),
            "distribution": {
                str(budget): budgets.count(budget)
                for budget in certification_config.m_grid
            },
        },
        "retention_distribution": {
            "mean": float(np.mean(retentions)),
            "median": float(np.median(retentions)),
            "p05": float(np.quantile(retentions, 0.05)),
            "minimum": float(np.min(retentions)),
        },
        "evidence_metrics": _aggregate_evidence(
            [value["evidence_metrics"] for value in values], cutoffs
        ),
        "fallback_n": int(sum(value["used_lid_fallback"] for value in values)),
        "saturated_n": int(sum(value["policy_saturated"] for value in values)),
        "candidate_saving_vs_frozen_fixed_reference": 1.0
        - mean_budget / certification_config.fixed_reference_budget,
        "coordinate_work": work,
        "coordinate_work_reduction_vs_frozen_fixed_reference": 1.0
        - work["total"] / fixed_work,
        "work_per_query": {
            "query_projection_coordinates": dimension * certification_config.m_prime,
            "projected_distance_evaluations": corpus_size,
            "projected_scan_count": 1,
            "original_rerank_distance_evaluations": mean_budget,
            "pilot_original_distances_reused_in_rerank": certification_config.m_pilot,
            "original_ground_truth_distance_evaluations_diagnostic": corpus_size,
        },
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# SciFact frozen-policy descriptive test",
        "",
        "This run evaluates all three already-frozen policies once on `query_test`. "
        "It is descriptive: test outcomes do not select a winner, change a policy, "
        "or create a new certificate. Certification decisions below are copied from "
        "the bound terminal `query_cert` result.",
        "",
        f"- test queries: {summary['n_queries']}",
        f"- corpus vectors: {summary['corpus_size']}",
        f"- projection dimension: {summary['m_prime']}",
        f"- fixed reference: `M={summary['fixed_reference_budget']}`",
        "",
        "| policy | terminal cert | test mean M | test mean retention | candidate saving | coordinate saving | fallback | saturation |",
        "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = (
        ("fixed reference", "fixed_reference"),
        ("monotone binned", "monotone_binned"),
        ("Tri-Predict", "tri_predict"),
    )
    for label, name in labels:
        value = summary["policies"][name]
        lines.append(
            f"| {label} | {value['terminal_certification_decision']} | "
            f"{value['budget']['mean']:.4f} | "
            f"{value['retention_distribution']['mean']:.6f} | "
            f"{value['candidate_saving_vs_frozen_fixed_reference']:.2%} | "
            f"{value['coordinate_work_reduction_vs_frozen_fixed_reference']:.2%} | "
            f"{value['fallback_n']} | {value['saturated_n']} |"
        )
    lines.extend(
        [
            "",
            "## Labeled-evidence metrics",
            "",
            "| path | cutoff | evidence hit | evidence recall | nDCG |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    evidence_sources = [("exact original reference", summary["exact_original_reference"])]
    evidence_sources.extend((label, summary["policies"][name]) for label, name in labels)
    for label, value in evidence_sources:
        for cutoff, metrics in value["evidence_metrics"].items():
            lines.append(
                f"| {label} | {cutoff} | {metrics['mean_evidence_hit']:.6f} | "
                f"{metrics['mean_evidence_recall']:.6f} | "
                f"{metrics['mean_ndcg']:.6f} |"
            )
    lines.extend(
        [
            "",
            "Candidate and coordinate savings are deterministic work proxies, not "
            "latency claims. Evidence metrics are not an embedding certificate or "
            "answer-quality result. No test-split confidence bound is produced.",
            "",
        ]
    )
    return "\n".join(lines)


def run_real_policy_test(
    config: RealPolicyTestConfig,
    certification_config: RealPolicyCertificationConfig,
    prepared_dir: Union[str, Path],
    embedding_config_path: Union[str, Path],
    embedding_cache_dir: Union[str, Path],
    policy_run_dir: Union[str, Path],
    certification_run_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    if config.evaluation_split != "query_test":
        raise RealPolicyTestError("descriptive test runner accepts query_test only")
    if certification_config.config_fingerprint != config.certification_config_fingerprint:
        raise RealPolicyTestError("certification config fingerprint mismatch")
    if max(config.evidence_cutoffs) > certification_config.k_gt or (
        certification_config.k_ctx not in config.evidence_cutoffs
        or certification_config.k_gt not in config.evidence_cutoffs
    ):
        raise RealPolicyTestError("evidence cutoffs disagree with frozen k_ctx/k_gt")
    prepared = Path(prepared_dir)
    embedding_cache = Path(embedding_cache_dir)
    policy_run = Path(policy_run_dir)
    certification_run = Path(certification_run_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite descriptive test output: {output}")

    # Validate both complete frozen inputs before any protected test data or qrels
    # are selected. Neither terminal cert decisions nor test outcomes select a policy.
    try:
        policy_bundle = _validate_policy_bundle(policy_run, certification_config)
    except RealPolicyCertificationError as exc:
        raise RealPolicyTestError(str(exc)) from exc
    certification_bundle = _validate_certification_bundle(
        certification_run, config, certification_config
    )

    embedding_config = load_text_embedding_config(embedding_config_path)
    if embedding_config.config_fingerprint != certification_config.embedding_config_fingerprint:
        raise RealPolicyTestError("embedding config fingerprint mismatch")
    validation = validate_text_embedding_cache(
        embedding_config, prepared, embedding_cache
    )
    dataset_manifest = validation["dataset_manifest"]
    embedding_manifest = validation["embedding_manifest"]
    if (
        dataset_manifest.get("fingerprint")
        != certification_config.dataset_manifest_fingerprint
        or validation.get("request_fingerprint")
        != certification_config.embedding_request_fingerprint
        or embedding_manifest.get("fingerprint")
        != certification_config.embedding_manifest_fingerprint
    ):
        raise RealPolicyTestError("dataset/embedding input identity mismatch")
    expected_split = dataset_manifest.get("splits", {}).get("query_test")
    if not isinstance(expected_split, dict) or (
        expected_split.get("n") != config.query_split_n
        or expected_split.get("id_hash") != config.query_split_id_hash
    ):
        raise RealPolicyTestError("dataset test split identity mismatch")

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
        raise RealPolicyTestError("query embedding rows do not match prepared query order")
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
        raise RealPolicyTestError("selected queries are not the frozen test IDs")

    qrel_rows = _load_jsonl(prepared / "qrels.jsonl", "prepared qrels")
    qrels_by_query: Dict[str, Dict[str, int]] = {
        query_id: {} for query_id in selected_query_ids
    }
    corpus_ids = corpus_table.ids.tolist()
    corpus_id_set = set(corpus_ids)
    for row in qrel_rows:
        if row.get("split") != config.evaluation_split:
            continue
        query_id = row.get("query_id")
        doc_id = row.get("doc_id")
        relevance = row.get("relevance")
        if query_id not in qrels_by_query:
            raise RealPolicyTestError("test qrel references unknown query")
        if doc_id not in corpus_id_set:
            raise RealPolicyTestError("test qrel references unknown document")
        if isinstance(relevance, bool) or not isinstance(relevance, int) or relevance < 1:
            raise RealPolicyTestError("test qrel relevance must be positive")
        if doc_id in qrels_by_query[query_id]:
            raise RealPolicyTestError("duplicate test qrel pair")
        qrels_by_query[query_id][doc_id] = relevance
    empty_qrels = [query_id for query_id, values in qrels_by_query.items() if not values]
    if empty_qrels:
        raise RealPolicyTestError(f"test queries have no positive qrels: {empty_qrels}")

    corpus = np.asarray(corpus_table.vectors, dtype=np.float32)
    test_queries = np.asarray(
        query_table.vectors[np.asarray(selected_rows, dtype=np.int64)],
        dtype=np.float32,
    )
    corpus_size, dimension = corpus.shape
    if (
        len(test_queries) != config.query_split_n
        or test_queries.shape[1] != dimension
        or dimension != embedding_config.model.embedding_dimension
        or certification_config.m_grid[-1] != corpus_size
        or policy_bundle.analytic_tri.corpus_size != corpus_size
        or certification_config.m_prime > dimension
    ):
        raise RealPolicyTestError("embedding shape is incompatible with frozen protocol")

    projection_started = perf_counter()
    matrix = dense_gaussian_projection(
        certification_config.m_prime,
        dimension,
        certification_config.projection_seed,
    )
    projected_corpus = project_rows(corpus, matrix)
    projected_queries = project_rows(test_queries, matrix)
    projection_ms = (perf_counter() - projection_started) * 1000.0
    metadata = projection_metadata(
        dimension=dimension,
        m_prime=certification_config.m_prime,
        seed=certification_config.projection_seed,
        normalization=True,
        embedding_model=(
            f"{embedding_config.model.name}@{embedding_config.model.revision}"
        ),
        corpus_hash=embedding_manifest["arrays"]["corpus"]["array_fingerprint"],
    )
    if metadata["fingerprint"] != certification_config.projection_fingerprint:
        raise RealPolicyTestError("reconstructed projection fingerprint mismatch")

    rankings, projected_search_ms = _exact_projected_rankings(
        projected_corpus,
        projected_queries,
        corpus_ids,
        k=corpus_size,
        batch_size=certification_config.query_batch_size,
    )
    ground_truth_started = perf_counter()
    original_index = ExactSquaredL2Index(
        corpus_ids, corpus, batch_size=certification_config.query_batch_size
    )
    original = original_index.search(
        test_queries, max(certification_config.k_gt, certification_config.s_lid)
    )
    ground_truth_ms = (perf_counter() - ground_truth_started) * 1000.0

    tie_rank = np.argsort(
        np.argsort(np.asarray(corpus_ids, dtype=str), kind="stable"), kind="stable"
    )
    lid_kwargs = {
        "s_lid": certification_config.s_lid,
        "min_neighbors": certification_config.min_lid_neighbors,
        "clip_min": certification_config.lid_clip_min,
        "clip_max": certification_config.lid_clip_max,
        "duplicate_tolerance": certification_config.duplicate_tolerance,
        "fallback": certification_config.lid_fallback,
    }
    records: list[Dict[str, Any]] = []
    pilot_ms = 0.0
    lid_ms = 0.0
    policy_ms = 0.0
    expansion_ms = 0.0
    rerank_ms = 0.0
    evidence_ms = 0.0
    combined_original_evaluations = 0
    for query_index, ranking in enumerate(rankings):
        query = np.asarray(test_queries[query_index], dtype=np.float64)
        pilot_rows = ranking[: certification_config.m_pilot]
        started = perf_counter()
        pilot_difference = np.asarray(corpus[pilot_rows], dtype=np.float64) - query
        pilot_squared = np.einsum("ij,ij->i", pilot_difference, pilot_difference)
        pilot_ms += (perf_counter() - started) * 1000.0

        started = perf_counter()
        pilot_lid = estimate_lid_from_squared_distances(pilot_squared, **lid_kwargs)
        oracle_lid = estimate_lid_from_squared_distances(
            original.squared_distances[query_index, : certification_config.s_lid],
            **lid_kwargs,
        )
        pilot_serialized = _serialize_lid(
            pilot_lid, certification_config.lid_decimal_places
        )
        oracle_serialized = _serialize_lid(
            oracle_lid, certification_config.lid_decimal_places
        )
        lid_ms += (perf_counter() - started) * 1000.0

        lid_value = float(pilot_serialized["clipped"])
        lid_valid = bool(pilot_serialized["valid"])
        started = perf_counter()
        fixed_decision = policy_bundle.fixed.choose(lid_value, lid_valid)
        monotone_decision = policy_bundle.monotone.choose(lid_value, lid_valid)
        analytic_decision = policy_bundle.analytic_tri.choose(lid_value, lid_valid)
        compiled_decision = policy_bundle.compiled_tri.choose(lid_value, lid_valid)
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
            raise RealPolicyTestError(
                "compiled Tri-Predict disagrees with analytic policy on query_test"
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
            decision.budget not in certification_config.m_grid
            or decision.budget < certification_config.m_pilot
            for decision in decisions.values()
        ):
            raise RealPolicyTestError("frozen policy emitted an unsafe budget")

        maximum_budget = max(decision.budget for decision in decisions.values())
        started = perf_counter()
        if maximum_budget > certification_config.m_pilot:
            tail_rows = ranking[certification_config.m_pilot : maximum_budget]
            tail_difference = np.asarray(corpus[tail_rows], dtype=np.float64) - query
            tail_squared = np.einsum("ij,ij->i", tail_difference, tail_difference)
            candidate_squared = np.concatenate((pilot_squared, tail_squared))
        else:
            candidate_squared = pilot_squared
        expansion_ms += (perf_counter() - started) * 1000.0
        combined_original_evaluations += maximum_budget

        exact_rows = set(
            original.rows[query_index, : certification_config.k_gt].tolist()
        )
        exact_ids = original.ids[
            query_index, : certification_config.k_gt
        ].tolist()
        started = perf_counter()
        try:
            policy_records = {
                name: _rerank_record(
                    decision=decision,
                    candidate_rows=ranking,
                    candidate_distances=candidate_squared,
                    tie_rank=tie_rank,
                    corpus_ids=corpus_ids,
                    exact_rows=exact_rows,
                    exact_ids=exact_ids,
                    k_gt=certification_config.k_gt,
                )
                for name, decision in decisions.items()
            }
        except RealPolicyCertificationError as exc:
            raise RealPolicyTestError(str(exc)) from exc
        rerank_ms += (perf_counter() - started) * 1000.0
        policy_records["tri_predict"]["compiled_decision_match"] = True

        relevance = qrels_by_query[selected_query_ids[query_index]]
        started = perf_counter()
        exact_evidence = _evidence_metrics(
            exact_ids, relevance, config.evidence_cutoffs
        )
        for value in policy_records.values():
            value["evidence_metrics"] = _evidence_metrics(
                value["reranked_top_k_ids"], relevance, config.evidence_cutoffs
            )
        evidence_ms += (perf_counter() - started) * 1000.0
        records.append(
            {
                "query_index": query_index,
                "query_id": selected_query_ids[query_index],
                "split": "query_test",
                "qrel_count": len(relevance),
                "relevance_by_doc_id": dict(sorted(relevance.items())),
                "pilot_lid": pilot_serialized,
                "oracle_lid": oracle_serialized,
                "oracle_lid_role": "diagnostic_only_not_used_for_policy_decisions",
                "pilot_candidate_ids": [corpus_ids[row] for row in pilot_rows],
                "exact_top_k_ids": exact_ids,
                "exact_original_evidence_metrics": exact_evidence,
                "projected_ranking_rows_sha256": _ranking_hash(ranking),
                "policies": policy_records,
            }
        )

    if (
        len(records) != config.query_split_n
        or {record["split"] for record in records} != {"query_test"}
        or stable_id_hash([record["query_id"] for record in records])
        != config.query_split_id_hash
    ):
        raise RealPolicyTestError("test records lost split identity")

    policy_fingerprints = {
        "fixed_reference": policy_bundle.fixed_fingerprint,
        "monotone_binned": policy_bundle.monotone.serialize()["fingerprint"],
        "tri_predict": policy_bundle.analytic_tri.serialize()["fingerprint"],
    }
    evaluations = {
        name: {
            "policy_fingerprint": policy_fingerprints[name],
            **_descriptive_policy_evaluation(
                name=name,
                records=records,
                certification_decision=certification_bundle.decisions[name],
                certification_config=certification_config,
                corpus_size=corpus_size,
                dimension=dimension,
                cutoffs=config.evidence_cutoffs,
            ),
        }
        for name in _POLICY_NAMES
    }
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
        "kind": "real_frozen_policy_descriptive_test_summary_v1",
        "data_scope": "query_test_only",
        "evaluation_role": config.reporting_role,
        "n_queries": len(records),
        "corpus_size": corpus_size,
        "embedding_dimension": dimension,
        "m_prime": certification_config.m_prime,
        "projection_seed": certification_config.projection_seed,
        "m_pilot": certification_config.m_pilot,
        "s_lid": certification_config.s_lid,
        "fixed_reference_budget": certification_config.fixed_reference_budget,
        "evidence_cutoffs": config.evidence_cutoffs,
        "certification_source": {
            "manifest_fingerprint": config.certification_source.manifest_fingerprint,
            "result_fingerprint": config.certification_source.result_fingerprint,
            "certificates_fingerprint": config.certification_source.certificates_fingerprint,
            "query_cert_id_hash": config.certification_source.query_cert_id_hash,
            "decisions": certification_bundle.decisions,
            "role": "terminal_query_cert_result_display_only_not_used_for_selection",
        },
        "policy_source_result_fingerprint": certification_config.policy_source.result_fingerprint,
        "policy_fingerprints": policy_fingerprints,
        "compiled_deployment_fingerprint": certification_config.policy_source.compiled_tri_predict_fingerprint,
        "compiled_reference_match_n": len(records),
        "policies": evaluations,
        "exact_original_reference": {
            "role": "diagnostic_full_original_space_top_k_not_a_deployment_policy",
            "evidence_metrics": _aggregate_evidence(
                [record["exact_original_evidence_metrics"] for record in records],
                config.evidence_cutoffs,
            ),
        },
        "lid_diagnostic": {
            "pilot_valid_n": int(sum(record["pilot_lid"]["valid"] for record in records)),
            "oracle_valid_n": int(sum(record["oracle_lid"]["valid"] for record in records)),
            "paired_valid_n": len(paired_lid_gaps),
            "mean_absolute_clipped_gap": (
                0.0
                if not paired_lid_gaps
                else _canonical_float(
                    float(np.mean(paired_lid_gaps)),
                    certification_config.lid_decimal_places,
                )
            ),
            "oracle_role": "diagnostic_only_not_used_for_policy_decisions",
        },
        "post_test_selection": _POST_TEST_SELECTION,
        "new_certification": _NEW_CERTIFICATION,
        "retuning": _RETUNING,
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
        "evidence_metrics_ms": evidence_ms,
        "projected_scan_count_per_query": 1,
        "projected_distance_evaluations": len(records) * corpus_size,
        "original_ground_truth_distance_evaluations_diagnostic": len(records)
        * corpus_size,
        "combined_shared_original_distance_evaluations": combined_original_evaluations,
        "counterfactual_original_rerank_distance_evaluations": {
            name: int(sum(record["policies"][name]["chosen_m"] for record in records))
            for name in _POLICY_NAMES
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        paths = {
            "manifest.json": temporary / "manifest.json",
            "per_query.jsonl": temporary / "per_query.jsonl",
            "summary.json": temporary / "summary.json",
            "timings.json": temporary / "timings.json",
            "report.md": temporary / "report.md",
        }
        _write_jsonl(paths["per_query.jsonl"], records)
        write_json(paths["summary.json"], summary)
        write_json(paths["timings.json"], timings)
        paths["report.md"].write_text(_report(summary), encoding="utf-8")
        result_artifacts = {
            name: _file_identity(paths[name]) for name in _TEST_RESULT_NAMES
        }
        result_identity = {
            "config_fingerprint": config.config_fingerprint,
            "certification_config_fingerprint": certification_config.config_fingerprint,
            "dataset_manifest_fingerprint": certification_config.dataset_manifest_fingerprint,
            "embedding_manifest_fingerprint": certification_config.embedding_manifest_fingerprint,
            "policy_source_result_fingerprint": certification_config.policy_source.result_fingerprint,
            "compiled_deployment_fingerprint": certification_config.policy_source.compiled_tri_predict_fingerprint,
            "certification_result_fingerprint": config.certification_source.result_fingerprint,
            "query_test_id_hash": config.query_split_id_hash,
            "artifacts": result_artifacts,
        }
        manifest = {
            "schema_version": 1,
            "kind": "real_frozen_policy_descriptive_test_manifest_v1",
            "data_scope": "query_test_only",
            "evaluation_role": config.reporting_role,
            "config_fingerprint": config.config_fingerprint,
            "certification_config_fingerprint": certification_config.config_fingerprint,
            "dataset_manifest_fingerprint": certification_config.dataset_manifest_fingerprint,
            "embedding_manifest_fingerprint": certification_config.embedding_manifest_fingerprint,
            "query_test_n": len(records),
            "query_test_id_hash": config.query_split_id_hash,
            "projection_fingerprint": certification_config.projection_fingerprint,
            "frozen_projection_fingerprint": certification_config.frozen_projection_fingerprint,
            "policy_source": {
                "manifest_fingerprint": certification_config.policy_source.manifest_fingerprint,
                "result_fingerprint": certification_config.policy_source.result_fingerprint,
                "selection_fingerprint": certification_config.policy_source.selection_fingerprint,
                "input_artifacts": policy_bundle.input_artifacts,
            },
            "certification_source": {
                "manifest_fingerprint": config.certification_source.manifest_fingerprint,
                "result_fingerprint": config.certification_source.result_fingerprint,
                "certificates_fingerprint": config.certification_source.certificates_fingerprint,
                "query_cert_n": config.certification_source.query_cert_n,
                "query_cert_id_hash": config.certification_source.query_cert_id_hash,
                "decisions": certification_bundle.decisions,
                "input_artifacts": certification_bundle.input_artifacts,
                "role": "terminal_query_cert_result_display_only_not_used_for_selection",
            },
            "policy_fingerprints": policy_fingerprints,
            "deployment": {
                "role": certification_config.policy_source.compiled_policy_role,
                "policy_fingerprint": certification_config.policy_source.compiled_tri_predict_fingerprint,
                "reference_policy_fingerprint": certification_config.policy_source.analytic_tri_predict_fingerprint,
                "artifact_sha256": certification_config.policy_source.compiled_artifact_sha256,
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
    parser.add_argument("--certification-config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--embedding-config", required=True, type=Path)
    parser.add_argument("--embedding-cache", required=True, type=Path)
    parser.add_argument("--policy-run", required=True, type=Path)
    parser.add_argument("--certification-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_real_policy_test_config(args.config)
    certification_config = load_real_policy_certification_config(
        args.certification_config
    )
    artifacts = run_real_policy_test(
        config,
        certification_config,
        args.dataset,
        args.embedding_config,
        args.embedding_cache,
        args.policy_run,
        args.certification_run,
        args.output,
    )
    print(f"completed frozen-policy descriptive test: {artifacts['report.md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
