"""Posthoc tune-only evidence, allocation, and shuffled-LID diagnostics."""

from __future__ import annotations

import argparse
import hashlib
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
from .projection import dense_gaussian_projection, project_rows, projection_metadata
from .real_dimension_sweep import _exact_projected_rankings, _ranking_hash
from .real_policy_certify import (
    RealPolicyCertificationConfig,
    RealPolicyCertificationError,
    _coordinate_work,
    _prefix_hash,
    _validate_policy_bundle,
    load_real_policy_certification_config,
)
from .text_embeddings import load_text_embedding_config, validate_text_embedding_cache
from .utils import fingerprint, stable_id_hash, write_json


class RealTuneDiagnosticsError(ValueError):
    pass


@dataclass(frozen=True)
class RealTuneDiagnosticsConfig:
    raw: Dict[str, Any]
    config_fingerprint: str
    policy_binding_config_fingerprint: str
    policy_source_result_fingerprint: str
    evaluation_split: str
    query_split_n: int
    query_split_id_hash: str
    evidence_cutoffs: list[int]
    shuffle_seed: int
    shuffle_repetitions: int
    fixed_quality_metrics: list[str]
    reporting_role: str


_SHA256_LENGTH = 64
_POLICY_NAMES = ("fixed_reference", "monotone_binned", "tri_predict")
_ADAPTIVE_NAMES = ("monotone_binned", "tri_predict")
_RESULT_NAMES = (
    "per_query.jsonl",
    "fixed_grid.json",
    "shuffled_controls.jsonl",
    "summary.json",
    "report.md",
)
_REPORTING_ROLE = "posthoc_tune_only_evidence_and_allocation_diagnostics"
_PROTECTED_ACCESS = "forbidden"
_POLICY_SELECTION = "forbidden"
_NEW_CERTIFICATION = "forbidden"
_RETUNING = "forbidden"
_FIXED_COST_RULE = "bracket_adaptive_mean_budget_on_frozen_grid"
_FIXED_QUALITY_RULE = (
    "smallest_frozen_fixed_budget_with_mean_metric_at_least_adaptive_mean"
)
_DIAGNOSTIC_ROLE = "posthoc_query_tune_diagnostic_not_policy_selection"
_SHUFFLE_UNIT = "permute_pilot_lid_and_validity_pairs_across_query_tune_ids"
_SHUFFLE_METRICS = [
    "embedding_retention",
    "candidate_evidence_recall",
    "final_context_evidence_recall_at_k_ctx",
]
_P_VALUE_RULE = (
    "one_plus_controls_at_least_observed_over_repetitions_plus_one"
)


def _exact_keys(value: Any, expected: set[str], name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RealTuneDiagnosticsError(f"{name} must be an object")
    if set(value) != expected:
        raise RealTuneDiagnosticsError(
            f"invalid {name} keys; missing={sorted(expected-set(value))}, "
            f"unknown={sorted(set(value)-expected)}"
        )
    return dict(value)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RealTuneDiagnosticsError(f"{name} must be a positive integer")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise RealTuneDiagnosticsError(f"{name} must be a SHA-256 value")
    result = value.strip().lower()
    if len(result) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise RealTuneDiagnosticsError(f"{name} must be a SHA-256 value")
    return result


def _integer_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise RealTuneDiagnosticsError(f"{name} must be a nonempty list")
    result = [_positive_integer(item, f"{name} item") for item in value]
    if result != sorted(set(result)):
        raise RealTuneDiagnosticsError(f"{name} must be strictly increasing")
    return result


def load_real_tune_diagnostics_config(
    path: Union[str, Path],
) -> RealTuneDiagnosticsConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealTuneDiagnosticsError(
            f"cannot load tune diagnostics config {config_path}: {exc}"
        ) from exc
    root = _exact_keys(
        raw,
        {
            "schema_version",
            "benchmark",
            "policy_binding_config_fingerprint",
            "policy_source_result_fingerprint",
            "evaluation_split",
            "query_split",
            "evidence",
            "matched_comparisons",
            "shuffled_lid",
            "reporting",
        },
        "root",
    )
    if (
        root["schema_version"] != 1
        or root["benchmark"] != "real_tune_only_evidence_diagnostics_v1"
        or root["evaluation_split"] != "query_tune"
    ):
        raise RealTuneDiagnosticsError(
            "diagnostics accept the posthoc query_tune protocol only"
        )
    query_split = _exact_keys(root["query_split"], {"n", "id_hash"}, "query_split")
    evidence = _exact_keys(
        root["evidence"],
        {
            "source",
            "cutoffs",
            "candidate_metrics",
            "final_context_metrics",
            "empty_qrel_behavior",
            "include_exact_original_reference",
        },
        "evidence",
    )
    if (
        evidence["source"] != "prepared_query_tune_qrels"
        or evidence["candidate_metrics"] != ["evidence_hit", "evidence_recall"]
        or evidence["final_context_metrics"]
        != ["evidence_hit", "evidence_recall", "ndcg"]
        or evidence["empty_qrel_behavior"]
        != "record_null_metrics_and_do_not_drop_query"
        or evidence["include_exact_original_reference"] is not True
    ):
        raise RealTuneDiagnosticsError("evidence diagnostic contract is not frozen")
    matched = _exact_keys(
        root["matched_comparisons"],
        {
            "fixed_cost_rule",
            "fixed_quality_metrics",
            "fixed_quality_rule",
            "role",
        },
        "matched_comparisons",
    )
    if (
        matched["fixed_cost_rule"] != _FIXED_COST_RULE
        or matched["fixed_quality_metrics"] != _SHUFFLE_METRICS
        or matched["fixed_quality_rule"] != _FIXED_QUALITY_RULE
        or matched["role"] != _DIAGNOSTIC_ROLE
    ):
        raise RealTuneDiagnosticsError("matched-comparison contract is not frozen")
    shuffled = _exact_keys(
        root["shuffled_lid"],
        {"seed", "repetitions", "unit", "policies", "metrics", "p_value"},
        "shuffled_lid",
    )
    if (
        shuffled["unit"] != _SHUFFLE_UNIT
        or shuffled["policies"] != list(_ADAPTIVE_NAMES)
        or shuffled["metrics"] != _SHUFFLE_METRICS
        or shuffled["p_value"] != _P_VALUE_RULE
    ):
        raise RealTuneDiagnosticsError("shuffled-LID contract is not frozen")
    reporting = _exact_keys(
        root["reporting"],
        {
            "role",
            "protected_split_access",
            "policy_selection",
            "new_certification",
            "retuning",
        },
        "reporting",
    )
    if reporting != {
        "role": _REPORTING_ROLE,
        "protected_split_access": _PROTECTED_ACCESS,
        "policy_selection": _POLICY_SELECTION,
        "new_certification": _NEW_CERTIFICATION,
        "retuning": _RETUNING,
    }:
        raise RealTuneDiagnosticsError("reporting safeguards are not frozen")
    return RealTuneDiagnosticsConfig(
        raw=root,
        config_fingerprint=fingerprint(root),
        policy_binding_config_fingerprint=_sha256(
            root["policy_binding_config_fingerprint"],
            "policy_binding_config_fingerprint",
        ),
        policy_source_result_fingerprint=_sha256(
            root["policy_source_result_fingerprint"],
            "policy_source_result_fingerprint",
        ),
        evaluation_split="query_tune",
        query_split_n=_positive_integer(query_split["n"], "query_split.n"),
        query_split_id_hash=_sha256(query_split["id_hash"], "query_split.id_hash"),
        evidence_cutoffs=_integer_list(evidence["cutoffs"], "evidence.cutoffs"),
        shuffle_seed=_positive_integer(shuffled["seed"], "shuffled_lid.seed"),
        shuffle_repetitions=_positive_integer(
            shuffled["repetitions"], "shuffled_lid.repetitions"
        ),
        fixed_quality_metrics=list(matched["fixed_quality_metrics"]),
        reporting_role=_REPORTING_ROLE,
    )


def _load_jsonl(path: Path, name: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RealTuneDiagnosticsError(
                        f"{name} line {line_number} must be an object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RealTuneDiagnosticsError(f"cannot load {name}: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
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


def _file_identity(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
    except OSError as exc:
        raise RealTuneDiagnosticsError(f"cannot hash artifact {path}: {exc}") from exc
    return {"bytes": size, "sha256": digest.hexdigest()}


def _ndcg(retrieved_ids: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    gains = [
        (2.0 ** relevance.get(doc_id, 0) - 1.0) / math.log2(rank + 2.0)
        for rank, doc_id in enumerate(retrieved_ids[:k])
    ]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    denominator = sum(
        (2.0 ** value - 1.0) / math.log2(rank + 2.0)
        for rank, value in enumerate(ideal)
    )
    if denominator <= 0.0:
        raise RealTuneDiagnosticsError("nDCG requires at least one positive qrel")
    return float(sum(gains) / denominator)


def _candidate_evidence(
    candidate_ids: Sequence[str], relevance: Mapping[str, int]
) -> Dict[str, Any]:
    if not relevance:
        return {
            "evidence_hit": None,
            "evidence_recall": None,
            "relevant_ids": [],
        }
    retained = sorted(set(candidate_ids).intersection(relevance))
    return {
        "evidence_hit": bool(retained),
        "evidence_recall": float(len(retained) / len(relevance)),
        "relevant_ids": retained,
    }


def _final_evidence(
    retrieved_ids: Sequence[str], relevance: Mapping[str, int], cutoffs: Sequence[int]
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for cutoff in cutoffs:
        if not relevance:
            result[str(cutoff)] = {
                "evidence_hit": None,
                "evidence_recall": None,
                "ndcg": None,
            }
            continue
        retained = set(retrieved_ids[:cutoff]).intersection(relevance)
        result[str(cutoff)] = {
            "evidence_hit": bool(retained),
            "evidence_recall": float(len(retained) / len(relevance)),
            "ndcg": _ndcg(retrieved_ids, relevance, cutoff),
        }
    return result


def _nullable_mean(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    observed = [float(value) for value in values if value is not None]
    return {
        "labeled_n": len(observed),
        "mean": None if not observed else float(np.mean(observed)),
    }


def _aggregate_candidate(values: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    hits = _nullable_mean(
        [None if value["evidence_hit"] is None else float(value["evidence_hit"]) for value in values]
    )
    recalls = _nullable_mean([value["evidence_recall"] for value in values])
    return {
        "labeled_n": recalls["labeled_n"],
        "empty_qrel_n": len(values) - recalls["labeled_n"],
        "mean_evidence_hit": hits["mean"],
        "mean_evidence_recall": recalls["mean"],
    }


def _aggregate_final(
    values: Sequence[Mapping[str, Mapping[str, Any]]], cutoffs: Sequence[int]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for cutoff in cutoffs:
        rows = [value[str(cutoff)] for value in values]
        hits = _nullable_mean(
            [None if row["evidence_hit"] is None else float(row["evidence_hit"]) for row in rows]
        )
        recalls = _nullable_mean([row["evidence_recall"] for row in rows])
        ndcgs = _nullable_mean([row["ndcg"] for row in rows])
        result[str(cutoff)] = {
            "labeled_n": recalls["labeled_n"],
            "empty_qrel_n": len(rows) - recalls["labeled_n"],
            "mean_evidence_hit": hits["mean"],
            "mean_evidence_recall": recalls["mean"],
            "mean_ndcg": ndcgs["mean"],
        }
    return result


def _budget_record(
    *,
    budget: int,
    ranking: np.ndarray,
    original_distances: np.ndarray,
    tie_rank: np.ndarray,
    corpus_ids: Sequence[str],
    exact_rows: set[int],
    exact_ids: Sequence[str],
    relevance: Mapping[str, int],
    k_gt: int,
    cutoffs: Sequence[int],
) -> Dict[str, Any]:
    candidate_rows = np.asarray(ranking[:budget], dtype=np.int64)
    candidate_ids = [corpus_ids[row] for row in candidate_rows]
    order = np.lexsort((tie_rank[candidate_rows], original_distances[:budget]))[:k_gt]
    reranked_rows = candidate_rows[order]
    reranked_ids = [corpus_ids[row] for row in reranked_rows]
    overlap = len(exact_rows.intersection(candidate_rows.tolist()))
    reranked_overlap = len(set(exact_ids).intersection(reranked_ids))
    if overlap != reranked_overlap:
        raise RealTuneDiagnosticsError(
            "candidate overlap disagrees with exact original rerank overlap"
        )
    return {
        "budget": int(budget),
        "embedding_retention": float(overlap / k_gt),
        "candidate_overlap": overlap,
        "reranked_overlap": reranked_overlap,
        "candidate_prefix_rows_sha256": _prefix_hash(candidate_rows),
        "candidate_set_evidence": _candidate_evidence(candidate_ids, relevance),
        "reranked_top_k_ids": reranked_ids,
        "final_context_evidence": _final_evidence(reranked_ids, relevance, cutoffs),
    }


def _aggregate_entries(
    entries: Sequence[Mapping[str, Any]], cutoffs: Sequence[int]
) -> Dict[str, Any]:
    retentions = [float(value["embedding_retention"]) for value in entries]
    budgets = [int(value["budget"]) for value in entries]
    return {
        "n": len(entries),
        "mean_budget": float(np.mean(budgets)),
        "mean_embedding_retention": float(np.mean(retentions)),
        "minimum_embedding_retention": float(np.min(retentions)),
        "candidate_set_evidence": _aggregate_candidate(
            [value["candidate_set_evidence"] for value in entries]
        ),
        "final_context_evidence": _aggregate_final(
            [value["final_context_evidence"] for value in entries], cutoffs
        ),
    }


def _metric_value(summary: Mapping[str, Any], metric: str, k_ctx: int) -> Optional[float]:
    if metric == "embedding_retention":
        return float(summary["mean_embedding_retention"])
    if metric == "candidate_evidence_recall":
        value = summary["candidate_set_evidence"]["mean_evidence_recall"]
    elif metric == "final_context_evidence_recall_at_k_ctx":
        value = summary["final_context_evidence"][str(k_ctx)][
            "mean_evidence_recall"
        ]
    else:
        raise RealTuneDiagnosticsError(f"unsupported comparison metric: {metric}")
    return None if value is None else float(value)


def _fixed_matches(
    *,
    adaptive: Mapping[str, Any],
    fixed_grid: Mapping[str, Mapping[str, Any]],
    grid: Sequence[int],
    metrics: Sequence[str],
    k_ctx: int,
) -> Dict[str, Any]:
    adaptive_budget = float(adaptive["mean_budget"])
    lower = max((value for value in grid if value <= adaptive_budget), default=grid[0])
    upper = min((value for value in grid if value >= adaptive_budget), default=grid[-1])
    quality: Dict[str, Any] = {}
    for metric in metrics:
        target = _metric_value(adaptive, metric, k_ctx)
        match: Optional[int] = None
        fixed_value: Optional[float] = None
        if target is not None:
            for budget in grid:
                candidate = _metric_value(fixed_grid[str(budget)], metric, k_ctx)
                if candidate is not None and candidate >= target:
                    match = int(budget)
                    fixed_value = candidate
                    break
        quality[metric] = {
            "adaptive_mean": target,
            "matched_fixed_budget": match,
            "matched_fixed_mean": fixed_value,
            "terminal_grid_not_met": target is not None and match is None,
        }
    return {
        "role": _DIAGNOSTIC_ROLE,
        "fixed_cost_bracket": {
            "adaptive_mean_budget": adaptive_budget,
            "lower_fixed_budget": int(lower),
            "upper_fixed_budget": int(upper),
        },
        "fixed_quality": quality,
    }


def _control_metric(entry: Mapping[str, Any], metric: str, k_ctx: int) -> Optional[float]:
    if metric == "embedding_retention":
        return float(entry["embedding_retention"])
    if metric == "candidate_evidence_recall":
        value = entry["candidate_set_evidence"]["evidence_recall"]
    else:
        value = entry["final_context_evidence"][str(k_ctx)]["evidence_recall"]
    return None if value is None else float(value)


def _run_shuffled_controls(
    *,
    config: RealTuneDiagnosticsConfig,
    source_records: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    policies: Mapping[str, Any],
    k_ctx: int,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(config.shuffle_seed)
    controls: list[Dict[str, Any]] = []
    observed: Dict[str, Dict[str, Any]] = {}
    for name in _ADAPTIVE_NAMES:
        entries = [record["policies"][name] for record in records]
        observed[name] = {
            "mean_budget": float(np.mean([entry["budget"] for entry in entries])),
            **{
                metric: _nullable_mean(
                    [_control_metric(entry, metric, k_ctx) for entry in entries]
                )["mean"]
                for metric in _SHUFFLE_METRICS
            },
        }
    for repetition in range(config.shuffle_repetitions):
        permutation = rng.permutation(len(records))
        row: Dict[str, Any] = {
            "schema_version": 1,
            "repetition": repetition,
            "permutation_sha256": hashlib.sha256(
                np.asarray(permutation, dtype="<i8").tobytes(order="C")
            ).hexdigest(),
            "policies": {},
        }
        for name in _ADAPTIVE_NAMES:
            policy = policies[name]
            budgets: list[int] = []
            metric_values: Dict[str, list[Optional[float]]] = {
                metric: [] for metric in _SHUFFLE_METRICS
            }
            for query_index, source_index in enumerate(permutation.tolist()):
                source = source_records[source_index]
                lid = source["pilot_lid"]
                decision = policy.choose(float(lid["clipped"]), bool(lid["valid"]))
                budget = int(decision.budget)
                budgets.append(budget)
                entry = records[query_index]["fixed_grid"][str(budget)]
                for metric in _SHUFFLE_METRICS:
                    metric_values[metric].append(_control_metric(entry, metric, k_ctx))
            row["policies"][name] = {
                "mean_budget": float(np.mean(budgets)),
                **{
                    metric: _nullable_mean(values)["mean"]
                    for metric, values in metric_values.items()
                },
            }
        controls.append(row)
    summary: Dict[str, Any] = {
        "seed": config.shuffle_seed,
        "repetitions": config.shuffle_repetitions,
        "unit": _SHUFFLE_UNIT,
        "p_value_rule": _P_VALUE_RULE,
        "policies": {},
    }
    for name in _ADAPTIVE_NAMES:
        value: Dict[str, Any] = {"observed": observed[name], "controls": {}}
        for metric in ["mean_budget", *_SHUFFLE_METRICS]:
            samples = [row["policies"][name][metric] for row in controls]
            numeric = [float(sample) for sample in samples if sample is not None]
            observed_value = observed[name][metric]
            control_summary = {
                "n": len(numeric),
                "mean": None if not numeric else float(np.mean(numeric)),
                "p05": None if not numeric else float(np.quantile(numeric, 0.05)),
                "p95": None if not numeric else float(np.quantile(numeric, 0.95)),
                "minimum": None if not numeric else float(np.min(numeric)),
                "maximum": None if not numeric else float(np.max(numeric)),
                "observed_minus_control_mean": (
                    None
                    if not numeric or observed_value is None
                    else float(observed_value - np.mean(numeric))
                ),
                "one_sided_p": (
                    None
                    if not numeric or observed_value is None
                    else float(
                        (1 + sum(sample >= observed_value for sample in numeric))
                        / (len(numeric) + 1)
                    )
                ),
            }
            value["controls"][metric] = control_summary
        summary["policies"][name] = value
    return controls, summary


def _report(summary: Mapping[str, Any]) -> str:
    def display(value: Optional[float], digits: int = 6) -> str:
        return "null" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "# SciFact tune-only evidence and allocation diagnostics",
        "",
        "This is a posthoc diagnostic on `query_tune`. It does not select or "
        "retune a policy, access a protected split, or create a certificate.",
        "",
        f"- queries: {summary['n_queries']}",
        f"- corpus vectors: {summary['corpus_size']}",
        f"- fixed projection dimension: {summary['m_prime']}",
        f"- shuffled-LID repetitions: {summary['shuffled_lid']['repetitions']}",
        "",
        "## Frozen-policy evidence",
        "",
        "| policy | mean M | retention | candidate recall | final recall@k_ctx |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for name in _POLICY_NAMES:
        value = summary["policies"][name]
        candidate = value["candidate_set_evidence"]["mean_evidence_recall"]
        final = value["final_context_evidence"][str(summary["k_ctx"])][
            "mean_evidence_recall"
        ]
        lines.append(
            f"| {name} | {value['mean_budget']:.3f} | "
            f"{value['mean_embedding_retention']:.6f} | "
            f"{display(candidate)} | {display(final)} |"
        )
    lines.extend(
        [
            "",
            "## Matched fixed comparisons",
            "",
            "| adaptive | fixed-cost bracket | retention match | candidate-recall match | final-recall match |",
            "| :--- | :---: | ---: | ---: | ---: |",
        ]
    )
    for name in _ADAPTIVE_NAMES:
        matched = summary["policies"][name]["matched_fixed"]
        bracket = matched["fixed_cost_bracket"]
        quality = matched["fixed_quality"]
        lines.append(
            f"| {name} | {bracket['lower_fixed_budget']}–"
            f"{bracket['upper_fixed_budget']} | "
            f"{quality['embedding_retention']['matched_fixed_budget']} | "
            f"{quality['candidate_evidence_recall']['matched_fixed_budget']} | "
            f"{quality['final_context_evidence_recall_at_k_ctx']['matched_fixed_budget']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen LID-bin relationships",
            "",
            "| bin | n | pilot LID | oracle LID | monotone M/R | Tri M/R |",
            "| ---: | ---: | ---: | ---: | :---: | :---: |",
        ]
    )
    for stratum in summary["lid_strata"]:
        monotone = stratum["policies"]["monotone_binned"]
        tri = stratum["policies"]["tri_predict"]
        lines.append(
            f"| {stratum['lid_bin']} | {stratum['n']} | "
            f"{display(stratum['pilot_lid_mean'], 3)} | "
            f"{display(stratum['oracle_lid_mean'], 3)} | "
            f"{monotone['mean_budget']:.1f}/{monotone['mean_embedding_retention']:.4f} | "
            f"{tri['mean_budget']:.1f}/{tri['mean_embedding_retention']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Shuffled-LID control",
            "",
            "| policy | observed retention | shuffled mean | observed - shuffled | p |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in _ADAPTIVE_NAMES:
        value = summary["shuffled_lid"]["policies"][name]
        control = value["controls"]["embedding_retention"]
        lines.append(
            f"| {name} | {value['observed']['embedding_retention']:.6f} | "
            f"{control['mean']:.6f} | "
            f"{control['observed_minus_control_mean']:.6f} | "
            f"{control['one_sided_p']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Matched fixed-cost and fixed-quality comparisons, the complete "
            "fixed grid, LID strata, and every shuffled repetition are saved in "
            "machine-readable artifacts. Timings are diagnostic and excluded "
            "from the result identity.",
            "",
        ]
    )
    return "\n".join(lines)


def run_real_tune_diagnostics(
    config: RealTuneDiagnosticsConfig,
    policy_binding_config: RealPolicyCertificationConfig,
    prepared_dir: Union[str, Path],
    embedding_config_path: Union[str, Path],
    embedding_cache_dir: Union[str, Path],
    policy_run_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    if config.evaluation_split != "query_tune":
        raise RealTuneDiagnosticsError("diagnostics accept query_tune only")
    if (
        policy_binding_config.config_fingerprint
        != config.policy_binding_config_fingerprint
        or policy_binding_config.policy_source.result_fingerprint
        != config.policy_source_result_fingerprint
    ):
        raise RealTuneDiagnosticsError("frozen policy binding mismatch")
    if (
        max(config.evidence_cutoffs) > policy_binding_config.k_gt
        or policy_binding_config.k_ctx not in config.evidence_cutoffs
        or policy_binding_config.k_gt not in config.evidence_cutoffs
    ):
        raise RealTuneDiagnosticsError("evidence cutoffs disagree with frozen search")
    prepared = Path(prepared_dir)
    embedding_cache = Path(embedding_cache_dir)
    policy_run = Path(policy_run_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite tune diagnostics output: {output}")

    # The complete frozen policy bundle is checked before dataset/query/qrel access.
    try:
        bundle = _validate_policy_bundle(policy_run, policy_binding_config)
    except RealPolicyCertificationError as exc:
        raise RealTuneDiagnosticsError(str(exc)) from exc
    source_records = _load_jsonl(policy_run / "per_query.jsonl", "policy records")
    if (
        len(source_records) != config.query_split_n
        or {record.get("split") for record in source_records} != {"query_tune"}
        or stable_id_hash([record.get("query_id") for record in source_records])
        != config.query_split_id_hash
    ):
        raise RealTuneDiagnosticsError("frozen policy records are not query_tune only")

    embedding_config = load_text_embedding_config(embedding_config_path)
    if embedding_config.config_fingerprint != policy_binding_config.embedding_config_fingerprint:
        raise RealTuneDiagnosticsError("embedding config fingerprint mismatch")
    validation = validate_text_embedding_cache(
        embedding_config, prepared, embedding_cache
    )
    dataset_manifest = validation["dataset_manifest"]
    embedding_manifest = validation["embedding_manifest"]
    if (
        dataset_manifest.get("fingerprint")
        != policy_binding_config.dataset_manifest_fingerprint
        or validation.get("request_fingerprint")
        != policy_binding_config.embedding_request_fingerprint
        or embedding_manifest.get("fingerprint")
        != policy_binding_config.embedding_manifest_fingerprint
    ):
        raise RealTuneDiagnosticsError("dataset/embedding input identity mismatch")
    split_identity = dataset_manifest.get("splits", {}).get("query_tune")
    if not isinstance(split_identity, dict) or (
        split_identity.get("n") != config.query_split_n
        or split_identity.get("id_hash") != config.query_split_id_hash
    ):
        raise RealTuneDiagnosticsError("dataset tune split identity mismatch")

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
        raise RealTuneDiagnosticsError(
            "query embedding rows do not match prepared query order"
        )
    selected_rows = [
        index
        for index, row in enumerate(query_rows)
        if row.get("split") == "query_tune"
    ]
    selected_query_ids = [prepared_query_ids[index] for index in selected_rows]
    if (
        len(selected_rows) != config.query_split_n
        or selected_query_ids != [record["query_id"] for record in source_records]
        or stable_id_hash(selected_query_ids) != config.query_split_id_hash
    ):
        raise RealTuneDiagnosticsError("selected queries are not the frozen tune IDs")

    corpus_ids = corpus_table.ids.tolist()
    corpus_id_set = set(corpus_ids)
    qrels_by_query: Dict[str, Dict[str, int]] = {
        query_id: {} for query_id in selected_query_ids
    }
    for row in _load_jsonl(prepared / "qrels.jsonl", "prepared qrels"):
        if row.get("split") != "query_tune":
            continue
        query_id = row.get("query_id")
        doc_id = row.get("doc_id")
        relevance = row.get("relevance")
        if query_id not in qrels_by_query:
            raise RealTuneDiagnosticsError("tune qrel references unknown query")
        if doc_id not in corpus_id_set:
            raise RealTuneDiagnosticsError("tune qrel references unknown document")
        if isinstance(relevance, bool) or not isinstance(relevance, int) or relevance < 1:
            raise RealTuneDiagnosticsError("tune qrel relevance must be positive")
        if doc_id in qrels_by_query[query_id]:
            raise RealTuneDiagnosticsError("duplicate tune qrel pair")
        qrels_by_query[query_id][doc_id] = relevance

    corpus = np.asarray(corpus_table.vectors, dtype=np.float32)
    tune_queries = np.asarray(
        query_table.vectors[np.asarray(selected_rows, dtype=np.int64)],
        dtype=np.float32,
    )
    corpus_size, dimension = corpus.shape
    if (
        tune_queries.shape != (config.query_split_n, dimension)
        or dimension != embedding_config.model.embedding_dimension
        or policy_binding_config.m_grid[-1] != corpus_size
        or bundle.analytic_tri.corpus_size != corpus_size
    ):
        raise RealTuneDiagnosticsError("embedding shape disagrees with frozen policies")

    projection_started = perf_counter()
    matrix = dense_gaussian_projection(
        policy_binding_config.m_prime,
        dimension,
        policy_binding_config.projection_seed,
    )
    projected_corpus = project_rows(corpus, matrix)
    projected_queries = project_rows(tune_queries, matrix)
    projection_ms = (perf_counter() - projection_started) * 1000.0
    metadata = projection_metadata(
        dimension=dimension,
        m_prime=policy_binding_config.m_prime,
        seed=policy_binding_config.projection_seed,
        normalization=True,
        embedding_model=(
            f"{embedding_config.model.name}@{embedding_config.model.revision}"
        ),
        corpus_hash=embedding_manifest["arrays"]["corpus"]["array_fingerprint"],
    )
    if metadata["fingerprint"] != policy_binding_config.projection_fingerprint:
        raise RealTuneDiagnosticsError("reconstructed projection fingerprint mismatch")
    rankings, projected_search_ms = _exact_projected_rankings(
        projected_corpus,
        projected_queries,
        corpus_ids,
        k=corpus_size,
        batch_size=policy_binding_config.query_batch_size,
    )

    tie_rank = np.argsort(
        np.argsort(np.asarray(corpus_ids, dtype=str), kind="stable"), kind="stable"
    )
    corpus_id_to_row = {doc_id: row for row, doc_id in enumerate(corpus_ids)}
    records: list[Dict[str, Any]] = []
    rerank_started = perf_counter()
    for query_index, (ranking, source) in enumerate(zip(rankings, source_records)):
        if _ranking_hash(ranking) != source.get("projected_ranking_rows_sha256"):
            raise RealTuneDiagnosticsError("projected ranking hash mismatch")
        exact_ids = source.get("exact_top_k_ids")
        if (
            not isinstance(exact_ids, list)
            or len(exact_ids) != policy_binding_config.k_gt
            or any(doc_id not in corpus_id_to_row for doc_id in exact_ids)
        ):
            raise RealTuneDiagnosticsError("invalid frozen exact top-k IDs")
        exact_rows = {corpus_id_to_row[doc_id] for doc_id in exact_ids}
        query = np.asarray(tune_queries[query_index], dtype=np.float64)
        difference = np.asarray(corpus[ranking], dtype=np.float64) - query
        original_distances = np.einsum("ij,ij->i", difference, difference)
        relevance = qrels_by_query[selected_query_ids[query_index]]
        fixed_grid: Dict[str, Any] = {}
        saved_grid = source.get("fixed_retention_by_budget")
        if not isinstance(saved_grid, dict):
            raise RealTuneDiagnosticsError("frozen tune record has no fixed budget grid")
        for budget in policy_binding_config.m_grid:
            entry = _budget_record(
                budget=budget,
                ranking=ranking,
                original_distances=original_distances,
                tie_rank=tie_rank,
                corpus_ids=corpus_ids,
                exact_rows=exact_rows,
                exact_ids=exact_ids,
                relevance=relevance,
                k_gt=policy_binding_config.k_gt,
                cutoffs=config.evidence_cutoffs,
            )
            if entry["embedding_retention"] != saved_grid.get(str(budget)):
                raise RealTuneDiagnosticsError("fixed-grid retention mismatch")
            fixed_grid[str(budget)] = entry
        lid = source.get("pilot_lid")
        if not isinstance(lid, dict) or lid.get("clipped") is None:
            raise RealTuneDiagnosticsError("frozen tune record has no deployable LID")
        lid_value = float(lid["clipped"])
        lid_valid = bool(lid.get("valid"))
        decisions = {
            "fixed_reference": bundle.fixed.choose(lid_value, lid_valid),
            "monotone_binned": bundle.monotone.choose(lid_value, lid_valid),
            "tri_predict": bundle.compiled_tri.choose(lid_value, lid_valid),
        }
        saved_policy_fields = {
            "monotone_binned": source.get("monotone_binned"),
            "tri_predict": source.get("tri_predict"),
        }
        policy_records: Dict[str, Any] = {}
        for name, decision in decisions.items():
            if decision.budget not in policy_binding_config.m_grid:
                raise RealTuneDiagnosticsError("policy emitted budget outside frozen grid")
            saved = saved_policy_fields.get(name)
            if name != "fixed_reference" and (
                not isinstance(saved, dict)
                or saved.get("chosen_m") != decision.budget
                or saved.get("embedding_retention")
                != fixed_grid[str(decision.budget)]["embedding_retention"]
            ):
                raise RealTuneDiagnosticsError("replayed policy disagrees with frozen tune result")
            policy_value = dict(fixed_grid[str(decision.budget)])
            policy_value.update(
                {
                    "lid_bin": int(decision.bin_index),
                    "used_lid_fallback": bool(decision.used_fallback),
                    "policy_saturated": bool(decision.saturated),
                    "predicted_retention": (
                        None if not isinstance(saved, dict) else saved.get("predicted_retention")
                    ),
                    "raw_predicted_retention": (
                        None
                        if not isinstance(saved, dict)
                        else saved.get("raw_predicted_retention")
                    ),
                }
            )
            policy_records[name] = policy_value
        policy_records["tri_predict"]["compiled_decision_match"] = True
        records.append(
            {
                "query_index": query_index,
                "query_id": selected_query_ids[query_index],
                "split": "query_tune",
                "qrel_count": len(relevance),
                "relevance_by_doc_id": dict(sorted(relevance.items())),
                "pilot_lid": source["pilot_lid"],
                "oracle_lid": source.get("oracle_lid"),
                "oracle_lid_role": "diagnostic_only_not_used_for_decisions",
                "exact_top_k_ids": exact_ids,
                "exact_original_evidence": _final_evidence(
                    exact_ids, relevance, config.evidence_cutoffs
                ),
                "projected_ranking_rows_sha256": _ranking_hash(ranking),
                "fixed_grid": fixed_grid,
                "policies": policy_records,
            }
        )
    full_grid_rerank_ms = (perf_counter() - rerank_started) * 1000.0

    fixed_grid_summary = {
        "schema_version": 1,
        "kind": "real_tune_fixed_grid_evidence_v1",
        "data_scope": "query_tune_only",
        "budgets": {
            str(budget): {
                **_aggregate_entries(
                    [record["fixed_grid"][str(budget)] for record in records],
                    config.evidence_cutoffs,
                ),
                "coordinate_work": _coordinate_work(
                    corpus_size=corpus_size,
                    dimension=dimension,
                    m_prime=policy_binding_config.m_prime,
                    mean_budget=float(budget),
                ),
            }
            for budget in policy_binding_config.m_grid
        },
    }
    fixed_grid_summary["fingerprint"] = fingerprint(fixed_grid_summary)
    policy_summaries: Dict[str, Any] = {}
    for name in _POLICY_NAMES:
        aggregate = _aggregate_entries(
            [record["policies"][name] for record in records],
            config.evidence_cutoffs,
        )
        aggregate["coordinate_work"] = _coordinate_work(
            corpus_size=corpus_size,
            dimension=dimension,
            m_prime=policy_binding_config.m_prime,
            mean_budget=aggregate["mean_budget"],
        )
        aggregate["policy_fingerprint"] = (
            bundle.fixed_fingerprint
            if name == "fixed_reference"
            else bundle.monotone.serialize()["fingerprint"]
            if name == "monotone_binned"
            else bundle.analytic_tri.serialize()["fingerprint"]
        )
        if name in _ADAPTIVE_NAMES:
            aggregate["matched_fixed"] = _fixed_matches(
                adaptive=aggregate,
                fixed_grid=fixed_grid_summary["budgets"],
                grid=policy_binding_config.m_grid,
                metrics=config.fixed_quality_metrics,
                k_ctx=policy_binding_config.k_ctx,
            )
        policy_summaries[name] = aggregate

    strata: list[Dict[str, Any]] = []
    bin_values = sorted(
        set(record["policies"]["monotone_binned"]["lid_bin"] for record in records)
    )
    for bin_index in bin_values:
        members = [
            record
            for record in records
            if record["policies"]["monotone_binned"]["lid_bin"] == bin_index
        ]
        pilot_values = [float(record["pilot_lid"]["clipped"]) for record in members]
        paired = [
            (
                float(record["pilot_lid"]["clipped"]),
                float(record["oracle_lid"]["clipped"]),
            )
            for record in members
            if isinstance(record.get("oracle_lid"), dict)
            and record["pilot_lid"].get("valid")
            and record["oracle_lid"].get("valid")
        ]
        strata.append(
            {
                "lid_bin": int(bin_index),
                "n": len(members),
                "pilot_lid_mean": float(np.mean(pilot_values)),
                "oracle_lid_mean": (
                    None if not paired else float(np.mean([value[1] for value in paired]))
                ),
                "pilot_oracle_mae": (
                    None
                    if not paired
                    else float(np.mean([abs(left - right) for left, right in paired]))
                ),
                "policies": {
                    name: _aggregate_entries(
                        [record["policies"][name] for record in members],
                        config.evidence_cutoffs,
                    )
                    for name in _POLICY_NAMES
                },
            }
        )

    shuffle_started = perf_counter()
    controls, shuffled_summary = _run_shuffled_controls(
        config=config,
        source_records=source_records,
        records=records,
        policies={
            "monotone_binned": bundle.monotone,
            "tri_predict": bundle.compiled_tri,
        },
        k_ctx=policy_binding_config.k_ctx,
    )
    shuffle_ms = (perf_counter() - shuffle_started) * 1000.0
    summary = {
        "schema_version": 1,
        "kind": "real_tune_evidence_allocation_diagnostics_summary_v1",
        "data_scope": "query_tune_only",
        "reporting_role": config.reporting_role,
        "n_queries": len(records),
        "empty_qrel_queries": int(sum(record["qrel_count"] == 0 for record in records)),
        "corpus_size": corpus_size,
        "embedding_dimension": dimension,
        "m_prime": policy_binding_config.m_prime,
        "projection_seed": policy_binding_config.projection_seed,
        "k_ctx": policy_binding_config.k_ctx,
        "k_gt": policy_binding_config.k_gt,
        "m_pilot": policy_binding_config.m_pilot,
        "evidence_cutoffs": config.evidence_cutoffs,
        "policy_source_result_fingerprint": config.policy_source_result_fingerprint,
        "fixed_grid_fingerprint": fixed_grid_summary["fingerprint"],
        "policies": policy_summaries,
        "exact_original_reference": {
            "final_context_evidence": _aggregate_final(
                [record["exact_original_evidence"] for record in records],
                config.evidence_cutoffs,
            )
        },
        "lid_strata": strata,
        "shuffled_lid": shuffled_summary,
        "protected_split_access": _PROTECTED_ACCESS,
        "policy_selection": _POLICY_SELECTION,
        "new_certification": _NEW_CERTIFICATION,
        "retuning": _RETUNING,
    }
    summary["fingerprint"] = fingerprint(summary)
    timings = {
        "role": "systems_diagnostic_excluded_from_result_identity",
        "projection_and_materialization_ms": projection_ms,
        "exact_projected_full_ranking_ms": projected_search_ms,
        "full_fixed_grid_original_distance_and_rerank_ms": full_grid_rerank_ms,
        "shuffled_lid_ms": shuffle_ms,
        "projected_scan_count_per_query": 1,
        "projected_distance_evaluations": len(records) * corpus_size,
        "original_distance_evaluations_full_grid_diagnostic": len(records)
        * corpus_size,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        paths = {
            "manifest.json": temporary / "manifest.json",
            "per_query.jsonl": temporary / "per_query.jsonl",
            "fixed_grid.json": temporary / "fixed_grid.json",
            "shuffled_controls.jsonl": temporary / "shuffled_controls.jsonl",
            "summary.json": temporary / "summary.json",
            "timings.json": temporary / "timings.json",
            "report.md": temporary / "report.md",
        }
        _write_jsonl(paths["per_query.jsonl"], records)
        write_json(paths["fixed_grid.json"], fixed_grid_summary)
        _write_jsonl(paths["shuffled_controls.jsonl"], controls)
        write_json(paths["summary.json"], summary)
        write_json(paths["timings.json"], timings)
        paths["report.md"].write_text(_report(summary), encoding="utf-8")
        result_artifacts = {
            name: _file_identity(paths[name]) for name in _RESULT_NAMES
        }
        result_identity = {
            "config_fingerprint": config.config_fingerprint,
            "policy_binding_config_fingerprint": (
                policy_binding_config.config_fingerprint
            ),
            "dataset_manifest_fingerprint": (
                policy_binding_config.dataset_manifest_fingerprint
            ),
            "embedding_manifest_fingerprint": (
                policy_binding_config.embedding_manifest_fingerprint
            ),
            "policy_source_result_fingerprint": (
                config.policy_source_result_fingerprint
            ),
            "query_tune_id_hash": config.query_split_id_hash,
            "artifacts": result_artifacts,
        }
        manifest = {
            "schema_version": 1,
            "kind": "real_tune_evidence_allocation_diagnostics_manifest_v1",
            "data_scope": "query_tune_only",
            "reporting_role": config.reporting_role,
            "config_fingerprint": config.config_fingerprint,
            "policy_binding_config_fingerprint": (
                policy_binding_config.config_fingerprint
            ),
            "dataset_manifest_fingerprint": (
                policy_binding_config.dataset_manifest_fingerprint
            ),
            "embedding_manifest_fingerprint": (
                policy_binding_config.embedding_manifest_fingerprint
            ),
            "query_tune_n": len(records),
            "query_tune_id_hash": config.query_split_id_hash,
            "projection_fingerprint": policy_binding_config.projection_fingerprint,
            "policy_source": {
                "manifest_fingerprint": (
                    policy_binding_config.policy_source.manifest_fingerprint
                ),
                "result_fingerprint": config.policy_source_result_fingerprint,
                "input_artifacts": bundle.input_artifacts,
            },
            "result_artifacts": result_artifacts,
            "result_fingerprint": fingerprint(result_identity),
            "timings_artifact": "timings.json",
            "protected_split_access": _PROTECTED_ACCESS,
            "policy_selection": _POLICY_SELECTION,
            "new_certification": _NEW_CERTIFICATION,
            "retuning": _RETUNING,
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
    parser.add_argument("--policy-binding-config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--embedding-config", required=True, type=Path)
    parser.add_argument("--embedding-cache", required=True, type=Path)
    parser.add_argument("--policy-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_real_tune_diagnostics_config(args.config)
    binding = load_real_policy_certification_config(args.policy_binding_config)
    artifacts = run_real_tune_diagnostics(
        config,
        binding,
        args.dataset,
        args.embedding_config,
        args.embedding_cache,
        args.policy_run,
        args.output,
    )
    print(f"completed tune-only diagnostics: {artifacts['report.md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
