"""Run a frozen, tune-only exact original-space retrieval baseline."""

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
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from .embeddings import load_embedding_array
from .indexes import ExactSquaredL2Index
from .text_embeddings import (
    load_text_embedding_config,
    validate_text_embedding_cache,
)
from .utils import fingerprint, stable_id_hash, write_json


class RealOriginalBaselineError(ValueError):
    pass


@dataclass(frozen=True)
class OriginalSearchConfig:
    normalized_inputs: bool
    distance: str
    arithmetic: str
    stable_tie_break: str
    query_batch_size: int
    cutoffs: list[int]
    k_ctx: int
    k_gt: int


@dataclass(frozen=True)
class RealOriginalBaselineConfig:
    schema_version: int
    benchmark: str
    dataset_manifest_fingerprint: str
    embedding_config_fingerprint: str
    embedding_request_fingerprint: str
    embedding_manifest_fingerprint: str
    evaluation_split: str
    search: OriginalSearchConfig
    raw: Dict[str, Any]
    config_fingerprint: str


_SHA256_LENGTH = 64


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RealOriginalBaselineError(f"{name} must be a nonempty string")
    return value.strip()


def _fingerprint_string(value: Any, name: str) -> str:
    result = _nonempty_string(value, name).lower()
    if len(result) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise RealOriginalBaselineError(f"{name} must be a SHA-256 fingerprint")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RealOriginalBaselineError(f"{name} must be a positive integer")
    return value


def load_real_original_baseline_config(
    path: Union[str, Path],
) -> RealOriginalBaselineConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealOriginalBaselineError(
            f"cannot load original baseline config {config_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise RealOriginalBaselineError("baseline config root must be an object")
    expected_root = {
        "schema_version",
        "benchmark",
        "dataset_manifest_fingerprint",
        "embedding_config_fingerprint",
        "embedding_request_fingerprint",
        "embedding_manifest_fingerprint",
        "evaluation_split",
        "search",
    }
    if set(raw) != expected_root:
        raise RealOriginalBaselineError(
            f"invalid root keys; missing={sorted(expected_root-set(raw))}, "
            f"unknown={sorted(set(raw)-expected_root)}"
        )
    if raw["schema_version"] != 1:
        raise RealOriginalBaselineError("schema_version must be 1")
    if raw["benchmark"] != "real_original_exact_v1":
        raise RealOriginalBaselineError(
            "benchmark must be real_original_exact_v1"
        )
    if raw["evaluation_split"] != "query_tune":
        raise RealOriginalBaselineError(
            "the pre-selection baseline accepts query_tune only"
        )
    search = raw["search"]
    expected_search = {
        "normalized_inputs",
        "distance",
        "arithmetic",
        "stable_tie_break",
        "query_batch_size",
        "cutoffs",
        "k_ctx",
        "k_gt",
    }
    if not isinstance(search, dict) or set(search) != expected_search:
        actual = set(search) if isinstance(search, dict) else set()
        raise RealOriginalBaselineError(
            f"invalid search keys; missing={sorted(expected_search-actual)}, "
            f"unknown={sorted(actual-expected_search)}"
        )
    if search["normalized_inputs"] is not True:
        raise RealOriginalBaselineError("search.normalized_inputs must be true")
    if search["distance"] != "squared_l2":
        raise RealOriginalBaselineError("search.distance must be squared_l2")
    if search["arithmetic"] != "numpy_float64":
        raise RealOriginalBaselineError(
            "search.arithmetic must be numpy_float64"
        )
    if search["stable_tie_break"] != "lexicographic_doc_id":
        raise RealOriginalBaselineError(
            "search.stable_tie_break must be lexicographic_doc_id"
        )
    cutoffs = search["cutoffs"]
    if not isinstance(cutoffs, list) or not cutoffs:
        raise RealOriginalBaselineError("search.cutoffs must be a nonempty list")
    validated_cutoffs = [
        _positive_integer(value, "search.cutoffs item") for value in cutoffs
    ]
    if validated_cutoffs != sorted(set(validated_cutoffs)):
        raise RealOriginalBaselineError(
            "search.cutoffs must be strictly increasing"
        )
    k_ctx = _positive_integer(search["k_ctx"], "search.k_ctx")
    k_gt = _positive_integer(search["k_gt"], "search.k_gt")
    if k_ctx not in validated_cutoffs or k_gt not in validated_cutoffs:
        raise RealOriginalBaselineError(
            "search.k_ctx and search.k_gt must appear in search.cutoffs"
        )
    if k_ctx > k_gt:
        raise RealOriginalBaselineError("search.k_ctx cannot exceed search.k_gt")
    return RealOriginalBaselineConfig(
        schema_version=1,
        benchmark="real_original_exact_v1",
        dataset_manifest_fingerprint=_fingerprint_string(
            raw["dataset_manifest_fingerprint"],
            "dataset_manifest_fingerprint",
        ),
        embedding_config_fingerprint=_fingerprint_string(
            raw["embedding_config_fingerprint"],
            "embedding_config_fingerprint",
        ),
        embedding_request_fingerprint=_fingerprint_string(
            raw["embedding_request_fingerprint"],
            "embedding_request_fingerprint",
        ),
        embedding_manifest_fingerprint=_fingerprint_string(
            raw["embedding_manifest_fingerprint"],
            "embedding_manifest_fingerprint",
        ),
        evaluation_split="query_tune",
        search=OriginalSearchConfig(
            normalized_inputs=True,
            distance="squared_l2",
            arithmetic="numpy_float64",
            stable_tie_break="lexicographic_doc_id",
            query_batch_size=_positive_integer(
                search["query_batch_size"], "search.query_batch_size"
            ),
            cutoffs=validated_cutoffs,
            k_ctx=k_ctx,
            k_gt=k_gt,
        ),
        raw=raw,
        config_fingerprint=fingerprint(raw),
    )


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealOriginalBaselineError(
            f"cannot load {description} {path}: {exc}"
        ) from exc


def _load_jsonl(path: Path, description: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RealOriginalBaselineError(
                        f"non-object {description} row at {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RealOriginalBaselineError(
            f"cannot load {description} {path}: {exc}"
        ) from exc
    if not rows:
        raise RealOriginalBaselineError(f"{description} cannot be empty")
    return rows


def _file_identity(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
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
        raise RealOriginalBaselineError("nDCG requires a positive qrel")
    return float(sum(gains) / denominator)


def _query_metrics(
    retrieved_ids: Sequence[str],
    relevance: Mapping[str, int],
    cutoffs: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    relevant_ids = set(relevance)
    if not relevant_ids:
        raise RealOriginalBaselineError("baseline query has no positive qrels")
    result: Dict[str, Dict[str, Any]] = {}
    for cutoff in cutoffs:
        retained = relevant_ids.intersection(retrieved_ids[:cutoff])
        result[str(cutoff)] = {
            "evidence_hit": bool(retained),
            "evidence_recall": float(len(retained) / len(relevant_ids)),
            "ndcg": _ndcg_at_k(retrieved_ids, relevance, cutoff),
        }
    return result


def _aggregate_records(
    records: Sequence[Mapping[str, Any]], cutoffs: Sequence[int]
) -> Dict[str, Any]:
    aggregates: Dict[str, Any] = {}
    for cutoff in cutoffs:
        values = [record["metrics"][str(cutoff)] for record in records]
        aggregates[str(cutoff)] = {
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
    return aggregates


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# SciFact tune-only original-space exact baseline",
        "",
        "This quality-reference run uses only `query_tune`. It does not inspect "
        "`query_cert` or `query_test`, choose a projection dimension, or certify "
        "an adaptive policy.",
        "",
        f"- queries: {summary['n_queries']}",
        f"- corpus vectors: {summary['corpus_size']}",
        f"- embedding dimension: {summary['embedding_dimension']}",
        "- distance: normalized original-space squared L2",
        "- backend arithmetic: NumPy float64",
        "- tie break: lexicographic stable document ID",
        "",
        "| cutoff | evidence hit | evidence recall | nDCG |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for cutoff, values in summary["metrics"].items():
        lines.append(
            f"| {cutoff} | {values['mean_evidence_hit']:.6f} | "
            f"{values['mean_evidence_recall']:.6f} | "
            f"{values['mean_ndcg']:.6f} |"
        )
    lines.extend(
        [
            "",
            "These are labeled-evidence metrics for the frozen embedding model. "
            "They are not embedding-neighbor retention, a policy certificate, or "
            "an answer-quality result.",
            "",
        ]
    )
    return "\n".join(lines)


def run_real_original_baseline(
    config: RealOriginalBaselineConfig,
    prepared_dir: Union[str, Path],
    embedding_config_path: Union[str, Path],
    embedding_cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    prepared = Path(prepared_dir)
    embedding_cache = Path(embedding_cache_dir)
    output = Path(output_dir)
    if config.evaluation_split != "query_tune":
        raise RealOriginalBaselineError(
            "the pre-selection baseline accepts query_tune only"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite baseline output: {output}")
    embedding_config = load_text_embedding_config(embedding_config_path)
    if embedding_config.config_fingerprint != config.embedding_config_fingerprint:
        raise RealOriginalBaselineError(
            "embedding config fingerprint does not match baseline config"
        )
    validation = validate_text_embedding_cache(
        embedding_config, prepared, embedding_cache
    )
    dataset_manifest = validation["dataset_manifest"]
    embedding_manifest = validation["embedding_manifest"]
    if dataset_manifest["fingerprint"] != config.dataset_manifest_fingerprint:
        raise RealOriginalBaselineError(
            "dataset fingerprint does not match baseline config"
        )
    if validation["request_fingerprint"] != config.embedding_request_fingerprint:
        raise RealOriginalBaselineError(
            "embedding request fingerprint does not match baseline config"
        )
    if embedding_manifest["fingerprint"] != config.embedding_manifest_fingerprint:
        raise RealOriginalBaselineError(
            "embedding cache fingerprint does not match baseline config"
        )

    corpus_table = load_embedding_array(
        embedding_cache / "corpus_embeddings.f32.npy",
        embedding_cache / "corpus_ids.json",
    )
    query_table = load_embedding_array(
        embedding_cache / "query_embeddings.f32.npy",
        embedding_cache / "query_ids.json",
    )
    corpus_rows = _load_jsonl(prepared / "corpus.jsonl", "corpus")
    query_rows = _load_jsonl(prepared / "queries.jsonl", "queries")
    qrel_rows = _load_jsonl(prepared / "qrels.jsonl", "qrels")
    expected_corpus_ids = [row.get("doc_id") for row in corpus_rows]
    expected_query_ids = [row.get("query_id") for row in query_rows]
    if corpus_table.ids.tolist() != expected_corpus_ids:
        raise RealOriginalBaselineError(
            "corpus embedding rows do not match prepared corpus order"
        )
    if query_table.ids.tolist() != expected_query_ids:
        raise RealOriginalBaselineError(
            "query embedding rows do not match prepared query order"
        )
    if config.search.cutoffs[-1] > len(corpus_table.ids):
        raise RealOriginalBaselineError("maximum cutoff exceeds corpus size")

    selected_rows = [
        index
        for index, row in enumerate(query_rows)
        if row.get("split") == config.evaluation_split
    ]
    selected_query_ids = [expected_query_ids[index] for index in selected_rows]
    expected_split = dataset_manifest["splits"][config.evaluation_split]
    if len(selected_rows) != expected_split["n"] or stable_id_hash(
        selected_query_ids
    ) != expected_split["id_hash"]:
        raise RealOriginalBaselineError(
            "selected query IDs do not match the frozen tune split"
        )
    qrels_by_query: Dict[str, Dict[str, int]] = {
        query_id: {} for query_id in selected_query_ids
    }
    corpus_id_set = set(corpus_table.ids.tolist())
    for row in qrel_rows:
        if row.get("split") != config.evaluation_split:
            continue
        query_id = row.get("query_id")
        doc_id = row.get("doc_id")
        relevance = row.get("relevance")
        if query_id not in qrels_by_query:
            raise RealOriginalBaselineError("tune qrel references unknown query")
        if doc_id not in corpus_id_set:
            raise RealOriginalBaselineError("tune qrel references unknown document")
        if (
            isinstance(relevance, bool)
            or not isinstance(relevance, int)
            or relevance < 1
        ):
            raise RealOriginalBaselineError("tune qrel relevance must be positive")
        if doc_id in qrels_by_query[query_id]:
            raise RealOriginalBaselineError("duplicate tune qrel pair")
        qrels_by_query[query_id][doc_id] = relevance
    empty_qrels = [
        query_id for query_id, relevance in qrels_by_query.items() if not relevance
    ]
    if empty_qrels:
        raise RealOriginalBaselineError(
            f"tune queries have no positive qrels: {empty_qrels}"
        )

    index_started = perf_counter()
    index = ExactSquaredL2Index(
        corpus_table.ids.tolist(),
        corpus_table.vectors,
        batch_size=config.search.query_batch_size,
    )
    index_build_ms = (perf_counter() - index_started) * 1000.0
    selected_queries = query_table.vectors[np.asarray(selected_rows)]
    search_result = index.search(selected_queries, config.search.cutoffs[-1])
    records = []
    for query_index, (query_id, retrieved) in enumerate(
        zip(selected_query_ids, search_result.ids.tolist())
    ):
        relevance = qrels_by_query[query_id]
        records.append(
            {
                "query_index": query_index,
                "query_id": query_id,
                "split": config.evaluation_split,
                "qrel_count": len(relevance),
                "relevance_by_doc_id": dict(sorted(relevance.items())),
                "exact_top_k_ids": retrieved,
                "metrics": _query_metrics(
                    retrieved, relevance, config.search.cutoffs
                ),
            }
        )

    summary = {
        "kind": "real_original_exact_tune_summary_v1",
        "data_scope": "query_tune_only",
        "n_queries": len(records),
        "empty_qrel_queries": 0,
        "corpus_size": len(corpus_table.ids),
        "embedding_dimension": int(corpus_table.vectors.shape[1]),
        "k_ctx": config.search.k_ctx,
        "k_gt": config.search.k_gt,
        "metrics": _aggregate_records(records, config.search.cutoffs),
    }
    summary["fingerprint"] = fingerprint(summary)
    timings = {
        "index_build_ms": index_build_ms,
        "search_total_ms": search_result.search_ms,
        "search_amortized_ms_per_query": search_result.search_ms / len(records),
        "distance_evaluations": len(records) * len(corpus_table.ids),
        "index_vector_bytes": int(index.vectors.nbytes),
        "source_embedding_bytes": int(corpus_table.vectors.nbytes),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        per_query_path = temporary / "per_query.jsonl"
        summary_path = temporary / "summary.json"
        timings_path = temporary / "timings.json"
        report_path = temporary / "report.md"
        manifest_path = temporary / "manifest.json"
        _write_jsonl(per_query_path, records)
        write_json(summary_path, summary)
        write_json(timings_path, timings)
        report_path.write_text(_report(summary), encoding="utf-8")
        result_artifacts = {
            path.name: _file_identity(path)
            for path in (per_query_path, summary_path, report_path)
        }
        result_identity = {
            "config_fingerprint": config.config_fingerprint,
            "dataset_manifest_fingerprint": dataset_manifest["fingerprint"],
            "embedding_manifest_fingerprint": embedding_manifest["fingerprint"],
            "query_split_id_hash": expected_split["id_hash"],
            "artifacts": result_artifacts,
        }
        manifest = {
            "schema_version": 1,
            "kind": "real_original_exact_tune_manifest_v1",
            "config_fingerprint": config.config_fingerprint,
            "dataset": {
                "manifest_fingerprint": dataset_manifest["fingerprint"],
                "query_split": config.evaluation_split,
                "query_split_n": expected_split["n"],
                "query_split_id_hash": expected_split["id_hash"],
            },
            "embedding": {
                "config_fingerprint": embedding_config.config_fingerprint,
                "request_fingerprint": validation["request_fingerprint"],
                "manifest_fingerprint": embedding_manifest["fingerprint"],
                "corpus_array_fingerprint": embedding_manifest["arrays"][
                    "corpus"
                ]["array_fingerprint"],
                "query_array_fingerprint": embedding_manifest["arrays"][
                    "queries"
                ]["array_fingerprint"],
            },
            "search": config.raw["search"],
            "data_scope": "query_tune_only",
            "result_artifacts": result_artifacts,
            "result_fingerprint": fingerprint(result_identity),
            "timings_artifact": "timings.json",
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
        }
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(manifest_path, manifest)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        name: output / name
        for name in (
            "manifest.json",
            "per_query.jsonl",
            "summary.json",
            "timings.json",
            "report.md",
        )
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--embedding-config", required=True, type=Path)
    parser.add_argument("--embedding-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_real_original_baseline_config(args.config)
    artifacts = run_real_original_baseline(
        config,
        args.dataset,
        args.embedding_config,
        args.embedding_cache,
        args.output,
    )
    print(f"completed tune-only original baseline: {artifacts['report.md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
