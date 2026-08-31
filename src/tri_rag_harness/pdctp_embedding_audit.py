"""Independently audit the frozen, label-free PDCTP FiQA E5 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

import numpy as np

from .text_embeddings import TextEmbeddingConfig, load_text_embedding_config
from .utils import array_fingerprint, fingerprint, stable_id_hash, write_json


class PDCTPEmbeddingAuditError(ValueError):
    pass


FIVE_ROLES = (
    "query_cal",
    "query_tune",
    "query_cert",
    "query_latency",
    "query_test",
)


def _file_identity(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _load_json(path: Path, name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PDCTPEmbeddingAuditError(f"cannot load {name}: {exc}") from exc


def _load_fingerprinted(path: Path, name: str) -> Dict[str, Any]:
    value = _load_json(path, name)
    if not isinstance(value, dict) or not isinstance(value.get("fingerprint"), str):
        raise PDCTPEmbeddingAuditError(f"{name} is not fingerprinted")
    body = dict(value)
    claimed = body.pop("fingerprint")
    if fingerprint(body) != claimed:
        raise PDCTPEmbeddingAuditError(f"{name} fingerprint mismatch")
    return value


def _read_jsonl(path: Path, name: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PDCTPEmbeddingAuditError(
                        f"non-object {name} row at line {line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise PDCTPEmbeddingAuditError(f"cannot load {name}: {exc}") from exc
    if not rows:
        raise PDCTPEmbeddingAuditError(f"{name} cannot be empty")
    return rows


def _verify_dataset_artifacts(prepared: Path, manifest: Mapping[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "corpus.jsonl",
        "queries.jsonl",
        "splits.json",
        "empty_documents.json",
        "formatted_text_hashes.json",
    }:
        raise PDCTPEmbeddingAuditError("text-only dataset artifacts are invalid")
    for name, expected in artifacts.items():
        if Path(name).name != name or not isinstance(expected, dict):
            raise PDCTPEmbeddingAuditError("unsafe dataset artifact metadata")
        path = prepared / name
        if not path.is_file() or _file_identity(path) != expected:
            raise PDCTPEmbeddingAuditError(f"dataset artifact mismatch: {name}")


def _formatted_pair_hashes(
    corpus: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    config: TextEmbeddingConfig,
) -> Dict[str, Any]:
    corpus_pairs = []
    for row in corpus:
        title, text = row.get("title", ""), row.get("text", "")
        if not isinstance(title, str) or not isinstance(text, str):
            raise PDCTPEmbeddingAuditError("corpus title/text is not a string")
        if config.formatting.strip_fields:
            title, text = title.strip(), text.strip()
        body = config.formatting.title_text_separator.join(
            value for value in (title, text) if value
        )
        if not body:
            raise PDCTPEmbeddingAuditError("corpus row formats to empty text")
        rendered = config.formatting.corpus_prefix + body
        corpus_pairs.append(
            {
                "id": row.get("doc_id"),
                "text_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            }
        )
    query_pairs = []
    role_pairs: Dict[str, list[Dict[str, Any]]] = {role: [] for role in FIVE_ROLES}
    for row in queries:
        text = row.get("text")
        role = row.get("role")
        if not isinstance(text, str) or role not in role_pairs:
            raise PDCTPEmbeddingAuditError("invalid query text/role")
        if config.formatting.strip_fields:
            text = text.strip()
        if not text:
            raise PDCTPEmbeddingAuditError("query formats to empty text")
        rendered = config.formatting.query_prefix + text
        item = {
            "id": row.get("query_id"),
            "text_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        }
        query_pairs.append(item)
        role_pairs[role].append(item)
    return {
        "corpus": {"n": len(corpus_pairs), "ordered_pair_hash": fingerprint(corpus_pairs)},
        "queries": {"n": len(query_pairs), "ordered_pair_hash": fingerprint(query_pairs)},
        "roles": {
            role: {"n": len(role_pairs[role]), "ordered_pair_hash": fingerprint(role_pairs[role])}
            for role in FIVE_ROLES
        },
    }


def _expected_request(
    config: TextEmbeddingConfig, dataset_manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "adapter": config.adapter,
        "config_fingerprint": config.config_fingerprint,
        "dataset_manifest_fingerprint": dataset_manifest["fingerprint"],
        "dataset_artifacts": dataset_manifest["artifacts"],
        "model": config.raw["model"],
        "formatting": config.raw["formatting"],
        "encoding": config.raw["encoding"],
        "required_packages": config.required_packages,
    }


def _array_audit(
    cache: Path,
    manifest: Mapping[str, Any],
    name: str,
    expected_ids: Sequence[str],
    expected_dimension: int,
) -> Dict[str, Any]:
    files = {
        "corpus": ("corpus_embeddings.f32.npy", "corpus_ids.json"),
        "queries": ("query_embeddings.f32.npy", "query_ids.json"),
    }
    array_name, id_name = files[name]
    array_path, id_path = cache / array_name, cache / id_name
    ids = _load_json(id_path, f"{name} embedding IDs")
    if ids != list(expected_ids):
        raise PDCTPEmbeddingAuditError(f"{name} embedding IDs are not row-aligned")
    try:
        values = np.load(array_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise PDCTPEmbeddingAuditError(f"cannot load {name} embeddings: {exc}") from exc
    if values.shape != (len(expected_ids), expected_dimension) or values.dtype != np.float32:
        raise PDCTPEmbeddingAuditError(f"{name} embedding shape/dtype mismatch")
    if not np.all(np.isfinite(values)):
        raise PDCTPEmbeddingAuditError(f"{name} embeddings contain nonfinite values")
    norms = np.linalg.norm(np.asarray(values, dtype=np.float64), axis=1)
    max_error = float(np.max(np.abs(norms - 1.0)))
    if max_error > 1e-5:
        raise PDCTPEmbeddingAuditError(f"{name} embeddings are not L2 normalized")
    actual = {
        "file": array_name,
        **_file_identity(array_path),
        "array_fingerprint": array_fingerprint(values),
        "shape": [len(expected_ids), expected_dimension],
        "dtype": "float32",
        "l2_normalized": True,
        "max_abs_norm_error": max_error,
        "id_file": id_name,
        "id_artifact": _file_identity(id_path),
        "id_hash": stable_id_hash(list(expected_ids)),
    }
    if manifest.get("arrays", {}).get(name) != actual:
        raise PDCTPEmbeddingAuditError(f"{name} cache metadata mismatch")
    return actual


def _validate_token_statistics(
    provider: Mapping[str, Any], counts: Mapping[str, Any], maximum: int
) -> Dict[str, Any]:
    runtime = provider.get("runtime")
    if not isinstance(runtime, dict):
        raise PDCTPEmbeddingAuditError("provider runtime metadata is missing")
    stats = runtime.get("input_token_lengths")
    if not isinstance(stats, dict) or set(stats) != {"corpus", "queries"}:
        raise PDCTPEmbeddingAuditError("input token statistics are incomplete")
    result: Dict[str, Any] = {}
    for name in ("corpus", "queries"):
        value = stats[name]
        required = {
            "n",
            "minimum",
            "maximum",
            "mean",
            "p95",
            "max_sequence_length",
            "truncated",
            "truncated_fraction",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise PDCTPEmbeddingAuditError(f"invalid {name} token statistics")
        if (
            value["n"] != counts[name]
            or value["max_sequence_length"] != maximum
            or isinstance(value["truncated"], bool)
            or not isinstance(value["truncated"], int)
            or not 0 <= value["truncated"] <= value["n"]
            or value["minimum"] < 1
            or value["maximum"] < value["minimum"]
            or not all(math.isfinite(float(value[key])) for key in ("mean", "p95", "truncated_fraction"))
            or not math.isclose(
                float(value["truncated_fraction"]),
                value["truncated"] / value["n"],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise PDCTPEmbeddingAuditError(f"inconsistent {name} token statistics")
        result[name] = dict(value)
    return result


def audit_pdctp_embedding_cache(
    protocol_freeze_path: Union[str, Path],
    protocol_state_path: Union[str, Path],
    role_assignments_path: Union[str, Path],
    continuity_config_path: Union[str, Path],
    embedding_config_path: Union[str, Path],
    prepared_dir: Union[str, Path],
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    protocol = _load_fingerprinted(Path(protocol_freeze_path), "protocol freeze")
    state = _load_fingerprinted(Path(protocol_state_path), "protocol state")
    roles = _load_fingerprinted(Path(role_assignments_path), "role assignments")
    prepared, cache, output = Path(prepared_dir), Path(cache_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite embedding audit: {output}")
    if (
        protocol.get("decision")
        != "READY_FOR_DATASET_AND_EMBEDDING_AUDIT_ONLY"
        or protocol.get("authorizes_method_evaluation") is not False
        or protocol.get("authorizes_protected_outcome_access") is not False
    ):
        raise PDCTPEmbeddingAuditError("protocol is not at the embedding-audit gate")
    if state.get("fingerprint") != protocol.get("initial_guard_state_fingerprint"):
        raise PDCTPEmbeddingAuditError("protocol state identity differs from the freeze")
    if any(
        state.get(key)
        for key in (
            "calibration_opened",
            "certification_opened",
            "latency_opened",
            "test_opened",
        )
    ) or any(
        state.get(key) is not None
        for key in (
            "selection_fingerprint",
            "certification_result_fingerprint",
            "latency_result_fingerprint",
        )
    ):
        raise PDCTPEmbeddingAuditError("a protected role or downstream result is already open")
    if state.get("config_fingerprint") != protocol.get("config_fingerprint"):
        raise PDCTPEmbeddingAuditError("protocol state is not bound to the freeze")
    if roles.get("fingerprint") != protocol.get("resolved_roles", {}).get(
        "assignment_fingerprint"
    ) or roles.get("all_roles_initially_closed") is not True:
        raise PDCTPEmbeddingAuditError("role assignments are not bound and closed")

    continuity_path = Path(continuity_config_path)
    protocol_embedding = protocol["protocol"]["embedding"]
    if _file_identity(continuity_path)["sha256"] != protocol_embedding[
        "continuity_source_config_sha256"
    ]:
        raise PDCTPEmbeddingAuditError("continuity embedding config identity mismatch")
    continuity = load_text_embedding_config(continuity_path)
    config = load_text_embedding_config(embedding_config_path)
    frozen_model = protocol_embedding["model"]
    for key in (
        "name",
        "revision",
        "embedding_dimension",
        "max_sequence_length",
        "trust_remote_code",
    ):
        if config.raw["model"][key] != frozen_model[key]:
            raise PDCTPEmbeddingAuditError(f"embedding model field changed: {key}")
    if config.raw["formatting"] != protocol_embedding["formatting"]:
        raise PDCTPEmbeddingAuditError("embedding formatting differs from protocol")
    for key, value in protocol_embedding["encoding"].items():
        if config.raw["encoding"].get(key) != value:
            raise PDCTPEmbeddingAuditError(f"embedding encoding field changed: {key}")
    if (
        config.raw["model"] != continuity.raw["model"]
        or config.required_packages != continuity.required_packages
    ):
        raise PDCTPEmbeddingAuditError("E5 model/runtime continuity changed")
    if any(
        config.required_packages.get(name) != version
        for name, version in protocol_embedding["runtime_packages"].items()
    ):
        raise PDCTPEmbeddingAuditError("protocol runtime package subset changed")

    dataset_manifest = _load_fingerprinted(
        prepared / "dataset_manifest.json", "dataset manifest"
    )
    if dataset_manifest.get("fingerprint") != config.dataset_manifest_fingerprint:
        raise PDCTPEmbeddingAuditError("embedding config is not bound to the dataset")
    if (
        dataset_manifest.get("adapter") != "pdctp_fiqa_text_only_v1"
        or dataset_manifest.get("protocol_freeze_fingerprint") != protocol["fingerprint"]
        or dataset_manifest.get("role_assignments_fingerprint") != roles["fingerprint"]
    ):
        raise PDCTPEmbeddingAuditError("dataset is not bound to the frozen protocol")
    expected_guards = {
        "contains_qrels": False,
        "contains_relevance_values": False,
        "contains_embeddings": False,
        "contains_retrieval_or_policy_outcomes": False,
        "opens_any_protocol_role": False,
        "authorizes_calibrator_fit": False,
    }
    if dataset_manifest.get("scope_guards") != expected_guards:
        raise PDCTPEmbeddingAuditError("dataset scope guards changed")
    _verify_dataset_artifacts(prepared, dataset_manifest)
    corpus = _read_jsonl(prepared / "corpus.jsonl", "corpus")
    queries = _read_jsonl(prepared / "queries.jsonl", "queries")
    counts = dataset_manifest["counts"]
    if len(corpus) != counts["corpus"] or len(queries) != counts["queries"]:
        raise PDCTPEmbeddingAuditError("dataset row counts changed")
    corpus_ids = [row.get("doc_id") for row in corpus]
    query_ids = [row.get("query_id") for row in queries]
    if not all(isinstance(value, str) for value in corpus_ids + query_ids):
        raise PDCTPEmbeddingAuditError("dataset IDs are invalid")
    if (
        stable_id_hash(corpus_ids)
        != dataset_manifest["ids"]["corpus_id_hash"]
        or stable_id_hash(query_ids)
        != dataset_manifest["ids"]["query_id_hash"]
    ):
        raise PDCTPEmbeddingAuditError("dataset ID hashes changed")
    splits = _load_json(prepared / "splits.json", "splits")
    if not isinstance(splits, dict) or set(splits) != set(FIVE_ROLES):
        raise PDCTPEmbeddingAuditError("dataset splits are invalid")
    expected_query_ids = []
    for role in FIVE_ROLES:
        frozen_ids = roles["roles"][role]["ordered_ids"]
        if splits[role] != frozen_ids:
            raise PDCTPEmbeddingAuditError(f"{role} IDs differ from the freeze")
        expected_query_ids.extend(frozen_ids)
    if query_ids != expected_query_ids:
        raise PDCTPEmbeddingAuditError("query rows are not in frozen role order")
    formatted = _formatted_pair_hashes(corpus, queries, config)
    if formatted != dataset_manifest["formatted_text_hashes"]:
        raise PDCTPEmbeddingAuditError("formatted text hashes changed")
    empty = _load_json(prepared / "empty_documents.json", "empty documents")
    empty_ids = [
        row.get("doc_id")
        for row in corpus
        if row.get("source_content_empty") is True
    ]
    if (
        len(empty_ids) != counts["empty_corpus_items"]
        or empty_ids != empty.get("ordered_doc_ids")
        or any(
            row.get("text") != "[EMPTY_DOCUMENT]" or row.get("title") != ""
            for row in corpus
            if row.get("source_content_empty") is True
        )
    ):
        raise PDCTPEmbeddingAuditError("empty-document replacement identities changed")

    embedding_manifest = _load_fingerprinted(
        cache / "embedding_manifest.json", "embedding manifest"
    )
    expected_request = _expected_request(config, dataset_manifest)
    request_fingerprint = fingerprint(expected_request)
    if (
        embedding_manifest.get("request") != expected_request
        or embedding_manifest.get("request_fingerprint") != request_fingerprint
        or embedding_manifest.get("dataset")
        != {
            "manifest_fingerprint": dataset_manifest["fingerprint"],
            "counts": dataset_manifest["counts"],
            "corpus_id_hash": dataset_manifest["ids"]["corpus_id_hash"],
            "query_id_hash": dataset_manifest["ids"]["query_id_hash"],
        }
        or embedding_manifest.get("kind") != "normalized_text_embedding_cache"
        or embedding_manifest.get("model")
        != {
            "name": config.model.name,
            "revision": config.model.revision,
            "homepage": config.model.homepage,
            "license": config.model.license,
            "embedding_dimension": config.model.embedding_dimension,
            "max_sequence_length": config.model.max_sequence_length,
        }
        or embedding_manifest.get("formatting") != config.raw["formatting"]
        or embedding_manifest.get("encoding") != config.raw["encoding"]
    ):
        raise PDCTPEmbeddingAuditError("cache request/dataset binding changed")
    arrays = embedding_manifest.get("arrays")
    if not isinstance(arrays, dict) or set(arrays) != {"corpus", "queries"}:
        raise PDCTPEmbeddingAuditError("cache arrays metadata is invalid")
    corpus_array = _array_audit(
        cache, embedding_manifest, "corpus", corpus_ids, config.model.embedding_dimension
    )
    query_array = _array_audit(
        cache, embedding_manifest, "queries", query_ids, config.model.embedding_dimension
    )
    provider = embedding_manifest.get("provider")
    if (
        not isinstance(provider, dict)
        or provider.get("provider") != "sentence_transformers_v1"
    ):
        raise PDCTPEmbeddingAuditError("cache was not built by the frozen provider")
    provider_model = provider.get("model")
    snapshot_files = (
        provider_model.get("snapshot_files")
        if isinstance(provider_model, dict)
        else None
    )
    if (
        not isinstance(provider_model, dict)
        or provider_model.get("name") != config.model.name
        or provider_model.get("revision") != config.model.revision
        or not isinstance(snapshot_files, dict)
        or set(snapshot_files) != set(config.model.snapshot_allow_patterns)
        or provider_model.get("snapshot_fingerprint") != fingerprint(snapshot_files)
    ):
        raise PDCTPEmbeddingAuditError("model snapshot identity is invalid")
    for name, identity in snapshot_files.items():
        if (
            not isinstance(identity, dict)
            or set(identity) != {"bytes", "sha256"}
            or not isinstance(identity["bytes"], int)
            or identity["bytes"] < 1
            or not isinstance(identity["sha256"], str)
            or len(identity["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in identity["sha256"]
            )
        ):
            raise PDCTPEmbeddingAuditError(f"invalid snapshot file identity: {name}")
    runtime = provider.get("runtime", {})
    device = runtime.get("device") if isinstance(runtime, dict) else None
    if (
        not isinstance(runtime, dict)
        or not isinstance(device, dict)
        or runtime.get("packages") != config.required_packages
        or not str(device.get("resolved", "")).startswith("cuda")
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("allow_tf32") is not False
        or runtime.get("model_dtype") != "float32"
        or runtime.get("attention_implementation") != "eager"
        or runtime.get("cublas_workspace_config") != ":4096:8"
        or runtime.get("cudnn_deterministic") is not True
        or runtime.get("cudnn_benchmark") is not False
    ):
        raise PDCTPEmbeddingAuditError("embedding runtime metadata differs from protocol")
    token_stats = _validate_token_statistics(
        provider,
        {"corpus": len(corpus_ids), "queries": len(query_ids)},
        config.model.max_sequence_length,
    )

    audit: Dict[str, Any] = {
        "schema_version": 1,
        "name": "pdctp_fiqa_e5_embedding_cache_audit",
        "audit_version": "pdctp_fiqa_e5_embedding_cache_audit_v1",
        "protocol_freeze_fingerprint": protocol["fingerprint"],
        "protocol_state_fingerprint": state["fingerprint"],
        "role_assignments_fingerprint": roles["fingerprint"],
        "dataset_manifest_fingerprint": dataset_manifest["fingerprint"],
        "embedding_config_fingerprint": config.config_fingerprint,
        "embedding_request_fingerprint": request_fingerprint,
        "embedding_manifest_fingerprint": embedding_manifest["fingerprint"],
        "counts": {
            "corpus": len(corpus_ids),
            "queries": len(query_ids),
            "empty_corpus_items": len(empty_ids),
        },
        "formatted_text_hashes": formatted,
        "arrays": {"corpus": corpus_array, "queries": query_array},
        "model_snapshot_fingerprint": provider_model["snapshot_fingerprint"],
        "input_token_lengths": token_stats,
        "checks": {
            "all_roles_remained_closed": True,
            "qrels_or_relevance_opened": False,
            "formatted_texts_match": True,
            "row_ids_match": True,
            "arrays_float32_l2_normalized": True,
            "model_snapshot_matches": True,
            "deterministic_runtime_matches": True,
            "cache_fingerprint_valid": True,
        },
        "scope_guards": {
            "contains_qrels_or_relevance": False,
            "contains_retrieval_or_policy_outcomes": False,
            "fits_or_selects_a_method": False,
            "runs_an_llm": False,
            "uses_an_approximate_index": False,
        },
        "decision": "READY_TO_OPEN_QUERY_CAL",
    }
    audit["fingerprint"] = fingerprint(audit)
    report = (
        "# PDCTP FiQA E5 embedding-cache audit\n\n"
        f"Decision: `{audit['decision']}`.\n\n"
        f"Dataset fingerprint: `{dataset_manifest['fingerprint']}`.\n\n"
        f"Embedding fingerprint: `{embedding_manifest['fingerprint']}`.\n\n"
        f"Rows: {len(corpus_ids):,} corpus and {len(query_ids):,} external queries; "
        f"{len(empty_ids)} frozen empty-document replacements.\n\n"
        f"Truncated inputs: corpus={token_stats['corpus']['truncated']}, "
        f"queries={token_stats['queries']['truncated']}.\n\n"
        "Every role remained closed. No qrel, relevance value, retrieval/policy "
        "outcome, LLM, or approximate index was accessed by this gate.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        write_json(temporary / "embedding_audit.json", audit)
        (temporary / "report.md").write_text(report, encoding="utf-8")
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "embedding_audit.json": output / "embedding_audit.json",
        "report.md": output / "report.md",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-freeze", required=True, type=Path)
    parser.add_argument("--protocol-state", required=True, type=Path)
    parser.add_argument("--role-assignments", required=True, type=Path)
    parser.add_argument("--continuity-config", required=True, type=Path)
    parser.add_argument("--embedding-config", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = audit_pdctp_embedding_cache(
        args.protocol_freeze,
        args.protocol_state,
        args.role_assignments,
        args.continuity_config,
        args.embedding_config,
        args.prepared,
        args.cache,
        args.output,
    )
    print(f"PDCTP embedding audit wrote {len(paths)} artifacts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
