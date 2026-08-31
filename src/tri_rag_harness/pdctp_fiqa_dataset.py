"""Prepare frozen FiQA corpus/query text without opening protected outcomes.

The output is deliberately text-only: it contains no qrel pairs, relevance
values, retrieval results, or policy outcomes.  All five frozen query roles are
encoded together so this gate does not open any role in the protocol guard.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Union

from .utils import fingerprint, stable_id_hash, write_json


class PDCTPFiQADatasetError(ValueError):
    pass


FIVE_ROLES = (
    "query_cal",
    "query_tune",
    "query_cert",
    "query_latency",
    "query_test",
)
_QUERY_PREFIX = "pdctp-beir-fiqa:query:"
_DOC_PREFIX = "pdctp-beir-fiqa:doc:"
_OUTPUT_NAMES = (
    "corpus.jsonl",
    "queries.jsonl",
    "splits.json",
    "empty_documents.json",
    "formatted_text_hashes.json",
    "dataset_manifest.json",
)


def _hash_file(path: Path) -> Dict[str, Any]:
    md5 = hashlib.md5(usedforsecurity=False)  # nosec B324: publisher identity
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            md5.update(block)
            sha256.update(block)
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _sha256_file(path: Path) -> Dict[str, Any]:
    identity = _hash_file(path)
    return {"bytes": identity["bytes"], "sha256": identity["sha256"]}


def _load_fingerprinted(path: Path, name: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PDCTPFiQADatasetError(f"cannot load {name}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("fingerprint"), str):
        raise PDCTPFiQADatasetError(f"{name} is not a fingerprinted object")
    body = dict(value)
    claimed = body.pop("fingerprint")
    if fingerprint(body) != claimed:
        raise PDCTPFiQADatasetError(f"{name} fingerprint mismatch")
    return value


def _hash_member(archive: zipfile.ZipFile, member: str) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(member, "r") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"path": member, "bytes": size, "sha256": digest.hexdigest()}


def _text_member(archive: zipfile.ZipFile, member: str):
    return io.TextIOWrapper(archive.open(member, "r"), encoding="utf-8", newline="")


def _source_id(value: Any, source: str, line_number: int) -> str:
    if isinstance(value, bool) or value is None:
        raise PDCTPFiQADatasetError(f"invalid _id in {source}:{line_number}")
    result = str(value).strip()
    if not result:
        raise PDCTPFiQADatasetError(f"empty _id in {source}:{line_number}")
    return result


def _load_corpus(
    archive: zipfile.ZipFile, member: str, marker: str
) -> tuple[Dict[str, Dict[str, Any]], list[str]]:
    rows: Dict[str, Dict[str, Any]] = {}
    empty_ids: list[str] = []
    with _text_member(archive, member) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PDCTPFiQADatasetError(
                    f"invalid corpus JSON in {member}:{line_number}"
                ) from exc
            if not isinstance(source, dict):
                raise PDCTPFiQADatasetError(
                    f"non-object corpus JSON in {member}:{line_number}"
                )
            source_id = _source_id(source.get("_id"), member, line_number)
            if source_id in rows:
                raise PDCTPFiQADatasetError(f"duplicate corpus _id {source_id!r}")
            title = source.get("title", "")
            text = source.get("text", "")
            if title is None:
                title = ""
            if text is None:
                text = ""
            if not isinstance(title, str) or not isinstance(text, str):
                raise PDCTPFiQADatasetError(
                    f"corpus title/text must be strings in {member}:{line_number}"
                )
            empty = not title.strip() and not text.strip()
            if empty:
                title = ""
                text = marker
                empty_ids.append(source_id)
            rows[source_id] = {
                "doc_id": _DOC_PREFIX + source_id,
                "source_doc_id": source_id,
                "title": title,
                "text": text,
                "source_content_empty": empty,
            }
    if not rows:
        raise PDCTPFiQADatasetError("corpus cannot be empty")
    return rows, sorted(empty_ids)


def _load_queries(archive: zipfile.ZipFile, member: str) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    with _text_member(archive, member) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PDCTPFiQADatasetError(
                    f"invalid query JSON in {member}:{line_number}"
                ) from exc
            if not isinstance(source, dict):
                raise PDCTPFiQADatasetError(
                    f"non-object query JSON in {member}:{line_number}"
                )
            source_id = _source_id(source.get("_id"), member, line_number)
            if source_id in rows:
                raise PDCTPFiQADatasetError(f"duplicate query _id {source_id!r}")
            text = source.get("text")
            if not isinstance(text, str) or not text.strip():
                raise PDCTPFiQADatasetError(
                    f"query text must be nonempty in {member}:{line_number}"
                )
            rows[source_id] = text
    if not rows:
        raise PDCTPFiQADatasetError("queries cannot be empty")
    return rows


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


def _formatted_hashes(
    corpus_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    formatting: Mapping[str, Any],
) -> Dict[str, Any]:
    if formatting != {
        "corpus_prefix": "passage: ",
        "query_prefix": "query: ",
        "title_text_separator": "\n",
        "strip_fields": True,
    }:
        raise PDCTPFiQADatasetError("embedding formatting differs from frozen E5 request")

    corpus_pairs = []
    for row in corpus_rows:
        parts = [str(row[key]).strip() for key in ("title", "text")]
        body = formatting["title_text_separator"].join(part for part in parts if part)
        if not body:
            raise PDCTPFiQADatasetError("canonical corpus row formats to empty text")
        rendered = formatting["corpus_prefix"] + body
        corpus_pairs.append(
            {"id": row["doc_id"], "text_sha256": hashlib.sha256(rendered.encode()).hexdigest()}
        )

    query_pairs = []
    by_role: Dict[str, list[Dict[str, str]]] = {role: [] for role in FIVE_ROLES}
    for row in query_rows:
        rendered = formatting["query_prefix"] + str(row["text"]).strip()
        item = {
            "id": row["query_id"],
            "text_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        }
        query_pairs.append(item)
        by_role[str(row["role"])].append(item)
    return {
        "schema": "pdctp_formatted_text_hashes_v1",
        "formatting": dict(formatting),
        "corpus": {"n": len(corpus_pairs), "ordered_pair_hash": fingerprint(corpus_pairs)},
        "queries": {"n": len(query_pairs), "ordered_pair_hash": fingerprint(query_pairs)},
        "roles": {
            role: {"n": len(by_role[role]), "ordered_pair_hash": fingerprint(by_role[role])}
            for role in FIVE_ROLES
        },
    }


def prepare_pdctp_fiqa_text_inputs(
    protocol_freeze_path: Union[str, Path],
    role_assignments_path: Union[str, Path],
    source_audit_path: Union[str, Path],
    archive_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Prepare a deterministic, qrel-free FiQA embedding input directory."""
    protocol_path = Path(protocol_freeze_path)
    roles_path = Path(role_assignments_path)
    source_path = Path(source_audit_path)
    archive_value = Path(archive_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset output: {output}")
    if not archive_value.is_file():
        raise FileNotFoundError(f"FiQA archive does not exist: {archive_value}")

    protocol = _load_fingerprinted(protocol_path, "protocol freeze")
    roles = _load_fingerprinted(roles_path, "role assignments")
    source = _load_fingerprinted(source_path, "source audit")
    if (
        protocol.get("decision") != "READY_FOR_DATASET_AND_EMBEDDING_AUDIT_ONLY"
        or protocol.get("authorizes_method_evaluation") is not False
        or protocol.get("authorizes_protected_outcome_access") is not False
    ):
        raise PDCTPFiQADatasetError("protocol does not authorize only the text/embedding gate")
    if roles.get("all_roles_initially_closed") is not True or roles.get(
        "authorizes_outcome_access"
    ) is not False:
        raise PDCTPFiQADatasetError("role assignments do not keep every role closed")
    resolved = protocol.get("resolved_inputs", {})
    if source.get("fingerprint") != resolved.get("source_audit", {}).get("fingerprint"):
        raise PDCTPFiQADatasetError("source audit fingerprint differs from protocol")
    if _sha256_file(source_path) != {
        "bytes": resolved.get("source_audit", {}).get("bytes"),
        "sha256": resolved.get("source_audit", {}).get("sha256"),
    }:
        raise PDCTPFiQADatasetError("source audit file identity differs from protocol")
    expected_assignment = protocol.get("resolved_roles", {}).get("assignment_fingerprint")
    if roles.get("fingerprint") != expected_assignment:
        raise PDCTPFiQADatasetError("role assignment fingerprint differs from protocol")

    role_objects = roles.get("roles")
    if not isinstance(role_objects, dict) or set(role_objects) != set(FIVE_ROLES):
        raise PDCTPFiQADatasetError("role assignments must contain exactly five roles")
    ids_by_role: Dict[str, list[str]] = {}
    all_ids: list[str] = []
    for role in FIVE_ROLES:
        value = role_objects[role]
        if not isinstance(value, dict) or set(value) != {"n", "ordered_id_hash", "ordered_ids"}:
            raise PDCTPFiQADatasetError(f"invalid role assignment object: {role}")
        ids = value["ordered_ids"]
        if not isinstance(ids, list) or ids != sorted(ids) or len(set(ids)) != len(ids):
            raise PDCTPFiQADatasetError(f"{role} IDs are not unique and ordered")
        if value["n"] != len(ids) or value["ordered_id_hash"] != fingerprint(ids):
            raise PDCTPFiQADatasetError(f"{role} count/hash mismatch")
        if any(not isinstance(item, str) or not item.startswith(_QUERY_PREFIX) for item in ids):
            raise PDCTPFiQADatasetError(f"{role} contains an invalid query ID")
        ids_by_role[role] = ids
        all_ids.extend(ids)
    if len(set(all_ids)) != len(all_ids):
        raise PDCTPFiQADatasetError("query IDs overlap across roles")
    frozen_roles = protocol["resolved_roles"]
    if frozen_roles.get("counts") != {role: len(ids_by_role[role]) for role in FIVE_ROLES}:
        raise PDCTPFiQADatasetError("role counts differ from protocol")
    if frozen_roles.get("ordered_id_hashes") != {
        role: fingerprint(ids_by_role[role]) for role in FIVE_ROLES
    }:
        raise PDCTPFiQADatasetError("role ordered-ID hashes differ from protocol")

    source_archive = source.get("source", {}).get("archive")
    actual_archive = _hash_file(archive_value)
    if actual_archive != source_archive:
        raise PDCTPFiQADatasetError(
            f"archive identity mismatch: expected={source_archive}, actual={actual_archive}"
        )
    source_config = protocol["protocol"]["dataset"]
    if (
        actual_archive["sha256"] != source_config["archive_sha256"]
        or source_config["corpus_size"] != source["counts"]["corpus"]
        or source_config["source_query_count"] != source["counts"]["source_queries"]
    ):
        raise PDCTPFiQADatasetError("protocol/source count or archive binding changed")
    audit_config = protocol["protocol"]["source_gate"]
    if audit_config["source_audit_fingerprint"] != source["fingerprint"]:
        raise PDCTPFiQADatasetError("protocol source gate changed")

    member_metadata = source["source"]["members"]
    corpus_member = member_metadata["corpus"]["path"]
    queries_member = member_metadata["queries"]["path"]
    # The archive hash binds qrels, but this gate deliberately opens only text
    # members.  Qrel members are left to future guard-controlled runners.
    opened_member_metadata = {
        "corpus": member_metadata["corpus"],
        "queries": member_metadata["queries"],
    }
    required_members = {item["path"] for item in opened_member_metadata.values()}
    try:
        with zipfile.ZipFile(archive_value, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise PDCTPFiQADatasetError("archive contains duplicate member names")
            missing = required_members - set(names)
            if missing:
                raise PDCTPFiQADatasetError(f"archive is missing members: {sorted(missing)}")
            for metadata in opened_member_metadata.values():
                if _hash_member(archive, metadata["path"]) != metadata:
                    raise PDCTPFiQADatasetError(
                        f"source member identity mismatch: {metadata['path']}"
                    )
            marker = source_config["empty_documents"]["replacement_text"]
            corpus, empty_source_ids = _load_corpus(archive, corpus_member, marker)
            queries = _load_queries(archive, queries_member)
    except zipfile.BadZipFile as exc:
        raise PDCTPFiQADatasetError(f"invalid ZIP archive: {archive_value}") from exc

    expected_empty = source_config["empty_documents"]
    if (
        len(corpus) != source_config["corpus_size"]
        or len(queries) != source_config["source_query_count"]
        or len(empty_source_ids) != expected_empty["source_count"]
        or expected_empty["silent_deletion_allowed"] is not False
    ):
        raise PDCTPFiQADatasetError("corpus/query/empty-document counts changed")
    expected_source_query_ids = {
        stable_id[len(_QUERY_PREFIX) :] for stable_id in all_ids
    }
    if set(queries) != expected_source_query_ids:
        raise PDCTPFiQADatasetError("archive queries differ from frozen role identities")

    corpus_rows = [corpus[source_id] for source_id in sorted(corpus)]
    query_rows: list[Dict[str, Any]] = []
    for role in FIVE_ROLES:
        for stable_id in ids_by_role[role]:
            source_id = stable_id[len(_QUERY_PREFIX) :]
            query_rows.append(
                {
                    "query_id": stable_id,
                    "source_query_id": source_id,
                    "role": role,
                    "text": queries[source_id],
                }
            )
    corpus_ids = [str(row["doc_id"]) for row in corpus_rows]
    query_ids = [str(row["query_id"]) for row in query_rows]
    if set(corpus_ids).intersection(query_ids):
        raise PDCTPFiQADatasetError("external query IDs overlap corpus IDs")
    formatting = protocol["protocol"]["embedding"]["formatting"]
    formatted = _formatted_hashes(corpus_rows, query_rows, formatting)
    empty_stable_ids = [_DOC_PREFIX + source_id for source_id in empty_source_ids]
    empty_artifact = {
        "schema": "pdctp_fiqa_empty_documents_v1",
        "policy": expected_empty["policy"],
        "replacement_text": marker,
        "formatted_embedding_text": expected_empty["formatted_embedding_text"],
        "n": len(empty_source_ids),
        "ordered_source_ids": empty_source_ids,
        "ordered_doc_ids": empty_stable_ids,
        "ordered_doc_id_hash": fingerprint(empty_stable_ids),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_jsonl(temporary / "corpus.jsonl", corpus_rows)
        _write_jsonl(temporary / "queries.jsonl", query_rows)
        write_json(temporary / "splits.json", ids_by_role)
        write_json(temporary / "empty_documents.json", empty_artifact)
        write_json(temporary / "formatted_text_hashes.json", formatted)
        artifact_names = _OUTPUT_NAMES[:-1]
        artifacts = {
            name: _sha256_file(temporary / name) for name in artifact_names
        }
        manifest: Dict[str, Any] = {
            "schema_version": 1,
            "adapter": "pdctp_fiqa_text_only_v1",
            "protocol_freeze_fingerprint": protocol["fingerprint"],
            "role_assignments_fingerprint": roles["fingerprint"],
            "source_audit_fingerprint": source["fingerprint"],
            "source": {
                "archive": actual_archive,
                "members": {
                    "corpus": member_metadata["corpus"],
                    "queries": member_metadata["queries"],
                },
            },
            "dataset": {
                "name": source_config["name"],
                "version": source_config["version"],
                "namespace": source_config["id_namespace"],
            },
            "counts": {
                "corpus": len(corpus_rows),
                "queries": len(query_rows),
                "empty_corpus_items": len(empty_source_ids),
            },
            "ids": {
                "corpus_id_hash": stable_id_hash(corpus_ids),
                "query_id_hash": stable_id_hash(query_ids),
                "queries_are_external": True,
                "roles_are_disjoint": True,
            },
            "roles": {
                role: {
                    "n": len(ids_by_role[role]),
                    "ordered_id_hash": fingerprint(ids_by_role[role]),
                }
                for role in FIVE_ROLES
            },
            "empty_documents": {
                "n": len(empty_source_ids),
                "ordered_doc_id_hash": fingerprint(empty_stable_ids),
                "replacement_text": marker,
            },
            "formatted_text_hashes": {
                "corpus": formatted["corpus"],
                "queries": formatted["queries"],
                "roles": formatted["roles"],
            },
            "scope_guards": {
                "contains_qrels": False,
                "contains_relevance_values": False,
                "contains_embeddings": False,
                "contains_retrieval_or_policy_outcomes": False,
                "opens_any_protocol_role": False,
                "authorizes_calibrator_fit": False,
            },
            "artifacts": artifacts,
        }
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(temporary / "dataset_manifest.json", manifest)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {name: output / name for name in _OUTPUT_NAMES}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-freeze", required=True, type=Path)
    parser.add_argument("--role-assignments", required=True, type=Path)
    parser.add_argument("--source-audit", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = prepare_pdctp_fiqa_text_inputs(
        args.protocol_freeze,
        args.role_assignments,
        args.source_audit,
        args.archive,
        args.output,
    )
    print(f"PDCTP FiQA text preparation wrote {len(paths)} artifacts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
