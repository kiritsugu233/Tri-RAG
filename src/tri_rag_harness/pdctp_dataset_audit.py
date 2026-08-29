"""Audit a pinned BEIR archive before any PDCTP method evaluation.

This module intentionally stops at source integrity, qrel integrity, normalized
query-text grouping, and a label-free five-role feasibility witness.  It does
not emit corpus/query text, embeddings, retrieval results, or method outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Sequence, Union

from .utils import fingerprint, write_json


class PDCTPDatasetAuditError(ValueError):
    pass


_ROLES = (
    "query_cal",
    "query_tune",
    "query_cert",
    "query_latency",
    "query_test",
)
_NATIVE_SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class PDCTPDatasetAuditConfig:
    raw: Dict[str, Any]
    config_fingerprint: str
    dataset_name: str
    dataset_version: str
    id_namespace: str
    minimum_relevance: int
    source: Dict[str, Any]
    license: Dict[str, Any]
    role_plan: Dict[str, Any]


def _object(
    raw: Mapping[str, Any], key: str, expected: set[str]
) -> Dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise PDCTPDatasetAuditError(f"{key} must be an object")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise PDCTPDatasetAuditError(
            f"invalid {key} keys; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return dict(value)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PDCTPDatasetAuditError(f"{name} must be a nonempty string")
    return value.strip()


def _hex(value: Any, name: str, length: int) -> str:
    result = _string(value, name).lower()
    if len(result) != length or any(c not in "0123456789abcdef" for c in result):
        raise PDCTPDatasetAuditError(
            f"{name} must contain exactly {length} lowercase hex digits"
        )
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PDCTPDatasetAuditError(f"{name} must be a positive integer")
    return value


def _member(value: Any, name: str) -> str:
    result = _string(value, name)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts:
        raise PDCTPDatasetAuditError(f"{name} must be a safe relative ZIP member")
    return str(path)


def load_pdctp_dataset_audit_config(
    path: Union[str, Path],
) -> PDCTPDatasetAuditConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PDCTPDatasetAuditError(
            f"cannot load dataset audit config {config_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise PDCTPDatasetAuditError("dataset audit config root must be an object")
    expected_root = {
        "schema_version",
        "audit",
        "dataset_name",
        "dataset_version",
        "id_namespace",
        "minimum_relevance",
        "source",
        "license",
        "role_plan",
    }
    if set(raw) != expected_root:
        raise PDCTPDatasetAuditError(
            f"invalid root keys; missing={sorted(expected_root-set(raw))}, "
            f"unknown={sorted(set(raw)-expected_root)}"
        )
    if raw["schema_version"] != 1:
        raise PDCTPDatasetAuditError("schema_version must be 1")
    if raw["audit"] != "pdctp_beir_source_audit_v1":
        raise PDCTPDatasetAuditError(
            "audit must be pdctp_beir_source_audit_v1"
        )

    source = _object(
        raw,
        "source",
        {
            "url",
            "metadata_url",
            "archive_md5",
            "archive_sha256",
            "archive_bytes",
            "archive_root",
            "corpus_member",
            "queries_member",
            "qrels_members",
        },
    )
    source["url"] = _string(source["url"], "source.url")
    source["metadata_url"] = _string(
        source["metadata_url"], "source.metadata_url"
    )
    if not source["url"].startswith("https://"):
        raise PDCTPDatasetAuditError("source.url must use HTTPS")
    source["archive_md5"] = _hex(
        source["archive_md5"], "source.archive_md5", 32
    )
    source["archive_sha256"] = _hex(
        source["archive_sha256"], "source.archive_sha256", 64
    )
    source["archive_bytes"] = _positive_int(
        source["archive_bytes"], "source.archive_bytes"
    )
    source["archive_root"] = _member(source["archive_root"], "source.archive_root")
    source["corpus_member"] = _member(
        source["corpus_member"], "source.corpus_member"
    )
    source["queries_member"] = _member(
        source["queries_member"], "source.queries_member"
    )
    qrels_members = source["qrels_members"]
    if not isinstance(qrels_members, dict) or set(qrels_members) != set(
        _NATIVE_SPLITS
    ):
        raise PDCTPDatasetAuditError(
            "source.qrels_members must contain exactly train, dev, and test"
        )
    source["qrels_members"] = {
        name: _member(value, f"source.qrels_members.{name}")
        for name, value in qrels_members.items()
    }

    license_config = _object(
        raw,
        "license",
        {
            "upstream_component",
            "upstream_identifier",
            "upstream_url",
            "commercial_use_permitted",
            "redistributor_disclaimer_url",
        },
    )
    for key in (
        "upstream_component",
        "upstream_identifier",
        "upstream_url",
        "redistributor_disclaimer_url",
    ):
        license_config[key] = _string(license_config[key], f"license.{key}")
    if license_config["commercial_use_permitted"] is not False:
        raise PDCTPDatasetAuditError(
            "FiQA audit must record commercial_use_permitted as false"
        )

    role_plan = _object(
        raw,
        "role_plan",
        {
            "seed",
            "query_cert_required",
            "cal_fraction_of_remaining_train",
            "minimum_role_sizes",
        },
    )
    seed = role_plan["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PDCTPDatasetAuditError("role_plan.seed must be a nonnegative integer")
    role_plan["query_cert_required"] = _positive_int(
        role_plan["query_cert_required"], "role_plan.query_cert_required"
    )
    fraction = role_plan["cal_fraction_of_remaining_train"]
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0.0 < float(fraction) < 1.0
    ):
        raise PDCTPDatasetAuditError(
            "role_plan.cal_fraction_of_remaining_train must lie in (0,1)"
        )
    role_plan["cal_fraction_of_remaining_train"] = float(fraction)
    minimums = role_plan["minimum_role_sizes"]
    if not isinstance(minimums, dict) or set(minimums) != set(_ROLES):
        raise PDCTPDatasetAuditError(
            "role_plan.minimum_role_sizes must contain exactly the five roles"
        )
    role_plan["minimum_role_sizes"] = {
        role: _positive_int(value, f"role_plan.minimum_role_sizes.{role}")
        for role, value in minimums.items()
    }
    if (
        role_plan["minimum_role_sizes"]["query_cert"]
        != role_plan["query_cert_required"]
    ):
        raise PDCTPDatasetAuditError(
            "query_cert minimum must equal query_cert_required"
        )

    return PDCTPDatasetAuditConfig(
        raw=raw,
        config_fingerprint=fingerprint(raw),
        dataset_name=_string(raw["dataset_name"], "dataset_name"),
        dataset_version=_string(raw["dataset_version"], "dataset_version"),
        id_namespace=_string(raw["id_namespace"], "id_namespace"),
        minimum_relevance=_positive_int(
            raw["minimum_relevance"], "minimum_relevance"
        ),
        source=source,
        license=license_config,
        role_plan=role_plan,
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


def _member_path(config: PDCTPDatasetAuditConfig, relative: str) -> str:
    return str(PurePosixPath(config.source["archive_root"]) / relative)


def _hash_member(archive: zipfile.ZipFile, member: str) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(member, "r") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"path": member, "bytes": size, "sha256": digest.hexdigest()}


def _text(archive: zipfile.ZipFile, member: str):
    return io.TextIOWrapper(archive.open(member, "r"), encoding="utf-8", newline="")


def _source_id(value: Any, member: str, line_number: int) -> str:
    if isinstance(value, bool) or value is None:
        raise PDCTPDatasetAuditError(f"invalid _id in {member}:{line_number}")
    result = str(value).strip()
    if not result:
        raise PDCTPDatasetAuditError(f"empty _id in {member}:{line_number}")
    return result


def _load_corpus_ids(
    archive: zipfile.ZipFile, member: str
) -> tuple[set[str], set[str]]:
    result: set[str] = set()
    empty_content_ids: set[str] = set()
    with _text(archive, member) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PDCTPDatasetAuditError(
                    f"invalid JSON in {member}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise PDCTPDatasetAuditError(
                    f"non-object JSON in {member}:{line_number}"
                )
            source_id = _source_id(row.get("_id"), member, line_number)
            if source_id in result:
                raise PDCTPDatasetAuditError(f"duplicate corpus _id {source_id!r}")
            title = row.get("title", "")
            text = row.get("text", "")
            if title is None:
                title = ""
            if text is None:
                text = ""
            if not isinstance(title, str) or not isinstance(text, str):
                raise PDCTPDatasetAuditError(
                    f"corpus title/text must be strings in {member}:{line_number}"
                )
            if not title.strip() and not text.strip():
                empty_content_ids.add(source_id)
            result.add(source_id)
    if not result:
        raise PDCTPDatasetAuditError("corpus cannot be empty")
    return result, empty_content_ids


def _load_queries(archive: zipfile.ZipFile, member: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    with _text(archive, member) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PDCTPDatasetAuditError(
                    f"invalid JSON in {member}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise PDCTPDatasetAuditError(
                    f"non-object JSON in {member}:{line_number}"
                )
            source_id = _source_id(row.get("_id"), member, line_number)
            if source_id in result:
                raise PDCTPDatasetAuditError(f"duplicate query _id {source_id!r}")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise PDCTPDatasetAuditError(
                    f"query text must be nonempty in {member}:{line_number}"
                )
            result[source_id] = text
    if not result:
        raise PDCTPDatasetAuditError("queries cannot be empty")
    return result


def _load_qrels(
    archive: zipfile.ZipFile,
    member: str,
    *,
    minimum_relevance: int,
    corpus_ids: set[str],
    empty_corpus_ids: set[str],
    query_ids: set[str],
) -> tuple[set[str], Dict[str, Any]]:
    pairs: set[tuple[str, str]] = set()
    positive_queries: set[str] = set()
    row_count = 0
    positive_rows = 0
    missing_queries: set[str] = set()
    missing_documents: set[str] = set()
    positive_rows_to_empty_corpus = 0
    with _text(archive, member) as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise PDCTPDatasetAuditError(f"empty qrels file {member}") from exc
        if [part.strip() for part in header[:3]] != [
            "query-id",
            "corpus-id",
            "score",
        ]:
            raise PDCTPDatasetAuditError(
                f"unexpected qrels header in {member}: {header[:3]!r}"
            )
        for line_number, row in enumerate(reader, start=2):
            if len(row) < 3:
                raise PDCTPDatasetAuditError(
                    f"qrels row has fewer than three columns in {member}:{line_number}"
                )
            query_id, doc_id = row[0].strip(), row[1].strip()
            if not query_id or not doc_id:
                raise PDCTPDatasetAuditError(
                    f"empty qrel ID in {member}:{line_number}"
                )
            try:
                relevance = int(row[2])
            except ValueError as exc:
                raise PDCTPDatasetAuditError(
                    f"invalid qrel relevance in {member}:{line_number}"
                ) from exc
            pair = (query_id, doc_id)
            if pair in pairs:
                raise PDCTPDatasetAuditError(
                    f"duplicate qrel pair {pair!r} in {member}"
                )
            pairs.add(pair)
            row_count += 1
            if query_id not in query_ids:
                missing_queries.add(query_id)
            if doc_id not in corpus_ids:
                missing_documents.add(doc_id)
            if relevance >= minimum_relevance:
                positive_rows += 1
                positive_queries.add(query_id)
                if doc_id in empty_corpus_ids:
                    positive_rows_to_empty_corpus += 1
    if missing_queries or missing_documents:
        raise PDCTPDatasetAuditError(
            f"invalid qrels in {member}; missing_queries={sorted(missing_queries)}, "
            f"missing_documents={sorted(missing_documents)}"
        )
    if not positive_queries:
        raise PDCTPDatasetAuditError(f"qrels contain no eligible queries: {member}")
    return positive_queries, {
        "rows": row_count,
        "positive_rows": positive_rows,
        "eligible_queries": len(positive_queries),
        "positive_rows_to_empty_corpus": positive_rows_to_empty_corpus,
    }


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _group_rank(seed: int, normalized_text: str) -> tuple[str, str]:
    return (
        hashlib.sha256(f"{seed}\0{normalized_text}".encode("utf-8")).hexdigest(),
        normalized_text,
    )


def _stable_query_id(namespace: str, source_id: str) -> str:
    return f"{namespace}:query:{source_id}"


def _ordered_role(role_ids: set[str]) -> list[str]:
    return sorted(role_ids)


def _role_witness(
    config: PDCTPDatasetAuditConfig,
    queries: Mapping[str, str],
    split_ids: Mapping[str, set[str]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    normalized = {
        query_id: _normalize(queries[query_id])
        for query_ids in split_ids.values()
        for query_id in query_ids
    }
    groups: Dict[str, Dict[str, set[str]]] = {}
    for split, query_ids in split_ids.items():
        for query_id in query_ids:
            groups.setdefault(
                normalized[query_id], {name: set() for name in _NATIVE_SPLITS}
            )[split].add(query_id)

    roles: Dict[str, set[str]] = {role: set() for role in _ROLES}
    excluded_test_matches: set[str] = set()
    latency_source_counts = {"train": 0, "dev": 0}
    training_groups: list[tuple[str, list[str]]] = []
    duplicate_groups = 0
    cross_native_groups = 0
    for text, by_split in groups.items():
        all_ids = set().union(*by_split.values())
        if len(all_ids) > 1:
            duplicate_groups += 1
        if sum(bool(by_split[name]) for name in _NATIVE_SPLITS) > 1:
            cross_native_groups += 1
        if by_split["test"]:
            roles["query_test"].update(by_split["test"])
            excluded_test_matches.update(by_split["train"])
            excluded_test_matches.update(by_split["dev"])
        elif by_split["dev"]:
            latency_ids = by_split["train"].union(by_split["dev"])
            roles["query_latency"].update(latency_ids)
            latency_source_counts["train"] += len(by_split["train"])
            latency_source_counts["dev"] += len(by_split["dev"])
        elif by_split["train"]:
            training_groups.append((text, sorted(by_split["train"])))

    training_groups.sort(key=lambda item: _group_rank(config.role_plan["seed"], item[0]))
    cert_required = config.role_plan["query_cert_required"]
    remaining_groups: list[tuple[str, list[str]]] = []
    for text, group_ids in training_groups:
        if len(roles["query_cert"]) < cert_required:
            roles["query_cert"].update(group_ids)
        else:
            remaining_groups.append((text, group_ids))

    remaining_count = sum(len(group_ids) for _, group_ids in remaining_groups)
    cal_target = int(
        remaining_count * config.role_plan["cal_fraction_of_remaining_train"]
    )
    for _, group_ids in remaining_groups:
        destination = (
            "query_cal" if len(roles["query_cal"]) < cal_target else "query_tune"
        )
        roles[destination].update(group_ids)

    stable_roles = {
        role: _ordered_role(
            {
                _stable_query_id(config.id_namespace, source_id)
                for source_id in source_ids
            }
        )
        for role, source_ids in roles.items()
    }
    role_sets = [set(stable_roles[role]) for role in _ROLES]
    ids_disjoint = not any(
        left.intersection(right)
        for index, left in enumerate(role_sets)
        for right in role_sets[index + 1 :]
    )
    texts_by_role = {
        role: {_normalize(queries[source_id]) for source_id in source_ids}
        for role, source_ids in roles.items()
    }
    text_sets = [texts_by_role[role] for role in _ROLES]
    texts_disjoint = not any(
        left.intersection(right)
        for index, left in enumerate(text_sets)
        for right in text_sets[index + 1 :]
    )
    minimums = config.role_plan["minimum_role_sizes"]
    size_gate = all(len(stable_roles[role]) >= minimums[role] for role in _ROLES)
    passed = ids_disjoint and texts_disjoint and size_gate

    witness: Dict[str, Any] = {
        "schema_version": 1,
        "name": "pdctp_fiqa_five_role_feasibility_witness",
        "witness_only": True,
        "authorizes_method_evaluation": False,
        "config_fingerprint": config.config_fingerprint,
        "assignment": {
            "seed": config.role_plan["seed"],
            "method": (
                "test-native priority; exclude non-test normalized-text matches; "
                "dev-native priority for label-free latency; sha256(seed\\0text) "
                "group ordering for train-only cert/cal/tune subdivision"
            ),
            "uses_relevance_magnitudes": False,
            "duplicate_groups_never_split": True,
        },
        "roles": stable_roles,
    }
    witness["fingerprint"] = fingerprint(witness)
    summary = {
        "passed": passed,
        "query_cert_required": cert_required,
        "minimum_role_sizes": minimums,
        "role_counts": {role: len(ids) for role, ids in stable_roles.items()},
        "ordered_id_hashes": {
            role: fingerprint(ids) for role, ids in stable_roles.items()
        },
        "ids_disjoint": ids_disjoint,
        "normalized_texts_disjoint": texts_disjoint,
        "duplicate_normalized_text_groups": duplicate_groups,
        "cross_native_normalized_text_groups": cross_native_groups,
        "excluded_non_test_queries_matching_test_text": len(excluded_test_matches),
        "latency_source_counts": latency_source_counts,
        "witness_fingerprint": witness["fingerprint"],
    }
    return witness, summary


def audit_pdctp_beir_source(
    config: PDCTPDatasetAuditConfig,
    archive_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    archive_value = Path(archive_path)
    output_value = Path(output_dir)
    if not archive_value.is_file():
        raise FileNotFoundError(f"BEIR archive does not exist: {archive_value}")
    if output_value.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output_value}")

    archive_hashes = _hash_file(archive_value)
    expected = {
        "bytes": config.source["archive_bytes"],
        "md5": config.source["archive_md5"],
        "sha256": config.source["archive_sha256"],
    }
    if archive_hashes != expected:
        raise PDCTPDatasetAuditError(
            f"archive identity mismatch: expected={expected}, actual={archive_hashes}"
        )

    corpus_member = _member_path(config, config.source["corpus_member"])
    queries_member = _member_path(config, config.source["queries_member"])
    qrel_members = {
        split: _member_path(config, member)
        for split, member in config.source["qrels_members"].items()
    }
    required = {corpus_member, queries_member, *qrel_members.values()}
    try:
        with zipfile.ZipFile(archive_value, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise PDCTPDatasetAuditError("archive contains duplicate member names")
            missing = required - set(names)
            if missing:
                raise PDCTPDatasetAuditError(
                    f"archive is missing required members: {sorted(missing)}"
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise PDCTPDatasetAuditError(f"ZIP CRC failed for {bad_member}")
            member_hashes = {
                "corpus": _hash_member(archive, corpus_member),
                "queries": _hash_member(archive, queries_member),
                **{
                    f"qrels_{split}": _hash_member(archive, member)
                    for split, member in qrel_members.items()
                },
            }
            corpus_ids, empty_corpus_ids = _load_corpus_ids(archive, corpus_member)
            queries = _load_queries(archive, queries_member)
            split_ids: Dict[str, set[str]] = {}
            qrel_counts: Dict[str, Dict[str, Any]] = {}
            for split in _NATIVE_SPLITS:
                ids, counts = _load_qrels(
                    archive,
                    qrel_members[split],
                    minimum_relevance=config.minimum_relevance,
                    corpus_ids=corpus_ids,
                    empty_corpus_ids=empty_corpus_ids,
                    query_ids=set(queries),
                )
                split_ids[split] = ids
                qrel_counts[split] = counts
    except zipfile.BadZipFile as exc:
        raise PDCTPDatasetAuditError(f"invalid ZIP archive: {archive_value}") from exc

    native_sets = [split_ids[split] for split in _NATIVE_SPLITS]
    native_ids_disjoint = not any(
        left.intersection(right)
        for index, left in enumerate(native_sets)
        for right in native_sets[index + 1 :]
    )
    if not native_ids_disjoint:
        raise PDCTPDatasetAuditError("native qrel query IDs overlap across splits")

    witness, role_summary = _role_witness(config, queries, split_ids)
    if not role_summary["passed"]:
        raise PDCTPDatasetAuditError(
            f"five-role capacity gate failed: {role_summary['role_counts']}"
        )

    output_value.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_value.name}.", dir=output_value.parent)
    )
    try:
        witness_path = temporary / "role_feasibility_witness.json"
        write_json(witness_path, witness)
        witness_hash = _hash_file(witness_path)
        report: Dict[str, Any] = {
            "schema_version": 1,
            "name": "pdctp_beir_source_audit",
            "audit_version": "pdctp_beir_source_audit_v1",
            "config_fingerprint": config.config_fingerprint,
            "dataset": {
                "name": config.dataset_name,
                "version": config.dataset_version,
                "namespace": config.id_namespace,
            },
            "source": {
                "url": config.source["url"],
                "metadata_url": config.source["metadata_url"],
                "archive": archive_hashes,
                "members": member_hashes,
            },
            "license": config.license,
            "integrity": {
                "archive_identity_verified": True,
                "zip_crc_verified": True,
                "required_members_present": True,
                "qrel_queries_exist": True,
                "qrel_documents_exist": True,
                "native_qrel_query_ids_disjoint": native_ids_disjoint,
            },
            "counts": {
                "corpus": len(corpus_ids),
                "empty_corpus_items": len(empty_corpus_ids),
                "source_queries": len(queries),
                "qrels": qrel_counts,
            },
            "eligibility": {
                "minimum_relevance": config.minimum_relevance,
                "requires_at_least_one_positive_qrel": True,
                "subdivision_uses_relevance_magnitudes": False,
                "five_role_capacity": role_summary,
            },
            "scope_guards": {
                "contains_query_or_corpus_text": False,
                "contains_qrel_pairs_or_relevance_values": False,
                "contains_embeddings": False,
                "contains_retrieval_or_policy_outcomes": False,
                "authorizes_method_evaluation": False,
            },
            "artifacts": {
                "role_feasibility_witness.json": witness_hash,
            },
            "decision": "GO_TO_PROTOCOL_FREEZE",
        }
        report["fingerprint"] = fingerprint(report)
        report_path = temporary / "source_audit.json"
        write_json(report_path, report)
        temporary.replace(output_value)
    except BaseException:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
        raise
    return {
        "source_audit.json": output_value / "source_audit.json",
        "role_feasibility_witness.json": output_value
        / "role_feasibility_witness.json",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = audit_pdctp_beir_source(
        load_pdctp_dataset_audit_config(args.config), args.archive, args.output
    )
    print(f"PDCTP source audit wrote {len(paths)} artifacts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
