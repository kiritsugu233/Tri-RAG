"""Prepare a pinned BEIR ZIP as canonical external-query retrieval artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, TextIO, Union

from .utils import fingerprint, stable_id_hash, write_json


class BEIRDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class BEIRSourceConfig:
    url: str
    archive_md5: str
    archive_root: str
    corpus_member: str
    queries_member: str
    qrels_members: Dict[str, str]
    homepage: str
    licenses: list[Dict[str, str]]


@dataclass(frozen=True)
class BEIRSplitConfig:
    seed: int
    development_qrels: str
    test_qrels: str
    tune_fraction: float


@dataclass(frozen=True)
class BEIRDatasetConfig:
    schema_version: int
    adapter: str
    dataset_name: str
    dataset_version: str
    id_namespace: str
    minimum_relevance: int
    source: BEIRSourceConfig
    splits: BEIRSplitConfig
    raw: Dict[str, Any]
    config_fingerprint: str


_MD5_RE = re.compile(r"[0-9a-f]{32}")
_SPLIT_ORDER = ("query_tune", "query_cert", "query_test")


def _object_section(
    raw: Mapping[str, Any], key: str, expected: set[str]
) -> Dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise BEIRDatasetError(f"{key} must be an object")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise BEIRDatasetError(
            f"invalid {key} keys; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return dict(value)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BEIRDatasetError(f"{name} must be a nonempty string")
    return value.strip()


def _relative_zip_member(value: Any, name: str) -> str:
    member = _nonempty_string(value, name)
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts:
        raise BEIRDatasetError(f"{name} must be a safe relative ZIP member")
    return str(path)


def load_beir_dataset_config(
    path: Union[str, Path],
) -> BEIRDatasetConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BEIRDatasetError(f"cannot load dataset config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BEIRDatasetError("dataset config root must be an object")
    expected_root = {
        "schema_version",
        "adapter",
        "dataset_name",
        "dataset_version",
        "id_namespace",
        "minimum_relevance",
        "source",
        "splits",
    }
    if set(raw) != expected_root:
        raise BEIRDatasetError(
            f"invalid root keys; missing={sorted(expected_root-set(raw))}, "
            f"unknown={sorted(set(raw)-expected_root)}"
        )
    if raw["schema_version"] != 1:
        raise BEIRDatasetError("schema_version must be 1")
    if raw["adapter"] != "beir_zip_v1":
        raise BEIRDatasetError("adapter must be beir_zip_v1")
    dataset_name = _nonempty_string(raw["dataset_name"], "dataset_name")
    dataset_version = _nonempty_string(raw["dataset_version"], "dataset_version")
    id_namespace = _nonempty_string(raw["id_namespace"], "id_namespace")
    if any(character.isspace() for character in id_namespace):
        raise BEIRDatasetError("id_namespace cannot contain whitespace")
    minimum_relevance = raw["minimum_relevance"]
    if (
        isinstance(minimum_relevance, bool)
        or not isinstance(minimum_relevance, int)
        or minimum_relevance < 1
    ):
        raise BEIRDatasetError("minimum_relevance must be a positive integer")

    source = _object_section(
        raw,
        "source",
        {
            "url",
            "archive_md5",
            "archive_root",
            "corpus_member",
            "queries_member",
            "qrels_members",
            "homepage",
            "licenses",
        },
    )
    archive_md5 = _nonempty_string(source["archive_md5"], "source.archive_md5").lower()
    if _MD5_RE.fullmatch(archive_md5) is None:
        raise BEIRDatasetError("source.archive_md5 must be 32 lowercase hex digits")
    archive_root = _relative_zip_member(source["archive_root"], "source.archive_root")
    corpus_member = _relative_zip_member(
        source["corpus_member"], "source.corpus_member"
    )
    queries_member = _relative_zip_member(
        source["queries_member"], "source.queries_member"
    )
    qrels_members = source["qrels_members"]
    if not isinstance(qrels_members, dict) or set(qrels_members) != {
        "development",
        "test",
    }:
        raise BEIRDatasetError(
            "source.qrels_members must contain exactly development and test"
        )
    qrels_members = {
        key: _relative_zip_member(value, f"source.qrels_members.{key}")
        for key, value in qrels_members.items()
    }
    licenses = source["licenses"]
    if not isinstance(licenses, list) or not licenses:
        raise BEIRDatasetError("source.licenses must be a nonempty list")
    validated_licenses = []
    for index, item in enumerate(licenses):
        if not isinstance(item, dict) or set(item) != {
            "component",
            "identifier",
            "url",
        }:
            raise BEIRDatasetError(
                f"source.licenses[{index}] must contain component, identifier, url"
            )
        validated_licenses.append(
            {
                key: _nonempty_string(value, f"source.licenses[{index}].{key}")
                for key, value in item.items()
            }
        )

    splits = _object_section(
        raw,
        "splits",
        {"seed", "development_qrels", "test_qrels", "tune_fraction"},
    )
    seed = splits["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise BEIRDatasetError("splits.seed must be a nonnegative integer")
    development_qrels = _nonempty_string(
        splits["development_qrels"], "splits.development_qrels"
    )
    test_qrels = _nonempty_string(splits["test_qrels"], "splits.test_qrels")
    if development_qrels not in qrels_members or test_qrels not in qrels_members:
        raise BEIRDatasetError("split qrels names must exist in source.qrels_members")
    if development_qrels == test_qrels:
        raise BEIRDatasetError("development and test qrels must be distinct")
    tune_fraction = splits["tune_fraction"]
    if (
        isinstance(tune_fraction, bool)
        or not isinstance(tune_fraction, (int, float))
        or not 0.0 < float(tune_fraction) < 1.0
    ):
        raise BEIRDatasetError("splits.tune_fraction must lie strictly in (0,1)")

    return BEIRDatasetConfig(
        schema_version=1,
        adapter="beir_zip_v1",
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        id_namespace=id_namespace,
        minimum_relevance=minimum_relevance,
        source=BEIRSourceConfig(
            url=_nonempty_string(source["url"], "source.url"),
            archive_md5=archive_md5,
            archive_root=archive_root,
            corpus_member=corpus_member,
            queries_member=queries_member,
            qrels_members=qrels_members,
            homepage=_nonempty_string(source["homepage"], "source.homepage"),
            licenses=validated_licenses,
        ),
        splits=BEIRSplitConfig(
            seed=seed,
            development_qrels=development_qrels,
            test_qrels=test_qrels,
            tune_fraction=float(tune_fraction),
        ),
        raw=raw,
        config_fingerprint=fingerprint(raw),
    )


def _file_hashes(path: Path) -> Dict[str, Any]:
    md5 = hashlib.md5(  # nosec B324 - required to match the publisher checksum
        usedforsecurity=False
    )
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            md5.update(block)
            sha256.update(block)
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _member_path(config: BEIRDatasetConfig, relative: str) -> str:
    return str(PurePosixPath(config.source.archive_root) / relative)


def _member_hash(archive: zipfile.ZipFile, member: str) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(member, "r") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return {"path": member, "bytes": size, "sha256": digest.hexdigest()}


def _text_member(archive: zipfile.ZipFile, member: str) -> TextIO:
    return io.TextIOWrapper(archive.open(member, "r"), encoding="utf-8", newline="")


def _source_id(value: Any, *, member: str, line_number: int) -> str:
    if isinstance(value, bool) or value is None:
        raise BEIRDatasetError(f"invalid _id in {member}:{line_number}")
    result = str(value).strip()
    if not result:
        raise BEIRDatasetError(f"empty _id in {member}:{line_number}")
    return result


def _load_corpus(
    archive: zipfile.ZipFile, member: str, namespace: str
) -> Dict[str, Dict[str, str]]:
    corpus: Dict[str, Dict[str, str]] = {}
    with _text_member(archive, member) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BEIRDatasetError(f"invalid JSON in {member}:{line_number}") from exc
            if not isinstance(value, dict):
                raise BEIRDatasetError(f"non-object JSON in {member}:{line_number}")
            source_id = _source_id(
                value.get("_id"), member=member, line_number=line_number
            )
            if source_id in corpus:
                raise BEIRDatasetError(f"duplicate corpus _id {source_id!r}")
            title = value.get("title", "")
            text = value.get("text", "")
            if title is None:
                title = ""
            if text is None:
                text = ""
            if not isinstance(title, str) or not isinstance(text, str):
                raise BEIRDatasetError(
                    f"corpus title/text must be strings in {member}:{line_number}"
                )
            if not title.strip() and not text.strip():
                raise BEIRDatasetError(
                    f"corpus row has no title or text in {member}:{line_number}"
                )
            corpus[source_id] = {
                "doc_id": f"{namespace}:doc:{source_id}",
                "source_doc_id": source_id,
                "title": title,
                "text": text,
            }
    if not corpus:
        raise BEIRDatasetError("corpus cannot be empty")
    return corpus


def _load_queries(archive: zipfile.ZipFile, member: str) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    with _text_member(archive, member) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BEIRDatasetError(f"invalid JSON in {member}:{line_number}") from exc
            if not isinstance(value, dict):
                raise BEIRDatasetError(f"non-object JSON in {member}:{line_number}")
            source_id = _source_id(
                value.get("_id"), member=member, line_number=line_number
            )
            if source_id in queries:
                raise BEIRDatasetError(f"duplicate query _id {source_id!r}")
            text = value.get("text")
            if not isinstance(text, str) or not text.strip():
                raise BEIRDatasetError(
                    f"query text must be nonempty in {member}:{line_number}"
                )
            queries[source_id] = text
    if not queries:
        raise BEIRDatasetError("queries cannot be empty")
    return queries


def _load_qrels(
    archive: zipfile.ZipFile,
    member: str,
    *,
    minimum_relevance: int,
) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with _text_member(archive, member) as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise BEIRDatasetError(f"empty qrels file {member}") from exc
        if [value.strip() for value in header[:3]] != [
            "query-id",
            "corpus-id",
            "score",
        ]:
            raise BEIRDatasetError(
                f"unexpected qrels header in {member}: {header[:3]!r}"
            )
        for line_number, row in enumerate(reader, start=2):
            if len(row) < 3:
                raise BEIRDatasetError(
                    f"qrels row has fewer than three columns in {member}:{line_number}"
                )
            query_id = row[0].strip()
            doc_id = row[1].strip()
            if not query_id or not doc_id:
                raise BEIRDatasetError(f"empty qrel ID in {member}:{line_number}")
            try:
                relevance = int(row[2])
            except ValueError as exc:
                raise BEIRDatasetError(
                    f"invalid qrel relevance in {member}:{line_number}"
                ) from exc
            if relevance < minimum_relevance:
                continue
            query_qrels = qrels.setdefault(query_id, {})
            if doc_id in query_qrels:
                raise BEIRDatasetError(
                    f"duplicate qrel pair ({query_id!r}, {doc_id!r}) in {member}"
                )
            query_qrels[doc_id] = relevance
    if not qrels:
        raise BEIRDatasetError(f"qrels contain no relevant pairs: {member}")
    return qrels


def _split_rank(seed: int, query_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{query_id}".encode("utf-8")).hexdigest()
    return digest, query_id


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


def prepare_beir_dataset(
    config: BEIRDatasetConfig,
    archive_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    archive_value = Path(archive_path)
    output_value = Path(output_dir)
    if not archive_value.is_file():
        raise FileNotFoundError(f"BEIR archive does not exist: {archive_value}")
    if output_value.exists():
        raise FileExistsError(f"refusing to overwrite dataset output: {output_value}")
    archive_hashes = _file_hashes(archive_value)
    if archive_hashes["md5"] != config.source.archive_md5:
        raise BEIRDatasetError(
            "BEIR archive MD5 mismatch: "
            f"expected {config.source.archive_md5}, got {archive_hashes['md5']}"
        )

    corpus_member = _member_path(config, config.source.corpus_member)
    queries_member = _member_path(config, config.source.queries_member)
    qrels_members = {
        key: _member_path(config, value)
        for key, value in config.source.qrels_members.items()
    }
    required_members = {corpus_member, queries_member, *qrels_members.values()}
    try:
        with zipfile.ZipFile(archive_value, "r") as archive:
            available = set(archive.namelist())
            missing = required_members - available
            if missing:
                raise BEIRDatasetError(
                    f"BEIR archive is missing members: {sorted(missing)}"
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise BEIRDatasetError(f"BEIR ZIP CRC failed for {bad_member}")
            member_hashes = {
                "corpus": _member_hash(archive, corpus_member),
                "queries": _member_hash(archive, queries_member),
                **{
                    f"qrels_{key}": _member_hash(archive, member)
                    for key, member in qrels_members.items()
                },
            }
            corpus = _load_corpus(archive, corpus_member, config.id_namespace)
            queries = _load_queries(archive, queries_member)
            source_qrels = {
                key: _load_qrels(
                    archive,
                    member,
                    minimum_relevance=config.minimum_relevance,
                )
                for key, member in qrels_members.items()
            }
    except zipfile.BadZipFile as exc:
        raise BEIRDatasetError(f"invalid BEIR ZIP archive: {archive_value}") from exc

    corpus_source_ids = set(corpus)
    for source_name, qrels in source_qrels.items():
        missing_queries = set(qrels) - set(queries)
        missing_docs = {
            doc_id
            for query_qrels in qrels.values()
            for doc_id in query_qrels
            if doc_id not in corpus_source_ids
        }
        if missing_queries or missing_docs:
            raise BEIRDatasetError(
                f"invalid {source_name} qrels; missing_queries="
                f"{sorted(missing_queries)}, missing_docs={sorted(missing_docs)}"
            )

    development_qrels = source_qrels[config.splits.development_qrels]
    test_qrels = source_qrels[config.splits.test_qrels]
    development_ids = set(development_qrels)
    test_ids = set(test_qrels)
    overlap = development_ids.intersection(test_ids)
    if overlap:
        raise BEIRDatasetError(
            f"development/test qrel query IDs overlap: {sorted(overlap)}"
        )
    ordered_development = sorted(
        development_ids,
        key=lambda query_id: _split_rank(config.splits.seed, query_id),
    )
    tune_n = int(math.floor(len(ordered_development) * config.splits.tune_fraction))
    if tune_n < 1 or tune_n >= len(ordered_development):
        raise BEIRDatasetError(
            "development qrels are too small for nonempty tune/cert splits"
        )
    source_ids_by_split = {
        "query_tune": ordered_development[:tune_n],
        "query_cert": ordered_development[tune_n:],
        "query_test": sorted(test_ids),
    }
    stable_ids_by_split = {
        split: [f"{config.id_namespace}:query:{source_id}" for source_id in source_ids]
        for split, source_ids in source_ids_by_split.items()
    }
    stable_split_sets = [set(stable_ids_by_split[split]) for split in _SPLIT_ORDER]
    if any(
        left.intersection(right)
        for index, left in enumerate(stable_split_sets)
        for right in stable_split_sets[index + 1 :]
    ):
        raise AssertionError("prepared query splits must be disjoint")
    corpus_ids = [row["doc_id"] for row in corpus.values()]
    query_ids = [
        query_id for split in _SPLIT_ORDER for query_id in stable_ids_by_split[split]
    ]
    if set(corpus_ids).intersection(query_ids):
        raise AssertionError("external query IDs overlap corpus IDs")

    output_value.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_value.name}.", dir=output_value.parent)
    )
    try:
        corpus_path = temporary / "corpus.jsonl"
        queries_path = temporary / "queries.jsonl"
        qrels_path = temporary / "qrels.jsonl"
        splits_path = temporary / "splits.json"
        manifest_path = temporary / "dataset_manifest.json"

        _write_jsonl(
            corpus_path,
            (corpus[source_id] for source_id in sorted(corpus)),
        )
        _write_jsonl(
            queries_path,
            (
                {
                    "query_id": stable_id,
                    "source_query_id": source_id,
                    "split": split,
                    "text": queries[source_id],
                }
                for split in _SPLIT_ORDER
                for source_id, stable_id in zip(
                    source_ids_by_split[split], stable_ids_by_split[split]
                )
            ),
        )

        qrel_rows = []
        for split in _SPLIT_ORDER:
            qrels = test_qrels if split == "query_test" else development_qrels
            for source_query_id in source_ids_by_split[split]:
                for source_doc_id, relevance in sorted(
                    qrels[source_query_id].items()
                ):
                    qrel_rows.append(
                        {
                            "doc_id": f"{config.id_namespace}:doc:{source_doc_id}",
                            "query_id": (
                                f"{config.id_namespace}:query:{source_query_id}"
                            ),
                            "relevance": relevance,
                            "source_doc_id": source_doc_id,
                            "source_query_id": source_query_id,
                            "split": split,
                        }
                    )
        _write_jsonl(qrels_path, qrel_rows)
        write_json(splits_path, stable_ids_by_split)

        artifact_hashes = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256_path(path)}
            for path in (corpus_path, queries_path, qrels_path, splits_path)
        }
        manifest: Dict[str, Any] = {
            "schema_version": 1,
            "adapter": config.adapter,
            "config_fingerprint": config.config_fingerprint,
            "dataset": {
                "name": config.dataset_name,
                "version": config.dataset_version,
                "homepage": config.source.homepage,
                "licenses": config.source.licenses,
            },
            "source": {
                "url": config.source.url,
                "archive": {
                    "expected_md5": config.source.archive_md5,
                    **archive_hashes,
                },
                "members": member_hashes,
            },
            "ids": {
                "namespace": config.id_namespace,
                "corpus_id_hash": stable_id_hash(corpus_ids),
                "query_id_hash": stable_id_hash(query_ids),
                "queries_are_external": True,
                "splits_are_disjoint": True,
            },
            "counts": {
                "corpus": len(corpus),
                "queries": len(query_ids),
                "qrels": len(qrel_rows),
            },
            "splits": {
                split: {
                    "n": len(stable_ids_by_split[split]),
                    "id_hash": stable_id_hash(stable_ids_by_split[split]),
                    "source_qrels": (
                        config.splits.test_qrels
                        if split == "query_test"
                        else config.splits.development_qrels
                    ),
                }
                for split in _SPLIT_ORDER
            },
            "split_rule": {
                "method": "sha256(seed\\0source_query_id), prefix to tune",
                "seed": config.splits.seed,
                "tune_fraction": config.splits.tune_fraction,
                "uses_qrel_labels": False,
            },
            "qrels": {
                "minimum_relevance": config.minimum_relevance,
                "all_queries_have_relevant_documents": True,
                "all_documents_exist": True,
            },
            "artifacts": artifact_hashes,
        }
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(manifest_path, manifest)
        temporary.rename(output_value)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        name: output_value / name
        for name in (
            "dataset_manifest.json",
            "corpus.jsonl",
            "queries.jsonl",
            "qrels.jsonl",
            "splits.json",
        )
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_beir_dataset_config(args.config)
    artifacts = prepare_beir_dataset(config, args.archive, args.output)
    print(f"prepared external retrieval dataset: {artifacts['dataset_manifest.json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
