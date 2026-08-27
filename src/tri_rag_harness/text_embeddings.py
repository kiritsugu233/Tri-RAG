"""Build or validate a fingerprinted text-embedding cache."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Union

import numpy as np

from .embeddings import normalize_rows
from .utils import array_fingerprint, fingerprint, stable_id_hash, write_json


class TextEmbeddingError(ValueError):
    pass


@dataclass(frozen=True)
class TextEmbeddingModelConfig:
    name: str
    revision: str
    homepage: str
    license: str
    embedding_dimension: int
    max_sequence_length: int
    snapshot_allow_patterns: List[str]
    trust_remote_code: bool


@dataclass(frozen=True)
class TextFormattingConfig:
    corpus_prefix: str
    query_prefix: str
    title_text_separator: str
    strip_fields: bool


@dataclass(frozen=True)
class TextEncodingConfig:
    batch_size: int
    model_dtype: str
    output_dtype: str
    l2_normalize: bool
    deterministic_algorithms: bool
    allow_tf32: bool
    attention_implementation: str
    cublas_workspace_config: str


@dataclass(frozen=True)
class TextEmbeddingConfig:
    schema_version: int
    adapter: str
    dataset_manifest_fingerprint: str
    model: TextEmbeddingModelConfig
    formatting: TextFormattingConfig
    encoding: TextEncodingConfig
    required_packages: Dict[str, str]
    raw: Dict[str, Any]
    config_fingerprint: str


class TextEmbeddingProvider(Protocol):
    def encode(
        self, texts: Sequence[str], *, batch_size: int, role: str
    ) -> np.ndarray:
        ...

    def metadata(self) -> Mapping[str, Any]:
        ...


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
def _section(
    raw: Mapping[str, Any], key: str, expected: set[str]
) -> Dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise TextEmbeddingError(f"{key} must be an object")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise TextEmbeddingError(
            f"invalid {key} keys; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return dict(value)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TextEmbeddingError(f"{name} must be a nonempty string")
    return value.strip()


def _nonempty_format_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TextEmbeddingError(f"{name} must be a nonempty string")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TextEmbeddingError(f"{name} must be boolean")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TextEmbeddingError(f"{name} must be a positive integer")
    return value


def _safe_snapshot_member(value: Any, name: str) -> str:
    member = _nonempty_string(value, name)
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or any(
        character in member for character in "*?[]"
    ):
        raise TextEmbeddingError(
            f"{name} must be an exact safe relative snapshot member"
        )
    return str(path)


def load_text_embedding_config(
    path: Union[str, Path],
) -> TextEmbeddingConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TextEmbeddingError(
            f"cannot load text embedding config {config_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise TextEmbeddingError("text embedding config root must be an object")
    expected_root = {
        "schema_version",
        "adapter",
        "dataset_manifest_fingerprint",
        "model",
        "formatting",
        "encoding",
        "runtime",
    }
    if set(raw) != expected_root:
        raise TextEmbeddingError(
            f"invalid root keys; missing={sorted(expected_root-set(raw))}, "
            f"unknown={sorted(set(raw)-expected_root)}"
        )
    if raw["schema_version"] != 1:
        raise TextEmbeddingError("schema_version must be 1")
    if raw["adapter"] != "sentence_transformers_v1":
        raise TextEmbeddingError("adapter must be sentence_transformers_v1")
    dataset_fingerprint = _nonempty_string(
        raw["dataset_manifest_fingerprint"], "dataset_manifest_fingerprint"
    ).lower()
    if _SHA256_RE.fullmatch(dataset_fingerprint) is None:
        raise TextEmbeddingError(
            "dataset_manifest_fingerprint must be 64 lowercase hex digits"
        )

    model = _section(
        raw,
        "model",
        {
            "name",
            "revision",
            "homepage",
            "license",
            "embedding_dimension",
            "max_sequence_length",
            "snapshot_allow_patterns",
            "trust_remote_code",
        },
    )
    revision = _nonempty_string(model["revision"], "model.revision").lower()
    if _COMMIT_RE.fullmatch(revision) is None:
        raise TextEmbeddingError("model.revision must be a 40-character commit SHA")
    patterns = model["snapshot_allow_patterns"]
    if not isinstance(patterns, list) or not patterns:
        raise TextEmbeddingError(
            "model.snapshot_allow_patterns must be a nonempty list"
        )
    validated_patterns = [
        _safe_snapshot_member(value, f"model.snapshot_allow_patterns[{index}]")
        for index, value in enumerate(patterns)
    ]
    if len(set(validated_patterns)) != len(validated_patterns):
        raise TextEmbeddingError("model.snapshot_allow_patterns must be unique")
    trust_remote_code = _boolean(
        model["trust_remote_code"], "model.trust_remote_code"
    )
    if trust_remote_code:
        raise TextEmbeddingError("trust_remote_code must remain false")

    formatting = _section(
        raw,
        "formatting",
        {
            "corpus_prefix",
            "query_prefix",
            "title_text_separator",
            "strip_fields",
        },
    )
    corpus_prefix = _nonempty_format_string(
        formatting["corpus_prefix"], "formatting.corpus_prefix"
    )
    query_prefix = _nonempty_format_string(
        formatting["query_prefix"], "formatting.query_prefix"
    )
    separator = formatting["title_text_separator"]
    if not isinstance(separator, str):
        raise TextEmbeddingError("formatting.title_text_separator must be a string")

    encoding = _section(
        raw,
        "encoding",
        {
            "batch_size",
            "model_dtype",
            "output_dtype",
            "l2_normalize",
            "deterministic_algorithms",
            "allow_tf32",
            "attention_implementation",
            "cublas_workspace_config",
        },
    )
    if encoding["model_dtype"] != "float32":
        raise TextEmbeddingError("encoding.model_dtype must be float32")
    if encoding["output_dtype"] != "float32":
        raise TextEmbeddingError("encoding.output_dtype must be float32")
    if encoding["attention_implementation"] != "eager":
        raise TextEmbeddingError(
            "encoding.attention_implementation must be eager"
        )
    if encoding["cublas_workspace_config"] != ":4096:8":
        raise TextEmbeddingError(
            "encoding.cublas_workspace_config must be :4096:8"
        )
    l2_normalize = _boolean(encoding["l2_normalize"], "encoding.l2_normalize")
    if not l2_normalize:
        raise TextEmbeddingError("encoding.l2_normalize must remain true")
    deterministic_algorithms = _boolean(
        encoding["deterministic_algorithms"],
        "encoding.deterministic_algorithms",
    )
    if not deterministic_algorithms:
        raise TextEmbeddingError(
            "encoding.deterministic_algorithms must remain true"
        )
    allow_tf32 = _boolean(encoding["allow_tf32"], "encoding.allow_tf32")
    if allow_tf32:
        raise TextEmbeddingError("encoding.allow_tf32 must remain false")

    runtime = _section(raw, "runtime", {"required_packages"})
    required_packages = runtime["required_packages"]
    if not isinstance(required_packages, dict) or not required_packages:
        raise TextEmbeddingError("runtime.required_packages must be a nonempty object")
    validated_packages: Dict[str, str] = {}
    for package, version in required_packages.items():
        package_name = _nonempty_string(package, "runtime package name")
        validated_packages[package_name] = _nonempty_string(
            version, f"runtime.required_packages.{package_name}"
        )

    return TextEmbeddingConfig(
        schema_version=1,
        adapter="sentence_transformers_v1",
        dataset_manifest_fingerprint=dataset_fingerprint,
        model=TextEmbeddingModelConfig(
            name=_nonempty_string(model["name"], "model.name"),
            revision=revision,
            homepage=_nonempty_string(model["homepage"], "model.homepage"),
            license=_nonempty_string(model["license"], "model.license"),
            embedding_dimension=_positive_integer(
                model["embedding_dimension"], "model.embedding_dimension"
            ),
            max_sequence_length=_positive_integer(
                model["max_sequence_length"], "model.max_sequence_length"
            ),
            snapshot_allow_patterns=validated_patterns,
            trust_remote_code=trust_remote_code,
        ),
        formatting=TextFormattingConfig(
            corpus_prefix=corpus_prefix,
            query_prefix=query_prefix,
            title_text_separator=separator,
            strip_fields=_boolean(
                formatting["strip_fields"], "formatting.strip_fields"
            ),
        ),
        encoding=TextEncodingConfig(
            batch_size=_positive_integer(
                encoding["batch_size"], "encoding.batch_size"
            ),
            model_dtype="float32",
            output_dtype="float32",
            l2_normalize=l2_normalize,
            deterministic_algorithms=deterministic_algorithms,
            allow_tf32=allow_tf32,
            attention_implementation="eager",
            cublas_workspace_config=":4096:8",
        ),
        required_packages=validated_packages,
        raw=raw,
        config_fingerprint=fingerprint(raw),
    )


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


def _load_json_object(path: Path, description: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TextEmbeddingError(f"cannot load {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TextEmbeddingError(f"{description} must be an object: {path}")
    return value


def _validate_dataset_manifest(
    prepared_dir: Path, expected_fingerprint: str
) -> Dict[str, Any]:
    manifest_path = prepared_dir / "dataset_manifest.json"
    manifest = _load_json_object(manifest_path, "dataset manifest")
    stored_fingerprint = manifest.get("fingerprint")
    fingerprint_input = dict(manifest)
    fingerprint_input.pop("fingerprint", None)
    if stored_fingerprint != fingerprint(fingerprint_input):
        raise TextEmbeddingError("dataset manifest fingerprint is invalid")
    if stored_fingerprint != expected_fingerprint:
        raise TextEmbeddingError(
            "dataset manifest does not match the frozen embedding config: "
            f"expected {expected_fingerprint}, got {stored_fingerprint}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TextEmbeddingError("dataset manifest artifacts must be an object")
    for name, expected in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise TextEmbeddingError(f"unsafe dataset artifact name: {name!r}")
        if not isinstance(expected, dict):
            raise TextEmbeddingError(f"invalid dataset artifact metadata: {name}")
        path = prepared_dir / name
        if not path.is_file():
            raise TextEmbeddingError(f"dataset artifact is missing: {path}")
        actual = _file_identity(path)
        if actual != {
            "bytes": expected.get("bytes"),
            "sha256": expected.get("sha256"),
        }:
            raise TextEmbeddingError(f"dataset artifact identity mismatch: {name}")
    for required in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl", "splits.json"):
        if required not in artifacts:
            raise TextEmbeddingError(f"dataset manifest is missing {required}")
    return manifest


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TextEmbeddingError(
                    f"invalid JSON in {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise TextEmbeddingError(f"non-object JSON in {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise TextEmbeddingError(f"JSONL input cannot be empty: {path}")
    return rows


def _string_field(row: Mapping[str, Any], field: str, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise TextEmbeddingError(f"{source} has invalid {field}")
    return value


def _format_inputs(
    config: TextEmbeddingConfig,
    prepared_dir: Path,
    dataset_manifest: Mapping[str, Any],
) -> tuple[List[str], List[str], List[str], List[str]]:
    corpus_rows = _read_jsonl(prepared_dir / "corpus.jsonl")
    query_rows = _read_jsonl(prepared_dir / "queries.jsonl")
    expected_counts = dataset_manifest.get("counts")
    if not isinstance(expected_counts, dict):
        raise TextEmbeddingError("dataset manifest counts must be an object")
    if len(corpus_rows) != expected_counts.get("corpus"):
        raise TextEmbeddingError("corpus row count does not match dataset manifest")
    if len(query_rows) != expected_counts.get("queries"):
        raise TextEmbeddingError("query row count does not match dataset manifest")

    corpus_ids: List[str] = []
    corpus_texts: List[str] = []
    for index, row in enumerate(corpus_rows):
        source = f"corpus row {index}"
        corpus_ids.append(_string_field(row, "doc_id", source))
        title = row.get("title", "")
        text = row.get("text", "")
        if not isinstance(title, str) or not isinstance(text, str):
            raise TextEmbeddingError(f"{source} title/text must be strings")
        if config.formatting.strip_fields:
            title = title.strip()
            text = text.strip()
        body = config.formatting.title_text_separator.join(
            part for part in (title, text) if part
        )
        if not body:
            raise TextEmbeddingError(f"{source} has no text after formatting")
        corpus_texts.append(config.formatting.corpus_prefix + body)

    query_ids: List[str] = []
    query_texts: List[str] = []
    for index, row in enumerate(query_rows):
        source = f"query row {index}"
        query_ids.append(_string_field(row, "query_id", source))
        text = _string_field(row, "text", source)
        if config.formatting.strip_fields:
            text = text.strip()
        if not text:
            raise TextEmbeddingError(f"{source} has no text after formatting")
        query_texts.append(config.formatting.query_prefix + text)

    if len(set(corpus_ids)) != len(corpus_ids):
        raise TextEmbeddingError("corpus IDs must be unique")
    if len(set(query_ids)) != len(query_ids):
        raise TextEmbeddingError("query IDs must be unique")
    if set(corpus_ids).intersection(query_ids):
        raise TextEmbeddingError("corpus and external-query IDs overlap")
    ids = dataset_manifest.get("ids")
    if not isinstance(ids, dict):
        raise TextEmbeddingError("dataset manifest IDs must be an object")
    if stable_id_hash(corpus_ids) != ids.get("corpus_id_hash"):
        raise TextEmbeddingError("corpus ID hash does not match dataset manifest")
    if stable_id_hash(query_ids) != ids.get("query_id_hash"):
        raise TextEmbeddingError("query ID hash does not match dataset manifest")
    return corpus_ids, corpus_texts, query_ids, query_texts


def _request_identity(
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


def _canonical_embeddings(
    values: np.ndarray,
    *,
    expected_rows: int,
    expected_dimension: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape != (expected_rows, expected_dimension):
        raise TextEmbeddingError(
            f"{name} provider output has shape {array.shape}, expected "
            f"({expected_rows}, {expected_dimension})"
        )
    if not np.all(np.isfinite(array)):
        raise TextEmbeddingError(f"{name} provider output contains non-finite values")
    normalized = normalize_rows(array).astype(np.float32)
    if not np.all(np.isfinite(normalized)):
        raise TextEmbeddingError(f"{name} normalized embeddings are non-finite")
    return np.ascontiguousarray(normalized)


def _array_metadata(array_path: Path, id_path: Path) -> Dict[str, Any]:
    array = np.load(array_path, allow_pickle=False)
    ids = json.loads(id_path.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
        raise TextEmbeddingError(f"invalid embedding ID artifact: {id_path}")
    if array.ndim != 2 or len(array) < 1 or len(ids) != len(array):
        raise TextEmbeddingError(
            f"embedding array and IDs are not row-aligned: {array_path}"
        )
    if len(set(ids)) != len(ids):
        raise TextEmbeddingError(f"embedding IDs are not unique: {id_path}")
    if not np.all(np.isfinite(array)):
        raise TextEmbeddingError(f"embedding array contains non-finite values: {array_path}")
    norms = np.linalg.norm(np.asarray(array, dtype=np.float64), axis=1)
    return {
        "file": array_path.name,
        **_file_identity(array_path),
        "array_fingerprint": array_fingerprint(array),
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "l2_normalized": True,
        "max_abs_norm_error": float(np.max(np.abs(norms - 1.0))),
        "id_file": id_path.name,
        "id_artifact": _file_identity(id_path),
        "id_hash": stable_id_hash(ids),
    }


def _validate_cache(
    output_dir: Path,
    *,
    expected_request_fingerprint: str,
    expected_dimension: int,
    expected_counts: Mapping[str, Any],
    expected_id_hashes: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest_path = output_dir / "embedding_manifest.json"
    manifest = _load_json_object(manifest_path, "embedding manifest")
    stored_fingerprint = manifest.get("fingerprint")
    fingerprint_input = dict(manifest)
    fingerprint_input.pop("fingerprint", None)
    if stored_fingerprint != fingerprint(fingerprint_input):
        raise TextEmbeddingError("embedding manifest fingerprint is invalid")
    if manifest.get("request_fingerprint") != expected_request_fingerprint:
        raise TextEmbeddingError(
            "embedding cache request fingerprint does not match the frozen request"
        )
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or set(arrays) != {"corpus", "queries"}:
        raise TextEmbeddingError("embedding manifest arrays are invalid")
    for name, metadata in arrays.items():
        if not isinstance(metadata, dict):
            raise TextEmbeddingError(f"embedding array metadata is invalid: {name}")
        expected_files = {
            "corpus": ("corpus_embeddings.f32.npy", "corpus_ids.json"),
            "queries": ("query_embeddings.f32.npy", "query_ids.json"),
        }
        expected_array_file, expected_id_file = expected_files[name]
        if metadata.get("file") != expected_array_file or metadata.get(
            "id_file"
        ) != expected_id_file:
            raise TextEmbeddingError(f"unsafe embedding cache paths: {name}")
        array_path = output_dir / expected_array_file
        id_path = output_dir / expected_id_file
        if not array_path.is_file() or not id_path.is_file():
            raise TextEmbeddingError(f"embedding cache artifact is missing: {name}")
        actual = _array_metadata(array_path, id_path)
        if actual != metadata:
            raise TextEmbeddingError(f"embedding cache artifact mismatch: {name}")
        if actual["dtype"] != "float32" or actual["shape"][1] != expected_dimension:
            raise TextEmbeddingError(f"embedding cache shape/dtype mismatch: {name}")
        count_key = "corpus" if name == "corpus" else "queries"
        id_hash_key = "corpus_id_hash" if name == "corpus" else "query_id_hash"
        if actual["shape"][0] != expected_counts.get(count_key):
            raise TextEmbeddingError(f"embedding cache row count mismatch: {name}")
        if actual["id_hash"] != expected_id_hashes.get(id_hash_key):
            raise TextEmbeddingError(f"embedding cache ID hash mismatch: {name}")
        if actual["max_abs_norm_error"] > 1e-5:
            raise TextEmbeddingError(f"embedding cache is not L2 normalized: {name}")
    return manifest


class SentenceTransformerProvider:
    def __init__(
        self,
        config: TextEmbeddingConfig,
        *,
        device: str,
        model_cache: Optional[Path] = None,
        local_files_only: bool = False,
    ) -> None:
        actual_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if actual_cublas_config != config.encoding.cublas_workspace_config:
            raise TextEmbeddingError(
                "CUBLAS_WORKSPACE_CONFIG mismatch: expected "
                f"{config.encoding.cublas_workspace_config!r}, got "
                f"{actual_cublas_config!r}"
            )
        actual_packages: Dict[str, str] = {}
        for package, expected in config.required_packages.items():
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as exc:
                raise TextEmbeddingError(
                    f"required package is missing: {package}"
                ) from exc
            actual_packages[package] = actual
            if actual != expected:
                raise TextEmbeddingError(
                    f"package version mismatch for {package}: "
                    f"expected {expected}, got {actual}"
                )

        try:
            import torch
            from huggingface_hub import snapshot_download
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise TextEmbeddingError(
                "sentence-transformers embedding dependencies are unavailable"
            ) from exc

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(
            config.encoding.deterministic_algorithms, warn_only=False
        )
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = config.encoding.allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = config.encoding.allow_tf32
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        snapshot = Path(
            snapshot_download(
                repo_id=config.model.name,
                revision=config.model.revision,
                allow_patterns=config.model.snapshot_allow_patterns,
                cache_dir=None if model_cache is None else str(model_cache),
                local_files_only=local_files_only,
            )
        )
        missing_snapshot_files = [
            name
            for name in config.model.snapshot_allow_patterns
            if not (snapshot / name).is_file()
        ]
        if missing_snapshot_files:
            raise TextEmbeddingError(
                f"model snapshot is missing files: {missing_snapshot_files}"
            )
        snapshot_files = {
            name: _file_identity(snapshot / name)
            for name in sorted(config.model.snapshot_allow_patterns)
        }

        self._model = SentenceTransformer(
            str(snapshot),
            device=device,
            trust_remote_code=config.model.trust_remote_code,
            model_kwargs={
                "torch_dtype": torch.float32,
                "attn_implementation": config.encoding.attention_implementation,
            },
        )
        self._model.max_seq_length = config.model.max_sequence_length
        self._model.eval()
        self._max_sequence_length = config.model.max_sequence_length
        self._input_statistics: Dict[str, Any] = {}
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension != config.model.embedding_dimension:
            raise TextEmbeddingError(
                f"model embedding dimension mismatch: expected "
                f"{config.model.embedding_dimension}, got {dimension}"
            )
        resolved_device = str(self._model.device)
        device_metadata: Dict[str, Any] = {
            "requested": device,
            "resolved": resolved_device,
        }
        if resolved_device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise TextEmbeddingError("CUDA device was requested but is unavailable")
            device_index = self._model.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_index)
            device_metadata.update(
                {
                    "index": int(device_index),
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version(),
                }
            )
        self._metadata: Dict[str, Any] = {
            "provider": "sentence_transformers_v1",
            "model": {
                "name": config.model.name,
                "revision": config.model.revision,
                "snapshot_files": snapshot_files,
                "snapshot_fingerprint": fingerprint(snapshot_files),
            },
            "runtime": {
                "python": platform.python_version(),
                "platform_system": platform.system(),
                "platform_machine": platform.machine(),
                "numpy": np.__version__,
                "packages": actual_packages,
                "device": device_metadata,
                "deterministic_algorithms": config.encoding.deterministic_algorithms,
                "allow_tf32": config.encoding.allow_tf32,
                "model_dtype": config.encoding.model_dtype,
                "attention_implementation": (
                    config.encoding.attention_implementation
                ),
                "cublas_workspace_config": actual_cublas_config,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "input_token_lengths": self._input_statistics,
            },
        }

    def _record_token_lengths(
        self, texts: Sequence[str], *, batch_size: int, role: str
    ) -> None:
        if role not in {"corpus", "queries"}:
            raise TextEmbeddingError(f"invalid embedding input role: {role}")
        if role in self._input_statistics:
            raise TextEmbeddingError(f"embedding input role encoded twice: {role}")
        lengths: List[int] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            tokenized = self._model.tokenizer(
                batch,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_length=True,
            )
            batch_lengths = tokenized.get("length")
            if batch_lengths is None:
                batch_lengths = [len(ids) for ids in tokenized["input_ids"]]
            lengths.extend(int(value) for value in batch_lengths)
        values = np.asarray(lengths, dtype=np.int64)
        truncated = int(np.count_nonzero(values > self._max_sequence_length))
        self._input_statistics[role] = {
            "n": int(len(values)),
            "minimum": int(np.min(values)),
            "maximum": int(np.max(values)),
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "max_sequence_length": self._max_sequence_length,
            "truncated": truncated,
            "truncated_fraction": float(truncated / len(values)),
        }

    def encode(
        self, texts: Sequence[str], *, batch_size: int, role: str
    ) -> np.ndarray:
        self._record_token_lengths(texts, batch_size=batch_size, role=role)
        return np.asarray(
            self._model.encode(
                list(texts),
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        )

    def metadata(self) -> Mapping[str, Any]:
        return self._metadata


def build_or_load_text_embedding_cache(
    config: TextEmbeddingConfig,
    prepared_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    provider: Optional[TextEmbeddingProvider] = None,
    device: str = "cuda",
    model_cache: Optional[Union[str, Path]] = None,
    local_files_only: bool = False,
) -> Dict[str, Any]:
    prepared_value = Path(prepared_dir)
    output_value = Path(output_dir)
    dataset_manifest = _validate_dataset_manifest(
        prepared_value, config.dataset_manifest_fingerprint
    )
    request = _request_identity(config, dataset_manifest)
    request_fingerprint = fingerprint(request)
    if output_value.exists():
        if not output_value.is_dir():
            raise TextEmbeddingError(f"embedding cache is not a directory: {output_value}")
        manifest = _validate_cache(
            output_value,
            expected_request_fingerprint=request_fingerprint,
            expected_dimension=config.model.embedding_dimension,
            expected_counts=dataset_manifest["counts"],
            expected_id_hashes=dataset_manifest["ids"],
        )
        return {
            "reused": True,
            "manifest": output_value / "embedding_manifest.json",
            "fingerprint": manifest["fingerprint"],
            "request_fingerprint": request_fingerprint,
        }

    corpus_ids, corpus_texts, query_ids, query_texts = _format_inputs(
        config, prepared_value, dataset_manifest
    )
    active_provider = provider
    if active_provider is None:
        active_provider = SentenceTransformerProvider(
            config,
            device=device,
            model_cache=None if model_cache is None else Path(model_cache),
            local_files_only=local_files_only,
        )
    corpus = _canonical_embeddings(
        active_provider.encode(
            corpus_texts,
            batch_size=config.encoding.batch_size,
            role="corpus",
        ),
        expected_rows=len(corpus_ids),
        expected_dimension=config.model.embedding_dimension,
        name="corpus",
    )
    queries = _canonical_embeddings(
        active_provider.encode(
            query_texts,
            batch_size=config.encoding.batch_size,
            role="queries",
        ),
        expected_rows=len(query_ids),
        expected_dimension=config.model.embedding_dimension,
        name="query",
    )

    output_value.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_value.name}.", dir=output_value.parent)
    )
    try:
        corpus_path = temporary / "corpus_embeddings.f32.npy"
        corpus_ids_path = temporary / "corpus_ids.json"
        query_path = temporary / "query_embeddings.f32.npy"
        query_ids_path = temporary / "query_ids.json"
        manifest_path = temporary / "embedding_manifest.json"
        np.save(corpus_path, corpus, allow_pickle=False)
        np.save(query_path, queries, allow_pickle=False)
        write_json(corpus_ids_path, corpus_ids)
        write_json(query_ids_path, query_ids)
        provider_metadata = dict(active_provider.metadata())
        manifest: Dict[str, Any] = {
            "schema_version": 1,
            "kind": "normalized_text_embedding_cache",
            "request": request,
            "request_fingerprint": request_fingerprint,
            "dataset": {
                "manifest_fingerprint": dataset_manifest["fingerprint"],
                "counts": dataset_manifest["counts"],
                "corpus_id_hash": dataset_manifest["ids"]["corpus_id_hash"],
                "query_id_hash": dataset_manifest["ids"]["query_id_hash"],
            },
            "model": {
                "name": config.model.name,
                "revision": config.model.revision,
                "homepage": config.model.homepage,
                "license": config.model.license,
                "embedding_dimension": config.model.embedding_dimension,
                "max_sequence_length": config.model.max_sequence_length,
            },
            "formatting": config.raw["formatting"],
            "encoding": config.raw["encoding"],
            "provider": provider_metadata,
            "arrays": {
                "corpus": _array_metadata(corpus_path, corpus_ids_path),
                "queries": _array_metadata(query_path, query_ids_path),
            },
        }
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(manifest_path, manifest)
        temporary.rename(output_value)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "reused": False,
        "manifest": output_value / "embedding_manifest.json",
        "fingerprint": manifest["fingerprint"],
        "request_fingerprint": request_fingerprint,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_text_embedding_config(args.config)
    result = build_or_load_text_embedding_cache(
        config,
        args.dataset,
        args.output,
        device=args.device,
        model_cache=args.model_cache,
        local_files_only=args.local_files_only,
    )
    state = "reused" if result["reused"] else "created"
    print(f"{state} text embedding cache: {result['manifest']}")
    print(f"embedding cache fingerprint: {result['fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
