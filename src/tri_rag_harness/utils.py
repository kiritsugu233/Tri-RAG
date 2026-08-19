from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def array_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(contiguous.shape)))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def stable_id_hash(ids: list[str]) -> str:
    return fingerprint(sorted(ids))
