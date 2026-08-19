from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np


@dataclass(frozen=True)
class EmbeddingTable:
    ids: np.ndarray
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError("embedding vectors must be a two-dimensional array")
        if self.ids.ndim != 1 or len(self.ids) != len(self.vectors):
            raise ValueError("embedding IDs must be one-dimensional and row-aligned")
        string_ids = [str(value) for value in self.ids.tolist()]
        if len(set(string_ids)) != len(string_ids):
            raise ValueError("embedding IDs must be unique")
        if not np.all(np.isfinite(self.vectors)):
            raise ValueError("embedding vectors must be finite")


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("vectors must be a finite two-dimensional array")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise ValueError("cannot normalize zero vectors")
    return values / norms[:, None]


def load_embedding_array(
    vector_path: Union[str, Path], id_path: Union[str, Path]
) -> EmbeddingTable:
    """Load a model-free NPY embedding matrix and a JSON list of stable IDs."""
    vectors = np.load(Path(vector_path), allow_pickle=False)
    ids_raw = json.loads(Path(id_path).read_text(encoding="utf-8"))
    if not isinstance(ids_raw, list) or not all(isinstance(value, str) for value in ids_raw):
        raise ValueError("ID artifact must be a JSON list of strings")
    return EmbeddingTable(np.asarray(ids_raw, dtype=str), np.asarray(vectors))


def make_table(ids: Sequence[str], vectors: np.ndarray) -> EmbeddingTable:
    if not all(isinstance(value, str) for value in ids):
        raise ValueError("all data-boundary IDs must be strings")
    return EmbeddingTable(np.asarray(ids, dtype=str), np.asarray(vectors, dtype=np.float64))
