from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    ids: np.ndarray
    rows: np.ndarray
    squared_distances: np.ndarray
    search_ms: float


class ExactSquaredL2Index:
    """CPU NumPy exact search with lexicographic stable-ID tie breaking."""

    def __init__(self, ids: Sequence[str], vectors: np.ndarray, batch_size: int = 256):
        values = np.asarray(vectors, dtype=np.float64)
        id_values = np.asarray(ids, dtype=str)
        if values.ndim != 2 or id_values.ndim != 1 or len(values) != len(id_values):
            raise ValueError("index IDs and vectors must be row-aligned")
        if len(set(id_values.tolist())) != len(id_values):
            raise ValueError("index IDs must be unique")
        if not np.all(np.isfinite(values)):
            raise ValueError("index vectors must be finite")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.ids = id_values
        self.vectors = values
        self.batch_size = batch_size
        self._norms = np.einsum("ij,ij->i", values, values)
        self._tie_rank = np.argsort(np.argsort(id_values, kind="stable"), kind="stable")

    def search(self, queries: np.ndarray, k: int) -> SearchResult:
        values = np.asarray(queries, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.vectors.shape[1]:
            raise ValueError("query shape does not match index dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("queries must be finite")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(self.ids):
            raise ValueError("k must lie between 1 and corpus size")
        started = perf_counter()
        all_rows = []
        all_distances = []
        for start in range(0, len(values), self.batch_size):
            query_batch = values[start : start + self.batch_size]
            query_norms = np.einsum("ij,ij->i", query_batch, query_batch)
            distances = (
                query_norms[:, None]
                + self._norms[None, :]
                - 2.0 * (query_batch @ self.vectors.T)
            )
            np.maximum(distances, 0.0, out=distances)
            for row_distances in distances:
                ordered = np.lexsort((self._tie_rank, row_distances))[:k]
                all_rows.append(ordered)
                all_distances.append(row_distances[ordered])
        elapsed_ms = (perf_counter() - started) * 1000.0
        rows = np.asarray(all_rows, dtype=np.int64)
        return SearchResult(
            ids=self.ids[rows],
            rows=rows,
            squared_distances=np.asarray(all_distances, dtype=np.float64),
            search_ms=elapsed_ms,
        )
