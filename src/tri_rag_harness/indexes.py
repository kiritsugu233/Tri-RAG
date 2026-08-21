from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    ids: np.ndarray
    rows: np.ndarray
    squared_distances: np.ndarray
    search_ms: float


@dataclass(frozen=True)
class StreamingSearchResult:
    rows: np.ndarray
    squared_distances: np.ndarray
    search_ms: float
    distance_evaluations: int
    scanned_vector_bytes: int


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


def _stable_top_k_rows(
    squared_distances: np.ndarray, rows: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return distance/row ordered top-k without fully sorting a large block."""
    distances = np.asarray(squared_distances)
    row_values = np.asarray(rows, dtype=np.int64)
    if distances.ndim != 1 or row_values.ndim != 1 or len(distances) != len(row_values):
        raise ValueError("distances and rows must be aligned one-dimensional arrays")
    if not 1 <= k <= len(distances):
        raise ValueError("k must lie between 1 and the candidate count")
    if len(distances) == k:
        selected = np.arange(k, dtype=np.int64)
    else:
        boundary = np.partition(distances, k - 1)[k - 1]
        lower = np.flatnonzero(distances < boundary)
        equal = np.flatnonzero(distances == boundary)
        needed = k - len(lower)
        equal_order = np.argsort(row_values[equal], kind="stable")[:needed]
        selected = np.concatenate([lower, equal[equal_order]])
    order = np.lexsort((row_values[selected], distances[selected]))
    selected = selected[order]
    return row_values[selected], distances[selected]


class StreamingExactSquaredL2Index:
    """Memory-bounded exact single-query search over ndarray or memmap rows.

    The backend never materializes a query-by-corpus matrix. Each call scans the
    corpus once in ``block_rows`` chunks and retains only the global top-k. Row
    number is the deterministic tie breaker because benchmark IDs are implicit
    stable ``doc-{row}`` strings.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        *,
        squared_norms: Optional[np.ndarray] = None,
        block_rows: int = 8192,
    ):
        values = np.asanyarray(vectors)
        if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating):
            raise ValueError("vectors must be a floating two-dimensional array")
        if isinstance(block_rows, bool) or not isinstance(block_rows, int) or block_rows < 1:
            raise ValueError("block_rows must be a positive integer")
        if squared_norms is None:
            norms = np.einsum("ij,ij->i", values, values, dtype=np.float64)
        else:
            norms = np.asarray(squared_norms)
        if norms.ndim != 1 or len(norms) != len(values):
            raise ValueError("squared_norms must align with vector rows")
        if not np.all(np.isfinite(norms)) or np.any(norms < 0.0):
            raise ValueError("squared_norms must be finite and nonnegative")
        self.vectors = values
        self.squared_norms = norms
        self.block_rows = block_rows
        self.scan_calls = 0

    def search_one(self, query: np.ndarray, k: int) -> StreamingSearchResult:
        query_value = np.asarray(query, dtype=self.vectors.dtype).reshape(-1)
        if len(query_value) != self.vectors.shape[1] or not np.all(
            np.isfinite(query_value)
        ):
            raise ValueError("query must be finite and match the index dimension")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(
            self.vectors
        ):
            raise ValueError("k must lie between 1 and corpus size")
        self.scan_calls += 1
        started = perf_counter()
        query_norm = float(np.dot(query_value, query_value))
        best_rows: Optional[np.ndarray] = None
        best_distances: Optional[np.ndarray] = None
        for start in range(0, len(self.vectors), self.block_rows):
            stop = min(start + self.block_rows, len(self.vectors))
            block = self.vectors[start:stop]
            distances = (
                query_norm
                + self.squared_norms[start:stop]
                - 2.0 * (block @ query_value)
            )
            distances = np.asarray(distances)
            np.maximum(distances, 0.0, out=distances)
            block_rows = np.arange(start, stop, dtype=np.int64)
            block_k = min(k, len(block_rows))
            rows, selected_distances = _stable_top_k_rows(
                distances, block_rows, block_k
            )
            if best_rows is None:
                best_rows = rows
                best_distances = selected_distances
            else:
                merged_rows = np.concatenate([best_rows, rows])
                merged_distances = np.concatenate([best_distances, selected_distances])
                merged_k = min(k, len(merged_rows))
                best_rows, best_distances = _stable_top_k_rows(
                    merged_distances, merged_rows, merged_k
                )
        if best_rows is None or best_distances is None:
            raise AssertionError("streaming index cannot search an empty corpus")
        return StreamingSearchResult(
            rows=best_rows,
            squared_distances=np.asarray(best_distances, dtype=np.float64),
            search_ms=(perf_counter() - started) * 1000.0,
            distance_evaluations=len(self.vectors),
            scanned_vector_bytes=int(self.vectors.nbytes),
        )
