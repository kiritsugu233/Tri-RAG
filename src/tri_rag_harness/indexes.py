from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Dict, Optional, Sequence

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
    query_upload_ms: float = 0.0
    backend_search_ms: float = 0.0
    result_download_ms: float = 0.0


@dataclass(frozen=True)
class FaissIndexBuildMetrics:
    backend: str
    faiss_version: str
    dimension: int
    vector_count: int
    vector_bytes: int
    host_index_build_ms: float
    host_to_device_ms: float
    gpu_device: Optional[int]
    transfer_timing_mode: str
    faiss_max_threads: Optional[int]

    def serialize(self) -> Dict[str, Any]:
        return asdict(self)


class FaissUnavailableError(RuntimeError):
    pass


class FaissBoundaryTieError(RuntimeError):
    pass


def _load_faiss() -> Any:
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise FaissUnavailableError(
            "FAISS is not installed; use a cluster FAISS CPU/GPU environment"
        ) from exc
    return faiss


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
        elapsed_ms = (perf_counter() - started) * 1000.0
        return StreamingSearchResult(
            rows=best_rows,
            squared_distances=np.asarray(best_distances, dtype=np.float64),
            search_ms=elapsed_ms,
            distance_evaluations=len(self.vectors),
            scanned_vector_bytes=int(self.vectors.nbytes),
            backend_search_ms=elapsed_ms,
        )


class FaissExactSquaredL2Index:
    """Exact float32 FAISS FlatL2 adapter for CPU or one NVIDIA GPU.

    FAISS does not expose a stable-ID tie contract at the top-k boundary. The
    adapter requests one extra result and refuses a boundary tie rather than
    silently returning a backend-dependent candidate set. Returned non-boundary
    ties are ordered by squared distance and then corpus row.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        *,
        device: str = "cpu",
        gpu_device: int = 0,
        faiss_threads: int = 1,
        faiss_module: Optional[Any] = None,
        enable_torch_transfer_timing: bool = True,
    ):
        values = np.asanyarray(vectors)
        if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating):
            raise ValueError("vectors must be a floating two-dimensional array")
        if not np.all(np.isfinite(values)):
            raise ValueError("index vectors must be finite")
        if device not in {"cpu", "gpu"}:
            raise ValueError("FAISS device must be 'cpu' or 'gpu'")
        if (
            isinstance(gpu_device, bool)
            or not isinstance(gpu_device, int)
            or gpu_device < 0
        ):
            raise ValueError("gpu_device must be a nonnegative integer")
        if (
            isinstance(faiss_threads, bool)
            or not isinstance(faiss_threads, int)
            or faiss_threads < 1
        ):
            raise ValueError("faiss_threads must be a positive integer")
        self.faiss = _load_faiss() if faiss_module is None else faiss_module
        if hasattr(self.faiss, "omp_set_num_threads"):
            self.faiss.omp_set_num_threads(faiss_threads)
        self.device = device
        self.gpu_device = gpu_device if device == "gpu" else None
        self.vectors = values
        self.vector_count = len(values)
        self.dimension = values.shape[1]
        self.scan_calls = 0
        self._torch = None
        self._torch_transfer_timing = False
        host_index = self.faiss.IndexFlatL2(self.dimension)
        host_started = perf_counter()
        host_index.add(np.ascontiguousarray(values, dtype=np.float32))
        host_build_ms = (perf_counter() - host_started) * 1000.0
        host_to_device_ms = 0.0
        transfer_timing_mode = "host_numpy_api_inclusive"
        self._gpu_resources = None
        if device == "gpu":
            required = ("StandardGpuResources", "index_cpu_to_gpu")
            if any(not hasattr(self.faiss, name) for name in required):
                raise FaissUnavailableError(
                    "installed FAISS build has no NVIDIA GPU support"
                )
            self._gpu_resources = self.faiss.StandardGpuResources()
            transfer_started = perf_counter()
            self.index = self.faiss.index_cpu_to_gpu(
                self._gpu_resources, gpu_device, host_index
            )
            host_to_device_ms = (perf_counter() - transfer_started) * 1000.0
            if enable_torch_transfer_timing:
                try:
                    import torch
                    import faiss.contrib.torch_utils  # type: ignore  # noqa: F401

                    if torch.cuda.is_available():
                        self._torch = torch
                        self._torch_transfer_timing = True
                        transfer_timing_mode = "torch_cuda_split"
                except (ImportError, RuntimeError):
                    pass
        else:
            self.index = host_index
        if int(self.index.ntotal) != self.vector_count:
            raise AssertionError("FAISS index did not retain every corpus vector")
        version = getattr(self.faiss, "__version__", None)
        if version is None and hasattr(self.faiss, "get_version"):
            version = self.faiss.get_version()
        self.build_metrics = FaissIndexBuildMetrics(
            backend=f"faiss_{device}_index_flat_l2",
            faiss_version=str(version) if version is not None else "unknown",
            dimension=self.dimension,
            vector_count=self.vector_count,
            vector_bytes=int(
                self.vector_count
                * self.dimension
                * np.dtype(np.float32).itemsize
            ),
            host_index_build_ms=host_build_ms,
            host_to_device_ms=host_to_device_ms,
            gpu_device=self.gpu_device,
            transfer_timing_mode=transfer_timing_mode,
            faiss_max_threads=(
                int(self.faiss.omp_get_max_threads())
                if hasattr(self.faiss, "omp_get_max_threads")
                else None
            ),
        )

    def _search_numpy(
        self, query: np.ndarray, requested_k: int
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        started = perf_counter()
        distances, rows = self.index.search(query, requested_k)
        search_ms = (perf_counter() - started) * 1000.0
        return distances, rows, 0.0, search_ms, 0.0

    def _search_torch(
        self, query: np.ndarray, requested_k: int
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        torch = self._torch
        if torch is None:
            raise AssertionError("torch transfer timing was not initialized")
        cuda_device = torch.device(f"cuda:{self.gpu_device}")
        host_query = torch.from_numpy(query)
        torch.cuda.synchronize(cuda_device)
        upload_started = perf_counter()
        device_query = host_query.to(cuda_device)
        torch.cuda.synchronize(cuda_device)
        upload_ms = (perf_counter() - upload_started) * 1000.0
        search_started = perf_counter()
        distances, rows = self.index.search(device_query, requested_k)
        torch.cuda.synchronize(cuda_device)
        search_ms = (perf_counter() - search_started) * 1000.0
        download_started = perf_counter()
        if torch.is_tensor(distances):
            distances = distances.cpu().numpy()
        else:
            distances = np.asarray(distances)
        if torch.is_tensor(rows):
            rows = rows.cpu().numpy()
        else:
            rows = np.asarray(rows)
        torch.cuda.synchronize(cuda_device)
        download_ms = (perf_counter() - download_started) * 1000.0
        return distances, rows, upload_ms, search_ms, download_ms

    def search_one(self, query: np.ndarray, k: int) -> StreamingSearchResult:
        query_value = np.asarray(query, dtype=np.float32).reshape(-1)
        if len(query_value) != self.dimension or not np.all(np.isfinite(query_value)):
            raise ValueError("query must be finite and match the index dimension")
        if (
            isinstance(k, bool)
            or not isinstance(k, int)
            or not 1 <= k <= self.vector_count
        ):
            raise ValueError("k must lie between 1 and corpus size")
        self.scan_calls += 1
        requested_k = min(self.vector_count, k + 1)
        query_matrix = np.ascontiguousarray(query_value[None, :], dtype=np.float32)
        total_started = perf_counter()
        if self._torch_transfer_timing:
            distances, rows, upload_ms, backend_ms, download_ms = self._search_torch(
                query_matrix, requested_k
            )
        else:
            distances, rows, upload_ms, backend_ms, download_ms = self._search_numpy(
                query_matrix, requested_k
            )
        total_ms = (perf_counter() - total_started) * 1000.0
        row_values = np.asarray(rows[0], dtype=np.int64)
        distance_values = np.asarray(distances[0], dtype=np.float64)
        if np.any(row_values < 0) or np.any(row_values >= self.vector_count):
            raise FloatingPointError("FAISS returned an invalid corpus row")
        if not np.all(np.isfinite(distance_values)) or np.any(distance_values < 0.0):
            raise FloatingPointError("FAISS returned an invalid squared L2 distance")
        if requested_k > k and distance_values[k - 1] == distance_values[k]:
            raise FaissBoundaryTieError(
                "FAISS top-k boundary is tied; stable candidate identity is undefined"
            )
        selected_rows = row_values[:k]
        selected_distances = distance_values[:k]
        order = np.lexsort((selected_rows, selected_distances))
        return StreamingSearchResult(
            rows=selected_rows[order],
            squared_distances=selected_distances[order],
            search_ms=total_ms,
            distance_evaluations=self.vector_count,
            scanned_vector_bytes=int(
                self.vector_count * self.dimension * np.dtype(np.float32).itemsize
            ),
            query_upload_ms=upload_ms,
            backend_search_ms=backend_ms,
            result_download_ms=download_ms,
        )
