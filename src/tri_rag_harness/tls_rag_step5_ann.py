"""ANN search, bounded exact reference, and persistent candidate membership.

An expansion restarts HNSW. No hidden traversal-state or exact-prefix reuse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np


def stable_top(ids, distances, k):
    ids = np.asarray(ids, dtype=str)
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1 or len(ids) != len(distances) or not 0 < k <= len(ids):
        raise ValueError("invalid top-k input")
    if not np.all(np.isfinite(distances)):
        raise ValueError("nonfinite distances")
    boundary = np.partition(distances, k - 1)[k - 1]
    lower = np.flatnonzero(distances < boundary)
    equal = np.flatnonzero(distances == boundary)
    chosen = np.concatenate((lower, equal[np.argsort(ids[equal], kind="stable")[:k - len(lower)]]))
    return chosen[np.lexsort((ids[chosen], distances[chosen]))]


def validate_vectors(ids, vectors):
    values = np.asarray(vectors)
    if values.ndim != 2 or len(ids) != len(values) or not len(values):
        raise ValueError("unaligned vectors")
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("IDs must be unique nonempty strings")
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite vectors")
    return values


class ExactCandidateIndex:
    """Float64 direct-difference reference, bounded by block_rows * dimension."""
    def __init__(self, ids, vectors, block_rows=8192):
        self.ids = tuple(ids)
        self.vectors = validate_vectors(ids, vectors)
        self.block_rows = block_rows
        if type(block_rows) is not int or block_rows < 1:
            raise ValueError("invalid block size")

    def search(self, query, k):
        query = np.asarray(query, dtype=np.float64)
        if query.shape != (self.vectors.shape[1],) or not np.all(np.isfinite(query)):
            raise ValueError("invalid query")
        if type(k) is not int or not 0 < k <= len(self.ids):
            raise ValueError("invalid k")
        keep_ids, keep_distances = [], np.empty(0)
        for start in range(0, len(self.ids), self.block_rows):
            difference = np.asarray(self.vectors[start:start + self.block_rows], dtype=np.float64) - query
            distances = np.einsum("ij,ij->i", difference, difference)
            ids = list(self.ids[start:start + self.block_rows]) + keep_ids
            distances = np.concatenate((distances, keep_distances))
            chosen = stable_top(ids, distances, min(k, len(ids)))
            keep_ids = np.asarray(ids)[chosen].tolist()
            keep_distances = distances[chosen]
        return tuple(keep_ids)


class HnswCandidateIndex:
    def __init__(self, ids, vectors, config, seed):
        import faiss
        self.ids = tuple(ids)
        values = np.ascontiguousarray(validate_vectors(ids, vectors), dtype=np.float32)
        if faiss.__version__ != config["faiss_version"]:
            raise ValueError("FAISS version must match the Step 5 protocol")
        faiss.omp_set_num_threads(config["threads"])
        self.config = config
        self.index = faiss.IndexHNSWFlat(values.shape[1], config["connections"], faiss.METRIC_L2)
        self.index.hnsw.efConstruction = config["ef_construction"]
        self.index.hnsw.rng = faiss.RandomGenerator(seed)
        self.index.add(values)  # fixed caller order, one build thread

    def search(self, query, k):
        if type(k) is not int or not 0 < k <= len(self.ids):
            raise ValueError("invalid ANN k")
        values = np.ascontiguousarray(query, dtype=np.float32)
        if values.shape != (self.index.d,) or not np.all(np.isfinite(values)):
            raise ValueError("invalid ANN query")
        self.index.hnsw.efSearch = max(self.config["ef_floor"], self.config["ef_multiplier"] * k)
        _, rows = self.index.search(values[None, :], k)
        found = rows[0].tolist()
        if len(set(found)) != k or any(i < 0 or i >= len(self.ids) for i in found):
            raise ValueError("ANN returned missing or duplicate IDs; no silent fallback")
        return tuple(self.ids[i] for i in found)

    def save(self, path):
        import faiss
        faiss.write_index(self.index, str(path))


class TorchExactGpuIndex:
    """Direct float64 squared L2 in GPU blocks, with string-ID boundary ties.

    Full block distances are transferred and stably merged on the host. This
    conservative reference charges transfers; it is not an optimized GPU ANN.
    """
    def __init__(self, ids, vectors, block_rows=8192):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU exact reference requires CUDA PyTorch")
        self.torch, self.ids, self.block_rows = torch, tuple(ids), block_rows
        self.values = torch.as_tensor(np.array(vectors, dtype=np.float64), device="cuda", dtype=torch.float64)
        torch.cuda.synchronize()

    def search(self, query, k):
        torch = self.torch
        torch.cuda.synchronize()
        q = torch.as_tensor(np.asarray(query), device="cuda", dtype=torch.float64)
        best_ids, best_distances = [], np.empty(0)
        for start in range(0, len(self.ids), self.block_rows):
            diff = self.values[start:start + self.block_rows] - q
            distances = torch.sum(diff * diff, dim=1).cpu().numpy()
            ids = list(self.ids[start:start + self.block_rows]) + best_ids
            distances = np.concatenate((distances, best_distances))
            selected = stable_top(ids, distances, min(k, len(ids)))
            best_ids, best_distances = np.asarray(ids)[selected].tolist(), distances[selected]
        torch.cuda.synchronize()
        return tuple(best_ids)


def expand_candidate_pool(previous, raw_ids, projected_distances, budget):
    if len(set(previous)) != len(previous) or len(set(raw_ids)) != len(raw_ids):
        raise ValueError("duplicate candidate IDs")
    if len(previous) > budget or len(raw_ids) != budget:
        raise ValueError("ANN must return the requested number of candidates")
    new = sorted(set(raw_ids) - set(previous), key=lambda i: (projected_distances[i], i))
    accepted = tuple(previous) + tuple(new[:budget - len(previous)])
    if len(accepted) != budget:
        raise ValueError("insufficient distinct ANN candidates")
    return accepted


@dataclass
class CandidateSession:
    ids: tuple
    original: np.ndarray
    projected: np.ndarray
    query: np.ndarray
    projected_query: np.ndarray
    index: object
    k_ctx: int
    row_by_id: dict = field(default_factory=dict)
    pool: tuple = ()
    original_cache: dict = field(default_factory=dict)
    projected_cache: dict = field(default_factory=dict)
    stage: int = -1

    def __post_init__(self):
        if not self.row_by_id:
            self.row_by_id = {item: row for row, item in enumerate(self.ids)}
        if len(self.row_by_id) != len(self.ids):
            raise ValueError("duplicate IDs")

    def expand(self, budget):
        if budget <= len(self.pool):
            raise ValueError("budgets must strictly increase")
        started = perf_counter()
        raw = self.index.search(self.projected_query, budget)
        searched = perf_counter()
        for item in raw:
            if item not in self.projected_cache:
                diff = np.asarray(self.projected[self.row_by_id[item]], dtype=np.float64) - self.projected_query
                self.projected_cache[item] = float(diff @ diff)
        previous = set(self.pool)
        self.pool = expand_candidate_pool(self.pool, raw, self.projected_cache, budget)
        gathered = perf_counter()
        new = [i for i in self.pool if i not in previous]
        rows = [self.row_by_id[i] for i in new]
        difference = np.asarray(self.original[rows], dtype=np.float64) - self.query
        distances = np.einsum("ij,ij->i", difference, difference)
        self.original_cache.update(zip(new, map(float, distances)))
        scored = perf_counter()
        reranked = tuple(sorted(self.pool, key=lambda i: (self.original_cache[i], i)))
        ended = perf_counter()
        self.stage += 1
        return {"stage": self.stage, "budget": budget, "raw_ids": list(raw),
                "candidate_ids": list(self.pool), "new_ids": new,
                "reranked_ids": list(reranked), "context_ids": list(reranked[:self.k_ctx]),
                "original_distances": len(new), "projected_refinements": len(raw),
                "ef_search": max(self.index.config["ef_floor"], self.index.config["ef_multiplier"] * budget)
                if isinstance(self.index, HnswCandidateIndex) else None,
                "timing_ms": {"search": (searched - started) * 1000,
                              "candidate_refine_merge": (gathered - searched) * 1000,
                              "gather_original_score": (scored - gathered) * 1000,
                              "rerank_context": (ended - scored) * 1000}}
