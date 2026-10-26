"""Isolated, approved CPU probe; no predictor and no production replacement."""
from __future__ import annotations

from time import perf_counter

import numpy as np

from mechanism.tls_rag_step5_audit import ExactScan
from mechanism.tls_rag_step5_exact_policy import PrefixSession
from tri_rag_harness.projection import project_rows
from tri_rag_harness.tls_rag_step5_ann import stable_top

GRID = (32, 64, 128, 256)
CASES = ('stream_fixed', 'stream_max', 'cache_reselect_fixed',
         'cache_reselect_lazy', 'cache_partition_fixed', 'cache_partition_lazy', 'cache_max')
SPEC = dict(schema='step5-reuse-probe-v1', corpus_size=1000000, dimension=384,
            m_prime=128, corpus_seed=55251, query_seed=55253, projection_seed=55117,
            order_seed=55259, bootstrap_seed=55261, block_rows=8192, k_ctx=5,
            queries=12, repeats=3, warmups=1, bootstrap_replicates=2000,
            budget_grid=list(GRID), cases=list(CASES))


class ExactDistanceSession:
    """One query, one exact scan; private cache, current-prefix-only results."""
    def __init__(self, ids, vectors, query, block_rows=8192, selector='reselect'):
        self._ids = np.asarray(ids, dtype=str)
        self._query = np.array(query, dtype=np.float64, copy=True)
        if (selector not in ('reselect', 'partition') or type(block_rows) is not int
                or block_rows < 1 or vectors.ndim != 2 or not len(self._ids)
                or vectors.shape[0] != len(self._ids)
                or self._query.shape != (vectors.shape[1],)
                or not np.all(np.isfinite(self._query))):
            raise ValueError('invalid cache geometry/query/selector')
        # ID uniqueness is a shared, fixture-level validation, outside all timings.
        self._selector, self._budget = selector, 0
        self._query.flags.writeable = False
        self._distances = np.empty(len(self._ids), dtype=np.float64)
        at = perf_counter()
        for start in range(0, len(self._ids), block_rows):
            end = min(start + block_rows, len(self._ids))
            diff = np.asarray(vectors[start:end], dtype=np.float64) - self._query
            distance = np.einsum('ij,ij->i', diff, diff)
            if not np.all(np.isfinite(distance)):
                raise ValueError('nonfinite exact distances')
            self._distances[start:end] = distance
        self.scan_write_ms = (perf_counter()-at)*1000
        self._order = np.arange(len(self._ids)) if selector == 'partition' else None
        self.distance_evaluations, self.scan_count = len(self._ids), 1
        self.selection_rows = 0
        self.selection_ms = 0.

    @property
    def retained_bytes(self):
        return self._distances.nbytes + (0 if self._order is None else self._order.nbytes)

    def expand(self, budget):
        if type(budget) is not int or not self._budget < budget <= len(self._ids):
            raise ValueError('budgets must be integer, increasing and in corpus')
        at = perf_counter()
        if self._selector == 'reselect':
            rows = stable_top(self._ids, self._distances, budget)
            self.selection_rows += len(self._ids)
        else:
            lo, need = self._budget, budget-self._budget
            tail = self._order[lo:]
            # All previous ranks are final. Partition only the remaining suffix.
            permutation = np.argpartition(self._distances[tail], need-1)
            self._order[lo:] = tail[permutation]
            del permutation
            tail = self._order[lo:]
            cutoff = self._distances[self._order[budget-1]]
            equal = np.flatnonzero(self._distances[tail] == cutoff)
            tie_rows = tail[equal]
            # Argpartition may split a cutoff tie arbitrarily. Reassign all equal
            # slots in position order so the prefix contains the smallest IDs.
            self._order[lo+equal] = tie_rows[np.argsort(self._ids[tie_rows], kind='stable')]
            new = self._order[lo:budget].copy()
            self._order[lo:budget] = new[np.lexsort((self._ids[new], self._distances[new]))]
            rows = self._order[:budget].copy()
            self.selection_rows += len(tail)
        self._budget = budget
        distances = self._distances[rows].copy()
        rows.flags.writeable = False
        distances.flags.writeable = False
        self.selection_ms += (perf_counter()-at)*1000
        return rows, distances


def run_case(reader, query, matrix, case, final_m, grid=GRID, block_rows=8192,
             validate=None):
    """Complete projected retrieval + exact reranking; no controller/label inputs.

    Optional validation runs after the timing boundary and cannot choose M.
    The caller validates shared IDs/shapes once for every case, before timing.
    """
    if case not in CASES or final_m not in grid:
        raise ValueError('unknown case/endpoint')
    at = perf_counter()
    qp = project_rows(query[None], matrix)[0]
    projection_ms = (perf_counter()-at)*1000
    schedule = [final_m] if case.endswith('_fixed') else [m for m in grid if m <= final_m]
    eager = case in ('stream_max', 'cache_max')
    requested = grid[-1] if eager else final_m
    cached = case.startswith('cache_')
    search_at = perf_counter()
    if cached:
        backend = ExactDistanceSession(reader.id_array, reader.projected, qp,
            block_rows, 'partition' if 'partition' in case else 'reselect')
        scan_ms = backend.scan_write_ms
        retained = backend.retained_bytes
        if eager:
            all_rows, all_d = backend.expand(requested)
    else:
        backend = ExactScan(reader.id_array, reader.projected, block_rows)
        all_rows, all_d, _ = backend.search(qp, requested)
        retained = all_rows.nbytes+all_d.nbytes
        scan_ms = (perf_counter()-search_at)*1000-backend.last_selection_ms
    cfg = {'k_ctx': SPEC['k_ctx']}
    session = PrefixSession(reader, query, qp, cfg)
    prefixes, records = [], []
    for budget in schedule:
        rows, dd = (backend.expand(budget) if cached and not eager
                    else (all_rows[:budget], all_d[:budget]))
        prefixes.append((rows, dd))
        records.append(session.expand(rows, dd))
    originals = session.original_cache.copy()
    selection_ms = backend.selection_ms if cached else backend.last_selection_ms
    selection_rows = backend.selection_rows if cached else None
    # Charge release of full-size cache/order/partition buffers in every case.
    del backend, session
    elapsed_ms = (perf_counter()-at)*1000
    result = dict(case=case, final_m=final_m, pipeline_ms=elapsed_ms,
        projection_ms=projection_ms, scan_ms=scan_ms, selection_ms=selection_ms,
        gather_score_ms=sum(r['gather_score_ms'] for r in records),
        rerank_ms=sum(r['rerank_ms'] for r in records),
        original_distances=sum(r['original_distances'] for r in records),
        projected_distances=len(reader.id_array), scans=1,
        selection_rows=selection_rows, retained_buffer_bytes=retained)
    if validate is not None:
        validate(prefixes, records, originals)
    return result
