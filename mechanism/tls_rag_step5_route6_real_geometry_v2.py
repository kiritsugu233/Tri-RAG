"""B0 v2: shared FP64 CUDA corpus, batched direct L2, CPU near-order repair."""
import time

import numpy as np

from mechanism import tls_rag_step5_route6_real_geometry_v1 as reference

ENVELOPE = 2.**-36


def repair_order(values, refine, envelope=ENVELOPE, check=lambda: None):
    """Refine every connected component of overlapping comparison intervals.

    If each GPU/CPU discrepancy is <= E, gaps > 2E cannot reverse. All other
    adjacent links are joined transitively before any values are changed.
    The envelope assumption is audited, not a formal kernel certificate.
    Mutates only the caller-owned working vector, never source embeddings.
    """
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype != np.float64 or not np.all(np.isfinite(values)):
        raise ValueError('finite FP64 score vector required')
    if envelope != ENVELOPE:
        raise ValueError('frozen comparison envelope')
    stamp = time.perf_counter()
    order = reference.stable_order(values)
    links = np.flatnonzero(np.diff(values[order]) <= 2*envelope)
    starts = links[np.r_[True, np.diff(links) > 1]] if len(links) else []
    stops = links[np.r_[np.diff(links) > 1, True]]+2 if len(links) else []
    maximum, count, largest = 0., 0, 0
    for start, stop in zip(starts, stops):
        check()
        rows = order[start:stop].copy()
        exact = np.asarray(refine(rows), dtype=np.float64)
        if exact.shape != rows.shape or not np.all(np.isfinite(exact)):
            raise ArithmeticError('invalid CPU reference scores')
        error = float(np.max(np.abs(exact-values[rows])))
        if error > envelope:
            raise ArithmeticError('FP64 comparison envelope violated')
        maximum = max(maximum, error)
        values[rows] = exact
        order[start:stop] = rows[np.lexsort((rows, exact))]
        count += len(rows)
        largest = max(largest, len(rows))
    return order, dict(envelope=envelope, clusters=len(starts), refined_rows=count,
                       largest_cluster=largest, max_checked_difference=maximum,
                       seconds=time.perf_counter()-stamp, roundoff_certified=False)


def cpu_scores(points, query, metric):
    x, q = np.array(points, dtype=np.float64, copy=True), np.asarray(query, dtype=np.float64)
    if metric == 'l2':
        x -= q
        return np.einsum('ij,ij->i', x, x)
    if metric == 'ip':
        return -(x@q)
    raise ValueError('reference metric')


class CudaCorpus:
    """Read the source once. No projection, normalization or ANN representation."""
    def __init__(self, corpus, cfg, check=lambda: None):
        from mechanism.tls_rag_step5_route6_a800_v1 import torch_backend
        self.torch, precision = torch_backend()
        t = self.torch
        self.cfg, self.check = cfg, check
        self.device = t.device('cuda:0')
        free, total = t.cuda.mem_get_info(0)
        if free < cfg['limits']['gpu_bytes']:
            raise RuntimeError('insufficient free device memory')
        t.cuda.set_per_process_memory_fraction(cfg['limits']['gpu_bytes']/total, 0)
        self.x = t.empty(corpus.shape, device=self.device, dtype=t.float64)
        stamp = time.perf_counter()
        self.max_norm_error, read_seconds, copy_seconds = 0., 0., 0.
        for start in range(0, len(corpus), cfg['block']):
            check()
            begin = time.perf_counter()
            x = np.array(corpus[start:start+cfg['block']], dtype=np.float64, copy=True)
            if not np.all(np.isfinite(x)):
                raise ValueError('nonfinite source')
            error = float(np.max(np.abs(np.linalg.norm(x, axis=1)-1)))
            self.max_norm_error = max(self.max_norm_error, error)
            if error > cfg['norm_tolerance']:
                raise ValueError('stored source norm')
            read_seconds += time.perf_counter()-begin
            begin = time.perf_counter()
            self.x[start:start+len(x)].copy_(t.as_tensor(x, dtype=t.float64, device=self.device))
            t.cuda.synchronize()
            copy_seconds += time.perf_counter()-begin
        self.identity = dict(backend='cuda-fp64-direct-with-cpu-order-repair', dtype='float64',
                             device=t.cuda.get_device_name(0), precision=precision,
                             source_device_bytes=self.x.numel()*self.x.element_size(),
                             source_read_norm_seconds=read_seconds, h2d_seconds=copy_seconds,
                             corpus_prepare_seconds=time.perf_counter()-stamp,
                             max_norm_error=self.max_norm_error)

    def close(self):
        self.x = None
        self.torch.cuda.empty_cache()

    def points(self, rows):
        ix = self.torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)
        return self.x[ix].cpu().numpy()

    def refine(self, rows, query, metric):
        result = np.empty(len(rows), dtype=np.float64)
        for start in range(0, len(rows), self.cfg['block']):
            self.check()
            selected = rows[start:start+self.cfg['block']]
            result[start:start+len(selected)] = cpu_scores(self.points(selected), query, metric)
        return result

    def batch(self, queries):
        queries = np.asarray(queries, dtype=np.float64)
        if (queries.ndim != 2 or queries.shape[1] != self.x.shape[1]
                or len(queries) > self.cfg['query_batch'] or not np.all(np.isfinite(queries))
                or np.max(np.abs(np.linalg.norm(queries, axis=1)-1)) > self.cfg['norm_tolerance']):
            raise ValueError('query batch/norm')
        t, n = self.torch, len(self.x)
        stamp = time.perf_counter()
        q = t.as_tensor(queries, dtype=t.float64, device=self.device)
        d2, ip = np.empty((len(q), n)), np.empty((len(q), n))
        kernel_seconds, d2h_seconds = 0., 0.
        for start in range(0, n, self.cfg['block']):
            self.check()
            x = self.x[start:start+self.cfg['block']]
            begin = time.perf_counter()
            difference = x[None, :, :]-q[:, None, :]
            distances = t.sum(difference*difference, dim=2, dtype=t.float64)
            scores = -(q@x.T)
            t.cuda.synchronize()
            kernel_seconds += time.perf_counter()-begin
            begin = time.perf_counter()
            d2[:, start:start+len(x)] = distances.cpu().numpy()
            ip[:, start:start+len(x)] = scores.cpu().numpy()
            d2h_seconds += time.perf_counter()-begin
            del difference, distances, scores
        common = dict(kernel_seconds=kernel_seconds, d2h_seconds=d2h_seconds,
                      batch_distance_seconds=time.perf_counter()-stamp, batch_queries=len(queries))
        for index, query in enumerate(queries):
            self.check()
            begin = time.perf_counter()
            order, l2_audit = repair_order(d2[index], lambda rows: self.refine(rows, query, 'l2'), check=self.check)
            ip_order, ip_audit = repair_order(ip[index], lambda rows: self.refine(rows, query, 'ip'), check=self.check)
            # Exact values at the reporting cutoff and every IP top-k row.
            rows = np.unique(np.r_[order[:self.cfg['k']+1], ip_order[:self.cfg['k']]])
            actual = self.refine(rows, query, 'l2')
            if np.max(np.abs(actual-d2[index, rows])) > ENVELOPE:
                raise ArithmeticError('reported cutoff audit failed')
            d2[index, rows] = actual
            yield dict(d2=d2[index], native_ip=ip[index], order=order, ip_order=ip_order,
                       zero_rows=np.flatnonzero(d2[index] == 0), max_norm_error=self.max_norm_error,
                       order_audit=dict(l2=l2_audit, ip=ip_audit),
                       costs=dict(query_order_seconds=time.perf_counter()-begin), batch_costs=common)


def device_self_test(cfg):
    """Deterministic tiny original-space test, not projection sampling/MC."""
    points = np.array([[1., 0., 0., 0.], [1., 1e-7, 0., 0.], [1., -1e-7, 0., 0.],
                       [0., 1., 0., 0.], [0., -1., 0., 0.], [0., 0., 1., 0.],
                       [.6, .8, 0., 0.], [.6, -.8, 0., 0.], [1., 0., 0., 0.]], dtype=np.float32)
    queries = points[[0, 3, 6]]
    local_cfg = dict(cfg, k=3)
    corpus = CudaCorpus(points, local_cfg)
    audits = []
    try:
        for q, row in zip(queries, corpus.batch(queries)):
            expected = reference.exact_geometry_search(points, q)
            if not np.array_equal(row['order'], expected['order']) or not np.array_equal(row['ip_order'], expected['ip_order']):
                raise ArithmeticError('tiny CPU/CUDA full order mismatch')
            audits.append(dict(order_audit=row['order_audit'], max_l2_error=float(np.max(np.abs(row['d2']-expected['d2'])))))
    finally:
        corpus.close()
    return dict(status='passed', deterministic_queries=3, corpus_rows=len(points),
                includes_zero_duplicate_neartie=True, new_mc_samples=0, audits=audits,
                roundoff_certified=False)
