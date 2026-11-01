"""Approved section38: full-corpus temporary banks and lifecycle kernels.

Truth and selection live outside the query API. Row numbers refer only to the
explicit bank ids.json map. Imports preserve the existing stable-L2 geometry.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time

import faiss
import numpy as np

from mechanism.tls_rag_step5_route4_screen_v1 import (
    ROOT, sha, fingerprint, checked, save, squared_l2, stable_top,
    gaussian_projection, check_versions,
)

NEW_PATHS = (
    'mechanism/tls_rag_step5_route4_scenarios_v1.py',
    'mechanism/tls_rag_step5_route4_advisor_v1.py',
    'configs/tls_rag_step5_route4_scenarios_v1.json',
    'tests/test_tls_rag_step5_route4_scenarios_v1.py',
    'scripts/run_tls_rag_step5_route4_scenarios_v1.py',
)
SOURCE_PATHS = NEW_PATHS + (
    'mechanism/tls_rag_step5_route4_screen_v1.py',
    'src/tri_rag_harness/tri_law.py', 'scripts/run_tests.sh',
)


def config():
    return json.loads((ROOT / NEW_PATHS[2]).read_text())


def sources():
    return {n: sha(ROOT / n) for n in SOURCE_PATHS}


def valid_vectors(x, d):
    if x.ndim != 2 or x.shape[1] != d or not np.isfinite(x).all():
        raise ValueError('vector domain')
    if not np.allclose(np.linalg.norm(x, axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError('original vectors must already be normalized')


def load_observed_inputs(s, role='advice', selection_seal=None):
    if role not in ('advice', 'select'):
        raise ValueError('closed query role')
    if role == 'select':
        if not isinstance(selection_seal,dict) or set(selection_seal)!={'path','sha256'}:
            raise ValueError('select requires a prior decision seal')
        checked(Path(selection_seal['path']),selection_seal['sha256'])
        sealed=json.loads(Path(selection_seal['path']).read_text())
        if sealed.get('select_inspected_for_choice') is not False or 'entries' not in sealed:
            raise ValueError('unsealed selection')
    base = ROOT / s['input_root']
    # No glob/parent-packet read: only four explicitly authorized old inputs.
    for n, digest in s['inputs'].items():
        checked(base / n, digest)
    X = np.load(base / 'corpus64.npy', mmap_mode='r', allow_pickle=False)
    ids = json.loads((base / 'ids.json').read_text())
    if X.dtype != np.float64 or X.shape != (s['parent_n'], s['d']):
        raise ValueError('parent vector schema')
    if len(ids) != len(X) or len(set(ids)) != len(ids) or not all(isinstance(i, str) for i in ids):
        raise ValueError('stable ID map')
    valid_vectors(X, s['d'])
    recs = json.loads((base / (role + '_queries.json')).read_text())
    if len(recs) != s['role_sizes'][role] or len({r['query_id'] for r in recs}) != len(recs):
        raise ValueError('query role identities')
    valid_vectors(np.asarray([r['query'] for r in recs]), s['d'])
    return X, ids, recs


def make_banks(ids, s):
    if len(ids) != len(set(ids)) or not all(isinstance(i, str) for i in ids):
        raise ValueError('bank IDs')
    import hashlib
    order = sorted(range(len(ids)), key=lambda j: (hashlib.sha256(
        (str(s['bank_seed']) + ':' + ids[j]).encode()).digest(), ids[j]))
    banks = {}
    for n, count in s['banks']:
        if n * count > len(ids) or n < s['k']:
            raise ValueError('nonoverlapping bank size')
        for b in range(count):
            banks[f'n{n}-b{b}'] = sorted(order[b*n:(b+1)*n], key=lambda j: ids[j])
    return banks


def base_key(spec):
    return '{}-{}-{}'.format(spec['kind'], spec.get('m', 0), spec.get('bits', 0))


def recipe_key(spec, M):
    return base_key(spec) + '-M' + str(M)


def base_specs(s):
    return ([dict(kind='original64'), dict(kind='original32')]
            + [dict(kind='gaussian', m=m) for m in s['gaussian_dimensions']]
            + [dict(kind='pca', m=m) for m in s['pca_dimensions']]
            + [dict(kind='pq', bits=b) for b in s['pq_bits']])


def budgets(spec, s):
    return [s['k']] if spec['kind'].startswith('original') else s['budgets']


def family_basis(X, family, s):
    if family == 'pca':
        c = X - X.mean(axis=0)
        _, v = np.linalg.eigh(c.T @ c / len(c))
        return v[:, ::-1].T
    if family == 'gaussian':
        return np.random.Generator(np.random.PCG64(s['production_seed'])).standard_normal(
            (max(s['gaussian_dimensions']), X.shape[1]))
    return None


class Model:
    def __init__(self, X, ids, spec):
        self.X, self.ids, self.spec = X, np.asarray(ids), dict(spec)
        self.rows = np.arange(len(X), dtype=np.int64)
        self.P = self.Y = self.index = None
        self.seed = 0

    @property
    def scan_bytes(self):
        if self.index is not None:
            return len(self.X) * self.index.code_size
        if self.Y is not None:
            return self.Y.nbytes
        return self.X.nbytes


def build_method(X, ids, spec, s, seed=None, basis=None, shared32=None):
    """No query, truth, or other bank is accepted by the builder."""
    model = Model(X, ids, spec)
    kind = spec['kind']
    if kind == 'original64':
        return model
    x = np.ascontiguousarray(X, dtype=np.float32) if shared32 is None else shared32
    if kind == 'original32':
        model.Y = x
    elif kind == 'gaussian':
        m = spec['m']
        model.seed = s['production_seed'] if seed is None else seed
        if basis is None:
            P = gaussian_projection(X.shape[1], m, s['production_seed'] if seed is None else seed,
                                    maximum=max(s['gaussian_dimensions']))
        else:
            P = basis[:m] / np.sqrt(m)
        model.P = P.astype(np.float32)
        model.Y = x @ model.P.T
    elif kind == 'pca':
        P = family_basis(X, 'pca', s) if basis is None else basis
        model.P = P[:spec['m']].astype(np.float32)
        model.Y = x @ model.P.T
    elif kind == 'pq':
        model.seed = s['pq_seed'] if seed is None else seed
        model.index = faiss.IndexPQ(X.shape[1], s['pq_subquantizers'], spec['bits'])
        model.index.pq.cp.niter = s['pq_iterations']
        model.index.pq.cp.seed = model.seed
        model.index.pq.cp.max_points_per_centroid = len(X)
        model.index.train(x)
        model.index.add(x)
    else:
        raise ValueError('unknown method')
    return model


def store_model(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    names = []
    for name in ('P', 'Y'):
        array = getattr(model, name)
        if array is not None:
            with (directory / (name + '.npy')).open('xb') as f:
                np.save(f, array, allow_pickle=False)
            names.append(name + '.npy')
    if model.index is not None:
        faiss.write_index(model.index, str(directory / 'pq.faiss'))
        names.append('pq.faiss')
    save(directory / 'model.json', dict(spec=model.spec, seed=model.seed,
         corpus_sha256=hashlib.sha256(np.ascontiguousarray(model.X).view(np.uint8)).hexdigest(),
         normalized_original=True, bank_ids=fingerprint(model.ids.tolist()),
         files={n: sha(directory / n) for n in names},
         original_vectors_shared=True, projected_renormalization=False))


def load_model(X, ids, directory, expected_spec=None, expected_seed=None):
    directory = Path(directory)
    meta = json.loads((directory / 'model.json').read_text())
    if meta['bank_ids'] != fingerprint(list(ids)) or (expected_spec is not None and meta['spec'] != expected_spec):
        raise ValueError('model fingerprint mismatch')
    if meta['corpus_sha256'] != hashlib.sha256(np.ascontiguousarray(X).view(np.uint8)).hexdigest() or not meta['normalized_original']:
        raise ValueError('model corpus mismatch')
    if expected_seed is not None and meta['seed'] != expected_seed:
        raise ValueError('model seed mismatch')
    for n, h in meta['files'].items():
        checked(directory / n, h)
    model = Model(X, ids, meta['spec'])
    model.seed = meta['seed']
    for n in ('P', 'Y'):
        if n + '.npy' in meta['files']:
            setattr(model, n, np.load(directory / (n + '.npy'), mmap_mode='r', allow_pickle=False))
    if 'pq.faiss' in meta['files']:
        model.index = faiss.read_index(str(directory / 'pq.faiss'))
    return model


def candidates(model, q, M):
    """Stable top-M. PQ uses native ADC; only an exact cutoff tie scans all."""
    start = time.perf_counter_ns()
    kind = model.spec['kind']
    q0 = np.asarray(q, dtype=np.float64)
    if q0.shape != (model.X.shape[1],) or not np.isfinite(q0).all():
        raise ValueError('query domain')
    q1 = q0 if kind == 'original64' or (model.P is not None and model.P.dtype==np.float64) else np.asarray(q0, dtype=np.float32)
    if model.P is not None:
        q1 = q1 @ model.P.T
    projected = time.perf_counter_ns()
    M = min(M, len(model.X))
    fallback = False
    if model.index is not None:
        D, I = model.index.search(q1[None], min(M + 1, len(model.X)))
        if M < len(model.X) and D[0, M-1] == D[0, M]:
            D, I = model.index.search(q1[None], len(model.X))
            fallback = True
        E = stable_top(D[0], I[0], M, model.ids)
    else:
        D = squared_l2(q1, model.X if model.Y is None else model.Y)
        E = stable_top(D, model.rows, M, model.ids)
    end = time.perf_counter_ns()
    return E, dict(query_projection_ns=projected-start, screen_ns=end-projected,
                   pq_cutoff_tie_full_scan=fallback)


def query_method(model, q, M, k):
    start = time.perf_counter_ns()
    E, stages = candidates(model, q, M)
    before_rerank = time.perf_counter_ns()
    if model.spec['kind'] == 'original64':
        final = E[:k]
    else:
        d = squared_l2(np.asarray(q, dtype=np.float64), model.X[E])
        final = stable_top(d, E, k, model.ids)
    reranked = time.perf_counter_ns()
    result = dict(candidate_rows=E.tolist(), final_ids=model.ids[final].tolist(),
                  **stages,rerank_ns=reranked-before_rerank,
                  logical_scan_bytes=model.scan_bytes,
                  logical_rerank_bytes=(0 if model.spec['kind']=='original64' else len(E)*model.X.shape[1]*8))
    end = time.perf_counter_ns()
    result['full_ns'] = end-start
    result['return_overhead_ns'] = (end-start)-sum(result[n] for n in ('query_projection_ns','screen_ns','rerank_ns'))
    return result


def truth(X, ids, recs, k):
    model = Model(X, ids, dict(kind='original64'))
    return {r['query_id']: query_method(model, np.asarray(r['query']), k, k)['final_ids'] for r in recs}


def quality(records, G, k):
    losses = [len(set(G[r['query_id']])-set(r['final_ids'])) for r in records]
    return dict(query_ids=[r['query_id'] for r in records], missed=losses,
                mean_loss=float(np.mean(losses)/k), worst_loss=float(max(losses, default=0)/k))


def eligible(value, s):
    return value['mean_loss'] <= s['maximum_mean_loss'] + 1e-12 and value['worst_loss'] <= s['maximum_query_loss'] + 1e-12


def measure_queries(model, recs, M, s):
    order = np.random.Generator(np.random.PCG64(s['schedule_seed'])).permutation(len(recs))
    records, samples = {}, {r['query_id']: [] for r in recs}
    for repeat in range(s['timing_repeats']):
        for j in order:
            r = recs[j]
            outer = time.perf_counter_ns()
            out = query_method(model, np.asarray(r['query']), M, s['k'])
            out['full_ns'] = time.perf_counter_ns()-outer
            out['return_overhead_ns'] = out['full_ns']-sum(out[n] for n in ('query_projection_ns','screen_ns','rerank_ns'))
            qid = r['query_id']
            compact = dict(query_id=qid, candidate_rows=out['candidate_rows'], final_ids=out['final_ids'])
            if qid in records and records[qid] != compact:
                raise ValueError('nonreproducible query output')
            records[qid] = compact
            samples[qid].append({k:v for k,v in out.items() if k not in ('candidate_rows','final_ids')})
    a = np.asarray([v['full_ns'] for values in samples.values() for v in values])
    return dict(records=[records[r['query_id']] for r in recs], samples=samples,
                mean_ns=float(a.mean()), p95_ns=float(np.quantile(a,.95)))


def probe_model(model, recs, allowed, s):
    """Offline trial: one distance/ADC pass and one max-M selection per query.

    Original scores on the max-M set are also paid once and reused by prefixes.
    Never used by the standalone fixed-M serving benchmark.
    """
    result = {str(M): [] for M in allowed}
    for r in recs:
        q = np.asarray(r['query'])
        E, _ = candidates(model, q, max(allowed))
        d = squared_l2(q, model.X[E])
        for M in allowed:
            take = min(M, len(E))
            F = stable_top(d[:take], E[:take], s['k'], model.ids)
            result[str(M)].append(dict(query_id=r['query_id'], candidate_rows=E[:take].tolist(), final_ids=model.ids[F].tolist()))
    return result


def lifecycle_seconds(build, cold_ns, hot_ns, Q):
    if min(build, cold_ns, hot_ns) < 0 or Q < 1:
        raise ValueError('lifecycle domain')
    return float(build + cold_ns/1e9 + (Q-1)*hot_ns/1e9)


def precision_probe(model, q, M, s):
    if model.P is None:
        return None
    p = (gaussian_projection(model.X.shape[1],model.spec['m'],model.seed,maximum=max(s['gaussian_dimensions']))
         if model.spec['kind']=='gaussian' else model.P.astype(np.float64))
    d = squared_l2(np.asarray(q) @ p.T, np.asarray(model.X, dtype=np.float64) @ p.T)
    ref = stable_top(d, model.rows, M, model.ids)
    observed = query_method(model, q, M, s['k'])
    return dict(candidate_membership_difference=len(set(ref)^set(observed['candidate_rows'])),
                reference_final_ids=model.ids[stable_top(squared_l2(q, model.X[ref]),ref,s['k'],model.ids)].tolist(),
                observed_final_ids=observed['final_ids'],
                caveat=('ideal Gaussian FP64 versus deployed FP32' if model.spec['kind']=='gaussian' else
                        'same rounded PCA matrix, accumulated in FP64; diagnostic only'))
