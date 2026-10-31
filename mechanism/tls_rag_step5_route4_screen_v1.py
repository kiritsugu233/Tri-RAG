"""Independent-pool Gaussian screen: approved route4 P1, no learned controller.

All truths/risk live outside Runtime. Kernels and candidate selection are shared
by quality replay and full-call measurements; no distance/result cache in timing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import faiss
import numpy as np
import scipy
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
NEW_PATHS = (
    'mechanism/tls_rag_step5_route4_screen_v1.py',
    'mechanism/tls_rag_step5_route4_risk_v1.py',
    'configs/tls_rag_step5_route4_screen_v1.json',
    'tests/test_tls_rag_step5_route4_screen_v1.py',
    'scripts/run_tls_rag_step5_route4_screen_v1.py',
)
SOURCE_PATHS = NEW_PATHS + ('src/tri_rag_harness/tri_law.py', 'scripts/run_tests.sh')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sources():
    return {p: sha(ROOT / p) for p in SOURCE_PATHS}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def config():
    return json.loads((ROOT / NEW_PATHS[2]).read_text())


def check_versions(s):
    actual = dict(numpy=np.__version__, scipy=scipy.__version__, faiss=faiss.__version__)
    if actual != s['versions']:
        raise ValueError(('fixed dependency mismatch', actual))
    faiss.omp_set_num_threads(1)
    return actual


def hashed_ids(ids, seed, n):
    if len(ids) != len(set(ids)) or not all(isinstance(x, str) for x in ids) or len(ids) < n:
        raise ValueError('duplicate/nonstring/insufficient IDs')
    return sorted(ids, key=lambda x: (hashlib.sha256((str(seed) + ':' + x).encode('utf-8')).digest(), x))[:n]


def checked(path, expected):
    if Path(path).is_symlink() or sha(path) != expected:
        raise ValueError('artifact identity mismatch: ' + str(path))


def load_verified_inputs(s):
    for name in ('corpus', 'ids', 'observed', 'request'):
        checked(ROOT / s[name], s[name + '_sha256'])
    ids = json.loads((ROOT / s['ids']).read_text())
    X = np.load(ROOT / s['corpus'], mmap_mode='r', allow_pickle=False)
    if X.shape != (100000, s['d']) or X.dtype != np.float64 or len(ids) != len(X):
        raise ValueError('parent corpus schema')
    rows = {x: j for j, x in enumerate(ids)}
    chosen = sorted(hashed_ids(ids, s['subset_seed'], s['n']))
    x = np.asarray(X[[rows[i] for i in chosen]])
    packet = json.loads((ROOT / s['observed']).read_text())
    request = json.loads((ROOT / s['request']).read_text())
    if packet['request'] != request or packet['opened_roles'] != ['tune'] or packet['forbidden_roles_opened']:
        raise ValueError('historical role boundary')
    records = [r for g in packet['groups'].values() for r in g]
    queries = {r['query_id']: r['query'] for r in records}
    expected = [q for g in request['groups'].values() for q in g]
    if len(records) != 768 or len(queries) != 768 or set(queries) != set(expected):
        raise ValueError('historical query identity')
    qids = hashed_ids(list(queries), s['query_seed'], sum(s['role_sizes'].values()))
    # MS MARCO query and passage IDs have separate namespaces. Numeric-string
    # collisions do not identify the same item. External-query provenance comes
    # from the checksum-bound historical query export, not untyped ID equality.
    roles, offset = {}, 0
    for role in ('advice', 'select', 'report'):
        take = qids[offset:offset + s['role_sizes'][role]]
        roles[role] = [dict(query_id=i, query=queries[i]) for i in take]
        offset += len(take)
    for values in [x] + [np.asarray([r['query'] for r in v]) for v in roles.values()]:
        if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, rtol=0, atol=1e-12):
            raise ValueError('normalization/nonfinite inputs')
    return x, chosen, roles


def gaussian_projection(d, m, seed, maximum=192):
    if not 1 <= m <= maximum:
        raise ValueError('projection dimension')
    return np.random.Generator(np.random.PCG64(seed)).standard_normal((maximum, d))[:m] / np.sqrt(m)


def stable_top(distances, rows, M, ids):
    d = np.asarray(distances)
    rows = np.asarray(rows, dtype=np.int64)
    # Uniqueness is established by input/list validation and subset selection;
    # do not rebuild a Python set on every measured distance scan.
    if d.shape != rows.shape or M < 1 or not np.isfinite(d).all():
        raise ValueError('selection domain')
    if not len(rows):
        return rows.copy()
    M = min(M, len(rows))
    threshold = np.partition(d, M - 1)[M - 1]
    take = np.flatnonzero(d <= threshold)
    order = np.lexsort((np.asarray(ids)[rows[take]], d[take]))
    return rows[take[order[:M]]]


def squared_l2(q, X):
    if X.dtype == np.float32:
        return faiss.pairwise_distances(np.ascontiguousarray(q[None], dtype=np.float32),
                                       np.ascontiguousarray(X, dtype=np.float32))[0]
    return cdist(np.asarray(q, dtype=np.float64)[None], np.asarray(X, dtype=np.float64), 'sqeuclidean')[0]


def candidate_pool(q, coarse, inverted_ids):
    """Only centroids and inverted IDs; no original candidate vectors or Pi."""
    distances = squared_l2(q, coarse)
    selected = np.lexsort((np.arange(len(coarse)), distances))[:inverted_ids['nprobe']]
    C = np.sort(np.concatenate([inverted_ids['lists'][int(i)] for i in selected]))
    return C, selected.astype(np.int64)[None], distances[selected].astype(np.float32)[None]


def build_independent_pool(X, s):
    """Corpus-only, executed before any Gaussian matrix exists."""
    x = np.ascontiguousarray(X, dtype=np.float32)
    km = faiss.Kmeans(x.shape[1], s['nlist'], niter=s['kmeans_iterations'], nredo=1,
                      seed=s['coarse_seed'], max_points_per_centroid=len(x))
    km.train(x)
    centroids = km.centroids.copy()
    quantizer = faiss.IndexFlatL2(x.shape[1])
    quantizer.add(centroids)
    assignment = quantizer.search(x, 1)[1][:, 0]
    lists = [np.flatnonzero(assignment == i).astype(np.int64) for i in range(s['nlist'])]
    pq = faiss.IndexIVFPQ(quantizer, x.shape[1], s['nlist'], s['pq_subquantizers'], s['pq_bits'])
    pq.nprobe = s['nprobe']
    pq.max_codes = 0
    pq.pq.cp.niter = s['kmeans_iterations']
    pq.pq.cp.seed = s['pq_seed']
    pq.pq.cp.max_points_per_centroid = len(x)
    pq.train(x)
    pq.add(x)
    for i, expected in enumerate(lists):
        actual = faiss.rev_swig_ptr(pq.invlists.get_ids(i), pq.invlists.list_size(i))
        if not np.array_equal(np.sort(actual), expected):
            raise ValueError('PQ changed independent pool assignment')
    return centroids, dict(nprobe=s['nprobe'], lists=lists), pq


def screen_exact(q_proj, projected, C, M, ids):
    if M >= len(C):
        return np.asarray(C).copy()
    return stable_top(squared_l2(q_proj, projected[C]), C, M, ids)


def rerank_original(q, X, E, k, ids):
    if not len(E):
        return np.asarray([], dtype=np.int64)
    return stable_top(squared_l2(q, X[E]), E, k, ids)


def loss_ledger(G, C, E, final):
    g, c, e, f = map(set, (G, C, E, final))
    if not e <= c or not f <= e or not g or g & e != g & f:
        raise ValueError('candidate inclusion or exact rerank contract')
    upstream, filtered = len(g - c), len((g & c) - e)
    if upstream + filtered != len(g - f):
        raise ValueError('loss decomposition')
    return dict(upstream_missed=upstream, filtering_missed=filtered,
                total_missed=upstream + filtered, retention=len(g & f) / len(g))


def spec_key(spec):
    return '{}-{}-{}'.format(spec['kind'], spec.get('m', 0), spec.get('M', 0))


def specifications(s):
    items = [dict(kind='original64', M=s['k'])]
    items += [dict(kind='original32', M=M) for M in s['budgets']]
    items += [dict(kind='pq', M=M) for M in s['budgets']]
    for kind in ('gaussian64', 'gaussian32', 'pca64', 'pca32'):
        items += [dict(kind=kind, m=m, M=M) for m in s['dimensions'] for M in s['budgets']]
    return items


def projection_name(kind, seed, m):
    return '{}-{}-{}.npy'.format(kind, seed, m)


class Runtime:
    """Truth-free query kernel. All setup/load costs reported outside call time."""
    def __init__(self, root, spec, seed, s, verify=True):
        self.root, self.spec, self.seed, self.s = Path(root), dict(spec), seed, s
        if spec not in specifications(s):
            raise ValueError('unfrozen method/budget/dimension')
        if spec['kind'].startswith('gaussian') and seed not in s['design_seeds'] + s['report_seeds']:
            raise ValueError('unfrozen Gaussian seed')
        manifest = json.loads((self.root / 'manifest.json').read_text())
        if manifest['sources'] != sources() or manifest['config'] != s:
            raise ValueError('source/config fingerprint mismatch')
        self.manifest = manifest
        self.loaded = []

        def load(name, numpy=True):
            p = self.root / name
            if name not in manifest['files']:
                raise ValueError('unbound runtime input')
            if verify:
                checked(p, manifest['files'][name]['sha256'])
            self.loaded.append(name)
            return np.load(p, mmap_mode='r', allow_pickle=False) if numpy else p

        self.X = load('corpus64.npy')
        self.ids = np.asarray(json.loads(load('ids.json', False).read_text()))
        self.coarse = load('coarse.npy')
        a = np.load(load('lists.npz', False), allow_pickle=False)
        self.inverted = dict(nprobe=s['nprobe'], lists=[a[str(i)] for i in range(s['nlist'])])
        self.projected, self.Pi, self.pq = None, None, None
        kind = spec['kind']
        if kind == 'original32':
            self.projected = load('corpus32.npy')
        elif kind.startswith(('gaussian', 'pca')):
            bank = seed if kind.startswith('gaussian') else 0
            self.projected = load(projection_name(kind, bank, spec['m']))
            self.Pi = load(projection_name('matrix-' + kind, bank, spec['m']))
        elif kind == 'pq':
            self.pq = faiss.read_index(str(load('pq.faiss', False)))
            self.pq.nprobe = s['nprobe']
            self.pq.max_codes = 0
        elif kind != 'original64':
            raise ValueError('unknown method')

    def candidates(self, q):
        return candidate_pool(q, self.coarse, self.inverted)

    def __call__(self, q):
        s, spec = self.s, self.spec
        start = time.perf_counter_ns()
        C, I, D = self.candidates(q)
        routed = time.perf_counter_ns()
        kind, M = spec['kind'], spec['M']
        query_projection_ns = 0
        if kind == 'original64':
            E = C
        elif kind == 'original32':
            t = time.perf_counter_ns()
            low_q = np.asarray(q, dtype=np.float32)
            query_projection_ns = time.perf_counter_ns() - t
            E = screen_exact(low_q, self.projected, C, M, self.ids)
        elif kind == 'pq':
            if M >= len(C):
                E = C.copy()
            else:
                # M+1 detects boundary ties; a full tied scan is paid when needed.
                scores, rows = self.pq.search_preassigned(np.asarray(q[None], dtype=np.float32), M + 1, I, D)
                if scores[0, M - 1] == scores[0, M]:
                    scores, rows = self.pq.search_preassigned(np.asarray(q[None], dtype=np.float32), len(C), I, D)
                E = stable_top(scores[0], rows[0], M, self.ids)
        else:
            t = time.perf_counter_ns()
            low_q = np.asarray(q, dtype=self.Pi.dtype) @ self.Pi.T
            query_projection_ns = time.perf_counter_ns() - t
            E = screen_exact(low_q, self.projected, C, M, self.ids)
        screened = time.perf_counter_ns()
        final = rerank_original(q, self.X, E, s['k'], self.ids)
        ended = time.perf_counter_ns()
        return dict(C=C, E=E, final=final, full_ns=ended - start,
                    route_ns=routed - start, query_projection_ns=query_projection_ns,
                    screen_ns=screened - routed - query_projection_ns,
                    rerank_ns=ended - screened,
                    logical_original_bytes=(len(C) * self.X.shape[1] * 4 if kind == 'original32' else 0)
                    + len(E) * self.X.shape[1] * 8,
                    logical_screen_bytes=0 if kind in ('original64', 'original32') else len(C) *
                    (self.s['pq_subquantizers'] if kind == 'pq' else self.projected.shape[1] * self.projected.dtype.itemsize))


def measure_full_call(runtime, queries, repeats, seed):
    samples, results = [], {}
    rng = np.random.Generator(np.random.PCG64(seed))
    for repeat in range(repeats):
        for j in rng.permutation(len(queries)):
            rec = queries[int(j)]
            q = np.asarray(rec['query'], dtype=np.float64)
            outer_start = time.perf_counter_ns()
            out = runtime(q)
            outer_end = time.perf_counter_ns()
            internal = out['full_ns']
            out['full_ns'] = outer_end - outer_start
            out['return_overhead_ns'] = out['full_ns'] - internal
            qid = rec['query_id']
            results[qid] = dict(E=out['E'].tolist(), final=out['final'].tolist())
            samples.append(dict(query_id=qid, repeat=repeat, **{k: int(v) for k, v in out.items()
                                                               if k not in ('C', 'E', 'final')}))
    return dict(samples=samples, results=results, loaded_files=runtime.loaded)


def open_role(root, role, manifest):
    if role not in ('advice', 'select', 'report'):
        raise ValueError('unapproved query role')
    root = Path(root)
    if role == 'report':
        seal = json.loads((root / 'selection_seal.json').read_text())
        selected = json.loads((root / 'selection.json').read_text())
        if seal != dict(selection=fingerprint(selected), inputs=fingerprint(manifest), sources=sources()):
            raise ValueError('selection seal changed')
        if selected['status'] != 'selected':
            raise ValueError('no candidate: report closed')
    name = role + '_queries.json'
    checked(root / name, manifest['files'][name]['sha256'])
    return json.loads((root / name).read_text())
