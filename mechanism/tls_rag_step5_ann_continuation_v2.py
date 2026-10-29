"""Approved CPU-only continuation study; historical development data, no certification.

Observed feature extraction and online decisions have no quality-label arguments.
Labels, calibration, future-gain diagnostics and nested evaluation are separate.
"""
from __future__ import annotations

import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

from mechanism import tls_rag_step5_ann_continuation_v1 as v1
from mechanism.tls_rag_step5_ann_enhancement import (
    sha, save, ordered, rerank, metrics, check_resources, paired_ratio,
)
from tri_rag_harness.tri_law import tri_law_probability
from tri_rag_harness.utils import fingerprint

ROOT = Path(__file__).resolve().parents[1]
NEW_PATHS = (
    'mechanism/tls_rag_step5_ann_continuation_v2.py',
    'mechanism/tls_rag_step5_ann_continuation_v2.cpp',
    'configs/tls_rag_step5_ann_continuation_v2.json',
    'tests/test_tls_rag_step5_ann_continuation_v2.py',
    'scripts/run_tls_rag_step5_ann_continuation_v2.py',
)
BIND_PATHS = NEW_PATHS + v1.BIND_PATHS + ('src/tri_rag_harness/tri_law.py',)


def config():
    s = json.loads((ROOT / NEW_PATHS[2]).read_text())
    if (s['active_routes'] != [2] or s['subset_size'] != 100000
            or any(s[k] for k in ('adapter_training', 'scale_1m', 'ssh',
                                  'formal_certification', 'fresh_independence'))):
        raise ValueError('only approved historical CPU100k route2 development')
    return s


def sources():
    return {p: sha(ROOT / p) for p in dict.fromkeys(BIND_PATHS)}


def nested_folds(qids, s):
    qids = list(qids)
    if len(qids) != 768 or len(set(qids)) != 768:
        raise ValueError('exactly768 historical query identities required')
    order = ordered(qids, 'continuation-v2-outer', s['seeds']['outer'])
    out = []
    for f in range(s['outer_folds']):
        outer = order[f * 192:(f + 1) * 192]
        rest = ordered(sorted(set(qids) - set(outer)), 'continuation-v2-inner-' + str(f),
                       s['seeds']['inner'])
        fold = dict(fit=rest[:384], cal=rest[384:480], select=rest[480:], outer=outer)
        if len(set(q for ids in fold.values() for q in ids)) != 768:
            raise ValueError('query-level fold overlap')
        out.append(fold)
    return out


class Inputs:
    def __init__(self, s):
        self.s = s
        self.truth = {}
        descriptors = {}
        for key, needed in [('', ['subset_corpus.npy', 'subset_ids.json',
                                 'subset_original.faiss', 'query_records.jsonl']),
                            ('second_', ['query_records.jsonl'])]:
            root = ROOT / s[key + 'parent_root']
            p = root / 'archive_verified.json'
            if sha(p) != s[key + 'parent_archive_sha256']:
                raise ValueError('parent archive changed')
            entries = json.loads(p.read_text())['files']
            for name in needed:
                p = root / name
                if (p.is_symlink() or p.stat().st_size != entries[name]['bytes']
                        or sha(p) != entries[name]['sha256']):
                    raise ValueError('input changed: ' + str(p))
            descriptors[key] = entries
            with (root / 'query_records.jsonl').open() as stream:
                for line in stream:
                    r = json.loads(line)
                    if r['method'] == 'equivalence':
                        q = r['query_id']
                        if q in self.truth and self.truth[q] != r['truth']:
                            raise ValueError('conflicting historical truth')
                        self.truth[q] = r['truth']
        root = Path(s['opened_input_root'])
        for name, key in [('observed.json', 'observed_sha256'), ('request.json', 'request_sha256')]:
            if sha(root / name) != s[key]:
                raise ValueError('opened query packet changed')
        packet = json.loads((root / 'observed.json').read_text())
        request = json.loads((root / 'request.json').read_text())
        if (packet['request'] != request or packet['opened_roles'] != ['tune']
                or packet['forbidden_roles_opened']):
            raise ValueError('forbidden role boundary')
        self.rows = {r['query_id']: r for group in packet['groups'].values() for r in group}
        qids = {q for group in request['groups'].values() for q in group}
        if set(self.rows) != qids or set(self.truth) != qids:
            raise ValueError('historical query/truth union changed')
        for r in self.rows.values():
            q = np.asarray(r['query'])
            if q.shape != (384,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-12:
                raise ValueError('query normalization changed')
        self.folds = nested_folds(qids, s)
        self.binding = fingerprint(dict(packet=s['observed_sha256'],
            parents=[s['parent_archive_sha256'], s['second_parent_archive_sha256']], folds=self.folds))
        parent = ROOT / s['parent_root']
        self.paths = dict(index=str(parent / 'subset_original.faiss'),
            index_sha=descriptors['']['subset_original.faiss']['sha256'],
            corpus=str(parent / 'subset_corpus.npy'), ids=str(parent / 'subset_ids.json'))
        self.seals = {}

    def fold_ids(self, fold, role):
        if role == 'outer' and len(self.seals) != 4:
            raise ValueError('all four fold selections must be sealed before outer evaluation')
        return list(self.folds[fold][role])

    def register_seal(self, fold, value):
        if (value['fold'] != fold or value['inputs'] != self.binding
                or value['sources'] != sources() or fold in self.seals):
            raise ValueError('fold seal invalid')
        self.seals[fold] = fingerprint(value)


def native_build(headers, omp_include, output, s):
    headers, omp_include, output = map(Path, (headers, omp_include, output))
    if sys.platform != 'darwin' or output.exists():
        raise ValueError('new macOS library path required')
    for p, key in [(headers / 'faiss/impl/HNSW.h', 'HNSW.h'), (omp_include / 'omp.h', 'omp.h')]:
        if sha(p) != s['native_headers'][key]:
            raise ValueError('fixed native header changed')
    cmd = ['clang++', '-std=c++17', '-O3', '-DNDEBUG', '-dynamiclib', '-undefined',
           'dynamic_lookup', '-I' + str(headers), '-I' + str(omp_include),
           str(ROOT / NEW_PATHS[1]), '-o', str(output)]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    return dict(command=cmd, compiler=subprocess.check_output(['clang++', '--version'], text=True),
                library_sha256=sha(output))


class Native(v1.Native):
    def __init__(self, path, s):
        super().__init__(path, s)
        self.advance_v2 = self.lib.continuation_v2_advance
        self.advance_v2.restype = C.c_int
        self.advance_v2.argtypes = [C.c_void_p, C.c_uint64, C.c_int] + [C.c_void_p] * 4 + [C.c_int]
        self.audit_v2 = self.lib.continuation_v2_audit
        self.audit_v2.restype = C.c_int
        self.audit_v2.argtypes = [C.c_void_p] * 5 + [C.c_int]
        self.predict_v2 = self.lib.continuation_v2_predict
        self.predict_v2.restype = C.c_int
        self.predict_v2.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_int] + [C.c_void_p] * 5 + [C.c_int, C.c_double, C.c_double, C.c_void_p]


class BoundarySession(v1.Session):
    def __init__(self, native, index, query, debug=False):
        self.state = None
        super().__init__(native, index, query, debug=debug)

    def advance(self, target=0, boundary=True):
        if self.state is None or not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < 2**63:
            raise ValueError('invalid or closed v2 session')
        if not isinstance(boundary, bool):
            raise ValueError('boundary must be bool')
        a, t, e = self._arrays()
        extra = np.zeros(3)
        rc = self.n.advance_v2(self.state, target, int(boundary), v1.ptr(a), v1.ptr(t),
                               v1.ptr(extra), e, len(e))
        if rc:
            raise RuntimeError(e.value.decode())
        return dict(self._stats(a, t), pending=float(extra[0]), pivot=float(extra[1]),
                    at_neighbor_boundary=bool(extra[2]))

    def audit(self):
        """Offline diagnostic only; never called by Runtime.call or decide."""
        if self.state is None:
            raise ValueError('closed diagnostic session')
        pending = np.full(64, -1, np.int32)
        frontier = np.full(self.n.s['ef_search'], -1, np.int32)
        visited = np.zeros(self.index.ntotal, np.uint8)
        e = C.create_string_buffer(1024)
        rc = self.n.audit_v2(self.state, v1.ptr(pending), v1.ptr(frontier), v1.ptr(visited), e, len(e))
        if rc:
            raise RuntimeError(e.value.decode())
        return pending[pending >= 0], frontier[frontier >= 0], visited


def observed_features(distances, rows, stats, previous, query, corpus, family, s):
    """Only the current scored pool/history; no truth, relevance or future state."""
    if family not in ('S', 'G', 'T', 'T0'):
        raise ValueError('unsupported feature family')
    d = np.asarray(distances, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int64)
    if d.shape != (64,) or rows.shape != (64,) or not np.isfinite(d).all():
        raise ValueError('complete scored pool required')
    start = perf_counter()
    threshold = max(float(d[-1]), 1e-30)
    front = stats['frontier_min']
    old = previous['stats'] if previous else None
    oldrows = previous['rows'] if previous else None
    change10 = 1 if oldrows is None else len(set(rows[:10]) - set(oldrows[:10])) / 10
    change64 = 1 if oldrows is None else len(set(rows) - set(oldrows)) / 64
    base = [np.log1p(stats[k]) for k in ('full_distances', 'upper_distances', 'expanded',
            'neighbor_slots', 'result_updates', 'discovered', 'frontier_size')]
    base += [stats['result_updates'] - (old['result_updates'] if old else 0),
             stats['full_distances'] - (old['full_distances'] if old else 0),
             (front - threshold) / threshold if front is not None else 0,
             float(front is not None), change10, change64, float(previous is not None),
             stats.get('pending', 0), stats.get('at_neighbor_boundary', 0)]
    base += (d[[0, 4, 9, 15, 31, 47, 63]] / threshold).tolist()
    base += [float(d.mean() / threshold), float(d.std() / threshold), float(d[0]), threshold]
    costs = dict(cheap_ms=(perf_counter() - start) * 1000, geometry_ms=0., tri_ms=0.)
    if family == 'S':
        return np.asarray(base), costs
    start = perf_counter()
    ranks = np.asarray(s['pair_ranks'], dtype=int) - 1
    unique = np.unique(ranks)
    vectors = np.asarray(corpus[rows[unique]], dtype=np.float64) - np.asarray(query, dtype=np.float64)
    sq = np.einsum('ij,ij->i', vectors, vectors)
    a, b = np.searchsorted(unique, ranks[:, 0]), np.searchsorted(unique, ranks[:, 1])
    lo, hi = np.minimum(sq[a], sq[b]), np.maximum(sq[a], sq[b])
    valid = (lo > 0) & (hi > lo)
    beta = np.ones(16)
    rho = np.zeros(16)
    beta[valid] = hi[valid] / lo[valid]
    dot = np.einsum('ij,ij->i', vectors[a], vectors[b])
    rho[valid] = np.clip(dot[valid] / np.sqrt(sq[a[valid]] * sq[b[valid]]), -1, 1)
    base += beta.tolist() + rho.tolist() + valid.astype(float).tolist()
    costs['geometry_ms'] = (perf_counter() - start) * 1000
    if family == 'G':
        return np.asarray(base), costs
    start = perf_counter()
    p = np.zeros(16)
    if valid.any():
        p[valid] = tri_law_probability(beta[valid], rho[valid] if family == 'T' else 0., s['tri_dimension'])
    base += p.tolist() + [float(p.mean()), float(p.max()), float(p.std())]
    costs['tri_ms'] = (perf_counter() - start) * 1000
    x = np.asarray(base)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite observed geometry')
    return x, costs


def export_model(model):
    roots, feature, threshold, left, right, value = [], [], [], [], [], []
    for estimator in model.estimators_.ravel():
        t = estimator.tree_
        offset = len(feature)
        roots.append(offset)
        feature.extend(t.feature.tolist())
        threshold.extend(t.threshold.tolist())
        left.extend((np.where(t.children_left < 0, -1, t.children_left + offset)).tolist())
        right.extend((np.where(t.children_right < 0, -1, t.children_right + offset)).tolist())
        value.extend(t.value[:, 0, 0].tolist())
    return dict(roots=roots, feature=feature, threshold=threshold, left=left, right=right,
                value=value, initial=float(model.init_.constant_[0, 0]), rate=float(model.learning_rate),
                n_features=int(model.n_features_in_))


class Predictor:
    def __init__(self, value, native):
        self.n = native
        self.initial, self.rate = value['initial'], value['rate']
        self.nf = value['n_features']
        self.a = {k: np.ascontiguousarray(value[k], dtype=np.int32 if k in ('roots', 'feature', 'left', 'right') else np.float64)
                  for k in ('roots', 'feature', 'threshold', 'left', 'right', 'value')}
        nn = len(self.a['value'])
        if (any(len(self.a[k]) != nn for k in ('feature', 'threshold', 'left', 'right'))
                or not nn or not len(self.a['roots'])
                or np.any((self.a['roots'] < 0) | (self.a['roots'] >= nn))):
            raise ValueError('invalid frozen tree arrays')

    def __call__(self, features):
        x = np.ascontiguousarray(features, dtype=np.float64)
        if x.shape != (self.nf,) or not np.isfinite(x).all():
            raise ValueError('model feature shape/value mismatch')
        a = self.a
        result = np.zeros(1)
        code = self.n.predict_v2(v1.ptr(x), len(x), v1.ptr(a['roots']), len(a['roots']),
            v1.ptr(a['feature']), v1.ptr(a['threshold']), v1.ptr(a['left']), v1.ptr(a['right']),
            v1.ptr(a['value']), len(a['value']), self.initial, self.rate, v1.ptr(result))
        if code:
            raise RuntimeError('native tree prediction failed: ' + str(code))
        return float(result[0])


def fit_models(traces, qids, s, native):
    from sklearn.ensemble import GradientBoostingRegressor
    import sklearn
    if sklearn.__version__ != '1.6.1':
        raise ValueError('fixed sklearn version required')
    out, checks = {}, {}
    for family in ('S', 'G', 'T', 'T0'):
        examples = [(r, tr[-1], len([x for x in tr if x['features']]))
                    for q in qids for tr in [traces[q]] for r in tr if r['features']]
        x = np.asarray([r['features'][family] for r, end, count in examples])
        weights = np.asarray([1 / count for r, end, count in examples])
        out[family], checks[family] = {}, {}
        for name in ('retention', 'hit'):
            y = np.asarray([max(0., end[name] - r[name]) for r, end, count in examples])
            m = GradientBoostingRegressor(**{k: s['model'][k] for k in
                ('n_estimators', 'max_depth', 'learning_rate', 'min_samples_leaf')},
                random_state=s['seeds']['model'])
            m.fit(x, y, sample_weight=weights)
            exported = export_model(m)
            predictor = Predictor(exported, native)
            native_values = np.array([predictor(z) for z in x])
            error = float(np.max(np.abs(native_values - m.predict(x))))
            if error > 1e-12:
                raise RuntimeError('native/Python model export mismatch')
            out[family][name] = exported
            checks[family][name] = dict(max_absolute_error=error, fit_queries=len(qids), fit_states=len(y))
    return out, checks


def predictors(models, native):
    return {family: {name: Predictor(v, native) for name, v in values.items()}
            for family, values in models.items()}


def predictions(traces, qids, predictors_by_family):
    out = {}
    for q in qids:
        out[q] = []
        for r in traces[q]:
            out[q].append({family: {name: max(0., p(r['features'][family])) for name, p in pred.items()}
                           for family, pred in predictors_by_family.items()} if r['features'] else {})
    return out


def calibrate(traces, qids, predictions_by_query, s):
    """Query-max residual on eligible decision checkpoints, not a theorem.

    Early states where no candidate policy may stop are not in this domain.
    Quantile/minimum choices are still checked as complete policies on cal/select.
    """
    out = {}
    for family in ('S', 'G', 'T', 'T0'):
        out[family] = {}
        for name in ('retention', 'hit'):
            residuals = []
            for q in qids:
                end = traces[q][-1]
                residuals.append(max([0.] + [max(0., end[name] - r[name]) - predictions_by_query[q][j][family][name]
                    for j, r in enumerate(traces[q]) if r['features']
                    and r['stats']['full_distances'] >= min(s['minimum_work'])]))
            out[family][name] = {str(t): float(max(0., np.quantile(residuals, t))) for t in s['residual_quantiles']}
    return out


def decide(scores, policy, margins, stats, s):
    """Current scores + frozen calibration only. No label/future parameters."""
    if stats['done'] or stats['full_distances'] < policy['minimum'] or policy.get('disabled'):
        return False
    family = policy['family']
    key = str(policy['quantile'])
    if family == 'Ts':
        cheap = scores['S']['retention'] + margins['S']['retention'][key]
        if cheap < max(0., policy['limit'] - s['selective_width']):
            family = 'S'
        elif cheap > policy['limit'] + s['selective_width']:
            return False
        else:
            family = 'T'
    score = scores[family]
    return (score['retention'] + margins[family]['retention'][key] <= policy['limit']
            and score['hit'] + margins[family]['hit'][key] <= s['hit_risk_limit'])


def replay(stages, scores, policy, margins, s):
    for j, r in enumerate(stages):
        if r['stats']['done'] or r.get('observed', True) and decide(scores[j], policy, margins, r['stats'], s):
            return j, r
    return len(stages) - 1, stages[-1]


def candidate_policies(family, s):
    return [dict(family=family, quantile=q, limit=t, minimum=n)
            for q in s['residual_quantiles'] for t in s['risk_limits'] for n in s['minimum_work']]


class Runtime(v1.Runtime):
    def call(self, query, method, policy=None, models=None, margins=None):
        if method.startswith(('fixed:', 'path:')):
            if int(method.split(':')[1]) not in self.s['checkpoints']:
                raise ValueError('work target outside frozen grid')
        elif method.startswith('policy:'):
            if policy is None or policy['family'] != method.split(':')[1]:
                raise ValueError('missing or mismatched frozen policy')
        elif method.startswith('candidate:'):
            if policy is None:
                raise ValueError('candidate requires explicit frozen policy')
        elif not (method.startswith(('original:', 'paid-noop:')) or method in ('natural', 'checkpoints', 'batch_gap', 'adj_gap')):
            raise ValueError('unknown v2 method')
        if method.startswith('original:'):
            return super().call(query, method)
        q = np.asarray(query, dtype=np.float64)
        boundary = method != 'batch_gap'
        checks, previous_updates, previous = 0, 0., None
        fees = dict(snapshot_ms=0., cheap_ms=0., geometry_ms=0., tri_ms=0., prediction_ms=0., decision_ms=0.)
        tri_calls, geometry_calls = 0, 0
        with BoundarySession(self.n, self.index, q) as session:
            if method == 'natural' or policy is not None and policy.get('disabled'):
                stats = session.advance()
            elif method.startswith('fixed:'):
                stats = session.advance(int(method.split(':')[1]))
            else:
                target = int(method.split(':')[1]) if method.startswith('path:') else 0
                for work in self.s['checkpoints']:
                    if target and work > target:
                        break
                    if previous is not None and previous['stats']['full_distances'] >= work:
                        continue
                    stats = session.advance(work, boundary)
                    checks += 1
                    if stats['done']:
                        break
                    stop = False
                    if method in ('batch_gap', 'adj_gap'):
                        start = perf_counter()
                        stop = v1.gap_policy(stats, previous_updates, 0., 4, 1024)
                        fees['decision_ms'] += (perf_counter() - start) * 1000
                    elif method.startswith(('policy:', 'candidate:', 'paid-noop:')):
                        start = perf_counter()
                        d, rows, _ = session.snapshot()
                        fees['snapshot_ms'] += (perf_counter() - start) * 1000
                        # Preserve observable history, but do not pay a model or
                        # geometric block when this policy cannot stop yet.
                        if policy is not None and stats['full_distances'] < policy['minimum']:
                            previous = dict(rows=rows, stats=stats)
                            previous_updates = stats['result_updates']
                            continue
                        family = policy['family'] if policy is not None else method.split(':')[1]
                        scores = {}
                        actual = ['S'] if family == 'Ts' else [family]
                        for fam in actual:
                            x, cost = observed_features(d, rows, stats, previous, q, self.x, fam, self.s)
                            for k in cost:
                                fees[k] += cost[k]
                            geometry_calls += int(fam != 'S')
                            tri_calls += int(fam in ('T', 'T0'))
                            start = perf_counter()
                            scores[fam] = {name: max(0., pred(x)) for name, pred in models[fam].items()}
                            fees['prediction_ms'] += (perf_counter() - start) * 1000
                        if family == 'Ts':
                            key = str(policy['quantile'])
                            z = scores['S']['retention'] + margins['S']['retention'][key]
                            if max(0., policy['limit'] - self.s['selective_width']) <= z <= policy['limit'] + self.s['selective_width']:
                                x, cost = observed_features(d, rows, stats, previous, q, self.x, 'T', self.s)
                                for k in cost:
                                    fees[k] += cost[k]
                                geometry_calls += 1
                                tri_calls += 1
                                start = perf_counter()
                                scores['T'] = {name: max(0., pred(x)) for name, pred in models['T'].items()}
                                fees['prediction_ms'] += (perf_counter() - start) * 1000
                        if policy is not None:
                            start = perf_counter()
                            stop = decide(scores, policy, margins, stats, self.s)
                            fees['decision_ms'] += (perf_counter() - start) * 1000
                        previous = dict(rows=rows, stats=stats)
                    previous_updates = stats['result_updates']
                    if previous is None:
                        previous = dict(rows=None, stats=stats)
                    else:
                        previous['stats'] = stats
                    if stop:
                        break
                if not stats['done'] and not target and not stop:
                    stats = session.advance()
            _, rows, _ = session.snapshot()
        start = perf_counter()
        pool, ctx, _ = rerank(self.x, self.ids, rows, q, self.s['k_gt'], self.s['k_ctx'])
        stats.update(rerank_ms=(perf_counter() - start) * 1000, checks=checks,
                     geometry_calls=geometry_calls, tri_calls=tri_calls, **fees)
        return dict(candidate_ids=pool, context_ids=ctx, stats=stats)

    def trace(self, query, boundary=True):
        q = np.asarray(query, dtype=np.float64)
        out = []
        with BoundarySession(self.n, self.index, q) as session:
            for target in self.s['checkpoints'] + [0]:
                stats = session.advance(target, boundary)
                d, rows, _ = session.snapshot()
                pool, ctx, _ = rerank(self.x, self.ids, rows, q, self.s['k_gt'], self.s['k_ctx'])
                out.append(dict(target=target, distances=d.tolist(), rows=rows.tolist(),
                                candidate_ids=pool, context_ids=ctx, stats=stats))
        return out


def paired_quality(rows, baseline, s):
    qids = sorted(rows)
    if set(qids) != set(baseline):
        raise ValueError('unpaired quality')
    rng = np.random.default_rng(s['seeds']['bootstrap'])
    ii = rng.integers(len(qids), size=(s['bootstrap_replicates'], len(qids)))
    out = {}
    from scipy.stats import beta as beta_dist
    for name in ('retention', 'hit'):
        delta = np.array([rows[q][name] - baseline[q][name] for q in qids])
        loss = np.maximum(0., -delta)
        events = int(np.sum(loss > 1e-12))
        event_upper = 1. if events == len(qids) else float(beta_dist.ppf(.95, events + 1, len(qids) - events))
        out[name] = dict(mean_delta=float(delta.mean()), interval95=np.quantile(delta[ii].mean(1), [.025, .975]).tolist(),
                         changed=int(np.sum(np.abs(delta) > 1e-12)), adverse_queries=events,
                         worst_query_loss=float(loss.max()), adverse_event_probability_upper95=event_upper,
                         conservative_mean_loss_upper95=event_upper,
                         caveat='historical development; event upper bounds bounded mean loss conservatively, not certification')
    return out
