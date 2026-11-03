"""R6-B0 real displacement geometry. No independent-error or tail approximation."""
import hashlib
import time

import numpy as np


def stable_order(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError('finite distance vector required')
    return np.lexsort((np.arange(len(values)), values))


def exact_geometry_search(corpus, query, block=65536, norm_tolerance=2.**-16,
                          check=lambda: None):
    """Full stored-vector FP64 L2 and IP, with row-ID ties and bounded blocks.

    IP here is an independent FP64 diagnostic, not the official FP32 backend.
    The source vectors are neither normalized nor otherwise rewritten.
    """
    q = np.asarray(query, dtype=np.float64)
    if q.ndim != 1 or corpus.shape[1:] != q.shape or not np.all(np.isfinite(q)):
        raise ValueError('query/corpus shape or finite values')
    if block < 1 or abs(np.linalg.norm(q)-1) > norm_tolerance:
        raise ValueError('block or stored query norm')
    n = len(corpus)
    d2, ip = np.empty(n), np.empty(n)
    max_norm_error, read_seconds, math_seconds = 0., 0., 0.
    for start in range(0, n, block):
        check()
        stamp = time.perf_counter()
        x = np.array(corpus[start:start+block], dtype=np.float64, copy=True)
        read_seconds += time.perf_counter()-stamp
        stamp = time.perf_counter()
        if not np.all(np.isfinite(x)):
            raise ValueError('nonfinite corpus')
        error = float(np.max(np.abs(np.linalg.norm(x, axis=1)-1)))
        max_norm_error = max(max_norm_error, error)
        if error > norm_tolerance:
            raise ValueError('stored corpus norm outside frozen gate')
        ip[start:start+len(x)] = -(x@q)
        x -= q
        d2[start:start+len(x)] = np.einsum('ij,ij->i', x, x)
        math_seconds += time.perf_counter()-stamp
    check()
    stamp = time.perf_counter()
    order, ip_order = stable_order(d2), stable_order(ip)
    return dict(d2=d2, native_ip=ip, order=order, ip_order=ip_order,
                zero_rows=np.flatnonzero(d2 == 0), max_norm_error=max_norm_error,
                costs=dict(read_seconds=read_seconds, distance_seconds=math_seconds,
                           sort_seconds=time.perf_counter()-stamp))


def query_seed(seed, query_id):
    return int.from_bytes(hashlib.sha256((str(seed)+'|'+query_id).encode()).digest()[:8], 'big')


def sample_geometry(order, query_id, cfg):
    """Sample the entire complement, including gaps between dense boundary bands."""
    order = np.asarray(order)
    n, k = len(order), cfg['k']
    if (order.ndim != 1 or order.dtype.kind not in 'iu' or not n
            or np.min(order) != 0 or np.max(order) != n-1
            or not np.all(np.bincount(order, minlength=n) == 1)):
        raise ValueError('complete corpus permutation required')
    b = set(range(k+1, k+cfg['near_boundary']+1))
    for factor in cfg['budget_factors']:
        m = k*factor
        b.update(range(m-cfg['half_width'], m+cfg['half_width']+1))
    if not b or min(b) <= k or max(b) >= n:
        raise ValueError('boundary must lie beyond GT and inside complete corpus')
    cutoff = max(b)
    endpoints = [cutoff]+[int(n*f) for f in cfg['far_fractions']]+[n]
    if any(right <= left for left, right in zip(endpoints, endpoints[1:])):
        raise ValueError('nonincreasing complement strata')
    pools = [np.array(sorted(set(range(k+1, cutoff+1))-b), dtype=np.int64)]
    pools += [np.arange(left+1, right+1) for left, right in zip(endpoints, endpoints[1:])]
    if len(pools) != len(cfg['far_samples']):
        raise ValueError('stratum count')
    rng = np.random.default_rng(query_seed(cfg['sample_seed'], query_id))
    ranks = {i: 'G' for i in range(1, k+1)}
    ranks.update({i: 'B' for i in b})
    strata = []
    for h, (pool, count) in enumerate(zip(pools, cfg['far_samples'])):
        if count < 2 or len(pool) < count:
            raise ValueError('insufficient stratum population')
        label = 'F%d' % h
        chosen = rng.choice(pool, size=count, replace=False)
        ranks.update({int(i): label for i in chosen})
        strata.append(dict(label=label, population=len(pool), sample=count,
                           inclusion=count/len(pool), rank_min=int(pool[0]), rank_max=int(pool[-1])))
    positions = sorted(ranks)
    if len(positions) > cfg['max_points'] or sum(r['population'] for r in strata)+k+len(b) != n:
        raise ValueError('point cap or population coverage')
    groups = [ranks[i] for i in positions]
    pi = pair_inclusion(groups, strata)
    return dict(rows=order[np.array(positions)-1], ranks=positions, groups=groups,
                strata=strata, pair_inclusion=pi, population=n, gt=k, boundary=len(b))


def pair_inclusion(groups, strata):
    lookup = {r['label']: r for r in strata}
    single = np.array([lookup[g]['inclusion'] if g in lookup else 1. for g in groups])
    pair = single[:, None]*single[None, :]
    for label, row in lookup.items():
        ix = np.flatnonzero(np.array(groups) == label)
        pair[np.ix_(ix, ix)] = row['sample']*(row['sample']-1)/(row['population']*(row['population']-1))
    np.fill_diagonal(pair, single)
    return pair


def query_direction_decomposition(points, query):
    x, q = np.asarray(points, dtype=float), np.asarray(query, dtype=float)
    if x.ndim != 2 or q.shape != x.shape[1:] or not np.all(np.isfinite(x)) or not np.all(np.isfinite(q)):
        raise ValueError('finite geometry required')
    if np.linalg.norm(q) == 0:
        raise ValueError('zero query direction')
    u = x-q
    a = u@(q/np.linalg.norm(q))
    residual = u-a[:, None]*(q/np.linalg.norm(q))
    gram, h = u@u.T, residual@residual.T
    return dict(displacements=u, gram=gram, d2=np.einsum('ij,ij->i', u, u),
                loading=a, residual_gram=h,
                decomposition_error=float(np.max(np.abs(gram-np.outer(a, a)-h))),
                orthogonality_error=float(np.max(np.abs(residual@q))))


def correlation(gram):
    d = np.diag(gram)
    scale = np.sqrt(d[:, None]*d[None, :])
    return np.divide(gram, scale, out=np.full_like(gram, np.nan), where=scale > 0)


def distribution(values, weights=None):
    v = np.asarray(values)
    w = np.ones(v.shape) if weights is None else np.asarray(weights)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[valid], w[valid]
    if not len(v):
        return dict(count=0, weighted_mean=None, quantiles=None)
    ix = np.argsort(v, kind='stable')
    v, w = v[ix], w[ix]
    cdf = np.cumsum(w)/np.sum(w)
    levels = [.05, .25, .5, .75, .95]
    return dict(count=len(v), weighted_mean=float(np.dot(v, w)/np.sum(w)),
                quantile_levels=levels,
                quantiles=[float(v[min(np.searchsorted(cdf, p), len(v)-1)]) for p in levels],
                minimum=float(v[0]), maximum=float(v[-1]))


def describe_geometry(geometry, sample):
    g, d, h = geometry['gram'], geometry['d2'], geometry['residual_gram']
    rho, residual_rho = correlation(g), correlation(h)
    i, j = np.triu_indices(len(d), 1)
    base_groups = np.array(['F' if s.startswith('F') else s for s in sample['groups']])
    keys = np.array(['-'.join(sorted((a, b), key=('G', 'B', 'F').index))
                     for a, b in zip(base_groups[i], base_groups[j])])
    weights = 1/sample['pair_inclusion'][i, j]
    margins = np.divide(d[j]-d[i], d[j]+d[i], out=np.full(len(i), np.nan), where=d[j]+d[i] > 0)
    blocks = {}
    for key in ('G-G', 'G-B', 'G-F', 'B-B', 'B-F', 'F-F'):
        mask = keys == key
        blocks[key] = {name: distribution(values[mask], weights[mask]) for name, values in
                       [('rho', rho[i, j]), ('error_correlation', rho[i, j]**2),
                        ('residual_rho', residual_rho[i, j]), ('margin', margins)]}
        finite = mask & np.isfinite(rho[i, j]) & np.isfinite(margins)
        hist, _, _ = np.histogram2d(np.clip(rho[i[finite], j[finite]], -1, 1), margins[finite],
                                   bins=(40, 40), range=((-1, 1), (0, 1)), weights=weights[finite])
        blocks[key]['rho_margin_histogram'] = hist.tolist()
    valid = d > 0
    c = rho[np.ix_(valid, valid)]
    eigenvalues, eigenvectors = np.linalg.eigh(c) if len(c) else (np.array([]), np.zeros((0, 0)))
    models = []
    dv = d[valid]
    gv = g[np.ix_(valid, valid)]
    near = np.array(sample['groups'])[valid]
    boundary = (near[:, None] == 'G') & (near[None, :] == 'B')
    actual_variance = dv[:, None]**2+dv[None, :]**2-2*gv**2
    for rank in (1, 2, 4, 8, 16):
        r = min(rank, len(c))
        if not r:
            continue
        low = (eigenvectors[:, -r:]*eigenvalues[-r:])@eigenvectors[:, -r:].T
        diagonal = np.diag(c-low)
        model = low+np.diag(diagonal)
        approx = model*np.sqrt(dv[:, None]*dv[None, :])
        approx_variance = dv[:, None]**2+dv[None, :]**2-2*approx**2
        mask = boundary & (actual_variance > 0)
        relative = np.abs(approx_variance[mask]-actual_variance[mask])/actual_variance[mask]
        models.append(dict(rank=rank, actual_rank=r, local_only=True,
                           min_residual_diagonal=float(np.min(diagonal)),
                           off_diagonal_frobenius_error=float(np.linalg.norm(c-model)),
                           boundary_variance_relative_error=distribution(relative)))
    return dict(blocks=blocks, local_correlation_eigenvalues=eigenvalues.tolist(),
                local_factor_diagnostics=models, zero_displacements=int(np.sum(~valid)),
                zero_residuals=int(np.sum(np.diag(h) == 0)),
                difference_variance_coefficient=dict(
                    minimum=float(np.min(d[:, None]**2+d[None, :]**2-2*g**2)),
                    negative_off_diagonal_count=int(np.sum((d[i]**2+d[j]**2-2*g[i, j]**2) < 0)),
                    zero_off_diagonal_count=int(np.sum((d[i]**2+d[j]**2-2*g[i, j]**2) == 0)),
                    nonpositive_pairs_excluded_only_from_relative_error=True),
                gram_min_eigenvalue=float(np.linalg.eigvalsh(g)[0]),
                residual_min_eigenvalue=float(np.linalg.eigvalsh(h)[0]),
                decomposition_error=geometry['decomposition_error'],
                orthogonality_error=geometry['orthogonality_error'],
                unit_norm_loading_identity_error=float(np.max(np.abs(geometry['loading']+d/2))),
                roundoff_certified=False,
                interpretation='descriptive, boundary-enriched local geometry; no macro-tail or full-corpus spectrum claim')
