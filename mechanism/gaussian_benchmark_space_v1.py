"""Fixed Gaussian representation diagnostics and paid FAISS search, v1."""
import time
import numpy as np
import faiss
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from mechanism.gaussian_benchmark_data_v1 import native_distances, native_recall, strict_retention, valid_ids


def project(vectors, matrix):
    return np.ascontiguousarray(np.asarray(vectors, dtype=np.float32) @ matrix)


def projection_matrix(d, m, seed):
    if not 0 < m <= d:
        raise ValueError('invalid projection dimension')
    return np.random.default_rng(seed).normal(0, 1 / np.sqrt(m), (d, m)).astype(np.float32)


def top_ids(distances, ids, count):
    """Partition then explicitly resolve threshold ties by corpus row ID."""
    if count > len(ids) or count < 1:
        raise ValueError('invalid top count')
    threshold = np.partition(distances, count - 1)[count - 1]
    below = np.flatnonzero(distances < threshold)
    equal = np.flatnonzero(distances == threshold)
    equal = equal[np.argsort(ids[equal], kind='stable')[:count-len(below)]]
    keep = np.concatenate((below, equal))
    return keep[np.lexsort((ids[keep], distances[keep]))]


def exact_candidates(vectors, queries, count, metric='l2', block_rows=32768, check=lambda: None):
    """Bounded Q-batch x block exact search; caller controls Q batch size."""
    if metric not in ('l2', 'native') or count > len(vectors) or count < 1:
        raise ValueError('invalid exact search')
    queries = np.asarray(queries, np.float32)
    best_ids = np.empty((len(queries), 0), np.int64)
    best_d = np.empty((len(queries), 0), np.float32)
    qnorm = np.sum(queries * queries, axis=1)
    for start in range(0, len(vectors), block_rows):
        check()
        x = np.asarray(vectors[start:start+block_rows], np.float32)
        if not np.all(np.isfinite(x)):
            raise ValueError('nonfinite corpus')
        dot = queries @ x.T
        distances = np.clip(1-dot, 0, 2) if metric == 'native' else np.maximum(qnorm[:, None] + np.sum(x*x, axis=1)[None, :] - 2*dot, 0)
        ids = np.arange(start, start+len(x), dtype=np.int64)
        take = min(count, best_ids.shape[1]+len(x))
        next_ids = np.empty((len(queries), take), np.int64)
        next_d = np.empty((len(queries), take), np.float32)
        for j in range(len(queries)):
            merged_i = np.concatenate((best_ids[j], ids))
            merged_d = np.concatenate((best_d[j], distances[j]))
            selected = top_ids(merged_d, merged_i, take)
            next_ids[j], next_d[j] = merged_i[selected], merged_d[selected]
        best_ids, best_d = next_ids, next_d
    return best_ids, best_d


def rerank(query, corpus, candidates, k):
    ids = valid_ids(candidates, len(corpus), k)
    distances = native_distances(query, corpus[ids])
    order = top_ids(distances, ids, k)
    return ids[order], distances[order]


def decompose_loss(truth, exact, actual, k):
    g, e, a = set(map(int, truth[:k])), set(map(int, exact)), set(map(int, actual))
    if len(g) != k:
        raise ValueError('invalid truth')
    return {'representation_loss': 1-len(g&e)/k, 'ann_removed': len(g&(e-a))/k,
            'ann_recovered': len(g&(a-e))/k, 'actual_loss': 1-len(g&a)/k}


def oracle_frontier(quality, cost, target):
    """LP randomized oracle lower bound with independently evaluated dual bound."""
    q, c = np.asarray(quality, float), np.asarray(cost, float)
    if q.ndim != 2 or q.shape != c.shape or not q.size or not np.all(np.isfinite(q)) or not np.all(np.isfinite(c)) or np.any(c < 0) or np.any((q < 0)|(q > 1)) or not 0 <= target <= 1:
        raise ValueError('invalid oracle matrix')
    n, a = q.shape
    maximum = float(q.max(axis=1).mean())
    if maximum < target:
        return {'status': 'quality_infeasible', 'maximum_mean_quality': maximum}
    rows = np.repeat(np.arange(n), a)
    eq = csr_matrix((np.ones(n*a), (rows, np.arange(n*a))), shape=(n, n*a))
    result = linprog(c.ravel()/n, A_ub=csr_matrix(-q.reshape(1, -1)/n), b_ub=[-target], A_eq=eq, b_eq=np.ones(n), bounds=(0, 1), method='highs')
    if not result.success:
        raise RuntimeError('oracle LP did not solve: ' + result.message)
    weights = result.x.reshape(n, a)
    lam = max(0.0, -float(result.ineqlin.marginals[0]))
    # For ANY lambda>=0 this is a valid dual lower bound; no solver feasibility assumption.
    lower = float(lam*target + np.min(c-lam*q, axis=1).mean())
    achieved = float((weights*q).sum()/n)
    upper = float((weights*c).sum()/n)
    if abs(weights.sum(axis=1)-1).max() > 1e-7 or achieved < target-1e-7 or lower > upper+1e-7:
        raise RuntimeError('oracle certificate failed')
    return {'status': 'randomized_lp_lower_bound', 'maximum_mean_quality': maximum,
            'dual_lambda': lam, 'cost_lower_bound_seconds': lower, 'primal_cost_seconds': upper,
            'primal_quality': achieved, 'duality_gap_seconds': upper-lower}


def build_backend(corpus, config, spec, projected=None, check=lambda: None):
    faiss.omp_set_num_threads(config['limits']['offline_threads'])
    x = corpus if projected is None else projected
    start = time.perf_counter()
    if spec['family'] in ('original_hnsw', 'gaussian_hnsw'):
        index = faiss.IndexHNSWFlat(x.shape[1], spec['degree'], faiss.METRIC_INNER_PRODUCT if projected is None else faiss.METRIC_L2)
        index.hnsw.efConstruction = config['hnsw']['ef_construction']
        index.hnsw.rng = faiss.RandomGenerator(config['seeds']['graph'])
    elif spec['family'] == 'ivfpq':
        settings = config['ivfpq']
        index = faiss.index_factory(x.shape[1], f"IVF{settings['nlist']},PQ{settings['subquantizers']}x{settings['bits']}fs", faiss.METRIC_INNER_PRODUCT)
        index.cp.seed = config['seeds']['training']
        index.pq.cp.seed = config['seeds']['training']
        train_ids = np.random.default_rng(config['seeds']['training']).choice(len(x), settings['training_count'], replace=False)
        check()
        index.train(np.ascontiguousarray(x[train_ids]))
    else:
        raise ValueError('unknown backend')
    for i in range(0, len(x), config['block_rows']):
        check()
        index.add(np.ascontiguousarray(x[i:i+config['block_rows']]))
    return index, time.perf_counter()-start


def run_backend(index, query, corpus, action, k, matrix=None):
    """Evaluation labels are absent from this timed retrieval API."""
    start = time.perf_counter()
    q = np.ascontiguousarray(query[None, :] if matrix is None else project(query[None, :], matrix))
    projected_at = time.perf_counter()
    if action.get('nprobe') is not None:
        index.nprobe = action['nprobe']
    else:
        index.hnsw.efSearch = action['ef_search']
    _, raw = index.search(q, action['M'])
    searched_at = time.perf_counter()
    ids = valid_ids(raw[0], len(corpus), k)
    selected_at = time.perf_counter()
    result, distances = rerank(query, corpus, ids, k)
    finished = time.perf_counter()
    return result, distances, ids, {'projection_seconds': projected_at-start,
        'search_seconds': searched_at-projected_at, 'selection_seconds': selected_at-searched_at,
        'rerank_seconds': finished-selected_at, 'total_seconds': finished-start,
        'pilot_seconds': 0.0, 'risk_seconds': 0.0, 'expansion_seconds': 0.0}
