"""R6-S0: stored-vector geometry, fitted representations and label-free search.

References are finite-precision audits, not real-arithmetic certificates. Nothing
in this module evaluates Tri-Law or assumes independence of projection errors.
"""
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mechanism.tls_rag_step5_route6_real_audit_v1 import (
    sha, canonical_vector_hash, stable_id)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def require_fingerprint(saved, expected):
    if saved != expected:
        raise ValueError('representation cache fingerprint mismatch')


def stable_top(scores, count, rows=None):
    """Partial selection, retaining all cutoff ties before canonical row sorting."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all() or count < 1 or count > len(scores):
        raise ValueError('finite scores and valid count required')
    ids = np.arange(len(scores), dtype=np.int64) if rows is None else np.asarray(rows, np.int64)
    if ids.shape != scores.shape or (rows is not None and (len(np.unique(ids)) != len(ids) or np.any(ids < 0))):
        raise ValueError('unique nonnegative explicit row map')
    threshold = np.partition(scores, count-1)[count-1]
    selected = np.flatnonzero(scores <= threshold)
    order = np.lexsort((ids[selected], scores[selected]))[:count]
    return ids[selected[order]]


def original_distances(raw, query):
    """Native stored-FP32 normalized-cosine formula; no normalization here."""
    return np.clip(np.float32(1)-np.asarray(raw, np.float32) @ np.asarray(query, np.float32), 0, 2)


def rerank(raw, query, candidates, k):
    ids = np.asarray(candidates, np.int64)
    if ids.ndim != 1 or len(ids) < k or np.any(ids < 0) or np.any(ids >= len(raw)) or len(np.unique(ids)) != len(ids):
        raise ValueError('invalid retrieval candidates')
    distances = original_distances(raw[ids], query)
    selected = stable_top(distances, k, ids)
    return selected, original_distances(raw[selected], query)


def read_payload(file, name, row=None):
    """Explicit closed-test trap, including when called outside the main runner."""
    if name not in ('train', 'learn', 'learn_neighbors'):
        raise ValueError('closed payload: '+name)
    if name != 'train' and row is None:
        raise ValueError('only approved individual learn rows may be read')
    return np.asarray(file[name][:] if row is None else file[name][row])


def load_screen_inputs(cfg, check=lambda: None, phase=lambda *a: None):
    import h5py
    source = cfg['source']
    input_start=time.monotonic()
    phase('source_full_sha')
    if Path(source['path']).stat().st_size != source['bytes'] or sha(source['path'], check) != source['sha256']:
        raise ValueError('source full SHA/size mismatch')
    if sha(cfg['roles_path'], check) != cfg['roles_sha256']:
        raise ValueError('original roles identity')
    source_roles_sha_seconds=time.monotonic()-input_start
    roles = json.loads(Path(cfg['roles_path']).read_text())['queries']
    if len(roles) != cfg['queries'] or len({q['row'] for q in roles}) != len(roles):
        raise ValueError('fixed unique query rows')
    with h5py.File(source['path'], 'r') as file:
        metric = file.attrs.get('distance')
        if isinstance(metric, bytes):
            metric = metric.decode()
        if metric != source['metric'] or int(file.attrs.get('dimension', -1)) != source['dimension']:
            raise ValueError('source native metadata')
        if file['train'].shape != (source['rows'], source['dimension']) or file['train'].dtype != np.dtype('float32'):
            raise ValueError('complete corpus shape/dtype')
        if (file['learn_neighbors'].shape != (len(file['learn']), cfg['k']) or
                file['learn_neighbors'].dtype.kind not in 'iu'):
            raise ValueError('published learn truth shape/dtype')
        phase('load_original_corpus')
        corpus_load_start=time.monotonic()
        raw = np.empty(file['train'].shape, np.float32)
        for start in range(0, len(raw), cfg['block']):
            check()
            block = read_payload(file, 'train', slice(start, start+cfg['block']))
            norms = np.linalg.norm(block.astype(np.float64), axis=1)
            if not np.isfinite(block).all() or np.max(np.abs(norms-1)) > cfg['norm_tolerance']:
                raise ValueError('stored corpus finite/unit-norm gate')
            raw[start:start+len(block)] = block
        corpus_load_seconds=time.monotonic()-corpus_load_start
        query_truth_start=time.monotonic()
        queries, truth = [], []
        for i, role in enumerate(roles):
            check()
            expected_role = 'geometry_discovery' if i < cfg['discovery_queries'] else 'geometry_check'
            q = read_payload(file, 'learn', role['row']).astype(np.float32)
            ids = read_payload(file, 'learn_neighbors', role['row']).astype(np.int64)
            if (role['role'] != expected_role or role['id'] != stable_id(source['sha256'], 'learn', role['row']) or
                    canonical_vector_hash(q) != role['vector_sha256'] or not np.isfinite(q).all() or
                    abs(np.linalg.norm(q.astype(float))-1) > cfg['norm_tolerance']):
                raise ValueError('fixed query identity/role/norm')
            if len(ids) != cfg['k'] or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(raw):
                raise ValueError('published truth duplicate/out-of-range ID')
            queries.append(q); truth.append(ids)
        metadata = {name: dict(shape=list(file[name].shape), dtype=str(file[name].dtype)) for name in file.keys()}
    return raw, np.asarray(queries), np.asarray(truth), roles, dict(
        source_sha256=source['sha256'], roles_sha256=cfg['roles_sha256'], metadata=metadata,
        allowed_payloads=['train', '128 fixed learn rows', 'corresponding 128 learn_neighbors rows'],
        test_payload_read=False, renormalized=False, independent_certification=False,
        source_roles_sha_seconds=source_roles_sha_seconds,corpus_load_seconds=corpus_load_seconds,
        query_truth_load_seconds=time.monotonic()-query_truth_start)


def load_prior_fp64(cfg, roles, check=lambda: None):
    """Read the closed B0 full-corpus ordering, never its local Gram as a corpus."""
    root = Path(cfg['old_reference'])
    if sha(root/'delivery_index.json', check) != cfg['old_index_sha256']:
        raise ValueError('closed B0 index identity')
    entries = json.loads((root/'delivery_index.json').read_text())['files']
    for entry in entries:
        check()
        path = root/entry['path']
        if path.resolve().parent != root.resolve() and root.resolve() not in path.resolve().parents:
            raise ValueError('unsafe prior index path')
        if path.stat().st_size != entry['bytes'] or sha(path, check) != entry['sha256']:
            raise ValueError('prior B0 product mutated')
    freeze = json.loads((root/'source_freeze.json').read_text())
    if freeze['config']['source']['sha256'] != cfg['source']['sha256']:
        raise ValueError('B0 source mismatch')
    rows = []
    for i, role in enumerate(roles):
        record = json.loads((root/'queries'/('%03d' % i)/'record.json').read_text())
        if record['query'] != role or record['rank_diagnostics']['l2_native_ip_topk_overlap'] != cfg['k']:
            raise ValueError('prior role/IP-L2 agreement mismatch')
        ids = record['corpus_id_map'][:cfg['k']]
        if [x['rank'] for x in ids] != list(range(1, cfg['k']+1)) or any(x['group'] != 'G' for x in ids):
            raise ValueError('prior full-corpus top-k map')
        rows.append([x['row'] for x in ids])
    return np.asarray(rows, np.int64)


def sample_fit_rows(n, size, seed):
    if not 1 <= size <= n:
        raise ValueError('fit corpus sample size')
    return np.sort(np.random.Generator(np.random.PCG64(seed)).choice(n, size, replace=False))


def fit_pca(sample):
    values = np.asarray(sample, np.float64)
    mean = values.mean(axis=0)
    centered = values-mean
    cov = centered.T @ centered / len(centered)
    eigenvalues, basis = np.linalg.eigh(cov)
    basis = basis[:, ::-1].copy()
    for col in range(basis.shape[1]):
        pivot = np.argmax(np.abs(basis[:, col]))
        if basis[pivot, col] < 0:
            basis[:, col] *= -1
    return dict(mean=mean, basis=basis, eigenvalues=eigenvalues[::-1])


def fit_representation(spec, sample, pca, cfg, check=lambda: None):
    import faiss
    check()
    fit_start=time.monotonic()
    d, m, kind = sample.shape[1], spec['dim'], spec['kind']
    mean = np.zeros(d, np.float64)
    matrix = np.eye(d, dtype=np.float64)
    if kind == 'pca':
        mean, matrix = pca['mean'], pca['basis'][:, :m].T.copy()
    elif kind == 'gaussian':
        matrix = np.random.Generator(np.random.PCG64(spec['seed'])).standard_normal((min(256, d), d))[:m]/np.sqrt(m)
    elif kind == 'hybrid':
        r, s = spec['retained'], m-spec['retained']
        mean = pca['mean']
        basis = pca['basis'][:, :r]
        g = np.random.Generator(np.random.PCG64(np.random.SeedSequence([spec['seed'],cfg['seeds']['hybrid_domain']]))).standard_normal((min(192, d), d))[:s]/np.sqrt(s)
        matrix = np.concatenate((basis.T, g-(g @ basis) @ basis.T))
    elif kind == 'opq':
        opq = faiss.OPQMatrix(d, spec['pq_bytes'])
        opq.niter, opq.niter_pq_0, opq.niter_pq = (cfg['training'][x] for x in ('opq_outer','opq_first','opq_later'))
        opq.max_train_points = cfg['fit_rows']
        opq_pq = faiss.ProductQuantizer(d, spec['pq_bytes'], 8)
        opq.pq = opq_pq
        opq.pq.cp.seed = cfg['seeds']['pq']
        opq.pq.cp.nredo = cfg['training']['nredo']
        opq.train(np.ascontiguousarray(sample))
        matrix = faiss.vector_to_array(opq.A).reshape(d, d).astype(np.float64)
        # OPQ v1.8 trains on centered data but deploys a rotation without bias.
    elif kind != 'raw':
        raise ValueError('unknown representation kind')
    if matrix.shape != (m, d):
        raise ValueError('representation output dimension')
    model = dict(spec=spec, mean=np.asarray(mean), matrix=np.asarray(matrix, np.float32))
    representation_fit_seconds=time.monotonic()-fit_start
    pq_started=time.monotonic()
    if 'pq_bytes' in spec:
        metric = faiss.METRIC_INNER_PRODUCT if kind in ('raw','opq') else faiss.METRIC_L2
        index = faiss.IndexPQ(m, spec['pq_bytes'], 8, metric)
        index.pq.cp.niter, index.pq.cp.nredo = cfg['training']['niter'], cfg['training']['nredo']
        index.pq.cp.seed = cfg['seeds']['pq']
        index.train(transform_batch(model, sample))
        model['index'] = index
        model['metric'] = 'ip' if metric == faiss.METRIC_INNER_PRODUCT else 'l2'
    else:
        model['metric'] = 'ip' if kind == 'raw' else 'l2'
    check()
    model['fit_costs']=dict(representation_seconds=representation_fit_seconds,
        pq_seconds=time.monotonic()-pq_started if 'pq_bytes' in spec else 0.)
    model['fingerprint'] = fingerprint(dict(spec=spec, matrix=hashlib.sha256(model['matrix'].tobytes()).hexdigest(),
        mean=hashlib.sha256(model['mean'].tobytes()).hexdigest(), source=cfg['source']['sha256'],
        roles=cfg.get('roles_sha256'), seeds=cfg['seeds'], training=cfg['training'], dtype='float32',
        no_renormalization=True, no_whitening=True, original_metric='clip(1-dot,0,2)', epsilon=cfg['epsilon']))
    return model


def transform_batch(model, vectors):
    x = np.asarray(vectors, np.float32)
    if model['spec']['kind'] == 'raw':
        return np.ascontiguousarray(x)
    return np.ascontiguousarray((x-model['mean'].astype(np.float32)) @ model['matrix'].T, dtype=np.float32)


def encode_pq(model, vectors, block=8192, check=lambda: None):
    """Actual IndexPQ codes; callers may persist codes, never decoded full floats."""
    if 'index' not in model:
        raise ValueError('PQ model required')
    index = model['index']
    for start in range(0, len(vectors), block):
        check()
        index.add(transform_batch(model, vectors[start:start+block]))
    import faiss
    codes = faiss.vector_to_array(index.codes).reshape(index.ntotal, index.code_size).copy()
    centers = faiss.vector_to_array(index.pq.centroids).reshape(index.pq.M, index.pq.ksub, index.pq.dsub).copy()
    return codes, centers


def adc_scores(query, codes, centroids, metric='l2'):
    codes = np.asarray(codes)
    centers = np.asarray(centroids, np.float64)
    q = np.asarray(query, np.float64).reshape(centers.shape[0], centers.shape[2])
    if codes.ndim != 2 or codes.shape[1] != len(centers) or codes.dtype != np.uint8:
        raise ValueError('8-bit PQ codes required')
    if metric == 'l2':
        tables = np.sum((centers-q[:, None, :])**2, axis=2)
    elif metric == 'ip':
        tables = -np.einsum('mcd,md->mc', centers, q)
    else:
        raise ValueError('ADC metric')
    return tables[np.arange(len(centers))[:, None], codes.T].sum(axis=0)


def decode_codes(codes, centers):
    return np.asarray(centers)[np.arange(len(centers))[:, None], np.asarray(codes).T].transpose(1,0,2).reshape(len(codes), -1)


def exact_representation_search(values, queries, count, metric='l2', codes=None, centers=None,
                                torch=None, check=lambda: None, batch=8, block=8192):
    """FP64 full scan of stored coordinates / the same frozen PQ codebook.

    CPU independent ADC accumulates FP64 tables. CUDA reference decodes the same
    codes in bounded chunks, then searches in FP64. These decoded audit vectors
    are NOT a compressed deployment backend or GPU Faiss.
    """
    queries = np.asarray(queries)
    n = len(codes) if codes is not None else len(values)
    result = []
    if torch is None:
        for q in queries:
            check()
            if codes is not None:
                scores = adc_scores(q, codes, centers, metric)
            else:
                x, q = np.asarray(values, np.float64), np.asarray(q, np.float64)
                scores = -x @ q if metric == 'ip' else np.sum((x-q)**2, axis=1)
            result.append(stable_top(scores, count))
        return np.asarray(result)
    # One serial reference corpus at a time; no query-by-corpus matrix on disk.
    dimension = queries.shape[1]
    device_values = torch.empty((n, dimension), dtype=torch.float64, device='cuda')
    norms = torch.empty(n,dtype=torch.float64,device='cuda') if metric=='l2' else None
    for start in range(0, n, block):
        check()
        x = decode_codes(codes[start:start+block], centers) if codes is not None else values[start:start+block]
        device_values[start:start+len(x)] = torch.as_tensor(x, dtype=torch.float64, device='cuda')
        if norms is not None:
            part=device_values[start:start+len(x)]
            norms[start:start+len(x)]=(part*part).sum(dim=1)
            del part
    for start in range(0, len(queries), batch):
        check()
        q = torch.as_tensor(queries[start:start+batch], dtype=torch.float64, device='cuda')
        scores = -(q @ device_values.T)
        if metric == 'l2':
            scores *= 2
            scores += norms[None, :]
            scores += (q*q).sum(dim=1)[:, None]
        host = scores.cpu().numpy()
        for row in host:
            result.append(stable_top(row, count))
        del q, scores, host
    del device_values, norms
    gc.collect(); torch.cuda.empty_cache()
    return np.asarray(result)


def make_flat(model, values, threads=14):
    import faiss
    faiss.omp_set_num_threads(threads)
    if 'index' in model:
        return model['index']
    index = faiss.IndexFlatIP(values.shape[1]) if model['metric']=='ip' else faiss.IndexFlatL2(values.shape[1])
    index.add(np.ascontiguousarray(values))
    return index


def make_ivf(model, sample, raw, cfg, check=lambda: None):
    import faiss
    d, nlist = model['spec']['dim'], cfg['ivf']['nlist']
    metric = faiss.METRIC_INNER_PRODUCT if model['metric']=='ip' else faiss.METRIC_L2
    quantizer = faiss.IndexFlatIP(d) if metric==faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
    if 'pq_bytes' in model['spec']:
        index = faiss.IndexIVFPQ(quantizer, d, nlist, model['spec']['pq_bytes'], 8, metric)
        index.pq.cp.niter, index.pq.cp.nredo = cfg['training']['niter'], cfg['training']['nredo']
        index.pq.cp.seed = cfg['seeds']['pq']
    else:
        index = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
    index.cp.niter, index.cp.nredo = cfg['training']['niter'], cfg['training']['nredo']
    index.cp.seed = cfg['seeds']['coarse']
    index.max_codes = 0
    index.train(transform_batch(model, sample))
    for start in range(0, len(raw), cfg['block']):
        check(); index.add(transform_batch(model, raw[start:start+cfg['block']]))
    return index


def faiss_search(index, query, count, metric):
    """Deterministic explicit-ID ties, growing beyond M+1 if necessary."""
    request = min(index.ntotal, count+1)
    while True:
        distances, ids = index.search(np.ascontiguousarray(query[None], np.float32), request)
        valid = ids[0] >= 0
        candidates, scores = ids[0][valid], distances[0][valid]
        if len(candidates) < count:
            raise ValueError('backend returned fewer than requested candidates')
        ordered_scores = -scores if metric=='ip' else scores
        chosen = stable_top(ordered_scores, count, candidates)
        kth = ordered_scores[np.flatnonzero(candidates == chosen[-1])[0]]
        if request == index.ntotal or len(candidates) < request or ordered_scores[-1] > kth:
            return chosen
        request = min(index.ntotal, max(request+1, request*2))


def search_backend(model, raw_query, count, index=None, device_values=None, torch=None):
    """Label-free deployment interface. Caller evaluates returned IDs separately."""
    q = transform_batch(model, np.asarray(raw_query)[None])[0]
    if index is not None:
        return faiss_search(index, q, count, model['metric'])
    if device_values is None or torch is None:
        raise ValueError('real backend handle required')
    device_q = torch.as_tensor(q, dtype=torch.float32, device='cuda')
    scores = -(device_values @ device_q)
    if model['metric']=='l2':
        scores = 2*scores + (device_values*device_values).sum(dim=1) + (device_q*device_q).sum()
    # Real GPU topk, then canonicalize returned boundary ties. Transfers paid.
    vals, ids = torch.topk(scores, min(count+1, len(scores)), largest=False, sorted=True)
    host_values, host_ids = vals.cpu().numpy(), ids.cpu().numpy()
    if len(host_values)>count and host_values[count-1] == host_values[count]:
        host_ids = torch.nonzero(scores <= vals[count-1], as_tuple=False).flatten().cpu().numpy()
        host_values = scores[torch.as_tensor(host_ids, device='cuda')].cpu().numpy()
    result = stable_top(host_values, count, host_ids)
    torch.cuda.synchronize()
    return result


def ivf_same_code_reference(index, queries, count):
    """Full probes of THIS IVF's residual codes, not another global PQ model."""
    old = index.nprobe
    try:
        index.nprobe = index.nlist
        return np.asarray([faiss_search(index, q, count,
            'ip' if index.metric_type == 0 else 'l2') for q in queries])
    finally:
        index.nprobe = old


class IVFDecodedView:
    """FP64 coarse+residual reconstruction from THIS IVF's 8-bit codes."""
    def __init__(self,index):
        import faiss
        self.index=index
        self.pq=hasattr(index,'pq')
        if self.pq:
            if index.pq.nbits!=8 or not index.by_residual:
                raise ValueError('frozen 8-bit residual IVFPQ required')
            self.coarse=index.quantizer.reconstruct_n(0,index.nlist).astype(np.float64)
            self.centers=faiss.vector_to_array(index.pq.centroids).reshape(index.pq.M,256,index.pq.dsub)
            self.codes=np.empty((index.ntotal,index.code_size),np.uint8)
            self.labels=np.empty(index.ntotal,np.int32)
            for list_id in range(index.nlist):
                size=index.invlists.list_size(list_id)
                ids=index.invlists.get_ids(list_id);codes=index.invlists.get_codes(list_id)
                try:
                    rows=faiss.rev_swig_ptr(ids,size)
                    self.labels[rows]=list_id
                    self.codes[rows]=faiss.rev_swig_ptr(codes,size*index.code_size).reshape(size,index.code_size)
                finally:
                    index.invlists.release_ids(list_id,ids);index.invlists.release_codes(list_id,codes)
    def __len__(self):return self.index.ntotal
    def __getitem__(self,key):
        if not isinstance(key,slice) or key.step is not None:
            raise ValueError('bounded contiguous IVF decode only')
        start=key.start or 0; stop=min(key.stop,self.index.ntotal)
        if self.pq:
            return decode_codes(self.codes[start:stop],self.centers).astype(np.float64)+self.coarse[self.labels[start:stop]]
        return self.index.reconstruct_n(start,stop-start)
