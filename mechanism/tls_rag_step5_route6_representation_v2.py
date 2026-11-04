"""Approved S0 v2: read-only source mapping and fingerprinted CUDA norm cache.

Unchanged transforms/ADC/ties are explicit calls to the frozen v1 implementation.
There is no independence model, new Tri formula, or projected renormalization.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mechanism import tls_rag_step5_route6_representation_v1 as frozen

fingerprint = frozen.fingerprint
require_fingerprint = frozen.require_fingerprint
stable_top = frozen.stable_top
original_distances = frozen.original_distances
rerank = frozen.rerank
sha = frozen.sha
canonical_vector_hash = frozen.canonical_vector_hash
stable_id = frozen.stable_id
read_payload = frozen.read_payload
sample_fit_rows = frozen.sample_fit_rows
fit_pca = frozen.fit_pca
transform_batch = frozen.transform_batch
encode_pq = frozen.encode_pq
adc_scores = frozen.adc_scores
decode_codes = frozen.decode_codes
exact_representation_search = frozen.exact_representation_search
make_flat = frozen.make_flat
make_ivf = frozen.make_ivf
faiss_search = frozen.faiss_search
ivf_same_code_reference = frozen.ivf_same_code_reference
IVFDecodedView = frozen.IVFDecodedView
load_prior_fp64 = frozen.load_prior_fp64


def source_mapping(cfg, verify_hash=False, check=lambda: None):
    """Map only contiguous native train bytes, never test or query payloads."""
    import h5py
    source = cfg['source']; path = Path(source['path'])
    if path.stat().st_size != source['bytes']:
        raise ValueError('source byte count')
    if verify_hash and sha(path, check) != source['sha256']:
        raise ValueError('source full SHA mismatch')
    with h5py.File(path, 'r') as file:
        metric = file.attrs.get('distance')
        if isinstance(metric, bytes): metric = metric.decode()
        train = file['train']
        layout = dict(offset=train.id.get_offset(), dtype=train.dtype.str, order='C',
                      chunks=train.chunks, compression=train.compression, virtual=bool(train.is_virtual))
        if (metric != source['metric'] or int(file.attrs.get('dimension', -1)) != source['dimension']
                or train.shape != (source['rows'], source['dimension']) or layout != source['layout']
                or layout['offset'] is None or train.dtype != np.dtype('<f4')):
            raise ValueError('source contiguous layout/metric/shape mismatch')
        if layout['offset'] + train.size * train.dtype.itemsize > source['bytes']:
            raise ValueError('source payload exceeds file')
    return np.memmap(path, mode='r', offset=layout['offset'], dtype='<f4',
                     shape=(source['rows'], source['dimension']), order='C')


def load_screen_inputs(cfg, check=lambda: None, phase=lambda *a, **k: None):
    import h5py
    start = time.monotonic(); phase('source_full_sha')
    raw = source_mapping(cfg, verify_hash=True, check=check)
    if sha(cfg['roles_path'], check) != cfg['roles_sha256']:
        raise ValueError('original roles identity')
    hash_seconds = time.monotonic() - start
    roles = json.loads(Path(cfg['roles_path']).read_text())['queries']
    if len(roles) != cfg['queries'] or len({r['row'] for r in roles}) != len(roles):
        raise ValueError('fixed unique query rows')
    phase('validate_mapped_corpus_norms'); tick = time.monotonic()
    for offset in range(0, len(raw), cfg['block']):
        check(); block = raw[offset:offset+cfg['block']]
        norms = np.linalg.norm(block.astype(np.float64), axis=1)
        if not np.isfinite(block).all() or np.max(np.abs(norms-1)) > cfg['norm_tolerance']:
            raise ValueError('stored corpus finite/unit-norm gate')
    norm_seconds = time.monotonic()-tick; tick = time.monotonic()
    queries, truth = [], []
    with h5py.File(cfg['source']['path'], 'r') as file:
        if (file['learn_neighbors'].shape != (len(file['learn']), cfg['k'])
                or file['learn_neighbors'].dtype.kind not in 'iu'):
            raise ValueError('published truth shape/dtype')
        for i, role in enumerate(roles):
            check(); q = read_payload(file, 'learn', role['row']).astype(np.float32)
            ids = read_payload(file, 'learn_neighbors', role['row']).astype(np.int64)
            expected = 'geometry_discovery' if i < cfg['discovery_queries'] else 'geometry_check'
            if (role['role'] != expected or role['id'] != stable_id(cfg['source']['sha256'], 'learn', role['row'])
                    or canonical_vector_hash(q) != role['vector_sha256'] or not np.isfinite(q).all()
                    or abs(np.linalg.norm(q.astype(float))-1) > cfg['norm_tolerance']):
                raise ValueError('fixed query identity/role/norm')
            if len(ids) != cfg['k'] or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(raw):
                raise ValueError('published truth duplicate/out-of-range')
            queries.append(q); truth.append(ids)
    return raw, np.asarray(queries), np.asarray(truth), roles, dict(
        source_sha256=cfg['source']['sha256'], roles_sha256=cfg['roles_sha256'],
        mapping=cfg['source']['layout'], raw_bytes=int(raw.nbytes), source_file_bytes=cfg['source']['bytes'],
        mode='read-only mmap', renormalized=False, test_payload_read=False, independent_certification=False,
        source_roles_sha_seconds=hash_seconds, corpus_norm_audit_seconds=norm_seconds,
        query_truth_load_seconds=time.monotonic()-tick)


def model_fingerprint(model, cfg):
    return fingerprint(dict(namespace=cfg['namespace'], spec=model['spec'], metric=model['metric'],
        matrix=hashlib.sha256(model['matrix'].tobytes()).hexdigest(),
        mean=hashlib.sha256(model['mean'].tobytes()).hexdigest(),source=cfg['source']['sha256'],
        layout=cfg['source']['layout'],roles=cfg['roles_sha256'],seeds=cfg['seeds'],training=cfg['training'],
        epsilon=cfg['epsilon'],dtype='float32',no_whitening=True,no_renormalization=True))


def fit_representation(spec, sample, pca, cfg, check=lambda: None):
    model = frozen.fit_representation(spec, sample, pca, cfg, check)
    model['v1_fingerprint'] = model['fingerprint']
    model['fingerprint'] = model_fingerprint(model,cfg)
    model['source_sha256'] = cfg['source']['sha256']
    return model


def cache_binding(model, source_sha, shape, device):
    return dict(namespace='s0-v2-cuda-norm-cache', model=model['fingerprint'],
                source=source_sha, shape=list(shape), dtype='float32', device=str(device), metric=model['metric'])


def prepare_cuda_index(values, model, cfg, torch, check=lambda: None):
    """Charge allocation/H2D and static norm preparation separately, once."""
    torch.cuda.synchronize(); tick = time.monotonic()
    device = torch.empty(values.shape, dtype=torch.float32, device='cuda')
    for start in range(0, len(values), cfg['block']):
        check()
        host = np.array(values[start:start+cfg['block']], dtype=np.float32, copy=True)
        device[start:start+len(host)] = torch.as_tensor(host, device='cuda')
    torch.cuda.synchronize(); placement = time.monotonic()-tick; tick = time.monotonic()
    norms = None
    if model['metric'] == 'l2':
        norms = torch.empty(len(values), dtype=torch.float32, device='cuda')
        for start in range(0, len(values), cfg['block']):
            check(); part = device[start:start+cfg['block']]
            norms[start:start+len(part)] = (part*part).sum(dim=1)
    torch.cuda.synchronize(); norm_seconds = time.monotonic()-tick
    binding = cache_binding(model, cfg['source']['sha256'], values.shape, device.device)
    return dict(values=device, norms=norms, binding=binding, fingerprint=fingerprint(binding),
                h2d_seconds=placement, norm_prepare_seconds=norm_seconds,
                payload_bytes=int(values.nbytes), norm_bytes=0 if norms is None else int(len(values)*4))


def validate_cuda_cache(model, cache):
    binding = cache['binding']; device = cache['values']
    if (cache['fingerprint'] != fingerprint(binding) or binding['model'] != model['fingerprint']
            or binding['metric'] != model['metric'] or binding['shape'] != list(device.shape)
            or binding['device'] != str(device.device) or str(device.dtype) != 'torch.float32'
            or binding['dtype'] != 'float32' or binding['source'] != model['source_sha256']):
        raise ValueError('CUDA cache fingerprint/shape/dtype/device mismatch')
    norms = cache['norms']
    if model['metric'] == 'l2' and (norms is None or tuple(norms.shape) != (len(device),)
            or norms.device != device.device or str(norms.dtype) != 'torch.float32'):
        raise ValueError('CUDA norm cache mismatch')


def cached_scores(values, query, norms, metric):
    scores = -(values @ query)
    if metric == 'l2':
        if norms is None or len(norms) != len(values): raise ValueError('static norms required')
        scores = 2*scores + norms + (query*query).sum()
    elif metric != 'ip': raise ValueError('cached metric')
    return scores


def query_coordinates(model, queries):
    """Match the deployed batch1 transform, including its FP32 operation path."""
    return np.asarray([transform_batch(model,np.asarray(q)[None])[0] for q in queries])


def search_backend(model, raw_query, count, index=None, cuda_index=None, torch=None):
    """Label-free actual backend; query transform, GPU topk/transfers paid."""
    q = transform_batch(model, np.asarray(raw_query)[None])[0]
    if index is not None:
        return faiss_search(index, q, count, model['metric'])
    if cuda_index is None or torch is None:
        raise ValueError('actual backend handle required')
    validate_cuda_cache(model, cuda_index)
    values = cuda_index['values']; device_q = torch.as_tensor(q, dtype=torch.float32, device=values.device)
    scores = cached_scores(values, device_q, cuda_index['norms'], model['metric'])
    vals, ids = torch.topk(scores, min(count+1, len(scores)), largest=False, sorted=True)
    host_values, host_ids = vals.cpu().numpy(), ids.cpu().numpy()
    if len(host_values) > count and host_values[count-1] == host_values[count]:
        host_ids = torch.nonzero(scores <= vals[count-1], as_tuple=False).flatten().cpu().numpy()
        host_values = scores[torch.as_tensor(host_ids, device=values.device)].cpu().numpy()
    result = stable_top(host_values, count, host_ids)
    torch.cuda.synchronize()
    return result


def model_from_products(root, spec, cfg, restore_pq=True):
    """Load saved FP32 matrices and exact PQ codebooks; never refit for timing."""
    import faiss
    root = Path(root); meta = json.loads((root/'model.json').read_text())
    if meta['spec'] != spec or meta['namespace'] != cfg['namespace']:
        raise ValueError('saved model spec/namespace')
    matrix = np.load(root/'matrix.npy', allow_pickle=False)
    mean = np.load(root/'mean.npy', allow_pickle=False)
    if sha(root/'matrix.npy') != meta['matrix_file_sha256'] or sha(root/'mean.npy') != meta['mean_file_sha256']:
        raise ValueError('saved model matrix/mean mutated')
    model = dict(spec=spec, matrix=matrix, mean=mean, metric=meta['metric'],
                 fingerprint=meta['fingerprint'], fit_costs=meta['fit_costs'],source_sha256=cfg['source']['sha256'])
    if model_fingerprint(model,cfg)!=model['fingerprint']:raise ValueError('saved model/config fingerprint mismatch')
    if 'pq_bytes' in spec and restore_pq:
        centers = np.load(root/'centroids.npy', allow_pickle=False)
        if sha(root/'centroids.npy') != meta['centroids_file_sha256']:
            raise ValueError('saved codebook mutated')
        index = faiss.IndexPQ(spec['dim'], spec['pq_bytes'], 8,
            faiss.METRIC_INNER_PRODUCT if model['metric']=='ip' else faiss.METRIC_L2)
        faiss.copy_array_to_vector(centers.reshape(-1), index.pq.centroids)
        index.is_trained = True
        model['index'] = index
    return model


def install_codes(model, path, expected_sha, check=lambda: None):
    import faiss
    if sha(path, check) != expected_sha:
        raise ValueError('saved code bytes mutated')
    codes = np.load(path, mmap_mode='r', allow_pickle=False)
    if codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[1] != model['spec']['pq_bytes']:
        raise ValueError('saved code shape/dtype')
    faiss.copy_array_to_vector(np.asarray(codes).reshape(-1), model['index'].codes)
    model['index'].ntotal = len(codes)
    return codes
