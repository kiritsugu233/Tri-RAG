"""Approved gate2: actual IVFPQ residual geometry and paid refinement controls.

No Tri predictor, adaptive dimension, or new query role. Probability statements
refer to ideal real Gaussian matrices; finite native results are development
audits under an explicit conservative floating-point error model, not certified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import norm

from mechanism import tls_rag_step5_pq_refinement_cost_v1 as cost
from mechanism import tls_rag_step5_route6_representation_screen_v1 as metrics

ROOT = cost.ROOT
PATHS = (
    'mechanism/tls_rag_step5_pq_refinement_geometry_v1.py',
    'configs/tls_rag_step5_pq_refinement_geometry_v1.json',
    'tests/test_tls_rag_step5_pq_refinement_geometry_v1.py',
    'scripts/run_tls_rag_step5_pq_refinement_geometry_v1.py',
    'docs/PQ_REFINEMENT_GEOMETRY_PROTOCOL_V1.md',
)
U32 = np.finfo(np.float32).eps / 2
U64 = np.finfo(np.float64).eps / 2


def gamma(n, unit=U64):
    if n * unit >= 1:
        raise ValueError('invalid accumulation envelope')
    return n * unit / (1 - n * unit)


def config():
    c = json.loads((ROOT / PATHS[1]).read_text())
    if (c['namespace'] != 'pq-refinement-geometry-v1' or c['tri_enabled']
            or c['formal_certification'] or c['nprobe'] != [64, 256]
            or c['search_budget'] != 6400 or c['k'] != 100
            or c['dimensions'] != [8, 16, 32] or c['delta'] != .01
            or c['scan_block'] != 256 or c['fit_rows'] != 32768
            or c['side_pq_bytes'] != [32, 64, 128]
            or c['timing'] != cost.config()['timing']):
        raise ValueError('outside approved gate2 scope')
    return c


def source_map(c):
    for p, expected in c['old_source_bindings'].items():
        if cost.sha(ROOT / p) != expected:
            raise ValueError('old source changed: ' + p)
    return {p: cost.sha(ROOT / p) for p in PATHS}


def outward_norm(values, error=0.):
    """Stored FP32 upper radii; error includes prior reconstruction uncertainty."""
    x = np.asarray(values, np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(error) or error < 0:
        raise ValueError('finite vectors and nonnegative envelope required')
    r = np.linalg.norm(x, axis=1)
    upper = r + error + 8 * gamma(x.shape[1]) * (r + 1)
    with np.errstate(over='ignore'):
        stored = np.nextafter(upper.astype(np.float32), np.float32(np.inf))
    if not np.isfinite(stored).all():
        raise ValueError('FP32 radius overflow')
    return stored


def extract_actual_ivf_codes(index, rows, saved_rows=None, saved_offsets=None):
    """Read THIS index's lists; only requested codes are copied, no direct map."""
    import faiss
    rows = np.asarray(rows, np.int64)
    if (rows.ndim != 1 or not len(rows) or np.any(np.diff(rows) <= 0)
            or rows[0] < 0 or rows[-1] >= index.ntotal or not index.by_residual
            or index.pq.nbits != 8 or index.code_size != index.pq.M):
        raise ValueError('sorted unique rows and original 8bit residual IVF required')
    codes = np.empty((len(rows), index.code_size), np.uint8)
    labels = np.empty(len(rows), np.int32)
    offsets = np.empty(len(rows), np.int64)
    found = np.zeros(len(rows), bool)
    if (saved_rows is None) != (saved_offsets is None):
        raise ValueError('both original list maps required')
    if saved_offsets is not None and (len(saved_offsets) != index.nlist + 1
            or saved_offsets[0] != 0 or saved_offsets[-1] != index.ntotal):
        raise ValueError('original list offset extent')
    for li in range(index.nlist):
        size = index.invlists.list_size(li)
        if not size:
            if saved_offsets is not None and saved_offsets[li] != saved_offsets[li+1]:
                raise ValueError('empty original list map')
            continue
        ip = index.invlists.get_ids(li)
        cp = index.invlists.get_codes(li)
        try:
            ids = faiss.rev_swig_ptr(ip, size)
            if saved_rows is not None:
                old = saved_rows[saved_offsets[li]:saved_offsets[li+1]]
                if not np.array_equal(ids, old):
                    raise ValueError('actual list differs from original row map')
            at = np.searchsorted(rows, ids)
            valid = at < len(rows)
            matches = np.flatnonzero(valid)
            matches = matches[rows[at[matches]] == ids[matches]]
            where = at[matches]
            if found[where].any() or len(np.unique(where)) != len(where):
                raise ValueError('duplicate requested ID inside actual IVF')
            all_codes = faiss.rev_swig_ptr(cp, size * index.code_size).reshape(size, index.code_size)
            codes[where] = all_codes[matches]
            labels[where] = li
            offsets[where] = matches
            found[where] = True
        finally:
            index.invlists.release_ids(li, ip)
            index.invlists.release_codes(li, cp)
    if not found.all():
        raise ValueError('requested original row absent from actual IVF')
    coarse = index.quantizer.reconstruct_n(0, index.nlist).astype(np.float32)
    centers = faiss.vector_to_array(index.pq.centroids).reshape(index.pq.M, 256, index.pq.dsub).copy()
    return dict(rows=rows, codes=codes, labels=labels, offsets=offsets, coarse=coarse, centers=centers)


def decode_actual_ivf_original(book, positions, inverse, mean):
    """Canonical yhat has original FP32 decode-plus-coarse rounding."""
    pos = np.asarray(positions, np.int64)
    codes = book['codes'][pos]
    parts = book['centers'][np.arange(codes.shape[1])[None, :], codes]
    residual = parts.reshape(len(pos), -1)
    y = (residual + book['coarse'][book['labels'][pos]]).astype(np.float32)
    xhat = y.astype(np.float64) @ np.asarray(inverse, np.float64).T + np.asarray(mean, np.float32).astype(np.float64)
    return xhat, y


def reconstruction_context(book, matrix, mean, c):
    A = np.asarray(matrix, np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not np.isfinite(A).all():
        raise ValueError('finite square OPQ required')
    condition = float(np.linalg.cond(A))
    if condition > c['numerics']['max_matrix_condition']:
        raise ValueError('ill-conditioned original OPQ')
    B = np.linalg.inv(A)
    defect = float(np.max(np.abs(A @ B - np.eye(len(A)))))
    if defect > c['numerics']['inverse_identity_max']:
        raise ValueError('inverse identity check')
    sy = (np.linalg.norm(book['coarse'].astype(float), axis=1).max()
          + np.sqrt(np.sum(np.max(np.sum(book['centers'].astype(float)**2, axis=2), axis=1))))
    bf = float(np.linalg.norm(B))
    # Real xhat means the stored inverse B applied to the native decoded yhat.
    # This does not assume that rounded B is an exact inverse of the original A.
    eta = 16 * gamma(len(A)) * (c['numerics']['raw_norm_bound'] + bf * (1 + 2*U32) * sy + 1)
    return dict(inverse=B, mean=np.asarray(mean, np.float32), dimension=len(A),
        condition=condition, inverse_identity_defect=defect, inverse_frobenius=bf,
        decoded_norm_bound=float(sy), residual_arithmetic_error=float(eta),
        raw_norm_bound=c['numerics']['raw_norm_bound'],
        base_centers=book['centers'].astype(np.float64),base_coarse=book['coarse'].astype(np.float64))


def build_residual_geometry(raw, book, ctx, c, phase=lambda *a, **k: None):
    """G0 residuals/norms, completed before G1 and any side-codebook fitting."""
    start = time.perf_counter()
    n, d = len(book['rows']), ctx['dimension']
    errors = np.empty((n, d), np.float64)
    for lo in range(0, n, c['build_block']):
        hi = min(lo + c['build_block'], n)
        x = np.asarray(raw[book['rows'][lo:hi]], np.float64)
        if not np.isfinite(x).all() or np.linalg.norm(x, axis=1).max() > ctx['raw_norm_bound']:
            raise ValueError('source outside recorded native-score norm envelope')
        xhat, _ = decode_actual_ivf_original(book, np.arange(lo, hi), ctx['inverse'], ctx['mean'])
        errors[lo:hi] = x - xhat
        phase('residual_build', rows=hi, total=n)
    radii = outward_norm(errors, ctx['residual_arithmetic_error'])
    costs = [dict(stage='actual_residual_and_norm', seconds=time.perf_counter()-start, rows=n)]
    return radii,errors,costs


def build_residual_summaries(raw, book, ctx, fit_positions, c, phase=lambda *a, **k: None, prepared=None):
    """G2/G3 paid side information. No query labels participate in fitting."""
    import faiss
    n,d=len(book['rows']),ctx['dimension']
    radii,errors,costs=build_residual_geometry(raw,book,ctx,c,phase) if prepared is None else prepared
    summaries = {'norm': dict(kind='norm', radii=radii, payload=4)}
    for seed in c['sketch_seeds']:
        G = np.random.Generator(np.random.PCG64(seed)).standard_normal((max(c['dimensions']), d))
        for m in c['dimensions']:
            begin = time.perf_counter();R = G[:m] / np.sqrt(m)
            z = np.empty((n, m), np.float32)
            for lo in range(0, n, c['build_block']):
                z[lo:lo+c['build_block']] = (errors[lo:lo+c['build_block']] @ R.T).astype(np.float32)
            name = 'gaussian_{}_{}'.format(seed, m)
            summaries[name] = dict(kind='gaussian', radii=radii, sketch=z, R=R,
                payload=4+4*m, seed=seed, dimension=m)
            costs.append(dict(stage=name, seconds=time.perf_counter()-begin, rows=n))
            phase('sketch_build', name=name)
    sample = np.ascontiguousarray(errors[fit_positions], np.float32)
    for width, seed in zip(c['side_pq_bytes'], c['side_pq_seeds']):
        begin = time.perf_counter();pq = faiss.ProductQuantizer(d, width, 8)
        pq.cp.niter = c['side_pq_training']['niter'];pq.cp.nredo = c['side_pq_training']['nredo'];pq.cp.seed = seed
        pq.train(sample);trained = time.perf_counter()
        centers = faiss.vector_to_array(pq.centroids).reshape(width, 256, d//width).copy()
        codes = np.empty((n, width), np.uint8);remaining = np.empty(n, np.float32)
        for lo in range(0, n, c['build_block']):
            e = errors[lo:lo+c['build_block']]
            cc = pq.compute_codes(np.ascontiguousarray(e, np.float32))
            decoded = pq.decode(cc).astype(np.float64)
            codes[lo:lo+len(e)] = cc
            remaining[lo:lo+len(e)] = outward_norm(e-decoded, ctx['residual_arithmetic_error'])
        name = 'sidepq_{}'.format(width)
        summaries[name] = dict(kind='sidepq', codes=codes, centers=centers,
            centers64=centers.astype(np.float64),radii=remaining,
            centroid_norm_bound=float(np.sqrt(np.sum(np.max(np.sum(centers.astype(float)**2,axis=2),axis=1)))),
            payload=width+4, seed=seed, dimension=d)
        costs.append(dict(stage=name, fit_seconds=trained-begin,
            encoding_seconds=time.perf_counter()-trained, seconds=time.perf_counter()-begin, rows=n,
            fit_rows=len(sample), no_query_labels=True))
        phase('side_pq_build', name=name)
    return summaries, errors, costs


def code_scores(query, codes, centers):
    """FP64 LUT, all table construction/access paid in the caller's clock."""
    q = np.asarray(query, np.float64).reshape(len(centers), -1)
    table = np.einsum('md,mcd->mc', q, centers.astype(np.float64,copy=False))
    return np.sum(table[np.arange(len(centers))[None, :], codes], axis=1)


def base_scores(query, book, pos, ctx):
    q = np.asarray(query, np.float64)
    dual = ctx['inverse'].T @ q
    # A real-valued sum LUT cannot reproduce native coarse+code FP32 addition;
    # its error is explicitly enclosed, rather than declaring ADC identical.
    coarse_table = ctx['base_coarse'] @ dual
    values = (coarse_table[book['labels'][pos]]
              + code_scores(dual, book['codes'][pos], ctx['base_centers'])
              + float(q @ ctx['mean'].astype(float)))
    Q = np.linalg.norm(q) * (1 + 8*gamma(len(q)))
    bound = Q * ctx['inverse_frobenius'] * ctx['decoded_norm_bound'] * (2*U32 + 16*gamma(len(q)))
    return values, float(bound), float(Q)


def score_bounds(query, book, pos, ctx, summary, c):
    """Lower native-distance bounds, no unread raw values or true top-k inputs."""
    pos = np.asarray(pos, np.int64);q = np.asarray(query, np.float64)
    if q.shape != (ctx['dimension'],) or not np.isfinite(q).all():
        raise ValueError('finite original query required')
    if np.linalg.norm(q) == 0:
        return np.full(len(pos), -np.inf), dict(fallback='zero_query')
    base, base_error, Q = base_scores(q, book, pos, ctx)
    r = summary['radii'][pos].astype(float)
    if not np.isfinite(r).all() or np.any(r < 0):
        return np.full(len(pos), -np.inf), dict(fallback='invalid_radius')
    info = dict(fallback=None, real_base_round_envelope=base_error)
    if summary['kind'] == 'norm':
        upper = base + Q*r + base_error
    elif summary['kind'] == 'sidepq':
        correction = code_scores(q, summary['codes'][pos], summary.get('centers64',summary['centers']))
        side_norm = summary.get('centroid_norm_bound')
        if side_norm is None:side_norm=np.sqrt(np.sum(np.max(np.sum(summary['centers'].astype(float)**2,axis=2),axis=1)))
        upper = base + correction + Q*r + base_error + 16*gamma(len(q))*Q*side_norm
    elif summary['kind'] == 'gaussian':
        R = summary['R'];m = len(R);u = q/np.linalg.norm(q);h = R @ u;T = float(h @ h)
        if not np.isfinite(T) or T < c['numerics']['minimum_T']:
            return np.full(len(pos), -np.inf), dict(fallback='small_or_invalid_T')
        z = summary['sketch'][pos].astype(float)
        ahat = (z @ h)/T
        # Covers stored sketch rounding, prior residual arithmetic, and FP64
        # operations under the standard bounded-error model. Not a BLAS proof.
        Rf = np.linalg.norm(R)
        ze = Rf * (2*ctx['residual_arithmetic_error'] + (2*U32+16*gamma(len(q)))*(r+ctx['residual_arithmetic_error']))
        guard = ze/np.sqrt(T) + 64*gamma(len(q))*Rf*(r+1)/T
        z_alpha = norm.isf(c['delta']/c['k'])
        directional = np.minimum(r, ahat + guard + z_alpha*r/np.sqrt(m*T))
        upper = base + Q*directional + base_error + 16*gamma(len(q))*Q*r
        info.update(T=T, sketch_error_guard_max=float(np.max(guard)))
    else:
        raise ValueError('unknown residual summary')
    # Includes FP32 native dot and subtraction error; clip is nonexpansive.
    native_error = (gamma(len(q), U32)*Q*ctx['raw_norm_bound']
                    + 4*U32*(1+Q*ctx['raw_norm_bound']))
    lower = np.clip(1-upper-native_error, 0, 2)
    with np.errstate(over='ignore', invalid='ignore'):
        lower = np.nextafter(lower.astype(np.float32), np.float32(-np.inf)).astype(float)
    if not np.isfinite(lower).all():
        return np.full(len(pos), -np.inf), dict(fallback='nonfinite_bound')
    info.update(native_score_envelope=float(native_error), formal_certification=False)
    return lower, info


def scan_with_bounds(raw, query, ids, lower, k, block, pilot_positions=None):
    """Label-free scan unless the separate oracle explicitly supplies a pilot."""
    ids = cost.validate_ids(raw, ids, k)
    lower = np.asarray(lower, float)
    if lower.shape != ids.shape or np.isnan(lower).any() or block < 1:
        raise ValueError('valid lower-bound vector and block required')
    pilot = np.arange(k) if pilot_positions is None else np.asarray(pilot_positions, np.int64)
    if len(pilot) != k or len(np.unique(pilot)) != k or np.any(pilot < 0) or np.any(pilot >= len(ids)):
        raise ValueError('exactly k unique pilot positions')
    read = np.zeros(len(ids), bool);read[pilot] = True
    initial = ids[pilot];dist = cost.native.original_distances(raw[initial], query)
    best = cost.native.stable_top(dist, k, initial)
    by_id = dict(zip(map(int, initial), map(float, dist)))
    trace = [dict(positions=pilot.tolist(), threshold=float(max(dist)), pilot=True)]
    batches = 1
    for lo in range(0, len(ids), block):
        available = np.arange(lo, min(lo+block, len(ids)))
        tau = max(by_id[int(row)] for row in best)
        take = available[(~read[available]) & (lower[available] <= tau)]
        if len(take):
            selected = ids[take];d = cost.native.original_distances(raw[selected], query)
            read[take] = True;batches += 1
            pool = np.r_[best, selected]
            values = np.r_[[by_id[int(row)] for row in best], d].astype(np.float32)
            best = cost.native.stable_top(values, k, pool)
            by_id = dict(zip(map(int, pool), map(float, values)))
        trace.append(dict(positions=take.tolist(), threshold=float(tau), pilot=False))
    final = cost.native.original_distances(raw[best], query);batches += 1
    return dict(returned=best, distances=final, used=ids[read], read_positions=np.flatnonzero(read),
        trace=trace, gather_batches=batches, logical_vectors=int(read.sum()+k))


def scan_and_refine(raw, query, ids, book, pos, ctx, summary, c):
    start = time.perf_counter();lower, audit = score_bounds(query, book, pos, ctx, summary, c)
    after = time.perf_counter();r = scan_with_bounds(raw, query, ids, lower, c['k'], c['scan_block'])
    r.update(aux_seconds=after-start, scan_seconds=time.perf_counter()-after, bound_audit=audit)
    return r


def ideal_threshold_diagnostic(raw, query, ids, book, pos, ctx, summary, c, known_top_ids):
    # Truth is confined to this explicitly named oracle; all its raw reads and
    # online bound construction are still paid. It is not inference input.
    pilot = np.flatnonzero(np.isin(ids, known_top_ids))
    start = time.perf_counter();lower, audit = score_bounds(query, book, pos, ctx, summary, c)
    after = time.perf_counter();r = scan_with_bounds(raw, query, ids, lower, c['k'], len(ids), pilot)
    r.update(aux_seconds=after-start, scan_seconds=time.perf_counter()-after, bound_audit=audit, oracle_only=True)
    return r


def load_geometry_inputs(c, phase=lambda *a, **k: None):
    source_map(c)
    old = Path(c['source_run'])
    if cost.sha(old/'delivery_index.json') != c['source_index_sha256']:
        raise ValueError('closed S0 index identity')
    original = {r['path']: r for r in json.loads((old/'delivery_index.json').read_text())['files']}
    for entry in c['products']:
        if original.get(entry['path']) != entry:
            raise ValueError('G product not in original closed inventory')
        cost.verify_product(old, entry)
    data = cost.load_cost_inputs(cost.config(), phase)
    original_freeze=json.loads((old/'source_freeze.json').read_text())['native_binary_sha256']
    if data['binary_sha256']['numpy']!=original_freeze['numpy']:
        raise ValueError('original native NumPy binary differs')
    if any(original_freeze.get(name)!=digest for name,digest in data['binary_sha256']['faiss'].items()):
        raise ValueError('original native FAISS binary differs')
    if c['query_indices'] != data['indices']:
        raise ValueError('gate2 cohort must remain C0/C1 CPU cohort')
    data['truth'] = np.load(old/'published_truth.npy', allow_pickle=False)
    data['cutoffs'] = np.load(old/'published_cutoffs.npy', allow_pickle=False)
    data['global_native'] = np.load(old/'native_fp32_top100.npy', allow_pickle=False)
    if (data['truth'].shape != (128, 100) or data['cutoffs'].shape != (128,)
            or data['global_native'].shape != (128, 100)):
        raise ValueError('original saved quality identities')
    fit = np.load(old/'fit_rows.npy', allow_pickle=False)[:c['fit_rows']].astype(np.int64)
    if (len(fit) != c['fit_rows'] or len(np.unique(fit)) != len(fit)
            or fit.min() < 0 or fit.max() >= len(data['raw'])):
        raise ValueError('original corpus-only fit rows')
    candidate_union = np.unique(np.concatenate([data['candidates'][p][data['indices']].reshape(-1) for p in c['nprobe']]))
    wanted = np.union1d(candidate_union, fit)
    phase('extract_actual_IVF_codes', requested_rows=len(wanted))
    book = extract_actual_ivf_codes(data['index'], wanted,
        np.load(old/'ivf/opq64/list_rows.npy', mmap_mode='r', allow_pickle=False),
        np.load(old/'ivf/opq64/list_offsets.npy', allow_pickle=False))
    ctx = reconstruction_context(book, data['model']['matrix'], data['model']['mean'], c)
    # One first requested position per actual list, fixed before seeing queries.
    _, representatives = np.unique(book['labels'], return_index=True)
    import faiss
    maximum_decode_difference = 0.
    for at in representatives:
        expected = np.empty(data['index'].d, np.float32)
        data['index'].reconstruct_from_offset(int(book['labels'][at]), int(book['offsets'][at]), faiss.swig_ptr(expected))
        _, actual = decode_actual_ivf_original(book, [at], ctx['inverse'], ctx['mean'])
        maximum_decode_difference = max(maximum_decode_difference, float(np.max(abs(actual[0]-expected))))
        if not np.array_equal(actual[0], expected):
            raise RuntimeError('own-code decode differs from actual FAISS reconstruction')
    data.update(book=book, ctx=ctx, fit_positions=np.searchsorted(wanted, fit),
        union_rows=len(candidate_union), decode_checks=len(representatives),
        decode_max_difference=maximum_decode_difference)
    return data


def save_array(path, **arrays):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)


def save_summaries(output, book, ctx, summaries, costs, c):
    save_array(output/'summaries/actual_code_book.npz', **book)
    save_array(output/'summaries/shared_inverse.npz', inverse=ctx['inverse'], mean=ctx['mean'])
    ledger = []
    for name, s in summaries.items():
        arrays = {k: v for k, v in s.items() if isinstance(v, np.ndarray)}
        save_array(output/'summaries'/(name+'.npz'), **arrays)
        shared = sum(v.nbytes for k, v in arrays.items() if k in ('R', 'centers','centers64'))
        shared_total=int(shared+ctx['inverse'].nbytes+ctx['mean'].nbytes
            +ctx['base_centers'].nbytes+ctx['base_coarse'].nbytes+book['centers'].nbytes+book['coarse'].nbytes)
        metadata_per_row=book['rows'].dtype.itemsize+book['labels'].dtype.itemsize+book['offsets'].dtype.itemsize
        # The original codebook is shared with the unchanged main index. Inverse
        # and scalar metadata are NEW side information and always charged.
        ledger.append(dict(name=name, bytes_per_vector=s['payload'],
            full_N_payload_bytes=c['source']['rows']*s['payload'],
            local_rows=len(book['rows']), local_payload_bytes=len(book['rows'])*s['payload'],
            new_shared_matrix_codebook_bytes=shared_total,
            new_shared_scalar_fields=8, scalar_bytes=8*8,
            local_all_comparator_arrays_resident=True, full_N_encoding_measured=False,
            original_vectors_retained=True,request_workspaces_and_IDs_extra=True,
            local_packed_summary_proxy=True,full_N_summary_layout_latency_measured=False,
            explicit_local_rowmap_bytes=int(book['rows'].nbytes+book['labels'].nbytes+book['offsets'].nbytes),
            extra_local_original_code_copy_bytes=int(book['codes'].nbytes),
            full_N_conservative_prototype_code_and_metadata_bytes=int(c['source']['rows']*(book['codes'].shape[1]+metadata_per_row)),
            full_N_conservative_prototype_side_state_bytes=int(c['source']['rows']*(s['payload']+book['codes'].shape[1]+metadata_per_row)+shared_total+64),
            hypothetical_dense_code_label_cache_bytes=int(c['source']['rows']*(book['codes'].shape[1]+4)),
            alternative_dense_layout_unimplemented=True,
            LUT_workspace_bytes=int(8*(len(book['centers'])*256+(len(s.get('centers',[]))*256 if s['kind']=='sidepq' else 0))),
            bound_score_mask_workspaces_extra=True))
    cost.unique_json(output/'offline_build_costs.json', costs)
    cost.unique_json(output/'storage_ledger.json', ledger)


def quality_record(result, reference, truth, cutoff, global_native, k):
    rr = result['returned'];dd = result['distances']
    base = metrics.benchmark_recall(dd, cutoff, truth, rr, k)
    return dict(**base, candidate_topk_retention=len(set(map(int, rr)) & set(map(int, reference)))/k,
        preserves_candidate_topk=bool(np.array_equal(rr, reference)),
        global_native_retention=len(set(map(int, rr)) & set(map(int, global_native)))/k)


def geometry_audit(raw, query, ids, pos, data, errors, summaries, c, target):
    """Saved exact geometry is diagnostic only; never a controller argument."""
    q64 = np.asarray(query, float);d = cost.native.original_distances(raw[ids], query)
    returned = cost.native.stable_top(d, c['k'], ids)
    tau = float(np.partition(d, c['k']-1)[c['k']-1]);Q = float(np.linalg.norm(q64))
    e = errors[pos];actual_correction = e @ q64
    raw_values = np.asarray(raw[ids], np.float64);raw_dot = raw_values @ q64
    reconstruct_score = raw_dot-actual_correction
    _, y32 = decode_actual_ivf_original(data['book'], pos, data['ctx']['inverse'], data['ctx']['mean'])
    transformed = cost.native.transform_batch(data['model'], query.reshape(1, -1))[0]
    adc = y32 @ transformed
    r = summaries['norm']['radii'][pos].astype(float)
    gap = tau-d.astype(float)
    normalized = np.divide(gap, Q*r, out=np.zeros_like(gap), where=(Q*r)>0)
    fields = dict(ids=ids, native_distances=d, native_topk=returned,
        real_raw_dot=raw_dot, actual_ADC_reconstructed_dot=adc,
        original_reconstruction_dot=reconstruct_score, residual_correction=actual_correction,
        residual_radius=r, signed_native_gap=gap, normalized_gap=normalized,
        zero_residual=np.linalg.norm(e, axis=1)==0,
        cutoff_tie=d==tau, clip_boundary=(d==0)|(d==2))
    bound_rows = []
    for name, summary in summaries.items():
        lower, info = score_bounds(query, data['book'], pos, data['ctx'], summary, c)
        fields['lower_'+name] = lower
        violation = lower > d.astype(float)
        topmask = np.isin(ids, returned)
        bound_rows.append(dict(name=name, candidate_bound_violations=int(violation.sum()),
            true_topk_bound_violations=int(np.count_nonzero(violation & topmask)),
            any_true_topk_violation=bool(np.any(violation & topmask)), bound_audit=info,
            beneficial_threshold_skip=int(np.count_nonzero((lower>tau) & ~topmask)),
            native_tau=tau, formal_certification=False))
    save_array(target, **fields)
    band_rows=[]
    for radius in c['boundary_bands']:
        mask=np.abs(normalized)<=radius
        band_rows.append(dict(normalized_band=radius, rows=int(mask.sum()),
            radius_quantiles=np.quantile(r[mask],[0,.5,.9,.99,1]).tolist() if mask.any() else None))
    return returned, dict(rows=len(ids), native_cutoff=tau,
        radius_quantiles=np.quantile(r,[0,.5,.9,.99,1]).tolist(),
        absolute_normalized_gap_quantiles=np.quantile(abs(normalized),[0,.5,.9,.99,1]).tolist(),
        ADC_vs_original_reconstruction_max_difference=float(np.max(abs(adc-reconstruct_score))),
        boundary_bands=band_rows, bounds=bound_rows)


def paired_geometry_replay(raw, query, search, ids, book, pos, ctx, summaries, c, mode, oracle_ids=None):
    start=time.perf_counter();actual=search(query);searched=time.perf_counter()
    if not np.array_equal(actual, ids):
        raise RuntimeError('gate2 upstream candidates differ from sealed snapshot')
    if mode=='baseline' or mode.startswith('prefix_'):
        used=actual if mode=='baseline' else actual[:int(mode.split('_')[1])]
        rr,dd=cost.native.rerank(raw,query,used,c['k'])
        r=dict(returned=rr,distances=dd,used=used,read_positions=np.arange(len(used)),
            trace=None,gather_batches=2,logical_vectors=len(used)+c['k'],
            aux_seconds=0.,scan_seconds=None,bound_audit=None)
    elif mode=='fullread':
        r=scan_with_bounds(raw,query,actual,np.full(len(actual),-np.inf),c['k'],c['scan_block'])
        r.update(aux_seconds=0.,scan_seconds=None,bound_audit=None)
    elif mode.startswith('oracle:'):
        if oracle_ids is None:raise ValueError('explicit diagnostic oracle identity required')
        live_pos=np.searchsorted(book['rows'],actual)
        if np.any(live_pos>=len(book['rows'])) or not np.array_equal(book['rows'][live_pos],actual):
            raise RuntimeError('actual candidate missing from paid summary map')
        r=ideal_threshold_diagnostic(raw,query,actual,book,live_pos,ctx,summaries[mode[7:]],c,oracle_ids)
    else:
        live_pos=np.searchsorted(book['rows'],actual)
        if np.any(live_pos>=len(book['rows'])) or not np.array_equal(book['rows'][live_pos],actual):
            raise RuntimeError('actual candidate missing from paid summary map')
        r=scan_and_refine(raw,query,actual,book,live_pos,ctx,summaries[mode],c)
    end=time.perf_counter();r.update(candidates=actual,seconds=end-start,
        search_seconds=searched-start,rerank_seconds=end-searched)
    return r


def summarize_geometry(records,c):
    # Reuse only the unchanged paired-statistics implementation; new judgement
    # fields are scoped to this separate gate2 namespace, never an old result.
    summary=cost.summarize_paired(records,c)
    summary.update(gate2_started=True,tri_implemented=False,formal_certification=False,
        judgement='joint development geometry/cost audit; quality and added cost reported per fixed method',
        numerical_scope='bounded-error IEEE model and native audits; not hardware or fixed-seed query-risk certification')
    for g in summary['groups']:
        rs=[r for r in records if (r['nprobe'],r['role'],r['mode'])==(g['nprobe'],g['role'],g['mode'])]
        first={r['query_index']:r for r in rs}
        g['quality']={key:float(np.mean([r['quality'][key] for r in first.values()]))
            for key in ['recall','strict_id','candidate_topk_retention','global_native_retention']}
        g['queries_with_candidate_topk_loss']=sum(not r['quality']['preserves_candidate_topk'] for r in first.values())
        g['mean_unique_original_rows']=float(np.mean([r['unique_read_rows'] for r in first.values()]))
        g['oracle_only']=g['mode'].startswith('oracle:')
        g['no_formal_quality_claim']=True
    return summary


def progress_callback(output):
    """Shared production/tiny reporter; summary name is metadata, not stage."""
    output=Path(output)
    def phase(stage,**fields):
        cost.atomic_json(output/'progress.json',dict(phase=stage,**fields))
    return phase


def run_science(c, output):
    output=Path(output);initial=source_map(c)
    phase=progress_callback(output)
    data=load_geometry_inputs(c,phase);raw=data['raw'];book=data['book'];ctx=data['ctx']
    cost.unique_json(output/'input_receipt.json',dict(source_index_sha256=c['source_index_sha256'],
        original_source_stat=data['original_source_stat'],query_indices=data['indices'],
        source_verify_seconds=data['source_verify_seconds'],hardware=cost.cpu_identity(),
        binary_sha256=data['binary_sha256'],versions=data['versions'],source_map=initial,
        union_rows=data['union_rows'],candidate_and_fit_rows=len(book['rows']),
        decode_checks=data['decode_checks'],decode_max_difference=data['decode_max_difference'],
        inverse_condition=ctx['condition'],inverse_identity_defect=ctx['inverse_identity_defect'],
        residual_arithmetic_error=ctx['residual_arithmetic_error'],
        new_test_or_cert_payload_read=False,saved_opened_quality_only=True))
    phase('G0_residual_geometry')
    prepared=build_residual_geometry(raw,book,ctx,c,phase)
    radii,errors,offline=prepared
    phase('G1_fixed_candidate_boundaries');begin=time.perf_counter();g1=[]
    for probe in c['nprobe']:
        for qi in data['indices']:
            ids=data['candidates'][probe][qi];q=data['queries'][qi];pos=np.searchsorted(book['rows'],ids)
            rr,gr=geometry_audit(raw,q,ids,pos,data,errors,{'norm':dict(kind='norm',radii=radii,payload=4)},c,
                output/'G1_geometry'/('{}-{}.npz'.format(probe,qi)))
            cost.compare_exact(dict(candidates=ids,returned=rr,distances=cost.native.original_distances(raw[rr],q)),
                ids,data['expected'][probe,qi])
            entry=dict(nprobe=probe,query_index=qi,role=data['roles'][qi]['role'],**gr)
            cost.unique_json(output/'G1_geometry'/('{}-{}.json'.format(probe,qi)),entry);g1.append(entry)
            if any(b['candidate_bound_violations'] for b in gr['bounds']):
                raise RuntimeError('G1 deterministic numerical envelope violated')
    offline.append(dict(stage='G1_diagnostic_boundary_audit',seconds=time.perf_counter()-begin,query_probe_pairs=len(g1)))
    cost.unique_json(output/'G1_summary.json',g1)
    phase('G2_G3_fixed_summaries')
    summaries,errors,offline=build_residual_summaries(raw,book,ctx,data['fit_positions'],c,phase,prepared=prepared)
    save_summaries(output,book,ctx,summaries,offline,c)
    expected_modes=['baseline','fullread']+['prefix_'+str(n) for n in c['fixed_prefix_counts']]+list(summaries)+['oracle:'+n for n in summaries]
    if c['modes']!=expected_modes:raise ValueError('fixed complete comparator grid changed')
    records=[];geometry=[];checks=0;total=0;rng=np.random.Generator(np.random.PCG64(c['schedule_seed']))
    with (output/'calls.jsonl').open('x') as stream:
        for probe in c['nprobe']:
            data['index'].nprobe=probe
            for qi in rng.permutation(data['indices']):
                qi=int(qi);q=data['queries'][qi];ids=data['candidates'][probe][qi]
                pos=np.searchsorted(book['rows'],ids)
                if not np.array_equal(book['rows'][pos],ids):raise RuntimeError('candidate summary map')
                search=lambda query:cost.stored.search_backend(data['model'],query,6400,index=data['index'])
                reference=cost.full_callback(raw,q,search,c['k'])
                cost.compare_exact(reference,ids,data['expected'][probe,qi]);checks+=1
                rr,gr=geometry_audit(raw,q,ids,pos,data,errors,summaries,c,
                    output/'geometry'/('{}-{}.npz'.format(probe,qi)))
                if not np.array_equal(rr,reference['returned']):raise RuntimeError('native full-C geometry reference')
                geometry.append(dict(nprobe=probe,query_index=qi,role=data['roles'][qi]['role'],**gr))
                cost.unique_json(output/'geometry'/('{}-{}.json'.format(probe,qi)),geometry[-1])
                if any(b['candidate_bound_violations'] for b in gr['bounds'] if not b['name'].startswith('gaussian_')):
                    cost.unique_json(output/'deterministic_bound_failure.json',geometry[-1])
                    raise RuntimeError('deterministic numerical envelope violated; retain and stop')
                first={}
                for repeat in range(-c['timing']['warmup'],c['timing']['repeats']):
                    for mode in rng.permutation(c['modes']):
                        mode=str(mode)
                        r=paired_geometry_replay(raw,q,search,ids,book,pos,ctx,summaries,c,mode,rr)
                        if mode in ('baseline','fullread'):
                            cost.compare_exact(r,ids,data['expected'][probe,qi])
                        else:
                            er,ed=cost.native.rerank(raw,q,r['used'],c['k'])
                            if not np.array_equal(er,r['returned']) or not np.array_equal(ed,r['distances']):
                                raise RuntimeError('native block/subset score mismatch; no tolerance change')
                        checks+=1
                        signature=cost.fingerprint([r['read_positions'].tolist(),r['returned'].tolist(),r['distances'].tolist()])
                        if mode in first and first[mode]!=signature:raise RuntimeError('fixed-policy repeat changed')
                        first[mode]=signature
                        if repeat>=0:
                            row=dict(nprobe=probe,query_index=qi,query_id=data['roles'][qi]['id'],role=data['roles'][qi]['role'],
                                mode=mode,repeat=repeat,seconds=r['seconds'],search_seconds=r['search_seconds'],
                                rerank_seconds=r['rerank_seconds'],aux_seconds=r['aux_seconds'],
                                scan_seconds=r['scan_seconds'],gather_batches=r['gather_batches'],
                                logical_original_payload_bytes=r['logical_vectors']*c['source']['dimension']*4,
                                unique_read_rows=len(r['read_positions']),read_positions_sha256=cost.fingerprint(r['read_positions'].tolist()),
                                read_trace_sha256=cost.fingerprint(r['trace']),
                                returned=r['returned'].tolist(),distances=r['distances'].tolist(),
                                candidate_sha256=cost.fingerprint(ids.tolist()),summary_kind=mode,
                                bound_audit=r['bound_audit'],quality=quality_record(r,rr,data['truth'][qi],data['cutoffs'][qi],data['global_native'][qi],c['k']),
                                actual_dram_bytes=None,actual_ssd_bytes=None,physical_requests=None)
                            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush();records.append(row);total+=1
                            if repeat==0:
                                cost.unique_json(output/'traces'/('{}-{}-{}.json'.format(probe,qi,mode.replace(':','_'))),
                                    dict(trace=r['trace'],signature=signature,oracle_only=mode.startswith('oracle:'),read_positions=r['read_positions'].tolist()))
                cost.unique_json(output/'query_observations'/('{}-{}.json'.format(probe,qi)),cost.process_observation())
                phase('paired_replay',nprobe=probe,query_index=qi,formal_calls=total,conformance_checks=checks)
    cost.unique_json(output/'geometry_summary.json',geometry)
    cost.unique_json(output/'summary.json',summarize_geometry(records,c))
    if initial!=source_map(c):raise RuntimeError('gate2 source changed during execution')
    path=Path(c['source']['path']);st=data['original_source_stat']
    if path.stat().st_size!=st['bytes'] or path.stat().st_mtime_ns!=st['mtime_ns']:
        raise RuntimeError('read-only source metadata changed')
    return dict(status='complete',formal_calls=total,conformance_checks=checks,
        source_map_unchanged=True,source_stat_unchanged=True,tri_implemented=False,
        formal_certification=False,no_automatic_gate3=True)
