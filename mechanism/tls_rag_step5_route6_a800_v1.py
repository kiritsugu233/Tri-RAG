"""Scheme6 finite geometry, shared Gaussian sampling and lazy A800 execution."""
from __future__ import annotations

from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from mechanism import tls_rag_step5_route6_bridge_v1 as bridge
from tri_rag_harness.tri_law import tri_law_probability


class GPUBlocked(RuntimeError):
    pass


def geometry_specs():
    specs = []
    for alpha in (0.0, .5):
        for kind in ('independent', 'equicorrelated', 'cycle_plus', 'cycle_minus'):
            specs.append({'id': f'single-{kind}-alpha{alpha}', 'kind': kind, 'alpha': alpha, 'beta': 2.0, 'k': 1, 'N': 5})
    for beta in (1.05, 1.25, 4.0):
        specs.append({'id': f'single-gap-{beta}', 'kind': 'independent', 'alpha': 0.0, 'beta': beta, 'k': 1, 'N': 5})
    specs.append({'id': 'single-repeated', 'kind': 'repeated', 'alpha': 0.0, 'beta': 2.0, 'k': 1, 'N': 5})
    for k in (4, 8):
        for kind in ('independent', 'equicorrelated'):
            specs.append({'id': f'macro-k{k}-{kind}', 'kind': kind, 'k': k, 'N': 64})
    return specs


def gram_factor(gram, rank, tolerance):
    """Declared rank is a geometry contract, never inferred by adding jitter."""
    gram = np.asarray(gram, np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not np.isfinite(gram).all() or not np.array_equal(gram, gram.T):
        raise ValueError('finite symmetric Gram required')
    values, vectors = np.linalg.eigh(gram)
    scale = max(1.0, float(np.linalg.norm(gram, 2)))
    if values[0] < -tolerance*scale or rank < 1 or rank > len(gram) or values[-rank] <= 0:
        raise ValueError('invalid PSD/rank contract')
    if len(gram) > rank and np.max(np.abs(values[:-rank])) > tolerance*scale:
        raise ValueError('unexpected nonzero eigenvalues')
    factor = vectors[:, -rank:]*np.sqrt(values[-rank:])
    if not np.allclose(factor@factor.T, gram, rtol=tolerance, atol=tolerance*scale):
        raise ValueError('Gram factor audit failed')
    return factor


def build_geometry(spec, config):
    n, k = bridge.integer(spec['N'], 'N'), bridge.integer(spec['k'], 'k')
    if not k < n or (k > 1 and spec['kind'] not in ('independent', 'equicorrelated')):
        raise ValueError('invalid geometry specification')
    tol = config['numeric']['geometry_audit_rtol']
    if k == 1:
        count = n-1
        kind = spec['kind']
        residual = np.eye(count)
        if kind == 'equicorrelated':
            residual = .5*np.ones((count, count))+.5*np.eye(count)
        elif kind.startswith('cycle_'):
            for i in range(count-1):
                residual[i, i+1] = residual[i+1, i] = .25
            residual[0, -1] = residual[-1, 0] = .25 if kind == 'cycle_plus' else -.25
        elif kind == 'repeated':
            residual = np.ones((count, count))
        elif kind != 'independent':
            raise ValueError('unknown geometry')
        alpha, beta = spec['alpha'], spec['beta']
        gram = np.empty((n, n))
        gram[0, 0] = 1
        gram[0, 1:] = gram[1:, 0] = alpha
        gram[1:, 1:] = alpha**2+(beta-alpha**2)*residual
        rank = 2 if kind == 'repeated' else n
        if kind == 'repeated':
            vectors = np.zeros((n, 2))
            vectors[0, 0] = 1
            vectors[1:, 1] = np.sqrt(beta)
        else:
            vectors = np.linalg.cholesky(gram)
    else:
        d2 = 1+np.arange(n)/100
        residual = np.eye(n) if spec['kind'] == 'independent' else .5*np.ones((n, n))+.5*np.eye(n)
        gram = np.sqrt(d2[:, None]*d2[None, :])*residual
        rank = n
        vectors = np.linalg.cholesky(gram)
    if config['dimension'] < rank+1:
        raise ValueError('ambient dimension insufficient for unit lift')
    c = -np.diag(gram)/2
    qmin, _, _, _ = np.linalg.lstsq(vectors, c, rcond=None)
    if not np.allclose(vectors@qmin, c, rtol=tol, atol=config['numeric']['geometry_audit_atol']):
        raise ValueError('unit lift range condition fails')
    squared = float(qmin@qmin)
    scale = np.sqrt(1/(4*(1+squared)))
    query = np.zeros(config['dimension'])
    query[:rank] = scale*qmin
    query[rank] = np.sqrt(1-scale**2*squared)
    corpus = np.broadcast_to(query, (n, len(query))).copy()
    corpus[:, :rank] += scale*vectors
    displacements = corpus-query
    actual = displacements@displacements.T
    if not np.allclose(actual, gram*scale**2, rtol=tol, atol=config['numeric']['geometry_audit_atol']) or not np.allclose(np.sum(corpus**2, axis=1), 1, rtol=0, atol=tol) or abs(query@query-1) > tol:
        raise ValueError('unit lift reconstruction failed')
    if not np.all(np.diag(actual)[k:] > np.diag(actual)[:k].max()):
        raise ValueError('strict GT boundary failed')
    factor = gram_factor(actual, rank, tol)
    ids = [f"route6:{spec['id']}:corpus:{i}" for i in range(n)]
    return {'spec': spec, 'query': query, 'corpus': corpus, 'gram': actual,
            'factor': factor, 'ids': ids, 'query_id': f"route6:{spec['id']}:query", 'rank': rank,
            'fingerprint': hashlib.sha256(query.tobytes()+corpus.tobytes()+factor.tobytes()).hexdigest()}


def seed_for(config, geometry_id, m, path):
    if path not in ('coordinate', 'gram'):
        raise ValueError('unknown random stream')
    payload = [config['namespace'], config['seeds'][path], geometry_id, int(m), path]
    return int.from_bytes(hashlib.sha256(json.dumps(payload, separators=(',', ':'), ensure_ascii=True).encode()).digest(), 'big')


def sample_shared_projection(rng, geometry, m, count, path):
    # C-order flattened draws preserve identities when only batch size changes.
    width = len(geometry['query']) if path == 'coordinate' else geometry['rank']
    if path not in ('coordinate', 'gram'):
        raise ValueError('unknown sample path')
    return rng.standard_normal((count, m, width))/np.sqrt(m)


def _rounding_factor(length, dtype):
    u = np.finfo(dtype).eps/2
    return length*u/(1-length*u)


def _evaluate_numpy(matrices, geometry, path, dtype=np.float64):
    p = np.asarray(matrices, dtype=dtype)
    m = p.shape[1]
    if path == 'coordinate':
        x, q = geometry['corpus'].astype(dtype), geometry['query'].astype(dtype)
        px, pq = p@x.T, p@q
        delta = px-pq[:, :, None]
        # Standard IEEE dot-product forward-error model; not an interval library.
        errors = _rounding_factor(p.shape[2], dtype)*((np.abs(p)@np.abs(x).T)+(np.abs(p)@np.abs(q))[:, :, None])
        errors += (np.finfo(dtype).eps/2)*(np.abs(px)+np.abs(pq[:, :, None]))
    elif path == 'gram':
        x = geometry['factor'].astype(dtype)
        delta = p@x.T
        errors = _rounding_factor(p.shape[2], dtype)*(np.abs(p)@np.abs(x).T)
    else:
        raise ValueError('unknown projection path')
    squared = np.sum(delta*delta, axis=1, dtype=dtype)
    bound = _rounding_factor(m+1, dtype)*np.sum(delta*delta, axis=1, dtype=dtype)
    bound += np.sum(2*np.abs(delta)*errors+errors**2, axis=1, dtype=dtype)
    if not np.isfinite(squared).all() or not np.isfinite(bound).all():
        raise FloatingPointError('nonfinite projected distances')
    return squared.astype(float), bound.astype(float)


def torch_backend():
    """No torch import/installation is required for CPU offline tests."""
    workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if workspace not in (':4096:8', ':16:8'):
        raise GPUBlocked('unsupported deterministic cuBLAS workspace configuration')
    try:
        import torch
    except ImportError as exc:
        raise GPUBlocked('PyTorch absent; server preflight/lock required') from exc
    if not torch.cuda.is_available():
        raise GPUBlocked('CUDA unavailable; no CPU fallback for A800 execution')
    if 'A800' not in torch.cuda.get_device_name(0).upper():
        raise GPUBlocked('visible cuda:0 is not an A800')
    # Do not mix the old and new TF32 control APIs.
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
        mode = {'api': 'fp32_precision', 'matmul': torch.backends.cuda.matmul.fp32_precision}
        if mode['matmul'] != 'ieee':
            raise GPUBlocked('IEEE matmul configuration did not stick')
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if torch.backends.cuda.matmul.allow_tf32:
            raise GPUBlocked('TF32 configuration did not stick')
        mode = {'api': 'allow_tf32', 'matmul': False}
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    mode.update(deterministic_algorithms=True, cublas_workspace=workspace)
    return torch, mode


def evaluate_projection(matrices, geometry, path, device='cpu', dtype='float64'):
    if dtype not in ('float64', 'float32'):
        raise ValueError('unsupported floating-point precision')
    if device == 'cpu':
        begin = time.perf_counter()
        squared, errors = _evaluate_numpy(matrices, geometry, path, np.float64 if dtype == 'float64' else np.float32)
        return squared, errors, {'wall_seconds': time.perf_counter()-begin, 'h2d_seconds': 0.0, 'kernel_seconds': 0.0, 'd2h_seconds': 0.0}
    if device != 'cuda':
        raise ValueError('unknown device')
    torch, _ = torch_backend()
    typ = torch.float64 if dtype == 'float64' else torch.float32
    np_type = np.float64 if dtype == 'float64' else np.float32
    torch.cuda.synchronize()
    begin = time.perf_counter()
    p = torch.as_tensor(matrices, dtype=typ, device='cuda:0')
    x = torch.as_tensor(geometry['corpus'] if path == 'coordinate' else geometry['factor'], dtype=typ, device='cuda:0')
    q = torch.as_tensor(geometry['query'], dtype=typ, device='cuda:0') if path == 'coordinate' else None
    torch.cuda.synchronize()
    uploaded = time.perf_counter()
    if path == 'coordinate':
        px, pq = p@x.T, p@q
        delta = px-pq[:, :, None]
        errors = _rounding_factor(p.shape[2], np_type)*(torch.abs(p)@torch.abs(x).T+(torch.abs(p)@torch.abs(q))[:, :, None])
        errors += (np.finfo(np_type).eps/2)*(torch.abs(px)+torch.abs(pq[:, :, None]))
    else:
        delta = p@x.T
        errors = _rounding_factor(p.shape[2], np_type)*(torch.abs(p)@torch.abs(x).T)
    squared = torch.sum(delta*delta, dim=1)
    bound = _rounding_factor(p.shape[1]+1, np_type)*torch.sum(delta*delta, dim=1)
    bound += torch.sum(2*torch.abs(delta)*errors+errors**2, dim=1)
    torch.cuda.synchronize()
    computed = time.perf_counter()
    squared, bound = squared.cpu().numpy().astype(float), bound.cpu().numpy().astype(float)
    torch.cuda.synchronize()
    finished = time.perf_counter()
    if not np.isfinite(squared).all() or not np.isfinite(bound).all():
        raise FloatingPointError('nonfinite GPU distances')
    return squared, bound, {'wall_seconds': finished-begin, 'h2d_seconds': uploaded-begin, 'kernel_seconds': computed-uploaded, 'd2h_seconds': finished-computed}


def budgets(geometry, config):
    k = geometry['spec']['k']
    return config['single_budgets'] if k == 1 else [k*f for f in config['macro_budget_factors']]


def boundary_indices(geometry, m, config):
    p, _, _ = bridge.geometry_probabilities(geometry['gram'], geometry['spec']['k'], m)
    return [np.lexsort((np.arange(len(row)), -row))[:config['boundary_size']].tolist() for row in p]


def _boundary_remainder_expectation(p, selected):
    """Sum the actual complement; subtraction can erase or negate its mass.

    ``selected`` is the trusted row-index output of boundary_indices. No clipping
    or tolerance changes the nonnegative remainder contract.
    """
    p = bridge.probabilities(p)
    outside = np.ones(len(p), dtype=bool)
    outside[np.asarray(selected, dtype=np.intp)] = False
    return float(p[outside].sum())


def unit_events(geometry, m, path, config):
    k, n = geometry['spec']['k'], geometry['spec']['N']
    selected = boundary_indices(geometry, m, config)
    events = []
    for i in range(k):
        for j in range(k, n):
            events.append({'type': 'marginal', 'i': i, 'j': j})
        for a, b in combinations(selected[i], 2):
            events.append({'type': 'joint', 'i': i, 'j': k+a, 'l': k+b})
    for M in budgets(geometry, config):
        for i in range(k):
            events.append({'type': 'target_missing', 'M': M, 'i': i})
        for ell in sorted(set((1, (k+1)//2, k))):
            events.append({'type': 'omission', 'M': M, 'ell': ell})
    for event in events:
        event.update(geometry=geometry['spec']['id'], m=m, path=path)
        event['id'] = bridge.fingerprint(event)
    return events


def build_catalogue(geometries, config):
    events = []
    for geometry in geometries:
        for m in config['single_m'] if geometry['spec']['k'] == 1 else config['macro_m']:
            for path in ('coordinate', 'gram'):
                events.extend(unit_events(geometry, m, path, config))
    if len({event['id'] for event in events}) != len(events):
        raise ValueError('duplicate catalogue IDs')
    return events


def event_observations(squared, error, geometry, events):
    n, k = geometry['spec']['N'], geometry['spec']['k']
    squared, error = np.asarray(squared), np.asarray(error)
    if squared.ndim != 2 or squared.shape[1] != n or error.shape != squared.shape or not np.isfinite(squared).all() or not np.isfinite(error).all() or np.any(error < 0):
        raise ValueError('invalid distance/error arrays')
    low, high = squared-error, squared+error
    definite = high[:, None, :] < low[:, :k, None]
    possible = low[:, None, :] <= high[:, :k, None]
    for i in range(k):
        definite[:, i, i] = possible[:, i, i] = False
    rank_low, rank_high = 1+definite.sum(axis=2), 1+possible.sum(axis=2)
    lower, upper = [], []
    for event in events:
        kind = event['type']
        if kind == 'marginal':
            a, b = definite[:, event['i'], event['j']], possible[:, event['i'], event['j']]
        elif kind == 'joint':
            a = definite[:, event['i'], event['j']] & definite[:, event['i'], event['l']]
            b = possible[:, event['i'], event['j']] & possible[:, event['i'], event['l']]
        elif kind == 'target_missing':
            a, b = rank_low[:, event['i']] > event['M'], rank_high[:, event['i']] > event['M']
        elif kind == 'omission':
            a = (rank_low > event['M']).sum(axis=1) >= event['ell']
            b = (rank_high > event['M']).sum(axis=1) >= event['ell']
        else:
            raise ValueError('unknown event')
        lower.append(a)
        upper.append(b)
    return np.array(lower).T, np.array(upper).T, {'rank_lower': rank_low, 'rank_upper': rank_high,
        'outside_definite': np.packbits(definite[:, :, k:], axis=2, bitorder='little'),
        'outside_possible': np.packbits(possible[:, :, k:], axis=2, bitorder='little')}


def sample_unit(geometry, m, path, config, catalogue_size, device, write_batch, check):
    rng = np.random.Generator(np.random.PCG64(seed_for(config, geometry['spec']['id'], m, path)))
    events = unit_events(geometry, m, path, config)
    low_count, high_count = np.zeros(len(events), np.int64), np.zeros(len(events), np.int64)
    start = time.perf_counter()
    timing = {'rng_seconds': 0.0, 'h2d_seconds': 0.0, 'kernel_seconds': 0.0, 'd2h_seconds': 0.0, 'projection_wall_seconds': 0.0, 'count_seconds': 0.0, 'artifact_seconds': 0.0}
    audits = []
    for offset in range(0, config['samples_per_path'], config['batch']):
        check()
        count = min(config['batch'], config['samples_per_path']-offset)
        t = time.perf_counter()
        matrices = sample_shared_projection(rng, geometry, m, count, path)
        timing['rng_seconds'] += time.perf_counter()-t
        squared, error, clocks = evaluate_projection(matrices, geometry, path, device)
        timing['projection_wall_seconds'] += clocks['wall_seconds']
        for key in ('h2d_seconds', 'kernel_seconds', 'd2h_seconds'):
            timing[key] += clocks[key]
        if offset == 0:
            audit_count = min(config['reference_samples'], count)
            ref, ref_error = _evaluate_numpy(matrices[:audit_count], geometry, path)
            if np.any(np.abs(squared[:audit_count]-ref) > error[:audit_count]+ref_error):
                raise ArithmeticError('CPU/GPU forward error audit failed')
            fp32, _ = _evaluate_numpy(matrices[:audit_count], geometry, path, np.float32)
            audits.append({'samples': audit_count, 'max_cpu_gpu_difference': float(np.abs(squared[:audit_count]-ref).max()),
                          'max_fp32_difference': float(np.abs(fp32-ref).max()), 'fp32_is_diagnostic_only': True})
        t = time.perf_counter()
        lo, hi, raw = event_observations(squared, error, geometry, events)
        low_count += lo.sum(axis=0)
        high_count += hi.sum(axis=0)
        timing['count_seconds'] += time.perf_counter()-t
        t = time.perf_counter()
        raw.update(sample_ids=np.arange(offset, offset+count), event_definite=np.packbits(lo, axis=1, bitorder='little'),
                   event_possible=np.packbits(hi, axis=1, bitorder='little'))
        write_batch(offset, raw)
        timing['artifact_seconds'] += time.perf_counter()-t
    lower, upper = bridge.clopper_pearson_envelope(low_count, high_count, config['samples_per_path'], catalogue_size, config['family_alpha'])
    records = [dict(event, definite_count=int(lo), possible_count=int(hi), lower=float(a), upper=float(b),
                    decisions={str(delta): bridge.decision((a, b), delta) for delta in config['risk_thresholds']})
               for event, lo, hi, a, b in zip(events, low_count, high_count, lower, upper)]
    return {'geometry': geometry['spec']['id'], 'm': m, 'path': path, 'events': records,
            'trials': config['samples_per_path'], 'catalogue_size': catalogue_size, 'timing': timing,
            'total_wall_seconds': time.perf_counter()-start, 'audits': audits,
            'numerics': 'IEEE forward-error model plus CPU audit, not formal directed-rounding certificate'}


def analyze_unit(geometry, m, config, coordinate_records):
    """Known-input bounds; S2 CP inputs are distinctly tagged and paid offline."""
    begin = time.perf_counter()
    k, n = geometry['spec']['k'], geometry['spec']['N']
    p, beta, rho = bridge.geometry_probabilities(geometry['gram'], k, m)
    selected = boundary_indices(geometry, m, config)
    records = []
    for M in budgets(geometry, config):
        h = M-k+1
        classical, conditional, s2, plain, boundary_s1 = [], [], [], [], []
        conditional_details = []
        s2_warnings = []
        for i in range(k):
            classical.append(bridge.marginal_bounds(p[i], h)[1])
            plain.append(min(1.0, float(p[i].sum()/h)))
            alpha = rho[i]*np.sqrt(beta[i])
            detail = bridge.conditional_marginal_bounds(beta[i], alpha, m, h, config['conditional_bins'])
            conditional_details.append(detail)
            conditional.append(min(classical[-1], detail['upper']))
            ids = selected[i]
            local = p[i, ids]
            eta = _boundary_remainder_expectation(p[i], ids)
            u = np.zeros((len(ids), len(ids)))
            for a, b in combinations(range(len(ids)), 2):
                j, l = sorted((k+ids[a], k+ids[b]))
                matched = [event for event in coordinate_records if event['type'] == 'joint' and event['i'] == i and sorted((event['j'], event['l'])) == [j, l]]
                if len(matched) != 1:
                    raise ValueError('missing presealed boundary joint interval')
                u[a, b] = u[b, a] = matched[0]['upper']
            candidates, baseline_candidates = [], []
            for t in range(h):
                threshold = h-t
                base = bridge.marginal_bounds(local, threshold)[1]
                moment = bridge.second_order_bounds(local, u, threshold)
                if moment['status'] == 'inconsistent_joint_constraints':
                    s2_warnings.append({'target_row': i, 'threshold': threshold, 'status': moment['status']})
                local_upper = min(base, moment['cantelli'], moment['factorial2'])
                candidates.append(bridge.boundary_remainder_bound(local_upper, eta, h, t))
                baseline_candidates.append(bridge.boundary_remainder_bound(base, eta, h, t))
            s2.append(min(candidates))
            boundary_s1.append(min(baseline_candidates))
        for ell in sorted(set((1, (k+1)//2, k))):
            row = {'M': M, 'ell': ell, 'cross_first_moment': bridge.cross_inversion_bound(float(p.sum()), n, k, M, ell),
                'target_markov_union': min(1.0, sum(plain)/ell), 'classical_marginal_union': min(1.0, sum(classical)/ell),
                'conditional_marginal_union': min(1.0, sum(conditional)/ell), 'boundary_s1': min(1.0, sum(boundary_s1)/ell),
                'boundary_s2_coordinate_cp': min(1.0, sum(s2)/ell), 'conditional_details': conditional_details,
                'S2_extra_input': 'coordinate-stream simultaneous joint CP upper endpoints; assess against independent Gram stream',
                'roundoff_certified': False, 's2_input_warnings': s2_warnings}
            if k == 1:
                row['classical_lower'] = bridge.marginal_bounds(p[0], M)[0]
            records.append(row)
    return {'geometry': geometry['spec']['id'], 'm': m, 'bounds': records, 'input_probability_count': int(p.size),
            'risk_compute_seconds': time.perf_counter()-begin, 'boundary_indices_relative_to_outsiders': selected}


def scalar_grid_audit(config):
    rows = []
    for m in config['scalar_grid']['m']:
        for beta in config['scalar_grid']['beta']:
            for rho in config['scalar_grid']['absolute_rho']:
                value = bridge.scalar_law(beta, rho, m)
                reference = float(tri_law_probability(beta, rho, m))
                probability = float(value['probability'])
                difference = abs(probability-reference)
                tol = config['numeric']['scalar_atol']+config['numeric']['scalar_rtol']*reference
                if difference > tol or float(value['power_upper'])+tol < reference or float(value['exponential_upper'])+tol < reference:
                    raise ArithmeticError('scalar law/envelope numerical audit failed')
                negative = bridge.scalar_law(beta, -rho, m)
                if probability != float(negative['probability']):
                    raise ArithmeticError('rho symmetry failed')
                rows.append({'m': m, 'beta': beta, 'absolute_rho': rho, 'reference': reference,
                    'probability': probability, 'gamma': float(value['gamma']), 'absolute_difference': difference,
                    'power_upper': float(value['power_upper']), 'exponential_upper': float(value['exponential_upper'])})
    return rows
