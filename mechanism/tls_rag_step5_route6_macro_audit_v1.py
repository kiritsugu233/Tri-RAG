"""Read-only bindings and ideal-versus-machine geometry diagnostics for R6-M0."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from mechanism import tls_rag_step5_route6_bridge_v1 as bridge


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_model_diagnostic(actual, d2, c, m):
    actual, d2 = np.asarray(actual, dtype=float), np.asarray(d2, dtype=float)
    if actual.shape != (len(d2), len(d2)) or not np.all(np.isfinite(actual)):
        raise ValueError('Gram shape/nonfinite')
    if not np.allclose(actual, actual.T, rtol=0, atol=2.**-40) or np.any(d2 <= 0):
        raise ValueError('Gram symmetry/scales')
    if c not in (0., .5) or isinstance(m, bool) or int(m) != m or m < 1:
        raise ValueError('model')
    ideal = np.sqrt(d2[:, None]*d2[None, :])*((1-c)*np.eye(len(d2))+c)
    # A common positive lift scale cancels in rankings. Fix alignment using row0,
    # without fitting geometry to risk labels or tuning a scale for smaller error.
    scale = float(actual[0, 0]/d2[0])
    if scale <= 0:
        raise ValueError('nonpositive lift scale')
    aligned = actual/scale
    values, vectors = np.linalg.eigh(ideal)
    if np.min(values) <= 0 or np.min(np.linalg.eigvalsh(aligned)) <= 0:
        raise ValueError('positive definite covariance required')
    inverse = (vectors/np.sqrt(values))@vectors.T
    error = inverse@(aligned-ideal)@inverse
    rho = float(np.max(np.abs(np.linalg.eigvalsh(error))))
    frobenius = float(np.linalg.norm(error))
    return dict(common_scale=scale, max_aligned_entry_error=float(np.max(np.abs(aligned-ideal))),
                relative_operator_error=rho, relative_frobenius_error=frobenius,
                tv_diagnostic=None if rho >= 1 else min(1., math.sqrt(m)*frobenius/math.sqrt(8*(1-rho))),
                roundoff_certified=False, scope='ideal-vs-floating geometry; not an arbitrary-Gram approximation theorem')


def load_frozen_macro_inputs(measurement_dir, index_path, cfg):
    root, index_path = Path(measurement_dir).resolve(), Path(index_path).resolve()
    binding = cfg['input_binding']
    if sha(index_path) != binding['index_sha256']:
        raise ValueError('frozen index SHA mismatch')
    index = json.loads(index_path.read_text())
    entries = {}
    for row in index['files']:
        relative = row['path']
        if Path(relative).is_absolute() or '..' in Path(relative).parts or relative in entries:
            raise ValueError('invalid/duplicate indexed path')
        entries[relative] = row
    verified = {}
    for relative in binding['selected_paths']:
        path = (root/relative).resolve()
        if root not in path.parents or relative not in entries:
            raise ValueError('missing or escaped frozen input')
        row = entries[relative]
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('frozen input SHA mismatch: '+relative)
        verified[relative] = row
    freeze = json.loads((root/'source_freeze.json').read_text())
    if freeze['commit'] != binding['source_commit'] or freeze['config'] != binding['old_config'] or freeze['sources'] != binding['old_sources']:
        raise ValueError('old source/config identity mismatch')
    models = []
    for c, kind in ((0., 'independent'), (.5, 'equicorrelated')):
        for k in cfg['k']:
            geometry = 'macro-k%d-%s' % (k, kind)
            ids = json.loads((root/('geometry/'+geometry+'/ids.json')).read_text())
            expected = ['route6:'+geometry+':corpus:'+str(i) for i in range(cfg['N'])]
            if ids['row_to_id'] != expected or ids['query_id'] != 'route6:'+geometry+':query' or len(set(expected)) != len(expected):
                raise ValueError('stable ID/row mapping mismatch')
            gram = np.load(root/('geometry/'+geometry+'/gram.npy'), allow_pickle=False)
            diagonal = np.diag(gram)
            if len(diagonal) != cfg['N'] or not np.all(np.diff(diagonal) > 0) or diagonal[k-1] >= diagonal[k]:
                raise ValueError('strict original GT boundary required')
            d2 = 1+np.arange(cfg['N'])/100.
            for m in cfg['m']:
                bounds = json.loads((root/('bounds/%s-m%d.json' % (geometry, m))).read_text())
                unit = json.loads((root/('units/%s-m%d-gram.json' % (geometry, m))).read_text())
                events = [(k*f, ell) for f in cfg['budget_factors'] for ell in sorted({1, (k+1)//2, k})]
                rows = [row for row in unit['events'] if row['type'] == 'omission']
                if (bounds['geometry'], bounds['m'], unit['geometry'], unit['m'], unit['path'], unit['trials']) != (geometry, m, geometry, m, 'gram', 65536):
                    raise ValueError('model/old stream identity mismatch')
                if [(r['M'], r['ell']) for r in rows] != events or [(r['M'], r['ell']) for r in bounds['bounds']] != events or len({r['id'] for r in rows}) != len(events):
                    raise ValueError('macro event identity mismatch')
                if any((r['geometry'], r['m'], r['path']) != (geometry, m, 'gram') for r in rows):
                    raise ValueError('event source mismatch')
                models.append(dict(geometry=geometry, k=k, m=m, c=c, d2=d2.tolist(), events=events,
                                   geometry_diagnostic=geometry_model_diagnostic(gram, d2, c, m),
                                   old_bounds=bounds, old_events=rows, gram=gram))
    return models, dict(index_sha256=sha(index_path), files=verified,
                        source_commit=freeze['commit'], role='released development consistency reference; no certification or fitting')


def compare_reference_records(model, reference):
    p, _, _ = bridge.geometry_probabilities(model['gram'], model['k'], model['m'])
    k, n = p.shape
    eta = [float(np.sum(np.delete(row, selected))) for row, selected in zip(p, model['old_bounds']['boundary_indices_relative_to_outsiders'])]
    result = []
    for row, old, cp in zip(reference['events'], model['old_bounds']['bounds'], model['old_events']):
        M, ell = row['M'], row['ell']
        if (M, ell) != (old['M'], old['ell']) or (M, ell) != (cp['M'], cp['ell']):
            raise ValueError('comparison event mismatch')
        q_upper = [bridge.marginal_bounds(p[:, j], ell)[1] for j in range(n)]
        pair_reaggregated = bridge.marginal_bounds(q_upper, M-k+ell)[1]
        h = M-k+1
        boundary_floor = min(1., sum(min(1., value/h) for value in eta)/ell)
        lo, hi = row['discretization_bracket']
        tv = model['geometry_diagnostic']['tv_diagnostic']
        overlap = None if tv is None else lo-tv <= cp['upper'] and hi+tv >= cp['lower']
        result.append(dict(row, old_event_id=cp['id'], old_gram_cp=[cp['lower'], cp['upper']],
                           old_cross_first_moment=old['cross_first_moment'], old_boundary_s1=old['boundary_s1'],
                           old_boundary_s2=old['boundary_s2_coordinate_cp'], pair_only_reaggregated=pair_reaggregated,
                           old_boundary_expression_floor=boundary_floor,
                           ideal_machine_tv_diagnostic=tv, diagnostic_cp_overlap=overlap,
                           reference_is_not_a_statistical_certification=True))
    return result
