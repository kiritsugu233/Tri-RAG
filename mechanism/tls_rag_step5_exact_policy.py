"""Matched exact-projection ablations, frozen selection, and one development eval.

Feature functions receive current-prefix sessions only. No truth, labels or
future prefix enters a state or a controller. Cached replay is never latency.
"""
from __future__ import annotations

import json
from time import perf_counter

import numpy as np

from tri_rag_harness.lid import estimate_lid_from_squared_distances
from tri_rag_harness.projection import project_rows
from tri_rag_harness.tls_rag_step3 import _observed_pair, _pair_summary, _distortion_summary
from tri_rag_harness.tls_rag_step5_policy import BASE_NAMES
from tri_rag_harness.utils import fingerprint
from mechanism.tls_rag_step5_audit import ExactScan, ResourceStop, save

D_NAMES = ('core_log_distortion_q90', 'shell_log_distortion_q90')
T_NAMES = ('boundary_risk_mean', 'boundary_risk_max', 'frontier_risk_mean', 'frontier_risk_max')


class PrefixSession:
    def __init__(self, reader, query, projected_query, cfg):
        self.reader, self.query, self.qp, self.cfg = reader, query, projected_query, cfg
        self.pool, self.original_cache, self.projected_cache = [], {}, {}
        self.stage = -1

    def expand(self, rows, distances):
        started = perf_counter()
        ids = self.reader.id_array[np.asarray(rows)].tolist()
        if (len(ids) <= len(self.pool) or ids[:len(self.pool)] != self.pool
                or len(ids) != len(set(ids))):
            raise ValueError('exact prefixes must nest and grow')
        new = ids[len(self.pool):]
        self.pool = ids
        self.projected_cache.update(zip(ids, map(float, distances)))
        rr = [self.reader.row_by_id[i] for i in new]
        diff = np.asarray(self.reader.corpus[rr], dtype=np.float64)-self.query
        original = np.einsum('ij,ij->i', diff, diff)
        self.original_cache.update(zip(new, map(float, original)))
        scored = perf_counter()
        ranked = sorted(self.pool, key=lambda i: (self.original_cache[i], i))
        self.stage += 1
        return {'stage': self.stage, 'budget': len(ids), 'candidate_ids': ids.copy(),
                'new_ids': new, 'reranked_ids': ranked, 'context_ids': ranked[:self.cfg['k_ctx']],
                'original_distances': len(new),
                'gather_score_ms': (scored-started)*1000,
                'rerank_ms': (perf_counter()-scored)*1000}


def feature_state(session, record, family, cfg):
    if family not in cfg['families'] or record['budget'] == cfg['budget_grid'][-1]:
        raise ValueError('unknown family or terminal state must not compute features')
    started = perf_counter()
    pool, context = record['candidate_ids'], record['context_ids']
    if set(pool) != set(session.original_cache) or len(pool) != record['budget']:
        raise ValueError('feature interface received future or missing scores')
    ordered = sorted(pool, key=lambda i: (session.projected_cache[i], i))
    count = min(cfg['features']['sample_size'], len(ordered))
    sample = [ordered[i] for i in np.linspace(0, len(ordered)-1, count, dtype=int)]
    shell = record['new_ids'][-cfg['features']['shell_size']:]
    sample = list(dict.fromkeys(sample+context+shell))
    original = np.asarray([session.original_cache[i] for i in pool])
    projected = np.asarray([session.projected_cache[i] for i in pool])
    positive = original > 0
    distortion = projected[positive]/original[positive]
    vectors = {i: np.asarray(session.reader.corpus[session.reader.row_by_id[i]], dtype=np.float64)
               for i in sample}
    vv = np.asarray([vectors[i] for i in sample])
    pair = vv[:,None,:]-vv[None,:,:]
    pair_d = np.einsum('ijd,ijd->ij', pair, pair)[np.triu_indices(len(sample), 1)]
    so = np.sort(original)
    lid = estimate_lid_from_squared_distances(so, s_lid=min(cfg['features']['lid_neighbors'],len(so)),
        min_neighbors=3, clip_min=1., clip_max=200., duplicate_tolerance=0., fallback=200.)
    base = [float(len(pool)), float(record['stage']), float(original.min()), float(original.max()),
            float(original.mean()), session.original_cache[context[-1]],
            float(so[-1]-so[-2]) if len(pool)>1 else 0.,
            float(distortion.mean()) if len(distortion) else 0.,
            float(distortion.max()) if len(distortion) else 0.,
            float(pair_d.mean()) if len(pair_d) else 0., float(np.count_nonzero(pair_d == 0)),
            float(lid.clipped), float(lid.valid)]
    base_ms = (perf_counter()-started)*1000
    d_values, t_values, d_valid, t_valid = [], [], True, True
    t_ms, d_ms, pair_count = 0., 0., 0
    if '+D' in family:
        at = perf_counter()
        core_d = _distortion_summary(context, session.projected_cache, session.original_cache)
        shell_d = _distortion_summary(shell, session.projected_cache, session.original_cache)
        d_values, d_valid = [core_d.q90, shell_d.q90], bool(core_d.valid and shell_d.valid)
        d_ms = (perf_counter()-at)*1000
    if '+T' in family:
        at = perf_counter()
        boundary = context[-1]
        bp = [_observed_pair(boundary, i, session.query, vectors, cfg['m_prime'])
              for i in sample if i not in set(context)]
        fp = [_observed_pair(c, s, session.query, vectors, cfg['m_prime'])
              for c in context for s in shell if c != s]
        bs, fs = _pair_summary(bp), _pair_summary(fp)
        t_values = [bs.probability_mean, bs.probability_maximum, fs.probability_mean, fs.probability_maximum]
        t_valid = bool(bs.valid_pair_count and fs.valid_pair_count
            and bs.zero_pair_count+fs.zero_pair_count+bs.nonfinite_pair_count+fs.nonfinite_pair_count == 0
            and bs.collinear_pair_count+fs.collinear_pair_count == 0)
        t_ms, pair_count = (perf_counter()-at)*1000, len(bp)+len(fp)
    state = {'stage': record['stage'], 'budget': record['budget'], 'base': base,
             'D': d_values, 'T': t_values,
             'base_valid': bool(np.all(positive) and np.all(np.isfinite(base))),
             'D_valid': d_valid, 'T_valid': t_valid,
             'sample_ids': sample, 'triplet_pair_count': pair_count,
             'timing_ms': {'base': base_ms, 'D': d_ms, 'T': t_ms}}
    return state


def values(state, family):
    if family not in ('B', 'B+D', 'B+T', 'B+D+T'):
        raise ValueError('invalid family')
    x = list(state['base'])
    if '+D' in family:
        x += state['D']
    if '+T' in family:
        x += state['T']
    expected = 13 + 2*('+D' in family) + 4*('+T' in family)
    if len(x) != expected:
        raise ValueError('feature schema/future feature mismatch')
    valid = (state['base_valid'] and ('+D' not in family or state['D_valid'])
             and ('+T' not in family or state['T_valid']))
    return np.asarray(x), bool(valid and np.all(np.isfinite(x)))


def fit_models(examples, cfg):
    models = {}
    for family in cfg['families']:
        for stage in range(len(cfg['budget_grid'])-1):
            xx, yy = [], []
            for state, target in examples:
                if state['stage'] != stage:
                    continue
                x, valid = values(state, family)
                if valid:
                    xx.append(x);yy.append(target)
            key = family+':'+str(stage)
            if len(xx) < cfg['minimum_fit']:
                models[key] = None
                continue
            x, y = np.asarray(xx), np.asarray(yy)
            if np.any(~np.isfinite(y)) or np.any((y < 0)|(y > 1)):
                raise ValueError('invalid retention targets')
            mean, scale = x.mean(0), x.std(0)
            scale = np.where(scale > cfg['scale_floor'], scale, 1.)
            design = np.column_stack((np.ones(len(x)), (x-mean)/scale))
            penalty = np.eye(design.shape[1])*cfg['ridge'];penalty[0,0]=0.
            coef = np.linalg.pinv(design.T@design+penalty)@design.T@y
            names = list(BASE_NAMES)+(list(D_NAMES) if '+D' in family else [])+(list(T_NAMES) if '+T' in family else [])
            models[key] = {'mean': mean.tolist(), 'scale': scale.tolist(),
                           'coefficients': coef.tolist(), 'n': len(xx), 'names': names}
    return models


def predict(state, family, models):
    x, valid = values(state, family)
    model = models.get(family+':'+str(state['stage']))
    if not valid or model is None:
        return None
    coef = np.asarray(model['coefficients'])
    return float(coef[0]+((x-model['mean'])/model['scale'])@coef[1:])


def calibrate(examples, models, cfg):
    margins, residuals = {}, {}
    for family in cfg['families']:
        for stage in range(len(cfg['budget_grid'])-1):
            key = family+':'+str(stage)
            residuals[key] = []
            for state, target in examples:
                if state['stage'] == stage:
                    score = predict(state, family, models)
                    if score is not None:
                        residuals[key].append(score-target)
            margins[key] = max(0., float(np.quantile(residuals[key], cfg['residual_quantile'], method='higher'))) if len(residuals[key]) >= cfg['minimum_fit'] else None
    return margins, residuals


def adjusted(state, family, models, margins):
    raw = predict(state, family, models)
    margin = margins.get(family+':'+str(state['stage']))
    return float(np.clip(raw-margin, 0, 1)) if raw is not None and margin is not None else None


def decide(state, policy, models, margins, cfg):
    if policy['family'] not in cfg['families'] or policy['threshold'] not in cfg['thresholds']:
        raise ValueError('unregistered controller')
    stage = state['stage']
    if not 0 <= stage < len(cfg['budget_grid'])-1 or cfg['budget_grid'][stage] != state['budget']:
        raise ValueError('terminal/no-feature or wrong grid state')
    score = adjusted(state, policy['family'], models, margins)
    return {'stop': score is not None and score >= policy['threshold'], 'score': score}


def metrics(record, truth, positives, cfg):
    t, c = set(truth['exact_top_k']), set(record['candidate_ids'])
    rerank_retention = len(t & set(record['reranked_ids'][:cfg['k_gt']]))/cfg['k_gt']
    retention = len(t & c)/cfg['k_gt']
    if retention != rerank_retention:
        raise ValueError('retention/rerank identity failed')
    exact_hit = float(bool(set(truth['exact_top_k'][:cfg['k_ctx']]) & set(positives)))
    if exact_hit != truth['exact_hit']:
        raise ValueError('saved exact Hit/context mismatch')
    hit = float(bool(set(record['context_ids']) & set(positives)))
    return {'retention': retention, 'hit': hit, 'exact_hit': exact_hit,
            'hit_delta': hit-exact_hit, 'original_distances': record['budget'],
            'pilot_distances': cfg['budget_grid'][0],
            'expansion_distances': record['budget']-cfg['budget_grid'][0]}


def trajectory(reader, group, qid, cfg, ledger, prefix=None):
    q = reader.query(group, qid)
    qp = project_rows(q[None], reader.matrix)[0]
    if prefix is None:
        scan = ExactScan(reader.id_array, reader.projected, cfg['block_rows'], ledger.check)
        rr, dd, _ = scan.search(qp, cfg['budget_grid'][-1])
    else:
        rr, dd = prefix
    session = PrefixSession(reader, q, qp, cfg)
    stages = []
    for budget in cfg['budget_grid']:
        # Only this prefix enters the session; no later scores/IDs are exposed.
        rec = session.expand(rr[:budget], dd[:budget])
        if budget != cfg['budget_grid'][-1]:
            rec['state'] = feature_state(session, rec, 'B+D+T', cfg)
        stages.append(rec)
    return {'query_id': qid, 'stages': stages}


def examples(trajectories):
    return [(s['state'], s['metrics']['retention']) for tr in trajectories for s in tr['stages'] if 'state' in s]


def replay(tr, policy, models, margins, cfg):
    if policy is None:
        return None
    for s in tr['stages']:
        if 'state' not in s:
            return {**s['metrics'], 'budget': s['budget'], 'reason': 'terminal_without_quality_assertion'}
        if decide(s['state'], policy, models, margins, cfg)['stop']:
            return {**s['metrics'], 'budget': s['budget'], 'reason': 'score_stop'}
    raise AssertionError('missing terminal fallback')


def pipeline(reader, group, qid, cfg, ledger, policy=None, fixed=None, models=None, margins=None):
    if (policy is None) == (fixed is None):
        raise ValueError('one fixed/adaptive method required')
    started = perf_counter()
    q = reader.query(group, qid)
    qp = project_rows(q[None], reader.matrix)[0]
    projected_at = perf_counter()
    scan = ExactScan(reader.id_array, reader.projected, cfg['block_rows'], ledger.check)
    maximum = fixed if fixed is not None else cfg['budget_grid'][-1]
    rr, dd, _ = scan.search(qp, maximum)
    searched_at = perf_counter()
    session = PrefixSession(reader, q, qp, cfg)
    budgets = [fixed] if fixed is not None else cfg['budget_grid']
    recs, feature_ms, controller_ms = [], 0., 0.
    for budget in budgets:
        rec = session.expand(rr[:budget], dd[:budget])
        recs.append(rec)
        if policy is not None and budget != cfg['budget_grid'][-1]:
            at = perf_counter()
            state = feature_state(session, rec, policy['family'], cfg)
            feature_ms += (perf_counter()-at)*1000
            at = perf_counter()
            decision = decide(state, policy, models, margins, cfg)
            controller_ms += (perf_counter()-at)*1000
            if decision['stop']:
                break
    elapsed = (perf_counter()-started)*1000
    return {'record': recs[-1], 'original_distances': sum(r['original_distances'] for r in recs),
            'pipeline_ms': elapsed, 'projection_ms': (projected_at-started)*1000,
            'scan_and_selection_ms': (searched_at-projected_at)*1000,
            'selection_ms': scan.last_selection_ms,
            'gather_score_ms': sum(r['gather_score_ms'] for r in recs),
            'rerank_ms': sum(r['rerank_ms'] for r in recs),
            'features_ms': feature_ms, 'controller_ms': controller_ms}


def interval(a, b=None, kind='difference', cfg=None):
    a = np.asarray(a, dtype=float)
    b = np.zeros_like(a) if b is None else np.asarray(b, dtype=float)
    if a.shape != b.shape or len(a) < 2 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('finite paired query blocks required')
    def statistic(aa, bb):
        if kind == 'difference':
            return float(aa.mean()-bb.mean())
        denominator = np.percentile(bb, 95) if kind == 'p95_ratio' else bb.mean()
        numerator = np.percentile(aa, 95) if kind == 'p95_ratio' else aa.mean()
        return None if denominator <= 0 else float(numerator/denominator)
    point = statistic(a, b)
    if point is None:
        return {'point': None, 'lower': None, 'upper': None, 'reason': 'zero denominator'}
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    result = []
    for _ in range(cfg['bootstrap_replicates']):
        ix = rng.integers(len(a), size=len(a))
        stat = statistic(a[ix], b[ix])
        if stat is None:
            return {'point': point, 'lower': None, 'upper': None, 'reason': 'bootstrap zero denominator'}
        result.append(stat)
    alpha = cfg['bootstrap_alpha']
    return {'point': point, 'lower': float(np.quantile(result, alpha/2, method='lower')),
            'upper': float(np.quantile(result, 1-alpha/2, method='higher')),
            'method': 'paired whole-query-block percentile bootstrap; descriptive development interval'}


def summarize(items):
    return {key: float(np.mean([r[key] for r in items])) for key in
            ['retention','hit','hit_delta','original_distances','pilot_distances','expansion_distances']}


def quality_ok(items, cfg, fixed=None):
    s = summarize(items)
    q = cfg['quality']
    return bool(s['retention'] >= q['retention'] and s['hit_delta'] >= -q['hit_drop']
            and (fixed is None or np.mean([a['retention']-b['retention'] for a,b in zip(items, fixed)]) >= -q['retention_drop']))


def prediction_report(trajectories, models, margins, cfg):
    reports, paired = {}, {}
    for family in cfg['families']:
        errors, scores, targets, valid_queries = [], [], [], []
        query_raw_mae = []
        for tr in trajectories:
            ee, rr = [], []
            for s in tr['stages'][:-1]:
                pred = adjusted(s['state'], family, models, margins)
                raw = predict(s['state'], family, models)
                target = s['metrics']['retention']
                ee.append(None if pred is None else abs(pred-target))
                rr.append(None if raw is None else abs(raw-target))
                if pred is not None:
                    errors.append(pred-target);scores.append(pred);targets.append(target)
            valid_queries.append(all(e is not None for e in ee))
            query_raw_mae.append(float(np.mean(rr)) if all(e is not None for e in rr) else None)
            paired.setdefault(tr['query_id'], {})[family] = float(np.mean(ee)) if all(e is not None for e in ee) else None
        e, score, target = np.asarray(errors), np.asarray(scores), np.asarray(targets)
        ece = 0.
        if len(score):
            bins = np.minimum((score*10).astype(int), 9)
            ece = sum(np.sum(bins==b)/len(score)*abs(float(np.mean(score[bins==b]-target[bins==b])))
                      for b in range(10) if np.any(bins==b))
        reports[family] = {'mae': float(np.mean(np.abs(e))) if len(e) else None,
            'rmse': float(np.sqrt(np.mean(e*e))) if len(e) else None,
            'calibration_error_10_equal_width_bins': ece if len(e) else None,
            'invalid_stage_fraction': 1-len(e)/(len(trajectories)*3),
            'complete_valid_queries': int(sum(valid_queries)),
            'raw_mae_complete_queries': float(np.mean([v for v in query_raw_mae if v is not None])) if any(v is not None for v in query_raw_mae) else None}
    use = [v for v in paired.values() if v['B+D+T'] is not None and v['B+D'] is not None]
    primary = interval([v['B+D+T'] for v in use], [v['B+D'] for v in use], 'ratio', cfg) if len(use)>1 else {'point':None,'lower':None,'upper':None}
    primary['paired_complete_queries'] = len(use)
    supporting = [v for v in paired.values() if v['B+T'] is not None and v['B'] is not None]
    secondary = interval([v['B+T'] for v in supporting], [v['B'] for v in supporting], 'ratio', cfg) if len(supporting)>1 else None
    return {'families': reports, 'primary_adjusted_mae_ratio': primary, 'supporting_B_T_vs_B': secondary,
            'per_query_mae': paired, 'equal_stage_weights': [1/3]*3,
            'missing_predictions': 'report invalids; positive prediction gate requires full paired coverage'}


def phase_b(reader, ledger, cfg, cached, binding):
    ledger.begin('phase_b_preseal')
    started = perf_counter()
    data = {}
    for group in ['fit', 'calibration', 'selection']:
        truth, labels = reader.truth(group), reader.labels(group)
        result = []
        with (ledger.root/(group+'.jsonl')).open('x') as f:
            for n, qid in enumerate(reader.group_ids(group)):
                ledger.check()
                tr = trajectory(reader, group, qid, cfg, ledger, cached.get(qid))
                for s in tr['stages']:
                    s['metrics'] = metrics(s, truth[qid], labels[qid], cfg)
                f.write(json.dumps(tr, sort_keys=True, allow_nan=False)+'\n');f.flush()
                result.append(tr)
                if (n+1) % 16 == 0:
                    ledger.status('running', group=group, completed_queries=n+1)
        data[group] = result
    models = fit_models(examples(data['fit']), cfg)
    margins, residuals = calibrate(examples(data['calibration']), models, cfg)
    save(ledger.root/'models.json', models, True)
    save(ledger.root/'margins.json', {'margins':margins,'residuals':residuals}, True)
    # Selection uses real fixed pipelines. Replay of trajectories supplies quality,
    # not an online timing claim. Fixed methods extract only their own budget.
    timing = {str(m): [] for m in cfg['budget_grid']}
    fixed_records, headroom = [], []
    rng = np.random.default_rng(cfg['timing_seed'])
    base_order = rng.permutation(cfg['budget_grid']).tolist()
    selection_ids = reader.group_ids('selection')
    with (ledger.root/'selection_fixed_timing.jsonl').open('x') as f:
        for n, qid in enumerate(selection_ids):
            ledger.check()
            order = base_order[n%4:]+base_order[:n%4]
            measured = {}
            for m in order:
                pipe = pipeline(reader,'selection',qid,cfg,ledger,fixed=m)
                measured[str(m)] = {k:v for k,v in pipe.items() if k != 'record'}
                timing[str(m)].append(pipe['pipeline_ms'])
            row = {'query_id':qid,'order':order,'methods':measured}
            f.write(json.dumps(row, sort_keys=True)+'\n');f.flush()
            fixed_records.append(row)
            if n < cfg['timing']['headroom_queries']:
                # Conservative upper: all fixed-256 selection, score and rerank
                # work could disappear. The shared distance scan cannot disappear.
                s = data['selection'][n]['stages'][0]
                at = perf_counter()
                decide(s['state'], {'family':'B+D+T','threshold':cfg['thresholds'][0]}, models,margins,cfg)
                controller = (perf_counter()-at)*1000
                comp = s['state']['timing_ms']
                mandatory = sum(comp.values())+controller
                p = measured[str(cfg['budget_grid'][-1])]
                removable = (p['pipeline_ms']-p['projection_ms']-p['scan_and_selection_ms']
                             +p['selection_ms'])
                headroom.append({'query_id':qid,'mandatory_pilot_features_control_ms':mandatory,
                                 'maximum_removable_ms':removable})
            if n == cfg['timing']['headroom_queries']-1:
                gap = interval([r['mandatory_pilot_features_control_ms'] for r in headroom],
                               [r['maximum_removable_ms'] for r in headroom], cfg=cfg)
                save(ledger.root/'headroom.json', {'records':headroom,'mandatory_minus_removable_ms':gap,
                    'no_headroom': gap['lower'] is not None and gap['lower']>0,
                    'measurement': 'components; not an adaptive online latency measurement'}, True)
                elapsed = perf_counter()-ledger.started
                future_selection = (len(selection_ids)-n-1)*np.mean([sum(x['methods'][str(m)]['pipeline_ms'] for m in cfg['budget_grid']) for x in fixed_records])/1000
                if elapsed+future_selection > cfg['limits']['seconds']:
                    raise ResourceStop('selection throughput forecast exceeds cumulative ceiling')
            if (n+1)%16 == 0:
                ledger.status('running',group='fixed_selection_timing',completed_queries=n+1)
    fixed_candidates = {}
    for j, m in enumerate(cfg['budget_grid']):
        rr = [tr['stages'][j]['metrics'] for tr in data['selection']]
        quality = summarize(rr)
        quality.update(pilot_distances=0., expansion_distances=0.)
        fixed_candidates[str(m)] = {'quality':quality,'eligible':quality_ok(rr,cfg),
                                    'mean_pipeline_ms':float(np.mean(timing[str(m)]))}
    eligible = [m for m in cfg['budget_grid'] if fixed_candidates[str(m)]['eligible']]
    fixed = min(eligible,key=lambda m:(fixed_candidates[str(m)]['mean_pipeline_ms'],m)) if eligible else None
    fixed_items = [tr['stages'][cfg['budget_grid'].index(fixed)]['metrics'] for tr in data['selection']] if fixed else None
    selected, candidate_summaries = {}, {}
    for family in cfg['families']:
        candidates = []
        for threshold in cfg['thresholds']:
            policy = {'family':family,'threshold':threshold}
            pp = [replay(tr,policy,models,margins,cfg) for tr in data['selection']]
            ident = fingerprint(policy)[:16]
            ok = fixed_items is not None and quality_ok(pp,cfg,fixed_items)
            candidate_summaries[ident] = {'policy':policy, 'eligible':ok, 'metrics':summarize(pp)}
            if ok:
                candidates.append((summarize(pp)['original_distances'],ident,policy))
        selected[family] = min(candidates,key=lambda x:(x[0],x[1]))[2] if candidates else None
    selection = {'fixed_budget':fixed,'fixed_candidates':fixed_candidates,
                 'selected_policies':selected,'candidate_summaries':candidate_summaries}
    save(ledger.root/'selection.json',selection,True)
    payload = {'sampling':fingerprint(reader.groups), 'models':fingerprint(models),
               'margins':fingerprint(margins), 'selection':selection,
               'analysis_and_sources':binding, 'preseal_files':{name:__import__('hashlib').sha256((ledger.root/name).read_bytes()).hexdigest()
                   for name in ['sampling.json','models.json','margins.json','selection.json','headroom.json']}}
    ledger.freeze(payload)
    ledger.authorize_evaluation()
    ledger.begin('phase_b_evaluation')
    truth, labels = reader.truth('evaluation'), reader.labels('evaluation')
    evaluation, per_query = [], []
    with (ledger.root/'evaluation.jsonl').open('x') as f:
        for n,qid in enumerate(reader.group_ids('evaluation')):
            ledger.check()
            tr = trajectory(reader,'evaluation',qid,cfg,ledger)
            for s in tr['stages']:
                s['metrics'] = metrics(s,truth[qid],labels[qid],cfg)
            tr['policies'] = {fam:replay(tr,policy,models,margins,cfg) for fam,policy in selected.items()}
            tr['fixed'] = {**tr['stages'][cfg['budget_grid'].index(fixed)]['metrics'],
                           'pilot_distances':0, 'expansion_distances':0} if fixed else None
            f.write(json.dumps(tr,sort_keys=True,allow_nan=False)+'\n');f.flush()
            evaluation.append(tr)
            per_query.append({'query_id':qid,'methods':tr['policies'],'fixed':tr['fixed']})
            if (n+1)%16 == 0:
                ledger.status('running',group='frozen_evaluation',completed_queries=n+1)
    prediction = prediction_report(evaluation,models,margins,cfg)
    save(ledger.root/'prediction.json',prediction,True)
    save(ledger.root/'evaluation_metrics.json',per_query,True)
    hr = json.loads((ledger.root/'headroom.json').read_text())
    latency = {'status':'unmeasured_no_headroom' if hr['no_headroom'] else 'unmeasured_no_candidate'}
    if not hr['no_headroom'] and fixed is not None and selected['B+D+T'] is not None and selected['B+D'] is not None:
        ledger.begin('phase_b_pipeline_timing')
        methods = ['B+D+T','B+D','fixed']
        for qid in selection_ids[:cfg['timing']['warmup_queries']]:
            for method in methods:
                pipeline(reader,'selection',qid,cfg,ledger, fixed=fixed if method=='fixed' else None,
                         policy=selected.get(method), models=models,margins=margins)
        times = {m: np.zeros((len(evaluation),cfg['timing']['repeats'])) for m in methods}
        timing_rng = np.random.default_rng(cfg['timing_seed'])
        with (ledger.root/'pipeline_timing.jsonl').open('x') as f:
            for repeat in range(cfg['timing']['repeats']):
                # Balance rotations over methods; shuffle query order per repeat.
                base = timing_rng.permutation(methods).tolist()
                for step,n in enumerate(timing_rng.permutation(len(evaluation))):
                    qid = evaluation[n]['query_id']
                    order = base[(step+repeat)%3:]+base[:(step+repeat)%3]
                    record = {'query_id':qid,'repeat':repeat,'order':order,'methods':{}}
                    for method in order:
                        ledger.check()
                        pipe = pipeline(reader,'evaluation',qid,cfg,ledger, fixed=fixed if method=='fixed' else None,
                            policy=selected.get(method),models=models,margins=margins)
                        expected = evaluation[n]['fixed'] if method=='fixed' else evaluation[n]['policies'][method]
                        actual = metrics(pipe['record'],truth[qid],labels[qid],cfg)
                        if actual['original_distances'] != expected['original_distances'] or actual['retention'] != expected['retention'] or actual['hit'] != expected['hit']:
                            raise ValueError('timed pipeline differs from frozen-policy replay')
                        times[method][n,repeat] = pipe['pipeline_ms']
                        record['methods'][method] = {k:v for k,v in pipe.items() if k!='record'}
                    f.write(json.dumps(record,sort_keys=True)+'\n');f.flush()
                    if (step+1)%16==0:
                        ledger.status('running',group='pipeline_timing',repeat=repeat,completed_queries=step+1)
        latency = {'status':'measured_warm_single_thread_cloud_backed', 'methods':{
            m:{'mean_ms':float(v.mean()),'pooled_call_p95_ms':float(np.percentile(v,95))} for m,v in times.items()},
            'comparisons':{m:{'mean_ratio':interval(times['B+D+T'],times[m],'ratio',cfg),
                              'p95_ratio':interval(times['B+D+T'],times[m],'p95_ratio',cfg)} for m in ['B+D','fixed']}}
        save(ledger.root/'latency.json',latency,True)
    primary = prediction['primary_adjusted_mae_ratio']
    prediction_pass = (primary['point'] is not None and primary['upper'] is not None
        and primary['paired_complete_queries']==len(evaluation)
        and primary['point'] <= 1-cfg['gates']['mae_reduction'] and primary['upper']<1)
    outcome, comparison = {}, {}
    for fam in cfg['families']:
        rr = [tr['policies'][fam] for tr in evaluation]
        if rr[0] is not None:
            outcome[fam] = {'means':summarize(rr), 'stop_histogram':{
                str(m):sum(r['budget']==m for r in rr) for m in cfg['budget_grid']}}
        else:
            outcome[fam] = {'status':'no_selection_candidate'}
    if fixed is not None:
        outcome['fixed'] = {'budget':fixed,'means':summarize([tr['fixed'] for tr in evaluation])}
    candidate_pass = False
    if fixed is not None and selected['B+D+T'] is not None and selected['B+D'] is not None:
        full = [tr['policies']['B+D+T'] for tr in evaluation]
        candidate_pass = quality_ok(full,cfg)
        for method in ['B+D','fixed']:
            other = [tr['fixed'] if method=='fixed' else tr['policies'][method] for tr in evaluation]
            delta = interval([r['retention'] for r in full],[r['retention'] for r in other],cfg=cfg)
            ratio = interval([r['original_distances'] for r in full],[r['original_distances'] for r in other],'ratio',cfg)
            comparison[method] = {'retention_difference':delta,'distance_ratio':ratio}
            candidate_pass = bool(candidate_pass and delta['lower'] is not None and delta['lower'] >= -cfg['quality']['retention_drop']
                                  and ratio['upper'] is not None and ratio['upper'] <= cfg['gates']['distance_ratio'])
    systems_pass = bool(latency['status'].startswith('measured') and all(
        c['mean_ratio']['upper'] is not None and c['mean_ratio']['upper']<cfg['gates']['mean_latency_ratio']
        and c['p95_ratio']['upper'] is not None and c['p95_ratio']['upper']<=cfg['gates']['p95_latency_ratio']
        for c in latency.get('comparisons',{}).values()))
    if primary['paired_complete_queries'] != len(evaluation):
        conclusion = 'inconclusive_invalid_prediction_coverage'
    elif not prediction_pass:
        conclusion = 'stop_tested_route_no_incremental_prediction_evidence'
    elif not candidate_pass:
        conclusion = 'prediction_only_current_decision_mechanism_not_useful'
    elif not systems_pass:
        conclusion = 'prediction_or_candidate_only_no_acceleration_evidence'
    else:
        conclusion = 'eligible_to_propose_separate_HNSW_study_only'
    return {'conclusion':conclusion, 'formal_certification':False,
            'history_exclusion_established':False, 'semantic_independence_established':False,
            'evaluation_parent_tune_previously_inspected':True,
            'phase_b_seconds':perf_counter()-started,'selection':selection,
            'prediction':{k:v for k,v in prediction.items() if k!='per_query_mae'},
            'quality_cost':outcome,'paired_comparisons':comparison,'latency':latency,
            'headroom':{k:v for k,v in hr.items() if k!='records'},
            'gates':{'prediction':prediction_pass,'candidate_quality_cost':candidate_pass,'systems':systems_pass}}
