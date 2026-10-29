#!/usr/bin/env python3
"""Approved route2 A then BC; no server, new index, closed role or formal claim."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
from scipy.optimize import linprog
from mechanism.tls_rag_step5_ann_continuation_v2 import (
    config, sources, sha, save, ordered, fingerprint, metrics, rerank, check_resources,
    paired_ratio, Inputs, Native, BoundarySession, Runtime, native_build,
    observed_features, fit_models, predictors, predictions, calibrate,
    candidate_policies, replay, paired_quality,
)
from mechanism.tls_rag_step5_ann_continuation_v1 import stopping_oracle


def jsonl(path, rows):
    if Path(path).exists():
        raise FileExistsError(path)
    with Path(path).open('x') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, allow_nan=False) + '\n')


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(x) for x in f]


def control_methods(s):
    return (['original:' + str(x) for x in s['fixed_ef']] + ['natural', 'checkpoints']
            + ['fixed:' + str(x) for x in s['checkpoints']]
            + ['path:' + str(x) for x in s['checkpoints']] + ['batch_gap', 'adj_gap'])


def feasible(quality, baseline, s):
    return all(np.mean([v[name] for v in quality.values()]) >=
               np.mean([v[name] for v in baseline.values()]) - s['quality_tolerance'][key] - 1e-12
               for name, key in [('retention', 'retention_drop'), ('hit', 'hit_drop')])


def quality_rows(runtime, inputs, qids, methods, bundle=None):
    pred = predictors(bundle['models'], runtime.n) if bundle else None
    out = {}
    for method in methods:
        out[method] = {}
        for q in qids:
            r = inputs.rows[q]
            value = runtime.call(r['query'], method,
                policy=bundle['policies'].get(method) if bundle else None,
                models=pred, margins=bundle['margins'] if bundle else None)
            out[method][q] = dict(**value, **metrics(value['candidate_ids'], value['context_ids'], inputs.truth[q], r['qrels']))
    return out


def measure(output, tag, inputs, qids, methods, s, started, expected, bundle_path=None, repeats=None):
    workers = output / 'workers'
    workers.mkdir(exist_ok=True)
    out = []
    rng = np.random.default_rng(s['seeds']['order'])
    library = output / 'continuation_v2.dylib'
    for repeat in range(s['repeats'] if repeats is None else repeats):
        for method in rng.permutation(methods):
            stem = tag + '-' + str(repeat) + '-' + method.replace(':', '_')
            job, result = workers / (stem + '.json'), workers / (stem + '-result.json')
            save(job, dict(sources=sources(), spec=s, paths=inputs.paths, library=str(library),
                library_sha256=sha(library), model_bundle=str(bundle_path) if bundle_path else None,
                model_sha256=sha(bundle_path) if bundle_path else None,
                method=str(method), repeat=repeat, tag=tag, result=str(result),
                queries=[dict(query_id=q, query=inputs.rows[q]['query']) for q in qids]))
            p = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(job)],
                               text=True, capture_output=True)
            if p.returncode:
                raise RuntimeError('worker failed: ' + p.stderr[-2500:])
            value = json.loads(result.read_text())
            if value['rss_highwater_bytes'] > s['limits']['rss_gib'] * 1024**3:
                raise RuntimeError('worker RSS resource stop')
            for row in value['timings']:
                ref = expected[method][row['query_id']]
                if (row['candidate_ids'] != ref['candidate_ids'] or row['context_ids'] != ref['context_ids']):
                    raise RuntimeError('worker output differs from expected: ' + str((tag, method, row['query_id'])))
            out.extend(value['timings'])
            check_resources(started, output, s)
        print(json.dumps(dict(stage='timing_progress', tag=tag, repeat=repeat + 1, methods=len(methods))), flush=True)
    jsonl(output / (tag + '-timings.jsonl'), out)
    return out


def summary(quality, timings, methods):
    out = {}
    for m in methods:
        tt = [r for r in timings if r['method'] == m]
        a = np.asarray([r['ms'] for r in tt])
        fields = set().union(*(r['stats'] for r in tt))
        out[m] = dict(mean_ms=float(a.mean()), median_ms=float(np.median(a)), p95_ms=float(np.quantile(a, .95)),
            retention=float(np.mean([r['retention'] for r in quality[m].values()])),
            hit=float(np.mean([r['hit'] for r in quality[m].values()])),
            timed_queries=len({r['query_id'] for r in tt}),
            repeats=len({r['repeat'] for r in tt}),
            stats={k: float(np.mean([r['stats'][k] for r in tt if r['stats'].get(k) is not None]))
                   for k in fields if any(r['stats'].get(k) is not None for r in tt)})
    return out


def stage_a(output, inputs, native, s, started):
    runtime = Runtime(inputs.paths, native, s)
    equivalence, traces, attribution = [], [], []
    idmap = {d: j for j, d in enumerate(runtime.ids)}
    autopsy = {'858803': 4096, '333885': 2048, '476068': 4096, '1037189': 4096}
    qids = sorted(inputs.rows)
    for n, qid in enumerate(qids):
        r = inputs.rows[qid]
        q = np.asarray(r['query'])
        runtime.index.hnsw.efSearch = 256
        d, ids = runtime.index.search(q.astype(np.float32)[None], 64)
        with BoundarySession(native, runtime.index, q, debug=True) as one:
            a = one.advance()
            od, oi, _ = one.snapshot()
        with BoundarySession(native, runtime.index, q, debug=True) as paused:
            previous_work = -1
            for target in s['checkpoints'] + [0]:
                active = target == 0 or previous_work < target
                stat = paused.advance(target)
                pd, pi, ps = paused.snapshot()
                _, _, again = paused.snapshot()
                if ps['state_hash'] != again['state_hash'] or stat['pending']:
                    raise RuntimeError('boundary/snapshot invariant failure')
                pool, ctx, _ = rerank(runtime.x, runtime.ids, pi, q)
                traces.append(dict(query_id=qid, target=target, observed=active,
                    distances=pd.tolist(), rows=pi.tolist(), candidate_ids=pool, context_ids=ctx,
                    stats=stat, **metrics(pool, ctx, inputs.truth[qid], r['qrels'])))
                previous_work = stat['full_distances']
            for x, y in [(od, d[0]), (oi, ids[0]), (pd, d[0]), (pi, ids[0])]:
                if not np.array_equal(x, y):
                    raise RuntimeError('FAISS/v2/full boundary bitwise mismatch')
            for key in ('full_distances', 'upper_distances', 'expanded', 'neighbor_slots',
                        'result_updates', 'discovered', 'batch4_calls', 'scalar_calls',
                        'distance_sequence', 'state_hash'):
                if a[key] != ps[key]:
                    raise RuntimeError('boundary resume changed full stream/state: ' + key)
        equivalence.append(dict(query_id=qid, stats=a, passed=True))
        if qid in autopsy:
            with BoundarySession(native, runtime.index, q, debug=True) as session:
                for target in s['checkpoints']:
                    st = session.advance(target, boundary=False)
                    if target != autopsy[qid]:
                        continue
                    bd, bi, _ = session.snapshot()
                    pending, front, visited = session.audit()
                    pool, ctx, _ = rerank(runtime.x, runtime.ids, bi, q)
                    missing = [x for x in inputs.truth[qid] if x not in pool]
                    membership = {x: ('pending' if idmap[x] in pending else
                        'scored_frontier' if idmap[x] in front else
                        'visited_not_frontier' if visited[idmap[x]] else 'not_discovered') for x in missing}
                    drained = session.advance(target, boundary=True)
                    dd, di, _ = session.snapshot()
                    dp, dc, _ = rerank(runtime.x, runtime.ids, di, q)
                    attribution.append(dict(query_id=qid, target=target, before=st,
                        missing=membership, pending_ids=[runtime.ids[int(x)] for x in pending],
                        before_quality=metrics(pool, ctx, inputs.truth[qid], r['qrels']),
                        after_drain=drained, after_drain_quality=metrics(dp, dc, inputs.truth[qid], r['qrels'])))
                    break
        if (n + 1) % 128 == 0:
            check_resources(started, output, s)
            print(json.dumps(dict(stage='A_equivalence', queries=n + 1)), flush=True)
    jsonl(output / 'equivalence.jsonl', equivalence)
    jsonl(output / 'boundary_traces.jsonl', traces)
    save(output / 'pending_attribution.json', attribution)
    methods = control_methods(s)
    quality = quality_rows(runtime, inputs, qids, methods)
    save(output / 'A-quality.json', quality)
    timed = ordered(qids, 'continuation-v2-A-timing', s['seeds']['order'])[:64]
    runtime.close()
    gc.collect()
    times = measure(output, 'A', inputs, timed, methods, s, started, quality, repeats=3)
    summ = summary(quality, times, methods)
    byq = {q: [t for t in traces if t['query_id'] == q] for q in qids}
    path = ['path:' + str(t) for t in s['checkpoints']] + ['checkpoints']
    ret = [[r['retention'] for r in byq[q]] for q in qids]
    hit = [[r['hit'] for r in byq[q]] for q in qids]
    base = [summ['original:256'][k] for k in ('retention', 'hit')]
    oracle = stopping_oracle(ret, hit, [summ[m]['mean_ms'] for m in path], base, s)
    fixed = [m for m in methods if m.startswith(('original:', 'fixed:')) or m == 'natural']
    values = np.array([[summ[m][k] for k in ('retention', 'hit')] for m in fixed])
    prices = np.array([summ[m]['mean_ms'] for m in fixed])
    target = np.maximum(0, np.array(base) - [s['quality_tolerance']['retention_drop'], s['quality_tolerance']['hit_drop']])
    lp = linprog(prices, A_ub=-values.T, b_ub=-target,
                 A_eq=np.ones((1, len(fixed))), b_eq=[1], bounds=(0, 1), method='highs')
    if not lp.success:
        raise RuntimeError('A fixed mixture infeasible')
    ratio = oracle['oracle_ms'] / float(lp.fun)
    out = dict(stage='A_complete', sources=sources(), inputs=inputs.binding, equivalence_queries=len(qids),
        summaries=summ, attribution=attribution, oracle=oracle, strongest_mixture_ms=float(lp.fun),
        strongest_mixture_methods=fixed, strongest_mixture_weights=lp.x.tolist(),
        conditional_headroom_ratio=ratio, proceed_B=ratio <= 1 - s['headroom_positive'],
        quality={m: paired_quality(quality[m], quality['original:256'], s) for m in ['batch_gap', 'adj_gap']},
        formal_certification=False, fresh_independence=False, resources=check_resources(started, output, s))
    save(output / 'stage_A.json', out)
    print(json.dumps(dict(stage='A_complete', proceed_B=out['proceed_B'], oracle_ratio=ratio,
        attribution=attribution, gap={m: summ[m] for m in ['batch_gap', 'adj_gap']})), flush=True)
    return out


def enrich(output, inputs, native, s, started):
    runtime = Runtime(inputs.paths, native, s)
    byq = {}
    fees = {f: [] for f in ('S', 'G', 'T', 'T0')}
    for r in read_jsonl(output / 'boundary_traces.jsonl'):
        q = r['query_id']
        tr = byq.setdefault(q, [])
        previous = next((x for x in reversed(tr) if x['observed']), None)
        prev = dict(rows=np.array(previous['rows']), stats=previous['stats']) if previous else None
        r['features'] = {}
        if r['observed'] and not r['stats']['done']:
            for fam in ('S', 'G', 'T', 'T0'):
                x, fee = observed_features(r['distances'], r['rows'], r['stats'], prev,
                                           inputs.rows[q]['query'], runtime.x, fam, s)
                r['features'][fam] = x.tolist()
                fees[fam].append(sum(fee.values()))
        tr.append(r)
    runtime.close()
    for tr in byq.values():
        for j, r in enumerate(tr):
            nxt = tr[min(j + 1, len(tr) - 1)]
            r['offline_labels'] = dict(
                remaining_retention_gain=max(0., tr[-1]['retention'] - r['retention']),
                remaining_hit_gain=max(0., tr[-1]['hit'] - r['hit']),
                remaining_distances=tr[-1]['stats']['full_distances'] - r['stats']['full_distances'],
                next_retention_gain=nxt['retention'] - r['retention'],
                next_distances=nxt['stats']['full_distances'] - r['stats']['full_distances'])
    jsonl(output / 'observed_features_and_labels.jsonl', [r for tr in byq.values() for r in tr])
    mean_fees = {f: float(np.mean(x)) for f, x in fees.items()}
    save(output / 'offline_feature_profile.json', dict(mean_state_feature_ms=mean_fees,
        caveat='offline construction profile only, paid online timings are separate'))
    check_resources(started, output, s)
    return byq, mean_fees


def predicted_choice_quality(traces, scores, qids, policy, margins, s):
    choices = {q: replay(traces[q], scores[q], policy, margins, s) for q in qids}
    return {q: r for q, (j, r) in choices.items()}, {q: j for q, (j, r) in choices.items()}


def train_selections(output, inputs, native, s, started, traces, fees, a):
    baselines = json.loads((output / 'A-quality.json').read_text())
    path = ['path:' + str(t) for t in s['checkpoints']] + ['checkpoints']
    prices = np.array([a['summaries'][m]['mean_ms'] for m in path])
    for fold in range(4):
        ids = inputs.folds[fold]
        models, export_checks = fit_models(traces, ids['fit'], s, native)
        pred = predictors(models, native)
        scores = predictions(traces, ids['cal'] + ids['select'], pred)
        margins = calibrate(traces, ids['cal'], scores, s)
        policies, expected, candidates = {}, {}, {}
        for family in s['families']:
            eligible = []
            for policy in candidate_policies(family, s):
                cq, _ = predicted_choice_quality(traces, scores, ids['cal'], policy, margins, s)
                if not feasible(cq, {q: baselines['original:256'][q] for q in ids['cal']}, s):
                    continue
                sq, choices = predicted_choice_quality(traces, scores, ids['select'], policy, margins, s)
                if not feasible(sq, {q: baselines['original:256'][q] for q in ids['select']}, s):
                    continue
                # Screening proxy only; finalists subsequently pay real path costs.
                estimates = []
                for q, j in choices.items():
                    cost = prices[j]
                    for k, t in enumerate(traces[q][:j + 1]):
                        if not t['features'] or t['stats']['full_distances'] < policy['minimum']:
                            continue
                        if family != 'Ts':
                            cost += fees[family]
                        else:
                            cost += fees['S']
                            z = scores[q][k]['S']['retention'] + margins['S']['retention'][str(policy['quantile'])]
                            if max(0., policy['limit'] - s['selective_width']) <= z <= policy['limit'] + s['selective_width']:
                                cost += fees['T']
                    estimates.append(cost)
                proxy = np.mean(estimates)
                eligible.append((float(proxy), fingerprint(policy), policy, sq))
            eligible.sort(key=lambda z: (z[0], z[1]))
            seen = set()
            final = []
            for proxy, fp, policy, sq in eligible:
                signature = tuple(tuple(sq[q]['candidate_ids']) for q in ids['select'])
                if signature in seen and family != 'Ts':
                    continue
                seen.add(signature)
                method = 'candidate:' + family + ':' + str(len(final))
                policies[method] = policy
                expected[method] = sq
                final.append(dict(method=method, proxy_ms=proxy, policy=policy))
                if len(final) == s['selection_finalists']:
                    break
            if not final:
                method = 'candidate:' + family + ':fallback'
                policies[method] = dict(family=family, quantile=.95, limit=0., minimum=1024, disabled=True)
                expected[method] = {q: traces[q][-1] for q in ids['select']}
                final.append(dict(method=method, fallback=True, policy=policies[method]))
            candidates[family] = dict(cal_select_eligible=len(eligible), finalists=final)
        # Include controls in each selection timing cohort; frozen choices remain frozen.
        controls = [m for m in control_methods(s) if not m.startswith('path:')]
        for method in controls:
            expected[method] = {q: baselines[method][q] for q in ids['select']}
        bundle = dict(models=models, margins=margins, policies=policies)
        bp = output / ('fold-' + str(fold) + '-candidate-models.json')
        save(bp, bundle)
        times = measure(output, 'select' + str(fold), inputs, ids['select'], list(expected), s,
                        started, expected, bp, repeats=s['selection_repeats'])
        summ = summary(expected, times, list(expected))
        choices = {fam: min(candidates[fam]['finalists'], key=lambda z: summ[z['method']]['mean_ms'])['method']
                   for fam in s['families']}
        fixed = [m for m in controls if m.startswith(('original:', 'fixed:')) or m == 'natural']
        viable = [m for m in fixed if feasible(expected[m], expected['original:256'], s)]
        best_fixed = min(viable, key=lambda m: summ[m]['mean_ms'])
        nontri = [choices['S'], choices['G'], 'adj_gap', best_fixed]
        nontri = [m for m in nontri if feasible(expected[m], expected['original:256'], s)]
        best_nontri = min(nontri, key=lambda m: summ[m]['mean_ms'])
        final_policies = {'policy:' + fam: policies[m] for fam, m in choices.items()}
        final_bundle = dict(models=models, margins=margins, policies=final_policies)
        fp = output / ('fold-' + str(fold) + '-models.json')
        save(fp, final_bundle)
        renamed = {m: 'policy:' + fam for fam, m in choices.items()}
        seal = dict(fold=fold, sources=sources(), inputs=inputs.binding, model_sha256=sha(fp),
            model_path=str(fp), choices=final_policies, best_fixed=best_fixed,
            best_nontri=renamed.get(best_nontri, best_nontri), train_ids=ids['fit'],
            cal_ids=ids['cal'], select_ids=ids['select'], outer_ids=ids['outer'],
            export_checks=export_checks, candidates=candidates, selection=summ,
            formal_certification=False, fresh_independence=False)
        save(output / ('fold-' + str(fold) + '-seal.json'), seal)
        inputs.register_seal(fold, seal)
        diagnostic = {}
        for fam in ('S', 'G', 'T', 'T0'):
            diagnostic[fam] = {}
            for name in ('retention', 'hit'):
                query_errors, late_errors, pairs = [], [], []
                for q in ids['select']:
                    error, late = [], []
                    for j, r in enumerate(traces[q]):
                        if not r['features']:
                            continue
                        target = max(0., traces[q][-1][name] - r[name])
                        estimate = scores[q][j][fam][name]
                        error.append(abs(estimate - target))
                        if r['stats']['full_distances'] >= 1024:
                            late.append(abs(estimate - target))
                        pairs.append((target, estimate))
                    if error:
                        query_errors.append(np.mean(error))
                    if late:
                        late_errors.append(np.mean(late))
                diagnostic[fam][name] = dict(query_equal_mae=float(np.mean(query_errors)),
                    late_query_equal_mae=float(np.mean(late_errors)), states=len(pairs),
                    actual_remaining_improvement_states=sum(y > 1e-12 for y, p in pairs),
                    missed_positive_states_at_score002=sum(y > 1e-12 and p <= .002 for y, p in pairs))
        save(output / ('fold-' + str(fold) + '-B-diagnostic.json'), dict(
            role='selection_development', metrics=diagnostic, export_checks=export_checks,
            caveat='not outer evaluation; residual gain is not a stopping probability'))
        print(json.dumps(dict(stage='B_fold_sealed', fold=fold, best_fixed=best_fixed,
            best_nontri=seal['best_nontri'], eligible={f: candidates[f]['cal_select_eligible'] for f in candidates})), flush=True)


def stage_bc(output, inputs, native, s, started):
    a = json.loads((output / 'stage_A.json').read_text())
    if a['sources'] != sources() or a['inputs'] != inputs.binding or not a['proceed_B']:
        raise ValueError('A evidence changed or paid continuation gate closed')
    traces, fees = enrich(output, inputs, native, s, started)
    train_selections(output, inputs, native, s, started, traces, fees, a)
    save(output / 'stage_B.json', dict(stage='B_complete', sources=sources(), inputs=inputs.binding,
        fold_seals=inputs.seals, offline_feature_profile_ms=fees, resources=check_resources(started, output, s),
        caveat='development models sealed; no outer result yet'))
    print(json.dumps(dict(stage='B_complete', feature_profile=fees)), flush=True)


def stage_c(output, inputs, native, s, started):
    b = json.loads((output / 'stage_B.json').read_text())
    if b['sources'] != sources() or b['inputs'] != inputs.binding:
        raise ValueError('B evidence changed')
    inputs.seals = {}
    for fold in range(4):
        seal = json.loads((output / ('fold-' + str(fold) + '-seal.json')).read_text())
        if fingerprint(seal) != b['fold_seals'][str(fold)] or sha(Path(seal['model_path'])) != seal['model_sha256']:
            raise ValueError('fold/model selection seal changed')
        inputs.register_seal(fold, seal)
    all_quality, all_times, folds = {}, [], []
    methods = [m for m in control_methods(s) if not m.startswith('path:')]
    methods += ['policy:' + f for f in s['families']] + ['paid-noop:' + f for f in ('S', 'G', 'T', 'T0')]
    for fold in range(4):
        qids = inputs.fold_ids(fold, 'outer')
        seal = json.loads((output / ('fold-' + str(fold) + '-seal.json')).read_text())
        bp = Path(seal['model_path'])
        bundle = json.loads(bp.read_text())
        runtime = Runtime(inputs.paths, native, s)
        quality = quality_rows(runtime, inputs, qids, methods, bundle)
        runtime.close()
        gc.collect()
        save(output / ('outer' + str(fold) + '-quality.json'), quality)
        times = measure(output, 'outer' + str(fold), inputs, qids, methods, s, started, quality, bp)
        summ = summary(quality, times, methods)
        qq = {m: paired_quality(quality[m], quality['original:256'], s) for m in methods}
        rr = {m: paired_ratio([dict(r, role='outer') for r in times], m, 'original:256', s)
              for m in methods if m != 'original:256'}
        result = dict(fold=fold, selection_best_fixed=seal['best_fixed'],
            selection_best_nontri=seal['best_nontri'], summaries=summ, quality=qq, ratios_to_original=rr,
            original_fixed_passes_mean=feasible(quality[seal['best_fixed']], quality['original:256'], s))
        save(output / ('fold-' + str(fold) + '-result.json'), result)
        folds.append(result)
        for m, values in quality.items():
            all_quality.setdefault(m, {}).update(values)
        for r in times:
            all_times.append(dict(r, fold=fold))
        print(json.dumps(dict(stage='C_fold_complete', fold=fold,
            policies={m: summ[m] for m in methods if m.startswith('policy:')})), flush=True)
    summ = summary(all_quality, all_times, methods)
    deltas = {m: paired_quality(all_quality[m], all_quality['original:256'], s) for m in methods}
    ratios = {m: paired_ratio([dict(r, role='outer') for r in all_times], m, 'original:256', s)
              for m in methods if m != 'original:256'}
    increments = {}
    for m in ['policy:T', 'policy:T0', 'policy:Ts']:
        increments[m] = {base: paired_ratio([dict(r, role='outer') for r in all_times], m, base, s)
                         for base in ['policy:S', 'policy:G']}
    jsonl(output / 'all_outer_timings.jsonl', all_times)
    save(output / 'all_outer_quality.json', all_quality)
    out = dict(stage='complete', sources=sources(), inputs=inputs.binding, folds=folds,
        summaries=summ, quality=deltas, ratios_to_original=ratios, tri_increment= increments,
        formal_certification=False, fresh_independence=False,
        limits='historical nested development; no unseen-query or cross-corpus confirmation; rare-loss bounds separate',
        resources=check_resources(started, output, s), completed_utc=datetime.now(timezone.utc).isoformat())
    save(output / 'result.json', out)
    print(json.dumps(dict(stage='complete', policies={m: summ[m] for m in methods if m.startswith('policy:')},
        resources=out['resources'])), flush=True)


def worker(path):
    job = json.loads(Path(path).read_text())
    s = config()
    if job['sources'] != sources() or job['spec'] != s or sha(Path(job['library'])) != job['library_sha256']:
        raise ValueError('worker source/library binding changed')
    bundle = None
    if job['model_bundle']:
        p = Path(job['model_bundle'])
        if sha(p) != job['model_sha256']:
            raise ValueError('worker model binding changed')
        bundle = json.loads(p.read_text())
    native = Native(job['library'], s)
    runtime = Runtime(job['paths'], native, s)
    pred = predictors(bundle['models'], native) if bundle else None
    kwargs = dict(policy=bundle['policies'].get(job['method']) if bundle else None,
                  models=pred, margins=bundle['margins'] if bundle else None)
    for row in job['queries'][:s['warmup']]:
        runtime.call(row['query'], job['method'], **kwargs)
    before = resource.getrusage(resource.RUSAGE_SELF)
    rng = np.random.default_rng(s['seeds']['order'] + job['repeat'])
    out = []
    for j in rng.permutation(len(job['queries'])):
        row = job['queries'][int(j)]
        start = perf_counter()
        value = runtime.call(row['query'], job['method'], **kwargs)
        out.append(dict(query_id=row['query_id'], method=job['method'], repeat=job['repeat'],
                        ms=(perf_counter() - start) * 1000, **value))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    save(job['result'], dict(timings=out, rss_highwater_bytes=usage.ru_maxrss,
        major_faults=usage.ru_majflt - before.ru_majflt, minor_faults=usage.ru_minflt - before.ru_minflt))


def main(args):
    s = config()
    output = Path(args.output).resolve()
    if args.phase == 'A':
        output.mkdir(parents=True, exist_ok=False)
        save(output / 'build.json', native_build(args.headers, args.omp_include, output / 'continuation_v2.dylib', s))
        state = dict(sources=sources(), started_utc=datetime.now(timezone.utc).isoformat(),
                     spec=s, platform=platform.platform(), python=sys.version, elapsed_before=0.)
        save(output / 'manifest.json', state)
    else:
        state = json.loads((output / 'manifest.json').read_text())
        if state['sources'] != sources() or state['spec'] != s:
            raise ValueError('frozen source/config changed between stages')
    previous = sum(json.loads(p.read_text())['stage_seconds'] for p in output.glob('phase-*-resources.json'))
    started = perf_counter() - previous
    phase_started = perf_counter()
    try:
        import scipy
        if np.__version__ != s['numpy_version'] or scipy.__version__ != s['scipy_version']:
            raise ValueError('fixed numeric versions changed')
        inputs = Inputs(s)
        native = Native(output / 'continuation_v2.dylib', s)
        if args.phase == 'A':
            save(output / 'input_binding.json', dict(inputs=inputs.binding, folds=inputs.folds, paths=inputs.paths))
            stage_a(output, inputs, native, s, started)
        elif args.phase == 'B':
            stage_bc(output, inputs, native, s, started)
        else:
            stage_c(output, inputs, native, s, started)
        save(output / ('phase-' + args.phase + '-resources.json'),
             dict(stage_seconds=perf_counter() - phase_started, cumulative_seconds=perf_counter() - started,
                  resources=check_resources(started, output, s)))
    except Exception as e:
        save(output / ('failure-' + args.phase + '.json'), dict(error=type(e).__name__ + ': ' + str(e),
            sources=sources(), seconds=perf_counter() - phase_started, cumulative_seconds=perf_counter() - started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--phase', choices=['A', 'B', 'C'])
    parser.add_argument('--headers')
    parser.add_argument('--omp-include')
    parser.add_argument('--worker')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif not args.output or not args.phase or args.phase == 'A' and not (args.headers and args.omp_include):
        parser.error('output/phase required; phaseA also requires fixed headers/omp-include')
    else:
        main(args)
