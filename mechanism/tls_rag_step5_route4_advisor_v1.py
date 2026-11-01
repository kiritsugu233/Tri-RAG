"""Section38 offline Gaussian risk intervals and paid direct-probe policies.

Bounds concern random matrices on a fixed finite query set. A large upper
bound is UNKNOWN, never evidence of high loss. No independence shortcut.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from tri_rag_harness.tri_law import tri_law_probability
from mechanism import tls_rag_step5_route4_scenarios_v1 as core


def count_risk_interval(mu_lower, mu_upper, k, g, M, pool_size):
    lo, hi = np.asarray(mu_lower, dtype=float), np.asarray(mu_upper, dtype=float)
    J = pool_size - g
    if not (k >= g >= 0 and M >= k and pool_size >= g):
        raise ValueError('rank domain')
    if lo.shape != (g,) or hi.shape != (g,) or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError('mean-count shape/domain')
    if np.any(lo < 0) or np.any(hi < lo) or np.any(hi > J + 1e-10):
        raise ValueError('mean-count interval')
    if M >= pool_size:
        lower, upper = np.zeros(g), np.zeros(g)
    else:
        lower = np.maximum(0, (lo-(M-1))/(J-(M-1))) if M <= J else np.zeros(g)
        upper = np.minimum(1, hi/(M-g+1))
    return dict(lower=float(lower.sum()/k), upper=float(upper.sum()/k),
                near_lower=lower.tolist(), near_upper=upper.tolist())


def assess_geometry(X, ids, recs, G, s):
    rows = {x:j for j,x in enumerate(ids)}
    allrows = np.arange(len(X))
    result = []
    for r in recs:
        q = np.asarray(r['query'])
        H = np.asarray([rows[x] for x in G[r['query_id']]], dtype=int)
        others = np.setdiff1d(allrows,H)
        sums = {m: [np.zeros(len(H)), np.zeros(len(H))] for m in s['gaussian_dimensions']}
        outside = 0
        for pos,i in enumerate(H):
            a = X[i]-q
            aa = float(a@a)
            for start in range(0,len(others),s['risk_block']):
                b = X[others[start:start+s['risk_block']]]-q
                bb = np.einsum('ij,ij->i',b,b)
                with np.errstate(divide='ignore',invalid='ignore'):
                    beta = bb/aa
                    rho = (b@a)/np.sqrt(aa*bb)
                valid = (aa>0)&np.isfinite(beta)&(beta>1)&np.isfinite(rho)&(np.abs(rho)<=1)
                outside += int((~valid).sum())
                for m in s['gaussian_dimensions']:
                    p = tri_law_probability(beta[valid],rho[valid],m) if valid.any() else np.array([])
                    sums[m][0][pos] += float(np.sum(p))
                    sums[m][1][pos] += float(np.sum(p)) + int((~valid).sum())
        result.append(dict(query_id=r['query_id'],near_ids=G[r['query_id']],competitors=len(others),
                           pairs=len(H)*len(others),conservative_pairs=outside,
                           dimensions={str(m):dict(mu_lower=lo.tolist(),mu_upper=hi.tolist(),budgets={
                               str(M):count_risk_interval(lo,hi,s['k'],len(H),M,len(X)) for M in s['budgets']})
                               for m,(lo,hi) in sums.items()}))
    return result


def expected_states(geometry, s, lower=True):
    states = {}
    for m in s['gaussian_dimensions']:
        for M in s['budgets']:
            intervals = [r['dimensions'][str(m)]['budgets'][str(M)] for r in geometry]
            L = np.asarray([v['lower'] for v in intervals]) if lower else np.zeros(len(intervals))
            U = np.asarray([v['upper'] for v in intervals])
            safe = U.mean() <= s['maximum_mean_loss'] and U.max() <= s['maximum_query_loss']
            bad = L.mean() > s['maximum_mean_loss'] or L.max() > s['maximum_query_loss']
            states[core.recipe_key(dict(kind='gaussian',m=m),M)] = dict(
                state='low_expected_risk' if safe else ('high_expected_risk' if bad else 'unknown'),
                mean_lower=float(L.mean()),mean_upper=float(U.mean()),
                worst_query_lower=float(L.max()),worst_query_upper=float(U.max()))
    return states


def run_probe_policy(X, ids, recs, s, policy, directory):
    """Independent from scenario A and all select/confirmation results."""
    if policy not in s['advisors']:
        raise ValueError('unfrozen advisor')
    started = time.perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    times = dict(truth_seconds=0.,geometry_seconds=0.,probe_seconds=0.,model_delivery_seconds=0.)
    geometry, states, G = [], {}, None
    attempted, observations, good, models = [], {}, [], {}
    shared32 = shared_gaussian = None
    if policy != 'always_unknown':
        t = time.perf_counter()
        G = core.truth(X,ids,recs,s['k'])
        times['truth_seconds'] = time.perf_counter()-t
    if policy.startswith('tri_'):
        t = time.perf_counter()
        geometry = assess_geometry(X,ids,recs,G,s)
        states = expected_states(geometry,s,lower=policy=='tri_interval')
        times['geometry_seconds'] = time.perf_counter()-t

    def probe(spec, allowed):
        nonlocal shared32, shared_gaussian
        t = time.perf_counter()
        if shared32 is None:
            shared32 = np.ascontiguousarray(X,dtype=np.float32)
        if spec['kind']=='gaussian' and shared_gaussian is None:
            shared_gaussian = core.family_basis(X,'gaussian',s)
        model = core.build_method(X,ids,spec,s,shared32=shared32,
                                  basis=shared_gaussian if spec['kind']=='gaussian' else None)
        values = core.probe_model(model,recs,allowed,s)
        key = core.base_key(spec)
        models[key] = model
        attempted.append(dict(spec=spec,budgets=list(allowed),actual_distance_passes=len(recs)))
        for M in allowed:
            q = core.quality(values[str(M)],G,s['k'])
            name = core.recipe_key(spec,M)
            observations[name] = dict(spec=spec,M=M,quality=q,records=values[str(M)])
            if core.eligible(q,s):
                # Declared byte proxy, not free scenario-A timing/quality labels.
                good.append((model.scan_bytes + M*X.shape[1]*8,name))
        times['probe_seconds'] += time.perf_counter()-t

    if policy == 'direct_mixed':
        jobs = [(dict(kind='gaussian',m=max(s['gaussian_dimensions'])),[max(s['budgets'])]),
                (dict(kind='pca',m=96 if 96 in s['pca_dimensions'] else max(s['pca_dimensions'])),[s['budgets'][-2]]),
                (dict(kind='pq',bits=min(s['pq_bits'])),[s['budgets'][-2]])]
    elif policy == 'direct_one':
        jobs = [(dict(kind='gaussian',m=max(s['gaussian_dimensions'])),s['budgets'])]
    elif policy == 'always_unknown':
        jobs = []
    else:
        jobs = [(dict(kind='gaussian',m=m),s['budgets']) for m in s['gaussian_dimensions']]

    # A certified-low expected-risk recipe still pays a concrete-matrix probe.
    if states:
        safe = [(m,M) for m in s['gaussian_dimensions'] for M in s['budgets']
                if states[core.recipe_key(dict(kind='gaussian',m=m),M)]['state']=='low_expected_risk']
        if safe:
            m,M = min(safe,key=lambda p:(len(X)*p[0]*4+p[1]*X.shape[1]*8,p))
            probe(dict(kind='gaussian',m=m),[M])
            if good:
                jobs = []
    for spec,allowed in jobs:
        if core.base_key(spec) in models:
            # Reuse the existing model; one new max-M pass if broader actions
            # remain. A failed early safe check is retained and billed.
            t = time.perf_counter()
            remaining = [M for M in allowed if core.recipe_key(spec,M) not in observations and
                         states.get(core.recipe_key(spec,M),{}).get('state')!='high_expected_risk']
            if not remaining:
                continue
            values = core.probe_model(models[core.base_key(spec)],recs,remaining,s)
            attempted.append(dict(spec=spec,budgets=remaining,actual_distance_passes=len(recs),reused_model=True))
            for M in remaining:
                q = core.quality(values[str(M)],G,s['k'])
                name = core.recipe_key(spec,M)
                observations[name] = dict(spec=spec,M=M,quality=q,records=values[str(M)])
                if core.eligible(q,s):
                    good.append((models[core.base_key(spec)].scan_bytes+M*X.shape[1]*8,name))
            times['probe_seconds'] += time.perf_counter()-t
            continue
        allowed = [M for M in allowed if states.get(core.recipe_key(spec,M),{}).get('state')!='high_expected_risk']
        if allowed:
            probe(spec,allowed)
    if good:
        name = min(good)[1]
        selected = dict(spec=observations[name]['spec'],M=observations[name]['M'],key=name)
        model = models[core.base_key(selected['spec'])]
        concrete_confirm = dict(status='reused_same_seed_advice_probe',quality=observations[name]['quality'])
    else:
        spec = dict(kind='original64')
        t = time.perf_counter()
        model = core.build_method(X,ids,spec,s)
        times['probe_seconds'] += time.perf_counter()-t
        selected = dict(spec=spec,M=s['k'],key=core.recipe_key(spec,s['k']))
        concrete_confirm = dict(status='exact_fallback',quality_guarantee='exact full-corpus L2')
    t = time.perf_counter()
    core.store_model(model,directory/'deployment_model')
    expected_seed = s['production_seed'] if selected['spec']['kind']=='gaussian' else (s['pq_seed'] if selected['spec']['kind']=='pq' else 0)
    delivered = core.load_model(X,ids,directory/'deployment_model',selected['spec'],expected_seed)
    first = core.query_method(delivered,np.asarray(recs[0]['query']),selected['M'],s['k'])
    times['model_delivery_seconds'] = time.perf_counter()-t
    # The returned recommendation precedes all select/extra-seed access.
    result = dict(policy=policy,selected=selected,states=states,geometry=geometry,
                  attempted=attempted,observations=observations,concrete_confirmation=concrete_confirm,
                  costs=times,total_seconds=time.perf_counter()-started,first_call_ns=first['full_ns'],
                  input_roles=['advice'],reads_scenario_A_results=False,
                  choice_proxy='scan bytes plus original FP64 rerank bytes',
                  fixed_seed=s['production_seed'],formal_certification=False)
    core.save(directory/'decision.json',result)
    return result


def evaluate_decisions(decision, measurements, G, s):
    q = core.quality(measurements['records'],G,s['k'])
    return dict(selected=decision['selected'],quality=q,quality_pass=core.eligible(q,s),
                compressed=decision['selected']['spec']['kind'] in ('gaussian','pca','pq'),
                decision_sha256=core.fingerprint(decision),mean_ns=measurements['mean_ns'],p95_ns=measurements['p95_ns'],
                advisor_total_seconds=decision['total_seconds'],
                theoretical_target='expectation over Gaussian matrices on fixed advice queries',
                observed_target='fixed deployed model on historically observed select queries',
                formal_certification=False,fresh_independence=False)
