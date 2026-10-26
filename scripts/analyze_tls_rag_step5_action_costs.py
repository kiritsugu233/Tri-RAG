#!/usr/bin/env python3
"""Read-only, non-deployable action cost screen on already-observed records.

No retrieval, index build, fitting, threshold selection or protected-role access.
Costs use action means, never the fastest noisy observation for each query.
Restart means a complete pilot call followed by a complete fresh call, returning
the latter result. It is NOT the old persistent-membership expansion policy.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, lil_matrix, vstack

LEGACY_SHA = 'f9942afa79b3a7bec1de0650831bbd1afb2fea736275304b6ff8fcc2194fc62e'
EXACT_TRAJECTORY_SHA = '0ec9856f37819010de0bed4d758d48b6be142498077e54a45d007ee833a329bd'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve(retention, hit, budgets, call_costs, target_r, target_h, restart):
    retention,hit=np.asarray(retention),np.asarray(hit)
    budgets,costs=np.asarray(budgets),np.asarray(call_costs,dtype=float)
    n,k=retention.shape
    if costs.ndim!=1 or costs.shape!=(k,) or hit.shape!=(n,k):
        raise ValueError('one action-mean cost per action; no per-query timing oracle')
    if np.any(~np.isfinite(retention)) or np.any((retention<0)|(retention>1)):
        raise ValueError('finite retention in [0,1] required')
    if np.any(~np.isfinite(costs)) or np.any(costs<=0):
        raise ValueError('positive finite action costs required')
    units=np.rint(retention*10)
    if np.max(np.abs(units/10-retention))>1e-12 or np.any((hit!=0)&(hit!=1)):
        raise ValueError('expected saved top10 retention counts and binary Hit')
    costs=costs.copy()
    charged=budgets.copy()
    if restart:
        costs[1:]+=costs[0]
        charged[1:]+=budgets[0]
    assignment=lil_matrix((n,n*k))
    for i in range(n):assignment[i,i*k:(i+1)*k]=1
    matrix=vstack([assignment,csr_matrix(units.reshape(1,-1)),
                   csr_matrix(hit.reshape(1,-1))]).tocsr()
    # Convert empirical targets to integer neighbor/hit counts, avoiding a
    # floating representation slightly above an exact integer count.
    required_r=math.ceil(float(target_r)*n*10-1e-8)
    required_h=math.ceil(float(target_h)*n-1e-8)
    constraints=LinearConstraint(matrix,np.r_[np.ones(n),required_r,required_h],
        np.r_[np.ones(n),np.inf,np.inf])
    result=milp(np.tile(costs,n),integrality=np.ones(n*k),bounds=Bounds(0,1),
        constraints=constraints,options={'time_limit':30.,'mip_rel_gap':0.})
    if not result.success:
        return {'optimal':False,'status':int(result.status),'message':result.message}
    selected=np.argmax(result.x.reshape(n,k),axis=1)
    assert np.max(np.abs(result.x-np.round(result.x)))<1e-7
    idx=np.arange(n)
    return {'optimal':True,'mip_gap':float(result.mip_gap),
        'model_mean_ms':float(np.mean(costs[selected])),
        'mean_final_M':float(np.mean(budgets[selected])),
        'mean_charged_original_scores':float(np.mean(charged[selected])),
        'retention':float(np.mean(retention[idx,selected])),
        'hit':float(np.mean(hit[idx,selected])),
        'budget_histogram':{str(b):int(np.count_nonzero(selected==i)) for i,b in enumerate(budgets)},
        'expanded_queries':int(np.count_nonzero(selected!=0)) if restart else None,
        'action_costs_ms':costs.tolist()}


def mixture(retention,hit,costs,target_r,target_h):
    r,h=np.mean(retention,axis=0),np.mean(hit,axis=0)
    result=linprog(costs,A_ub=np.array([-r,-h]),b_ub=[-target_r,-target_h],
        A_eq=np.ones((1,len(costs))),b_eq=[1.],bounds=(0,1),method='highs')
    if not result.success:return {'feasible':False,'message':result.message}
    return {'feasible':True,'model_mean_ms':float(result.fun),
        'weights':result.x.tolist(),'expected_retention':float(r@result.x),
        'expected_hit':float(h@result.x)}


def scenario(name,r,h,budgets,costs,target_r,target_h,original=None):
    eligible=[i for i in range(len(budgets))
              if np.mean(r,axis=0)[i]>=target_r-1e-12
              and np.mean(h,axis=0)[i]>=target_h-1e-12]
    controls={}
    if eligible:
        i=min(eligible,key=lambda j:costs[j])
        controls['eligible_fixed']={'M':budgets[i],'model_mean_ms':float(costs[i])}
    mix=mixture(r,h,costs,target_r,target_h)
    if mix['feasible']:controls['query_independent_mixture']=mix
    if original is not None and original['retention']>=target_r-1e-12 and original['hit']>=target_h-1e-12:
        controls['original_ANN64']=original
    if not controls:return {'name':name,'feasible_control':False}
    best_name=min(controls,key=lambda n:controls[n]['model_mean_ms'])
    baseline=controls[best_name]['model_mean_ms']
    analyses={}
    for restart in (False,True):
        oracle=solve(r,h,budgets,costs,target_r,target_h,restart)
        if oracle['optimal']:
            gap=baseline-oracle['model_mean_ms']
            oracle.update(gross_headroom_ms=gap,gross_saving_fraction=gap/baseline,
                extra_cost_budget_ms={str(frac):baseline*(1-frac)-oracle['model_mean_ms']
                    for frac in (0.,.05,.10)})
        analyses['pilot_then_fresh_restart' if restart else 'free_before_search']=oracle
        print(name, 'restart' if restart else 'free_before_search',
              oracle.get('model_mean_ms',oracle.get('message')),flush=True)
    return {'name':name,'queries':len(r),'quality_target':{'retention':target_r,'hit':target_h},
        'budgets':budgets,'single_call_action_mean_ms':list(map(float,costs)),
        'controls':controls,'strongest_control':best_name,'oracles':analyses}


def load_legacy(root):
    path=root/'tune_compact.jsonl.gz'
    if sha(path)!=LEGACY_SHA:raise ValueError('legacy compact SHA mismatch')
    with gzip.open(path,'rt') as f:rows=[json.loads(s) for s in f]
    assert len(rows)==2048 and len({r['query_id'] for r in rows})==2048
    budgets=[64,256,1024,4096]
    ret=np.array([[r['metrics'][f'fixed:{b}']['retention'] for b in budgets] for r in rows])
    hit=np.array([[r['metrics'][f'fixed:{b}']['hit'] for b in budgets] for r in rows])
    costs=np.array([np.mean([r['metrics'][f'fixed:{b}']['timing_ms'] for r in rows]) for b in budgets])
    original={k:float(np.mean([r['metrics']['original:64'][k] for r in rows]))
              for k in ('retention','hit','timing_ms')}
    original['model_mean_ms']=original.pop('timing_ms')
    h0=float(np.mean([r['exact_hit'] for r in rows]))
    return ret,hit,budgets,costs,original,h0,{'tune_compact.jsonl.gz':LEGACY_SHA}


def load_exact(root,review):
    hashes=review['selection_cost_audit']['source_artifacts_sha256']
    for n,h in hashes.items():
        if sha(root/n)!=h:raise ValueError('exact artifact SHA mismatch: '+n)
    manifest=json.loads((root/'artifacts.json').read_text())
    name='selection.jsonl'
    if manifest[name]!=EXACT_TRAJECTORY_SHA or sha(root/name)!=EXACT_TRAJECTORY_SHA:
        raise ValueError('selection trajectory SHA mismatch')
    rows=[json.loads(s) for s in (root/name).read_text().splitlines()]
    timed=[json.loads(s) for s in (root/'selection_fixed_timing.jsonl').read_text().splitlines()]
    assert len(rows)==len(timed)==256
    assert len({r['query_id'] for r in rows})==256
    assert {r['query_id'] for r in rows}=={r['query_id'] for r in timed}
    budgets=[32,64,128,256]
    assert all([s['budget'] for s in r['stages']]==budgets for r in rows)
    ret=np.array([[s['metrics']['retention'] for s in r['stages']] for r in rows])
    hit=np.array([[s['metrics']['hit'] for s in r['stages']] for r in rows])
    costs=np.array([np.mean([r['methods'][str(b)]['pipeline_ms'] for r in timed]) for b in budgets])
    h0=float(np.mean([r['stages'][0]['metrics']['exact_hit'] for r in rows]))
    return ret,hit,budgets,costs,h0,{**hashes,name:manifest[name]}


def self_test():
    r=np.array([[0.,1.],[1.,1.]]);h=np.ones((2,2));costs=np.array([2.,5.])
    direct=solve(r,h,[32,64],costs,1.,1.,False)
    restart=solve(r,h,[32,64],costs,1.,1.,True)
    assert direct['model_mean_ms']==3.5 and restart['model_mean_ms']==4.5
    assert direct['mean_charged_original_scores']==48.
    assert restart['mean_charged_original_scores']==64.
    assert mixture(r,h,costs,1.,1.)['model_mean_ms']==5.
    try:solve(r,h,[32,64],np.array([[2.,5.],[9.,1.]]),1.,1.,False)
    except ValueError:pass
    else:raise AssertionError('per-query noisy cost oracle accepted')
    assert not solve(r,h,[32,64],costs,1.1,1.,False)['optimal']
    try:solve(np.full((2,2),np.nan),h,[32,64],costs,1.,1.,False)
    except ValueError:pass
    else:raise AssertionError('nonfinite quality accepted')
    print('Self-checks: pilot charge, score accounting, mixture, noisy-cost refusal, infeasible/nonfinite targets passed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--legacy-root',type=Path,default=Path('/private/tmp/step5-analysis-UZ5raZ/compact'))
    parser.add_argument('--exact-root',type=Path,default=Path('/private/tmp/step5-mechanism-results-9affeea8/mechanism-v1-20261025-9affeea8'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.self_test:self_test();return
    if args.output is None or args.output.exists():raise ValueError('a new output file is required')
    repository=Path(__file__).resolve().parents[1]
    review=json.loads((repository/'docs/step5_tri_feature_cost_review_evidence.json').read_text())
    r,h,b,c,orig,exacth,legacy_hashes=load_legacy(args.legacy_root)
    scenarios=[scenario('ANN_existing_quality_target',r,h,b,c,.90,exacth-.02,orig),
               scenario('ANN_match_original64_quality',r,h,b,c,orig['retention'],orig['hit'],orig)]
    er,eh,eb,ec,exacth,exact_hashes=load_exact(args.exact_root,review)
    scenarios.append(scenario('Exact_existing_quality_target',er,eh,eb,ec,.90,exacth-.02))
    result={'schema':'step5_action_cost_screen_v1','date':'2026-10-26',
        'scope':'Observed development data only; no new retrieval, fitting, deployed policy or certification.',
        'formal_certification':False,'independent_evaluation':False,'router_fitted':False,
        'cost_model':'Mean of real complete calls at each action. All methods pay the same mean for an action; no cherry-picking query-level timing noise.',
        'restart_definition':'A full pilot call, then a full fresh call if selected, returning the fresh result. Two real search requests and both scoring costs are charged. No persistent-union quality is inferred.',
        'free_before_search_definition':'Perfect knowledge selects one real action before any search; no pilot or features. Strictly optimistic information/cost model, not a proposed deployable Tri policy.',
        'required_improvement_sensitivity':[0.,.05,.10],
        'source_hashes':{'legacy':legacy_hashes,'exact':exact_hashes},
        'source_code_sha256':{n:sha(repository/n) for n in [
            'scripts/analyze_tls_rag_step5_action_costs.py',
            'src/tri_rag_harness/tls_rag_step5_ann.py',
            'src/tri_rag_harness/tls_rag_step5_benchmark.py',
            'mechanism/tls_rag_step5_exact_policy.py']},
        'scenarios':scenarios,
        'limitations':['Scenarios use different development populations and timing conditions; do not compare their absolute milliseconds across scenarios.',
            'All quality records have already been observed; formal_certification=false and no independent generalization claim.',
            'Summed mean-call costs are a model, not newly measured restart-pipeline latency.',
            'New feature, controller and data preparation costs must fit inside the remaining headroom.',
            'Mixture quality is expected quality on the empirical workload, not a newly executed randomized policy.',
            '5% and10% are reported sensitivities, not approved new acceptance thresholds.',
            'The fresh-replacement action differs from frozen persistent membership; implementation needs separate scoped approval.']}
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':main()
