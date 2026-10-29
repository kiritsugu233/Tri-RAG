#!/usr/bin/env python3
"""Single approved CPU100k route2 trial. No trained/paid Tri controller."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from scipy.optimize import linprog
from mechanism.tls_rag_step5_ann_continuation_v1 import (config,sources,sha,save,ordered,
    metrics,check_resources,paired_ratio,Native,Session,Runtime,OpenedContinuationInputs,
    seal_selection,gap_methods,native_build,stopping_oracle)


class Trial:
    def __init__(self,inputs,native,output,s,started):
        self.inputs=inputs;self.native=native;self.output=output;self.s=s;self.started=started
        self.runtime=Runtime(inputs.paths,native,s);self.traces=[];self.quality=[];self.timings=[];self.equivalence=[]

    def prepare(self,group):
        r=self.runtime
        for row in self.inputs.records(group):
            q=np.asarray(row['query']);r.index.hnsw.efSearch=self.s['ef_search']
            d,i=r.index.search(q.astype(np.float32)[None],self.s['return_pool'])
            with Session(self.native,r.index,q,debug=True) as one:
                one.advance();nd,ni,ns=one.snapshot()
            with Session(self.native,r.index,q,debug=True) as paused:
                for budget in self.s['checkpoints']:
                    paused.advance(budget);pd,pi,ps=paused.snapshot();_,_,again=paused.snapshot()
                    if ps['state_hash']!=again['state_hash']:raise RuntimeError('snapshot mutated live state')
                paused.advance();pd,pi,ps=paused.snapshot()
            for a,b in [(nd,d[0]),(ni,i[0]),(pd,d[0]),(pi,i[0])]:
                if not np.array_equal(a,b):raise RuntimeError('FAISS/native/resume bitwise mismatch')
            for name in ['full_distances','upper_distances','expanded','neighbor_slots','result_updates',
                'discovered','batch4_calls','scalar_calls','distance_sequence','state_hash']:
                if ns[name]!=ps[name]:raise RuntimeError('pause/resume changed work/state: '+name)
            self.equivalence.append(dict(group=group,query_id=row['query_id'],stats=ns,distances=nd.tolist(),rows=ni.tolist()))
            for stage in r.trace(q):
                self.traces.append(dict(group=group,query_id=row['query_id'],**stage,
                    **metrics(stage['candidate_ids'],stage['context_ids'],row['truth'],row['qrels'])))
        check_resources(self.started,self.output,self.s);self.persist()
        print(json.dumps(dict(stage='verified_and_traced',group=group,queries=len(self.inputs.records(group)))),flush=True)

    def score(self,group,methods):
        existing={(x['group'],x['method']) for x in self.quality}
        for method in methods:
            if (group,method) in existing:continue
            for row in self.inputs.records(group):
                value=self.runtime.call(row['query'],method)
                self.quality.append(dict(group=group,query_id=row['query_id'],method=method,**value,
                    **metrics(value['candidate_ids'],value['context_ids'],row['truth'],row['qrels'])))
            check_resources(self.started,self.output,self.s)
        self.persist()

    def quality_mean(self,group,method):
        rows=[x for x in self.quality if x['group']==group and x['method']==method]
        if not rows:raise ValueError('unscored method')
        return [float(np.mean([x[k] for x in rows])) for k in ('retention','hit')]

    def feasible(self,group,method):
        a,b=self.quality_mean(group,method),self.quality_mean(group,'original:256');t=self.s['quality_tolerance']
        return a[0]>=b[0]-t['retention_drop'] and a[1]>=b[1]-t['hit_drop']

    def measure(self,group,methods):
        methods=list(dict.fromkeys(methods));rows=self.inputs.records(group)
        byid={r['query_id']:r for r in rows}
        timed=ordered(list(byid),'continuation-timing-v1',self.s['seeds']['timing'])[:self.s['timing_queries']]
        self.runtime.close();self.runtime=None;gc.collect();workers=self.output/'workers';workers.mkdir(exist_ok=True)
        rng=np.random.default_rng(self.s['seeds']['order'])
        for repeat in range(self.s['repeats']):
            for method in rng.permutation(methods):
                tag=group+'-'+str(repeat)+'-'+str(method).replace(':','_');job=workers/(tag+'.json');result=workers/(tag+'-result.json')
                save(job,dict(sources=sources(),spec=self.s,paths=self.inputs.paths,library=str(self.output/'continuation.dylib'),
                    group=group,repeat=repeat,method=str(method),result=str(result),
                    queries=[dict(query_id=q,query=byid[q]['query']) for q in timed]))
                p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(job)],text=True,capture_output=True)
                if p.returncode:raise RuntimeError('timing worker failed: '+p.stderr[-1800:])
                value=json.loads(result.read_text());expected={r['query_id']:r for r in self.quality if r['group']==group and r['method']==method}
                for row in value['timings']:
                    ref=expected[row['query_id']]
                    if row['candidate_ids']!=ref['candidate_ids'] or row['context_ids']!=ref['context_ids']:
                        raise RuntimeError('timed worker output differs from correctness pass')
                if value['rss_highwater_bytes']>self.s['limits']['rss_gib']*1024**3:raise RuntimeError('resource_stop: worker RSS')
                self.timings.extend(value['timings'])
            check_resources(self.started,self.output,self.s)
        self.runtime=Runtime(self.inputs.paths,self.native,self.s);self.persist()
        print(json.dumps(dict(stage='measured',group=group,methods=len(methods),timed_queries=len(timed))),flush=True)

    def summary(self,group,method):
        times=[r for r in self.timings if r['group']==group and r['method']==method]
        if not times:raise ValueError('unmeasured method')
        ret,hit=self.quality_mean(group,method);keys=set().union(*(r['stats'] for r in times))
        return dict(retention=ret,hit=hit,mean_ms=float(np.mean([x['ms'] for x in times])),
            median_ms=float(np.median([x['ms'] for x in times])),p95_ms=float(np.quantile([x['ms'] for x in times],.95)),
            timed_queries=len({x['query_id'] for x in times}),
            stats_timed={k:float(np.mean([r['stats'][k] for r in times if r['stats'].get(k) is not None]))
                for k in keys if any(r['stats'].get(k) is not None for r in times)})

    def choose(self,group,methods):
        return min((self.summary(group,m)['mean_ms'],m) for m in methods if self.feasible(group,m))[1]

    def ratio(self,group,a,b):
        rows=[dict(r,role=r['group']) for r in self.timings if r['group']==group and r['method'] in (a,b)]
        return paired_ratio(rows,a,b,self.s)

    def quality_delta(self,group,method):
        a={r['query_id']:r for r in self.quality if r['group']==group and r['method']==method}
        b={r['query_id']:r for r in self.quality if r['group']==group and r['method']=='original:256'}
        if set(a)!=set(b):raise ValueError('quality not paired')
        rng=np.random.default_rng(self.s['seeds']['bootstrap'])
        ii=rng.integers(len(a),size=(self.s['bootstrap_replicates'],len(a)));out={}
        for name in ('retention','hit'):
            delta=np.array([a[q][name]-b[q][name] for q in sorted(a)])
            out[name]=dict(mean=float(delta.mean()),interval95=np.quantile(delta[ii].mean(1),[.025,.975]).tolist(),changed_queries=int(np.sum(delta!=0)))
        return out

    def diagnostic(self,group,fixed):
        targets=self.s['checkpoints']+[0];path_methods=['path:'+str(t) for t in targets[:-1]]+['checkpoints']
        qids=self.inputs.groups[group]
        trace={(r['query_id'],r['target']):r for r in self.traces if r['group']==group}
        a=np.array([[trace[q,t]['retention'] for t in targets] for q in qids])
        b=np.array([[trace[q,t]['hit'] for t in targets] for q in qids])
        prices=[self.summary(group,m)['mean_ms'] for m in path_methods]
        out=stopping_oracle(a,b,prices,self.quality_mean(group,'original:256'),self.s)
        values=np.array([self.quality_mean(group,m) for m in fixed]);cost=np.array([self.summary(group,m)['mean_ms'] for m in fixed])
        target=out['target'];lp=linprog(cost,A_ub=-values.T,b_ub=-np.array(target),A_eq=np.ones((1,len(fixed))),b_eq=[1],bounds=(0,1),method='highs')
        if not lp.success:raise RuntimeError('fixed cohort mixture infeasible')
        best=self.choose(group,fixed);out.update(best_fixed=best,best_fixed_ms=self.summary(group,best)['mean_ms'],
            strongest_mixture_ms=float(lp.fun),strongest_mixture_methods=fixed,strongest_mixture_weights=lp.x.tolist(),
            path_methods=path_methods,query_ids=qids,
            oracle_ratio_to_best_fixed=out['oracle_ms']/self.summary(group,best)['mean_ms'],
            oracle_ratio_to_strongest_mixture=out['oracle_ms']/float(lp.fun))
        # Conditional descriptive bootstrap: allocation remains fixed. It does not
        # correct optimism of knowing future quality or refit the oracle each time.
        rng=np.random.default_rng(self.s['seeds']['bootstrap']);weights=np.array(out['oracle_weights'])
        timed=sorted({x['query_id'] for x in self.timings if x['group']==group})
        methods=path_methods+fixed
        samples={m:np.array([np.mean([r['ms'] for r in self.timings if r['group']==group and r['method']==m and r['query_id']==q]) for q in timed]) for m in methods}
        ii=rng.integers(len(timed),size=(self.s['bootstrap_replicates'],len(timed)))
        qq=rng.integers(len(qids),size=(self.s['bootstrap_replicates'],len(qids)))
        cp=np.maximum.accumulate(np.stack([samples[m][ii].mean(1) for m in path_methods],axis=1),axis=1)
        oc=(weights[qq].mean(1)*cp).sum(1)
        mc=np.stack([samples[m][ii].mean(1) for m in fixed],axis=1)@lp.x
        out['conditional_ratio_interval95_to_mixture']=np.quantile(oc/mc,[.025,.975]).tolist()
        out['interval_limitation']='descriptive fixed future-quality allocation and mixture, not oracle refit or independent certification'
        return out

    def persist(self):
        for name,rows in [('equivalence.jsonl',self.equivalence),('checkpoint_records.jsonl',self.traces),
            ('query_records.jsonl',self.quality),('timing_records.jsonl',self.timings)]:
            with (self.output/name).open('w') as stream:
                for row in rows:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')


def worker(path):
    job=json.loads(Path(path).read_text());s=job['spec']
    if job['sources']!=sources() or s!=config():raise ValueError('worker seal changed')
    native=Native(job['library'],s);r=Runtime(job['paths'],native,s)
    for row in job['queries'][:s['warmup']]:r.call(row['query'],job['method'])
    before=resource.getrusage(resource.RUSAGE_SELF);times=[];rng=np.random.default_rng(s['seeds']['order']+job['repeat'])
    for j in rng.permutation(len(job['queries'])):
        row=job['queries'][int(j)];t=perf_counter();value=r.call(row['query'],job['method']);ms=(perf_counter()-t)*1000
        times.append(dict(group=job['group'],query_id=row['query_id'],method=job['method'],repeat=job['repeat'],ms=ms,**value))
    usage=resource.getrusage(resource.RUSAGE_SELF)
    save(job['result'],dict(timings=times,rss_highwater_bytes=usage.ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        major_faults=usage.ru_majflt-before.ru_majflt,minor_faults=usage.ru_minflt-before.ru_minflt))


def main(args):
    import scipy
    s=config()
    if np.__version__!=s['numpy_version'] or scipy.__version__!=s['scipy_version']:raise ValueError('numeric versions changed')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False);started=perf_counter();trial=None
    state=dict(stage='initializing',sources=sources(),started_utc=datetime.now(timezone.utc).isoformat(),
        formal_certification=False,fresh_independence=False,predictor_training=False,paid_tri=False,scale_1m=False)
    try:
        save(output/'build.json',native_build(args.headers,args.omp_include,output/'continuation.dylib'))
        started=perf_counter();native=Native(output/'continuation.dylib',s);inputs=OpenedContinuationInputs(s)
        save(output/'manifest.json',dict(state,spec=s,runtime=native.receipt,platform=platform.platform(),python=sys.version))
        save(output/'input_binding.json',dict(inputs=inputs.binding,groups=inputs.groups,paths=inputs.paths))
        trial=Trial(inputs,native,output,s,started)
        for group in ('fit','cal','select'):trial.prepare(group)
        fixed=['original:'+str(ef) for ef in s['fixed_ef']]+['native']+['fixed:'+str(t) for t in s['checkpoints']]
        paths=['path:'+str(t) for t in s['checkpoints']]+['checkpoints'];controls=fixed+paths;gaps=gap_methods(s)
        trial.score('cal',controls+gaps);eligible=[m for m in gaps if trial.feasible('cal',m)]
        trial.score('select',controls+eligible);trial.measure('select',controls+eligible)
        choice=dict(best_fixed=trial.choose('select',fixed),cal_eligible=eligible,
            best_gap=trial.choose('select',[m for m in eligible if trial.feasible('select',m)]+['checkpoints']),
            active_routes=[2],paid_tri=False)
        select_diagnostic=trial.diagnostic('select',fixed)
        seal=seal_selection(output/'selection_seal.json',inputs,choice);inputs.open_report(output/'selection_seal.json',seal)
        trial.prepare('report');methods=list(dict.fromkeys(controls+[choice['best_gap']]))
        trial.score('report',methods);trial.measure('report',methods);diag=trial.diagnostic('report',fixed)
        ratios={m:trial.ratio('report',m,'original:256') for m in methods if m!='original:256'}
        lo,hi=diag['conditional_ratio_interval95_to_mixture']
        signal='positive_conditional_headroom' if hi<=1-s['headroom_positive'] else 'negative_on_measured_actions' if lo>=1-s['headroom_negative'] else 'inconclusive'
        state.update(stage='complete',selection=choice,selection_seal=seal,select_diagnostic=select_diagnostic,
            select={m:trial.summary('select',m) for m in controls+eligible},report={m:trial.summary('report',m) for m in methods},
            report_diagnostic=diag,report_ratios=ratios,headroom_signal=signal,
            report_quality_deltas={m:trial.quality_delta('report',m) for m in methods},
            tri_increment='not_measured',resources=check_resources(started,output,s),
            completed_utc=datetime.now(timezone.utc).isoformat())
        trial.persist();save(output/'result.json',state)
    except Exception as error:
        if trial:trial.persist()
        state.update(stage='failed',error=type(error).__name__+': '+str(error),seconds=perf_counter()-started)
        save(output/'failure.json',state);raise
    print(json.dumps(dict(stage='complete',selection=choice,headroom=signal,resources=state['resources'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--headers');p.add_argument('--omp-include');p.add_argument('--worker')
    args=p.parse_args()
    if args.worker:worker(args.worker)
    elif not all((args.output,args.headers,args.omp_include)):p.error('output, headers and omp-include required')
    else:main(args)
