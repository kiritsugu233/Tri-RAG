#!/usr/bin/env python3
"""Approved CPU-only comparator screen. Reuses graphs; never trains an adapter."""
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
from mechanism.tls_rag_step5_ann_comparators_v2 import (Native,OpenedInputs,Context,config,sources,
    sha,save,ordered,rerank,metrics,exact_truth,check_resources,paired_ratio,prepare_cache,
    native_build,seal_selection,EVENT_FIELDS,fingerprint)


def parse_method(method):
    rule,m,a,variant=method.split(':')
    if variant not in ('point','batch','prog'):raise ValueError('invalid method variant')
    return rule,int(m),float(a),variant


class Runtime:
    def __init__(self,paths,native,spec):
        import faiss
        self.paths=paths;self.n=native;self.s=spec;self.contexts={}
        if sha(paths['index'])!=paths['index_sha']:raise ValueError('index SHA changed')
        self.index=faiss.read_index(paths['index']);self.x=np.load(paths['corpus'],mmap_mode='r')
        self.ids=json.loads(Path(paths['ids']).read_text())
        if self.index.d!=self.x.shape[1] or self.index.ntotal!=len(self.x) or len(self.ids)!=len(self.x):
            raise ValueError('index/corpus identity geometry mismatch')
        for j in np.linspace(0,len(self.x)-1,17,dtype=int):
            if not np.array_equal(self.index.reconstruct(int(j)),self.x[j].astype(np.float32)):
                raise ValueError('index coordinates disagree with corpus')

    def context(self,m,progressive=False):
        key=(m,progressive)
        if key not in self.contexts:
            self.contexts[key]=Context(self.paths['cache'],m,self.s,self.paths['index_sha'],progressive)
        return self.contexts[key]

    def call(self,query,method,shadow=False,events=False):
        q=np.asarray(query,dtype=np.float64);q32=np.ascontiguousarray(q,dtype=np.float32)
        stats={};ev=None
        if method.startswith('original:'):
            _,ef,k=method.split(':');self.index.hnsw.efSearch=int(ef)
            t=perf_counter();D,I=self.index.search(q32[None],int(k));rows=I[0]
            stats['native_ms']=(perf_counter()-t)*1000
        elif method=='noop':
            _,rows,stats,ev=self.n.search(self.index,q32,events=events)
        else:
            overhead=method.startswith('overhead:')
            if overhead:method=method.removeprefix('overhead:')
            rule,m,alpha,variant=parse_method(method)
            _,rows,stats,ev=self.n.search(self.index,q32,self.context(m,variant=='prog'),
                rule=rule,alpha=alpha,mode=1 if shadow or overhead else 2,
                skip='batch' if variant=='batch' else 'point',events=events)
        t=perf_counter();pool,context,_=rerank(self.x,self.ids,rows,q,self.s['k_gt'],self.s['k_ctx'])
        stats['rerank_ms']=(perf_counter()-t)*1000
        return dict(candidate_ids=pool,context_ids=context,stats=stats),ev

    def close(self):
        self.index=None;self.contexts.clear()
        if hasattr(self,'x'):self.x._mmap.close();del self.x
        gc.collect()


class Screen:
    def __init__(self,inputs,paths,native,output,spec,started,prefix=''):
        self.inputs=inputs;self.paths=paths;self.native=native;self.output=output;self.s=spec
        self.started=started;self.r=Runtime(paths,native,spec);self.records=[];self.timings=[]
        self.truth=dict(inputs.truth);self.prefix=prefix

    def verify(self,records,role,shadow=False):
        event_root=self.output/'events';event_root.mkdir(exist_ok=True)
        for j,record in enumerate(records):
            q=np.array(record['query']);qid=record['query_id'];ix=self.r.index
            if qid not in self.truth:self.truth[qid]=exact_truth(self.r.x,self.r.ids,q,self.s['k_gt'])
            ix.hnsw.efSearch=self.s['ef_search'];d,i=ix.search(q.astype(np.float32)[None],self.s['return_pool'])
            nd,ni,stats,_=self.native.search(ix,q)
            if not np.array_equal(d[0],nd) or not np.array_equal(i[0],ni):raise RuntimeError('v2 no-op not bitwise FAISS')
            self.records.append(dict(role=role,query_id=qid,method='equivalence',truth=self.truth[qid],
                rows=ni.tolist(),distances=nd.tolist(),stats=stats))
            if shadow:
                for rule in ('polar','normal','student'):
                    nd,ni,ss,ev=self.native.search(ix,q,self.r.context(32),rule=rule,mode=1,events=j<16)
                    if not np.array_equal(d[0],nd) or not np.array_equal(i[0],ni):raise RuntimeError('v2 shadow changed path')
                    if ev is not None:
                        name=f'{role}-{j}-{rule}.npz';np.savez_compressed(event_root/name,events=ev)
                        self.records.append(dict(role=role,query_id=qid,method='shadow:'+rule,stats=ss,
                            event_file='events/'+name,sha256=sha(event_root/name)))
            if j%16==0:check_resources(self.started,self.output,self.s)
        print(json.dumps(dict(stage='verified',role=role,queries=len(records))),flush=True)

    def measure(self,records,methods,role):
        methods=list(dict.fromkeys(methods))
        for method in methods:
            for record in records:
                value,_=self.r.call(record['query'],method)
                self.records.append(dict(role=role,query_id=record['query_id'],method=method,**value,
                    **metrics(value['candidate_ids'],value['context_ids'],self.truth[record['query_id']],record['qrels'])))
            check_resources(self.started,self.output,self.s)
        timed=ordered([r['query_id'] for r in records],'comparators-v2-timing',self.s['seeds']['timing'])[:self.s['timing_queries']]
        byid={r['query_id']:r for r in records};rng=np.random.default_rng(self.s['seeds']['order'])
        workers=self.output/'workers';workers.mkdir(exist_ok=True);tag=role+'-'+str(len(self.timings))
        self.r.close();self.r=None
        for repeat in range(self.s['repeats']):
            for method in rng.permutation(methods):
                name=tag+'-'+str(repeat)+'-'+str(method).replace(':','_')
                result=workers/(name+'-result.json');job=workers/(name+'.json')
                save(job,dict(sources=sources(),spec=self.s,paths=self.paths,library=str(self.output/'comparators.dylib'),
                    method=str(method),role=role,repeat=repeat,result=str(result),
                    queries=[dict(query_id=q,query=byid[q]['query']) for q in timed]))
                p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(job)],capture_output=True,text=True)
                if p.returncode:raise RuntimeError('independent worker failed: '+p.stderr[-2000:])
                value=json.loads(result.read_text());expected={r['query_id']:r for r in self.records if r['role']==role and r['method']==method}
                for row in value['timings']:
                    ref=expected[row['query_id']]
                    if row['candidate_ids']!=ref['candidate_ids'] or row['context_ids']!=ref['context_ids']:
                        raise RuntimeError('worker differs from correctness pass')
                if value['rss_highwater_bytes']>self.s['limits']['rss_gib']*1024**3:raise RuntimeError('resource_stop: worker RSS')
                self.timings.extend(value['timings'])
            check_resources(self.started,self.output,self.s)
        self.r=Runtime(self.paths,self.native,self.s);self.persist()
        print(json.dumps(dict(stage='measured',role=role,methods=len(methods),queries=len(records))),flush=True)

    def summary(self,role,method):
        q=[r for r in self.records if r['role']==role and r['method']==method and 'retention' in r]
        t=[r for r in self.timings if r['role']==role and r['method']==method]
        if not q or not t:raise ValueError('unmeasured method')
        out=dict(queries=len(q),timed_queries=len({r['query_id'] for r in t}),
            retention=float(np.mean([r['retention'] for r in q])),hit=float(np.mean([r['hit'] for r in q])),
            mean_ms=float(np.mean([r['ms'] for r in t])),median_ms=float(np.median([r['ms'] for r in t])),
            p95_ms=float(np.quantile([r['ms'] for r in t],.95)))
        keys=set().union(*(r['stats'] for r in q))
        out['stats']={k:float(np.mean([r['stats'][k] for r in q if k in r['stats']])) for k in keys}
        return out

    def ratio(self,role,a,b):return paired_ratio([r for r in self.timings if r['role']==role and r['method'] in (a,b)],a,b,self.s)

    def feasible(self,role,method):
        r,b=self.summary(role,method),self.summary(role,'original:256:64');tol=self.s['quality_tolerance']
        return r['retention']>=b['retention']-tol['retention_drop'] and r['hit']>=b['hit']-tol['hit_drop']

    def choose(self,role,methods):
        eligible=[(self.summary(role,m)['mean_ms'],m) for m in methods if self.feasible(role,m)]
        return min(eligible)[1] if eligible else None

    def quality_delta(self,role,method):
        a={r['query_id']:r for r in self.records if r['role']==role and r['method']==method and 'retention' in r}
        b={r['query_id']:r for r in self.records if r['role']==role and r['method']=='original:256:64'}
        if set(a)!=set(b):raise ValueError('quality not paired')
        rng=np.random.default_rng(self.s['seeds']['bootstrap'])
        ii=rng.integers(len(a),size=(self.s['bootstrap_replicates'],len(a)))
        out={}
        for name in ('retention','hit'):
            diff=np.array([a[q][name]-b[q][name] for q in sorted(a)])
            out[name]=dict(mean=float(diff.mean()),interval95=np.quantile(diff[ii].mean(1),[.025,.975]).tolist(),
                changed_queries=int(np.sum(diff!=0)))
        return out

    def persist(self):
        for name,rows in [('query_records.jsonl',self.records),('timing_records.jsonl',self.timings)]:
            with (self.output/(self.prefix+name)).open('w') as stream:
                for r in rows:stream.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')


def worker(path):
    job=json.loads(Path(path).read_text())
    if job['sources']!=sources() or job['spec']!=config():raise ValueError('worker source/config mismatch')
    native=Native(job['library'],job['spec']);r=Runtime(job['paths'],native,job['spec']);s=job['spec']
    for q in job['queries'][:s['warmup']]:r.call(q['query'],job['method'])
    before=resource.getrusage(resource.RUSAGE_SELF);times=[]
    rng=np.random.default_rng(s['seeds']['order']+job['repeat'])
    for j in rng.permutation(len(job['queries'])):
        record=job['queries'][int(j)];t=perf_counter();value,_=r.call(record['query'],job['method']);ms=(perf_counter()-t)*1000
        times.append(dict(role=job['role'],query_id=record['query_id'],method=job['method'],repeat=job['repeat'],ms=ms,**value))
    usage=resource.getrusage(resource.RUSAGE_SELF)
    save(job['result'],dict(timings=times,rss_highwater_bytes=usage.ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        major_faults=usage.ru_majflt-before.ru_majflt,minor_faults=usage.ru_minflt-before.ru_minflt))


def main(args):
    import scipy
    s=config()
    if np.__version__!=s['numpy_version'] or scipy.__version__!=s['scipy_version']:raise ValueError('numeric environment changed')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    state=dict(stage='initializing',sources=sources(),started_utc=datetime.now(timezone.utc).isoformat(),
        formal_certification=False,adapter_training=False,deferred_routes_executed=False)
    started=perf_counter();screen=None
    try:
        save(output/'build.json',native_build(args.headers,args.omp_include,output/'comparators.dylib'))
        started=perf_counter();native=Native(output/'comparators.dylib',s);inputs=OpenedInputs(s)
        save(output/'manifest.json',dict(state,spec=s,runtime=native.receipt,python=sys.version,platform=platform.platform()))
        save(output/'input_binding.json',dict(inputs=inputs.binding,groups=inputs.groups,source_index_sha=inputs.index_sha))
        t=perf_counter();cache=prepare_cache(inputs.x,inputs.base_matrix,output/'cache',s,inputs.index_sha)
        save(output/'setup.json',dict(cache_seconds=perf_counter()-t,cache_fingerprint=cache['fingerprint'],
            index_builds=0,old_index_reused=True,setup_seconds=perf_counter()-started))
        inputs.x._mmap.close();inputs.x=None
        paths=dict(index=str(inputs.parent/'subset_original.faiss'),index_sha=inputs.index_sha,
            corpus=str(inputs.parent/'subset_corpus.npy'),ids=str(inputs.parent/'subset_ids.json'),cache=str(output/'cache'))
        screen=Screen(inputs,paths,native,output,s,started);dev=inputs.records('development')
        screen.verify(dev,'development',shadow=True)
        controls=['original:'+str(ef)+':'+str(k) for k in s['fixed_pool'] for ef in s['fixed_ef'] if ef>=k]+['noop','overhead:chi:32:0.05:point']
        fixed=[f'{rule}:{m}:{a}:point' for rule in s['rules'] for m in s['dimensions'] for a in s['alphas']]
        strict=[f'{rule}:64:{s["strict_alpha"]}:point' for rule in ('chi','polar','normal','student')]
        screen.measure(dev,controls+fixed+strict,'development')
        all_active=list(fixed);batch=[]
        for rule in s['batch_controls']:
            best=screen.choose('development',[m for m in fixed if m.startswith(rule+':')])
            if best:batch.append(best.removesuffix(':point')+':batch')
        if batch:screen.measure(dev,batch,'development');all_active.extend(batch)
        progressive=[];best=screen.choose('development',all_active)
        if best and screen.ratio('development',best,'original:256:64')['ratio']<=1-s['progressive_trigger_gain']:
            for rule in s['rules']:
                b=screen.choose('development',[m for m in fixed if m.startswith(rule+':')])
                if b:progressive.append(rule+':64:'+b.split(':')[2]+':prog')
            if progressive:screen.measure(dev,progressive,'development');all_active.extend(progressive)
        family={rule:screen.choose('development',[m for m in all_active if m.startswith(rule+':')]) for rule in s['rules']}
        choice=dict(best=screen.choose('development',all_active),best_direct_tri=family['polar'],
            best_nontri=screen.choose('development',[m for m in all_active if not m.startswith('polar:')]),
            family=family,batch_tested=batch,progressive_tested=progressive,strict_for_selection=False)
        binding=seal_selection(output/'selection_seal.json',inputs,choice);inputs.open_report(output/'selection_seal.json',binding)
        screen.persist();report=inputs.records('report');screen.verify(report,'report')
        methods=list(dict.fromkeys(controls+[m for m in list(family.values())+[choice['best'],choice['best_nontri']] if m]))
        screen.measure(report,methods,'report')
        report_ratios={m:screen.ratio('report',m,'original:256:64') for m in methods if m!='original:256:64'}
        deltas={m:screen.quality_delta('report',m) for m in methods if m!='original:256:64'}
        best=choice['best'];quality_confident=bool(best and all(deltas[best][name]['interval95'][0]>=-s['quality_tolerance'][key]
            for name,key in [('retention','retention_drop'),('hit','hit_drop')]))
        scale_allowed=bool(best and screen.feasible('report',best) and quality_confident and report_ratios[best]['interval95'][1]<=1-s['net_gain'])
        state.update(stage='subset_complete',selection=choice,selection_seal=binding,
            development={m:screen.summary('development',m) for m in controls+fixed+strict+batch+progressive},
            report={m:screen.summary('report',m) for m in methods},report_ratios=report_ratios,report_quality_deltas=deltas,scale_allowed=scale_allowed,
            resources=check_resources(started,output,s),fresh_independence=False)
        if choice['best_direct_tri'] and choice['best_nontri']:
            state['direct_tri_vs_nontri']=screen.ratio('report',choice['best_direct_tri'],choice['best_nontri'])
        save(output/'subset_result.json',state)
        # Scale runs are strictly conditional on measured report net gain.
        if scale_allowed:
            run_scale(screen,inputs,native,output,s,started,choice,state)
        else:state['scale_status']='not_run_no_100k_net_gain'
        state.update(stage='complete',completed_utc=datetime.now(timezone.utc).isoformat(),resources=check_resources(started,output,s))
        save(output/'result.json',state)
    except Exception as error:
        if screen is not None:screen.persist()
        state.update(stage='failed',error=type(error).__name__+': '+str(error),seconds=perf_counter()-started)
        save(output/'failure.json',state);raise
    print(json.dumps(dict(stage=state['stage'],selection=state['selection'],scale=state.get('scale_status'),resources=state['resources'])),flush=True)


def run_scale(screen,inputs,native,output,s,started,choice,state):
    """A frozen 64-query development sensitivity; no scale-driven selection."""
    screen.r.close()
    root=inputs.full_root
    for name in ('corpus.npy','ids.json','original.faiss'):
        p=root/name;desc=inputs.full_descriptors[name]
        if p.is_symlink() or p.stat().st_size!=desc['bytes'] or sha(p)!=desc['sha256']:raise ValueError('full-scale input changed')
    x=np.load(root/'corpus.npy',mmap_mode='r')
    cache=prepare_cache(x,inputs.base_matrix,output/'cache_1m',s,inputs.full_descriptors['original.faiss']['sha256'])
    x._mmap.close();del x
    paths=dict(index=str(root/'original.faiss'),index_sha=inputs.full_descriptors['original.faiss']['sha256'],
        corpus=str(root/'corpus.npy'),ids=str(root/'ids.json'),cache=str(output/'cache_1m'))
    scale=Screen(inputs,paths,native,output,s,started,prefix='scale_');scale.records=[];scale.timings=[]
    records=inputs.records('development')[:64]
    scale.truth={r['query_id']:r['truth'] for r in records}
    try:
        scale.verify(records,'scale')
        methods=list(dict.fromkeys(['original:256:64','noop']+[m for m in (choice['best'],choice['best_direct_tri'],choice['best_nontri']) if m]))
        scale.measure(records,methods,'scale')
        state.update(scale_status='complete',scale={m:scale.summary('scale',m) for m in methods},
            scale_ratios={m:scale.ratio('scale',m,'original:256:64') for m in methods if m!='original:256:64'})
    finally:
        # Preserve partial scale evidence without overwriting subset records.
        scale.persist();screen.persist()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output')
    parser.add_argument('--headers');parser.add_argument('--omp-include');parser.add_argument('--worker')
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif not all((args.output,args.headers,args.omp_include)):parser.error('output, headers and omp-include required')
    else:main(args)
