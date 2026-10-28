#!/usr/bin/env python3
"""Bounded local route-1 screen. No SSH, closed roles, routes 2/3 or training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import platform
import resource
import subprocess
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from mechanism.tls_rag_step5_ann_enhancement import (Native,ObservedInputs,config,
    dense_gaussian_projection,sha,save,sources,ordered,rerank,metrics,exact_truth,
    check_resources,seal_selection,paired_ratio,native_build,EVENT_FIELDS)


def oracle_from_events(events,n):
    """Diagnostic future information only; never passed to a deployable rule."""
    cache=np.full(n,np.nan,np.float32)
    rows=events[:,0].astype(int)
    if len(set(rows))!=len(rows):raise RuntimeError('duplicate base-layer observation')
    cache[rows]=events[:,3]
    return cache


class Screen:
    def __init__(self,inputs,corpus,ids,index,projected,matrix,native,output,spec,started):
        self.inputs=inputs;self.x=corpus;self.ids=ids;self.index=index;self.p=projected
        self.matrix=matrix;self.native=native;self.output=output;self.s=spec;self.started=started
        self.records=[];self.timings=[];self.truth={};self.oracles={}
        self.scale=len(ids)==spec['corpus_size']

    def restore(self):
        import faiss
        if not hasattr(self.inputs,'corpus'):
            self.inputs.corpus=np.load(self.inputs.root/'corpus.npy',mmap_mode='r')
        self.x=self.inputs.corpus if self.scale else np.load(self.output/'subset_corpus.npy',mmap_mode='r')
        self.index=faiss.read_index(str(self.inputs.root/'original.faiss' if self.scale else self.output/'subset_original.faiss'))
        self.p=(np.ascontiguousarray(np.load(self.inputs.root/'projected.npy',mmap_mode='r'),dtype=np.float32)
            if self.scale else np.load(self.output/'subset_projected.npy',mmap_mode='r'))

    def call(self,record,method):
        # Labels are deliberately unavailable to this search function.
        q=np.asarray(record['query'],dtype=np.float64)
        q32=np.ascontiguousarray(q,dtype=np.float32);stats={}
        if method.startswith('original_ef'):
            parts=method.split('_');self.index.hnsw.efSearch=int(parts[1][2:])
            k=self.s['return_pool'] if len(parts)==2 else int(parts[2][4:])
            _,rows=self.index.search(q32[None],k);rows=rows[0]
        elif method=='native_noop':
            _,rows,stats,_=self.native.search(self.index,q32)
        elif method=='free_oracle':
            _,rows,stats,_=self.native.search(self.index,q32,mode=3,oracle=self.oracles[record['query_id']])
        else:
            rule,alpha=method.split(':');mode=4 if rule.startswith('overhead_') else 2
            rule=rule.removeprefix('overhead_')
            qp=np.ascontiguousarray(q@self.matrix.T,dtype=np.float32)
            _,rows,stats,_=self.native.search(self.index,q32,self.p,qp,mode=mode,
                rule=rule,alpha=float(alpha),audit=self.s['audit_limit'] if mode==2 else 0)
        pool,context,_=rerank(self.x,self.ids,rows,q,self.s['k_gt'],self.s['k_ctx'])
        return pool,context,stats

    def prepare(self,records,group,shadow=False):
        log=self.output/'events';log.mkdir(exist_ok=True)
        for record in records:
            qid=record['query_id'];q=np.asarray(record['query'],dtype=np.float64)
            self.truth[qid]=exact_truth(self.x,self.ids,q,self.s['k_gt'])
            self.index.hnsw.efSearch=self.s['ef_search']
            fd,fi=self.index.search(q.astype(np.float32)[None],self.s['return_pool'])
            d,i,stats,events=self.native.search(self.index,q,events=True)
            if not np.array_equal(fd[0],d) or not np.array_equal(fi[0],i):
                raise RuntimeError('no-op differs from fixed FAISS; no filtering allowed')
            self.oracles[qid]=oracle_from_events(events,len(self.ids))
            od,oi,os,_=self.native.search(self.index,q,mode=3,oracle=self.oracles[qid])
            if not np.array_equal(d,od) or not np.array_equal(i,oi):
                raise RuntimeError('safe-skip oracle altered baseline path/output')
            if shadow:
                qp=np.ascontiguousarray(q@self.matrix.T,dtype=np.float32)
                sd,si,ss,se=self.native.search(self.index,q,self.p,qp,mode=1,rule='tri',events=True)
                if not np.array_equal(d,sd) or not np.array_equal(i,si):
                    raise RuntimeError('shadow changed search')
                filename=log/(sha_id(qid)+'.npz')
                np.savez_compressed(filename,events=se)
                self.records.append(dict(group=group,query_id=qid,method='shadow',stats=ss,
                    event_file=str(filename.relative_to(self.output)),sha256=sha(filename)))
            self.records.append(dict(group=group,query_id=qid,method='equivalence',
                stats=stats,oracle_stats=os,truth=self.truth[qid],candidate_rows=i.tolist()))
            if len(self.records)%16==0:check_resources(self.started,self.output,self.s)
        print(json.dumps(dict(stage='prepared',group=group,queries=len(records))),flush=True)

    def measure(self,records,methods,group):
        for method in methods:
            for record in records:
                pool,context,stats=self.call(record,method)
                self.records.append(dict(group=group,query_id=record['query_id'],method=method,
                    candidate_ids=pool,context_ids=context,stats=stats,
                    **metrics(pool,context,self.truth[record['query_id']],record['qrels'])))
        timed=ordered([r['query_id'] for r in records],'ann-enhancement-timing',self.s['seeds']['timing'])[:self.s['timing_queries']]
        byid={r['query_id']:r for r in records};rng=np.random.default_rng(self.s['seeds']['order'])
        jobs=self.output/'workers';jobs.mkdir(exist_ok=True)
        tag=group+'-'+str(len(self.timings))
        oracle_path=jobs/(tag+'-oracle.npz')
        np.savez_compressed(oracle_path,**{q:self.oracles[q] for q in timed})
        job=dict(sources=sources(),spec=self.s,scale=self.scale,input_root=str(self.inputs.root),
            output_root=str(self.output),oracle_path=str(oracle_path),
            queries=[dict(query_id=q,query=byid[q]['query']) for q in timed],group=group)
        # Release parent working sets before a single-method worker is launched.
        self.index=None;self.p=None;self.x=None
        self.inputs.corpus._mmap.close();del self.inputs.corpus;gc.collect()
        for repeat in range(self.s['repeats']):
            for method in rng.permutation(methods):
                key=tag+'-'+str(repeat)+'-'+str(method).replace(':','-')
                path=jobs/(key+'.json');result=jobs/(key+'-result.json')
                save(path,dict(job,method=str(method),repeat=repeat,result_path=str(result)))
                completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--timing-worker',str(path)],
                    capture_output=True,text=True)
                if completed.returncode:raise RuntimeError('timing worker: '+completed.stderr[-2000:])
                record=json.loads(result.read_text())
                expected={r['query_id']:r for r in self.records if r['group']==group and r['method']==method and 'candidate_ids' in r}
                for row in record['timings']:
                    ref=expected[row['query_id']]
                    if row['candidate_ids']!=ref['candidate_ids'] or row['context_ids']!=ref['context_ids']:
                        raise RuntimeError('timed worker differs from correctness pass')
                self.timings.extend(record['timings'])
                if record['rss_highwater_bytes']>self.s['limits']['rss_gib']*1024**3:
                    raise RuntimeError('resource_stop: worker RSS')
            check_resources(self.started,self.output,self.s)
        self.restore()
        print(json.dumps(dict(stage='measured',group=group,methods=methods,queries=len(records))),flush=True)

    def summary(self,group,method):
        quality=[r for r in self.records if r['group']==group and r['method']==method and 'retention' in r]
        time=[r['ms'] for r in self.timings if r['group']==group and r['method']==method]
        if not quality or not time:raise ValueError('missing measured method')
        return dict(queries=len(quality),retention=float(np.mean([r['retention'] for r in quality])),
            hit=float(np.mean([r['hit'] for r in quality])),mean_ms=float(np.mean(time)),
            median_ms=float(np.median(time)),p95_ms=float(np.quantile(time,.95)),
            full_distances=mean_stat(quality,'full_distances'),projected_distances=mean_stat(quality,'projected_distances'))

    def persist(self):
        # A separate final write preserves every completed query if a later stage fails.
        for name,rows in [('query_records.jsonl',self.records),('timing_records.jsonl',self.timings)]:
            with (self.output/name).open('w') as stream:
                for row in rows:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')


def sha_id(qid):
    import hashlib
    return hashlib.sha256(qid.encode()).hexdigest()


def mean_stat(rows,key):
    values=[r['stats'][key] for r in rows if key in r['stats']]
    return None if not values else float(np.mean(values))


def timing_worker(path):
    """One method/one repeat/process. Load and warm up outside the measured call."""
    from types import SimpleNamespace
    import faiss
    job=json.loads(Path(path).read_text())
    if job['sources']!=sources() or job['spec']!=config():raise ValueError('worker source/config changed')
    spec=job['spec'];output=Path(job['output_root']);root=Path(job['input_root']);scale=job['scale']
    faiss.omp_set_num_threads(spec['threads'])
    native=Native(output/'route1.dylib',spec)
    index=faiss.read_index(str(root/'original.faiss' if scale else output/'subset_original.faiss'))
    x=np.load(root/'corpus.npy' if scale else output/'subset_corpus.npy',mmap_mode='r')
    ids=json.loads((root/'ids.json' if scale else output/'subset_ids.json').read_text())
    filtered=job['method'] not in ('native_noop','free_oracle') and not job['method'].startswith('original_ef')
    p=None
    if filtered:
        p=(np.ascontiguousarray(np.load(root/'projected.npy',mmap_mode='r'),dtype=np.float32)
            if scale else np.load(output/'subset_projected.npy',mmap_mode='r'))
    matrix=dense_gaussian_projection(spec['m_prime'],spec['dimension'],
        spec['scale_projection_seed'] if scale else spec['seeds']['projection'])
    screen=Screen(SimpleNamespace(root=root),x,ids,index,p,matrix,native,output,spec,perf_counter())
    if job['method']=='free_oracle':
        with np.load(job['oracle_path']) as data:screen.oracles={k:data[k] for k in data.files}
    for record in job['queries'][:spec['warmup']]:screen.call(record,job['method'])
    usage_before=resource.getrusage(resource.RUSAGE_SELF)
    timing=[];rng=np.random.default_rng(spec['seeds']['order']+job['repeat'])
    for j in rng.permutation(len(job['queries'])):
        record=job['queries'][int(j)];t=perf_counter();pool,context,_=screen.call(record,job['method'])
        timing.append(dict(query_id=record['query_id'],method=job['method'],repeat=job['repeat'],group=job['group'],
            ms=(perf_counter()-t)*1000,candidate_ids=pool,context_ids=context))
    usage=resource.getrusage(resource.RUSAGE_SELF)
    save(job['result_path'],dict(timings=timing,rss_highwater_bytes=usage.ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        minor_faults=usage.ru_minflt-usage_before.ru_minflt,major_faults=usage.ru_majflt-usage_before.ru_majflt))


def choose(screen,group,methods):
    baseline=screen.summary(group,'original_ef256');tol=screen.s['quality_tolerance']
    feasible=[]
    for method in methods:
        result=screen.summary(group,method)
        if (result['retention']>=baseline['retention']-tol['retention_drop'] and
            result['hit']>=baseline['hit']-tol['hit_drop']):feasible.append((result['mean_ms'],method))
    return min(feasible)[1] if feasible else None


def run(args):
    import faiss, scipy
    spec=config()
    if (np.__version__,scipy.__version__)!=(spec['numpy_version'],spec['scipy_version']):
        raise ValueError('frozen numeric environment changed')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=perf_counter();screen=None;state=dict(stage='initializing',formal_certification=False,
        active_routes=[1],deferred_routes=[2,3],sources=sources(),started_utc=datetime.now(timezone.utc).isoformat())
    try:
        build=native_build(args.headers,args.omp_include,output/'route1.dylib')
        save(output/'build.json',build)
        started=perf_counter()  # compilation is recorded, outside the compute budget
        native=Native(output/'route1.dylib',spec)
        save(output/'manifest.json',dict(spec=spec,sources=sources(),runtime=native.receipt,
            python=sys.version,platform=platform.platform(),machine=platform.machine(),
            environment={k:__import__('os').environ.get(k) for k in
                ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            source_headers=build['headers'],started_utc=state['started_utc'],formal_certification=False))
        inputs=ObservedInputs(args.inputs or spec['input_root'],spec)
        save(output/'input_binding.json',dict(binding=inputs.binding,groups=inputs.groups,
            packet_sha256=spec['input_observed_sha256'],opened_roles=['previously_opened_tune']))
        byid={i:n for n,i in enumerate(inputs.ids)}
        selected=ordered(inputs.ids,'ann-enhancement-corpus-v1',spec['seeds']['corpus'])[:spec['subset_size']]
        rows=np.array([byid[i] for i in selected]);del byid
        x=np.asarray(inputs.corpus[rows],dtype=np.float64);ids=selected
        matrix=dense_gaussian_projection(spec['m_prime'],spec['dimension'],spec['seeds']['projection'])
        p=np.ascontiguousarray(x@matrix.T,dtype=np.float32)
        np.save(output/'subset_rows.npy',rows);np.save(output/'projection.npy',matrix)
        np.save(output/'subset_corpus.npy',x)
        np.save(output/'subset_projected.npy',p);save(output/'subset_ids.json',ids)
        faiss.omp_set_num_threads(spec['threads'])
        index=faiss.IndexHNSWFlat(spec['dimension'],spec['degree'])
        index.hnsw.efConstruction=spec['ef_construction'];index.hnsw.efSearch=spec['ef_search']
        index.hnsw.rng=faiss.RandomGenerator(spec['seeds']['index'])
        t=perf_counter();index.add(np.ascontiguousarray(x,dtype=np.float32));build_seconds=perf_counter()-t
        faiss.write_index(index,str(output/'subset_original.faiss'))
        save(output/'setup.json',dict(index_build_seconds=build_seconds,setup_seconds=perf_counter()-started,
            files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir() if p.is_file()},
            post_projection_normalized=False))
        screen=Screen(inputs,x,ids,index,p,matrix,native,output,spec,started)
        del x,index,p
        for group in ('fit','cal','select'):screen.prepare(inputs.records(group),group,shadow=group in ('fit','cal'))
        controls=['original_ef'+str(ef) for ef in spec['fixed_ef']]+[
            'original_ef'+str(ef)+'_pool128' for ef in spec['fixed_ef'] if ef>=128]
        basic=controls+['native_noop','free_oracle','overhead_pair:0.01','overhead_tri:0.01']
        selection=inputs.records('select');screen.measure(selection,basic,'select')
        comparisons={m:screen.summary('select',m) for m in basic}
        bound=paired_ratio([r for r in screen.timings if r['group']=='select'],
            'free_oracle','native_noop',spec)
        headroom=1-bound['ratio']
        # This is an optimistic measured path bound, not a universal ANN optimum.
        active=[]
        if headroom>=spec['headroom_minimum']:
            active=[rule+':'+str(a) for rule in spec['rules'] for a in spec['alphas']]
            screen.measure(selection,active,'select')
        best=choose(screen,'select',active) if active else None
        cheap=choose(screen,'select',[m for m in active if not m.startswith('tri:')]) if active else None
        stop_reason='no_free_path_headroom' if not active else ('no_quality_feasible_rule' if best is None else 'fixed_rule_screen_completed')
        choice=dict(best=best,best_nontri=cheap,reason=stop_reason,training='not_started',
            free_path_headroom=headroom,diagnostic_only=True)
        seal=seal_selection(output/'selection_seal.json',inputs,choice,dict(comparisons=comparisons,oracle_ratio=bound))
        screen.persist();inputs.open_report(output/'selection_seal.json',seal)
        report=inputs.records('report');screen.prepare(report,'report')
        methods=list(dict.fromkeys(controls+['native_noop','free_oracle']+[m for m in (best,cheap) if m]))
        screen.measure(report,methods,'report');screen.persist()
        summaries={group:{m:screen.summary(group,m) for m in
            (basic+active if group=='select' else methods)} for group in ('select','report')}
        ratios={m:paired_ratio([r for r in screen.timings if r['group']=='report'],m,'original_ef256',spec)
            for m in methods if m!='original_ef256'}
        state.update(stage='subset_complete',selection=choice,selection_seal=seal,
            summaries=summaries,report_ratios=ratios,resources=check_resources(started,output,spec))
        save(output/'subset_result.json',state)
        # 1M scale sensitivity reuses the old index/matrix and only opened select IDs.
        # No second graph is built; no report retuning follows this result.
        del screen.index,screen.p,screen.x;gc.collect()
        index=faiss.read_index(str(inputs.root/'original.faiss'))
        old=np.load(inputs.root/'projected.npy',mmap_mode='r',allow_pickle=False)
        p=np.ascontiguousarray(old,dtype=np.float32);del old
        matrix=dense_gaussian_projection(spec['m_prime'],spec['dimension'],spec['scale_projection_seed'])
        scale=Screen(inputs,inputs.corpus,inputs.ids,index,p,matrix,native,output,spec,started)
        del index,p
        scale_records=selection[:spec['scale_queries']]
        # Archived 1M exact truth is already opened; no new full scan is needed.
        for record in scale_records:
            q=np.asarray(record['query']);scale.index.hnsw.efSearch=spec['ef_search']
            fd,fi=scale.index.search(q.astype(np.float32)[None],spec['return_pool'])
            d,i,st,events=native.search(scale.index,q,events=True)
            if not np.array_equal(fd[0],d) or not np.array_equal(fi[0],i):raise RuntimeError('1M no-op mismatch')
            scale.truth[record['query_id']]=record['truth']
            scale.oracles[record['query_id']]=oracle_from_events(events,len(inputs.ids))
            od,oi,ost,_=native.search(scale.index,q,mode=3,oracle=scale.oracles[record['query_id']])
            if not np.array_equal(d,od) or not np.array_equal(i,oi):raise RuntimeError('1M oracle mismatch')
            scale.records.append(dict(group='scale',query_id=record['query_id'],method='equivalence',stats=st,oracle_stats=ost))
        scale_methods=list(dict.fromkeys(['original_ef256','native_noop','free_oracle','overhead_pair:0.01',
            'overhead_tri:0.01']+[m for m in (best,cheap) if m]))
        scale.measure(scale_records,scale_methods,'scale')
        for name,values in [('scale_query_records.jsonl',scale.records),('scale_timing_records.jsonl',scale.timings)]:
            with (output/name).open('x') as stream:
                for value in values:stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
        state.update(stage='complete',scale_summaries={m:scale.summary('scale',m) for m in scale_methods},
            scale_projection_seed=spec['scale_projection_seed'],resources=check_resources(started,output,spec),
            report_independence='previously observed development identities; not fresh certification',
            deferred_routes_executed=False,completed_utc=datetime.now(timezone.utc).isoformat())
        save(output/'result.json',state)
    except Exception as error:
        if screen is not None:screen.persist()
        state.update(stage='failed',error=type(error).__name__+': '+str(error),elapsed_seconds=perf_counter()-started)
        save(output/'failure.json',state);raise
    print(json.dumps(dict(stage=state['stage'],output=str(output),selection=state['selection'],
        resources=state['resources'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output');parser.add_argument('--inputs')
    parser.add_argument('--headers');parser.add_argument('--omp-include')
    parser.add_argument('--timing-worker')
    args=parser.parse_args()
    if args.timing_worker:timing_worker(args.timing_worker)
    elif not all((args.output,args.headers,args.omp_include)):parser.error('output, headers and omp-include required')
    else:run(args)
