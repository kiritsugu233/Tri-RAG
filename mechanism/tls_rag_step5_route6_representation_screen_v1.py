"""R6-S0 development screening, complete-call accounting and oracle diagnostics."""
import gc
import itertools
import json
from pathlib import Path
import time

import numpy as np

from mechanism import tls_rag_step5_route6_representation_v1 as rep


def benchmark_recall(distances, cutoff, truth, returned, k, epsilon=.001):
    distances, returned, truth = np.asarray(distances), np.asarray(returned), np.asarray(truth)
    if (len(distances) != k or len(returned) != k or len(truth) != k or
            len(np.unique(returned)) != k or not np.isfinite(distances).all()):
        raise ValueError('exactly k unique reranked items required')
    return dict(recall=float(np.count_nonzero(distances <= cutoff+epsilon)/k),
                strict_id=float(len(set(map(int, returned)) & set(map(int, truth)))/k))


def loss_transitions(truth, before, after):
    truth, before, after = set(map(int, truth)), set(map(int, before)), set(map(int, after))
    kept_before, kept_after = truth & before, truth & after
    return dict(retained=sorted(kept_before & kept_after), lost=sorted(kept_before-kept_after),
                rescued=sorted(kept_after-kept_before), omitted=sorted(truth-(before|after)),
                signed_retention_change=(len(kept_after)-len(kept_before))/len(truth))


def measure_call(call, synchronize=lambda: None):
    synchronize()
    start = time.perf_counter()
    value = call()
    synchronize()
    return value, time.perf_counter()-start


def select_fixed(points, target, metric='recall', role='geometry_discovery'):
    """Only discovery rows enter selection, even when check records are present."""
    eligible = [p for p in points if p['role']==role and p['metric']==metric and p['quality']>=target]
    return min(eligible, key=lambda p:(p['seconds'], p['resident_bytes'], p['id'])) if eligible else None


def lifecycle_cost(offline_seconds, query_seconds, raw_bytes, payload_bytes, index_bytes=0,
                   transfer_seconds=0., horizons=(1,1000,100000,1000000)):
    if min(offline_seconds,query_seconds,transfer_seconds,raw_bytes,payload_bytes,index_bytes)<0 or raw_bytes<=0:
        raise ValueError('raw retention and all nonnegative costs required')
    return dict(offline_seconds=float(offline_seconds), query_seconds=float(query_seconds),
                transfer_seconds_included=float(transfer_seconds), raw_bytes=int(raw_bytes),
                payload_bytes=int(payload_bytes), index_extra_bytes=int(index_bytes),
                resident_bytes=int(raw_bytes+payload_bytes+index_bytes),
                horizons={str(q):float(offline_seconds+q*query_seconds) for q in horizons})


def break_even(a_offline, a_query, b_offline, b_query):
    if a_query==b_query:
        return None
    q=(b_offline-a_offline)/(a_query-b_query)
    return float(q) if q>=0 else None


def oracle_budget_bound(quality, seconds, target, solver_seconds=60, deterministic=True):
    """Mean-quality LP lower bound; optional 0/1 MILP with explicit solver status."""
    from scipy.optimize import linprog, milp, Bounds, LinearConstraint
    from scipy.sparse import csr_matrix, vstack
    quality, seconds = np.asarray(quality, float), np.asarray(seconds, float)
    if (quality.shape != seconds.shape or quality.ndim!=2 or not np.isfinite(quality).all() or
            not np.isfinite(seconds).all() or np.any(seconds<0) or np.any(quality<0) or
            np.any(quality>1) or not 0<=target<=1):
        raise ValueError('finite measured oracle coefficient table')
    n, m=quality.shape
    eq=csr_matrix(np.kron(np.eye(n), np.ones((1,m))))
    c=seconds.ravel()/n
    quality_constraint=quality.ravel()[None,:]/n
    started=time.monotonic()
    lp=linprog(c, A_ub=-quality_constraint, b_ub=[-target], A_eq=eq, b_eq=np.ones(n),
               bounds=(0,1), method='highs', options={'time_limit':float(solver_seconds)})
    def result(solution):
        x=solution.x
        feasible=(x is not None and np.all(x>=-1e-7) and np.all(x<=1+1e-7) and
                  np.max(np.abs(eq@x-1))<=1e-7 and float(quality.ravel()@x/n)>=target-1e-7)
        return dict(status=int(solution.status), message=str(solution.message),
            optimal=bool(solution.status==0 and feasible), feasible=bool(feasible),
            objective_seconds=float(c@x) if feasible else None,
            achieved_quality=float(quality.ravel()@x/n) if feasible else None,
            x=x.reshape(n,m).tolist() if feasible else None)
    out=dict(lp=result(lp), label='randomized clairvoyant LP relaxation, not a deployable policy',
             mean_quality_constraint=True, measured_coefficients_only=True)
    if deterministic:
        matrix=vstack((eq,csr_matrix(quality_constraint)),format='csc')
        mip=milp(c,integrality=np.ones(n*m),bounds=Bounds(0,1),
                 constraints=LinearConstraint(matrix,np.r_[np.ones(n),target],np.r_[np.ones(n),np.inf]),
                 options={'time_limit':float(solver_seconds)})
        out['milp']=result(mip)
        out['milp'].update(gap=float(mip.mip_gap) if getattr(mip,'mip_gap',None) is not None else None,
                           dual_bound=float(mip.mip_dual_bound) if getattr(mip,'mip_dual_bound',None) is not None and np.isfinite(mip.mip_dual_bound) else None)
    out['solver_seconds']=time.monotonic()-started
    fixed=[dict(budget_column=j,quality=float(quality[:,j].mean()),seconds=float(seconds[:,j].mean()))
           for j in range(m) if quality[:,j].mean()>=target]
    out['best_same_pipeline_fixed']=min(fixed,key=lambda p:p['seconds']) if fixed else None
    # One global randomized mixture; this does not see individual-query truth.
    mix=linprog(seconds.mean(axis=0), A_ub=-quality.mean(axis=0)[None],b_ub=[-target],
                A_eq=np.ones((1,m)),b_eq=[1.],bounds=(0,1),method='highs',
                options={'time_limit':float(solver_seconds)})
    out['fixed_mixture']=dict(status=int(mix.status),optimal=bool(mix.status==0),
        seconds=float(mix.fun) if mix.success else None,weights=mix.x.tolist() if mix.success else None)
    out['solver_seconds']=time.monotonic()-started
    return out


def seal_report(completed, expected, claims=None):
    missing=sorted(set(expected)-set(completed))
    return dict(status='complete' if not missing else 'incomplete',completed=sorted(completed),missing=missing,
                missing_is_negative_evidence=False,formal_test_score=False,tri_validated=False,
                independent_certification=False,claims=claims or [])


class Products:
    """Exclusive, bounded scientific writes; failure evidence is retained."""
    def __init__(self, root, limit, check=lambda: None):
        self.root, self.limit, self.check=Path(root),int(limit),check
        self.root.mkdir(parents=True,exist_ok=True)

    def reserve(self, size):
        self.check()
        used=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        if size<0 or used+size>self.limit:
            raise RuntimeError('artifact write budget')

    def json(self, name, value):
        data=(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
        self.reserve(len(data)); path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data)

    def array(self,name,value):
        value=np.asarray(value); self.reserve(value.nbytes+1024)
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)


def _evaluate(raw,queries,truth,cutoffs,candidates,budgets,k,epsilon,roles,identity,reference=None):
    records=[]
    for qi,(q,ids) in enumerate(zip(queries,candidates)):
        for m in budgets:
            returned, distances=rep.rerank(raw,q,ids[:m],k)
            metric=benchmark_recall(distances,cutoffs[qi],truth[qi],returned,k,epsilon)
            row=dict(query_id=roles[qi]['id'],role=roles[qi]['role'],query_index=qi,
                     pipeline=identity,budget=m,returned=returned.tolist(),distances=distances.tolist(),**metric)
            if reference is not None:
                row['transition']=loss_transitions(truth[qi],reference[qi,:m],ids[:m])
            records.append(row)
    return records


def _quality_means(records,metric,role,budget):
    return float(np.mean([r[metric] for r in records if r['role']==role and r['budget']==budget]))


def _timings(model,raw,queries,roles,budgets,index,cfg,check,torch=None,device_values=None,
             threads=None,subset=None):
    import faiss
    faiss.omp_set_num_threads(threads or cfg['timing']['faiss_threads'])
    rng=np.random.Generator(np.random.PCG64(cfg['seeds']['schedule']))
    schedule=list(itertools.product(range(len(queries)) if subset is None else subset,budgets))
    rng.shuffle(schedule)
    rows=[]
    sync=torch.cuda.synchronize if device_values is not None else lambda:None
    for qi,m in schedule:
        for repeat in range(-cfg['timing']['warmup'],cfg['timing']['repeats']):
            check()
            def call():
                t=time.perf_counter()
                ids=rep.search_backend(model,queries[qi],m,index=index,device_values=device_values,torch=torch)
                search=time.perf_counter()-t
                t=time.perf_counter()
                result,dist=rep.rerank(raw,queries[qi],ids,cfg['k'])
                return ids,result,dist,search,time.perf_counter()-t
            (ids,result,dist,search,rerank),elapsed=measure_call(call,sync)
            if repeat>=0:
                rows.append(dict(query_index=int(qi),query_id=roles[qi]['id'],role=roles[qi]['role'],
                    budget=int(m),repeat=repeat,seconds=elapsed,search_transform_transfer_seconds=search,
                    rerank_seconds=rerank,candidate_sha256=rep.fingerprint(ids.tolist()),
                    returned=result.tolist(),distances=dist.tolist()))
    faiss.omp_set_num_threads(cfg['timing']['faiss_threads'])
    return rows


def _timing_points(identity,quality,timings,resident):
    points=[]
    for role in ('geometry_discovery','geometry_check'):
        for m in sorted({r['budget'] for r in timings}):
            times=[r['seconds'] for r in timings if r['role']==role and r['budget']==m]
            if not times:continue
            for metric in ('recall','strict_id'):
                points.append(dict(id=identity+'-M'+str(m),pipeline=identity,role=role,budget=m,metric=metric,
                    quality=_quality_means(quality,metric,role,m),seconds=float(np.mean(times)),
                    median_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)),
                    resident_bytes=int(resident),hardware_profile='cuda' if ':cuda' in identity else 'cpu14',
                    logical_resident_layout='CPU raw+search index/model' if ':cuda' not in identity else 'CPU raw/model + CUDA FP32 search payload',
                    physical_peak_is_separate=True))
    return points


def _selected_budgets(quality,cfg):
    chosen=set()
    for target in cfg['targets']:
        valid=[m for m in cfg['budgets'] if _quality_means(quality,'recall','geometry_discovery',m)>=target]
        if valid:chosen.add(min(valid))
    return sorted(chosen)


def _save_model(products,prefix,model):
    import faiss
    products.array(prefix+'/matrix.npy',model['matrix']);products.array(prefix+'/mean.npy',model['mean'])
    metadata=dict(spec=model['spec'],fingerprint=model['fingerprint'],metric=model['metric'])
    if 'index' in model:
        index=model['index']
        centers=faiss.vector_to_array(index.pq.centroids)
        products.array(prefix+'/centroids.npy',centers)
        metadata['centroids_sha256']=rep.fingerprint(centers.tolist())
    products.json(prefix+'/model.json',metadata)


def run_screen(cfg,root,torch,check,phase,environment):
    """One serial full-corpus development run; all incomplete stages stay explicit."""
    import faiss
    products=Products(root,cfg['limits']['artifact_bytes'],check)
    faiss.omp_set_num_threads(cfg['timing']['faiss_threads'])
    started=time.monotonic()
    input_start=time.monotonic()
    raw,queries,truth,roles,input_receipt=rep.load_screen_inputs(cfg,check,phase)
    input_load_sha_seconds=time.monotonic()-input_start
    products.json('inputs.json',input_receipt);products.json('roles.json',dict(queries=roles))
    products.array('published_truth.npy',truth)
    products.array('corpus_id_map/rows.npy',np.arange(len(raw),dtype=np.int32))
    products.json('corpus_id_map/identity.json',dict(source_sha256=cfg['source']['sha256'],dataset='train',
        row_map='rows.npy',stable_id_format='vibe:sha256:<source>:train:%08d',
        candidate_arrays_are_explicit_train_row_map_references=True))
    cutoffs=np.asarray([rep.original_distances(raw[t],q).max() for t,q in zip(truth,queries)])
    products.array('published_cutoffs.npy',cutoffs)
    phase('validate_closed_fp64_reference')
    prior=rep.load_prior_fp64(cfg,roles,check)
    products.array('prior_fp64_top100.npy',prior)
    phase('fit_corpus_sample')
    tick=time.monotonic()
    fit_rows=rep.sample_fit_rows(len(raw),cfg['fit_rows'],cfg['seeds']['fit'])
    sample=raw[fit_rows].copy()
    products.array('fit_rows.npy',fit_rows)
    products.json('fit_id_map.json',dict(source_sha256=cfg['source']['sha256'],dataset='train',
        row_map='fit_rows.npy',stable_id_format='vibe:sha256:<source>:train:%08d',
        sha256=rep.sha(Path(root)/'fit_rows.npy'),seed=cfg['seeds']['fit']))
    sample_seconds=time.monotonic()-tick
    pca_start=time.monotonic()
    phase('fit_sample_pca',component=True)
    pca=rep.fit_pca(sample)
    pca_seconds=time.monotonic()-pca_start
    shared_fit_seconds=sample_seconds+pca_seconds
    for key,value in pca.items():products.array('shared_pca/'+key+'.npy',value)
    all_points=[];all_quality={};configs={s['id']:s for s in cfg['representations']};completed=[]
    ledger=[];models={};runtime_map={};fixed_records={}
    order=list(cfg['representations'])
    # Raw first supplies native truth. Remaining visit order is fixed before labels.
    rest=order[1:];np.random.Generator(np.random.PCG64(cfg['seeds']['schedule'])).shuffle(rest)
    order=[order[0]]+rest
    products.json('schedule.json',dict(configuration_order=[s['id'] for s in order],
        seed=cfg['seeds']['schedule'],cpu_one_thread_query_ids=sorted(
          [q['id'] for q in roles[:cfg['discovery_queries']]],
          key=lambda q:rep.fingerprint([cfg['seeds']['sensitivity'],q]))[:cfg['timing']['sensitivity_queries']]))
    sensitivity_ids=json.loads((Path(root)/'schedule.json').read_text())['cpu_one_thread_query_ids']
    sensitivity=[i for i,r in enumerate(roles) if r['id'] in sensitivity_ids]
    for spec in order:
        ident=spec['id'];prefix='representations/'+ident
        phase(ident+':fit',component=True);tick=time.monotonic()
        model=rep.fit_representation(spec,sample,pca,cfg,check)
        fit_seconds=time.monotonic()-tick
        write_start=time.monotonic()
        _save_model(products,prefix,model)
        model_write_seconds=time.monotonic()-write_start
        models[ident]={k:v for k,v in model.items() if k!='index'}
        phase(ident+':encode_transform',component=True);tick=time.monotonic()
        codes=centers=values=None
        if 'pq_bytes' in spec:
            codes,centers=rep.encode_pq(model,raw,cfg['block'],check)
            products.array(prefix+'/codes.npy',codes)
            # Independently check a stored-code slice against explicit reconstruction.
            q=rep.transform_batch(model,queries[:1])[0]
            decoded=rep.decode_codes(codes[:cfg['block']],centers).astype(float)
            adc=rep.adc_scores(q,codes[:len(decoded)],centers,model['metric'])
            brute=-decoded@q.astype(float) if model['metric']=='ip' else np.sum((decoded-q.astype(float))**2,axis=1)
            if not np.allclose(adc,brute,rtol=1e-12,atol=1e-12):raise RuntimeError('ADC decoding conformance')
        else:
            values=raw if spec['kind']=='raw' else np.empty((len(raw),spec['dim']),np.float32)
            if values is not raw:
                for start in range(0,len(raw),cfg['block']):
                    check();values[start:start+cfg['block']]=rep.transform_batch(model,raw[start:start+cfg['block']])
        transform_seconds=time.monotonic()-tick
        phase(ident+':fp64_reference');tick=time.monotonic()
        transformed_q=np.asarray([rep.transform_batch(model,q[None])[0] for q in queries])
        reference=rep.exact_representation_search(values,transformed_q,max(cfg['budgets']),model['metric'],
            codes,centers,torch,check,cfg['query_batch'],cfg['block'])
        reference_seconds=time.monotonic()-tick
        products.array(prefix+'/fp64_candidates.npy',reference.astype(np.int32))
        ref_quality=_evaluate(raw,queries,truth,cutoffs,reference,cfg['budgets'],cfg['k'],cfg['epsilon'],roles,ident+':fp64')
        products.json(prefix+'/fp64_quality.json',ref_quality)
        if codes is not None and model['metric']=='ip':
            phase(ident+':same_code_l2_diagnostic')
            l2ref=rep.exact_representation_search(None,transformed_q,max(cfg['budgets']),'l2',codes,centers,
                torch,check,cfg['query_batch'],cfg['block'])
            products.array(prefix+'/same_code_l2_candidates.npy',l2ref.astype(np.int32))
            products.json(prefix+'/same_code_l2_quality.json',_evaluate(raw,queries,truth,cutoffs,l2ref,
                cfg['budgets'],cfg['k'],cfg['epsilon'],roles,ident+':same_code_l2',reference))
        phase(ident+':flat_build',component=True);tick=time.monotonic()
        index=rep.make_flat(model,values,cfg['timing']['faiss_threads'])
        build_seconds=time.monotonic()-tick
        if spec['kind']=='raw' and 'pq_bytes' not in spec:
            phase('native_truth_gate')
            native=np.asarray([rep.faiss_search(index,q,cfg['k'],'ip') for q in queries])
            native_cutoffs=np.asarray([rep.original_distances(raw[t],q).max() for t,q in zip(native,queries)])
            delta=np.abs(native_cutoffs-cutoffs)
            prior_cutoffs=np.asarray([rep.original_distances(raw[t],q).max() for t,q in zip(prior,queries)])
            prior_delta=np.abs(prior_cutoffs-cutoffs)
            products.json('truth_gate.json',dict(passed=bool(max(np.max(delta),np.max(prior_delta))<=cfg['epsilon']),
                cutoff_differences=delta.tolist(),prior_fp64_id_cutoff_differences=prior_delta.tolist(),native_published_overlap=[len(set(a)&set(b)) for a,b in zip(native,truth)],
                prior_fp64_published_overlap=[len(set(a)&set(b)) for a,b in zip(prior,truth)],
                prior_fp64_native_overlap=[len(set(a)&set(b)) for a,b in zip(prior,native)],
                truth_label='published-learn-truth development metric, same VIBE distance-threshold formula',
                formal_test=False,prior_fp64_is_reused_closed_full_corpus=True))
            products.array('native_fp32_top100.npy',native)
            if max(np.max(delta),np.max(prior_delta))>cfg['epsilon']:raise RuntimeError('published/native truth cutoff gate')
        phase(ident+':faiss_quality')
        candidate=np.asarray([rep.faiss_search(index,q,max(cfg['budgets']),model['metric']) for q in transformed_q])
        products.array(prefix+'/faiss_candidates.npy',candidate.astype(np.int32))
        quality=_evaluate(raw,queries,truth,cutoffs,candidate,cfg['budgets'],cfg['k'],cfg['epsilon'],roles,ident+':cpu_flat',reference)
        products.json(prefix+'/faiss_quality.json',quality)
        all_quality[ident+':cpu_flat']=quality
        budgets=_selected_budgets(quality,cfg) or [cfg['budgets'][0],cfg['budgets'][-1]]
        resident=raw.nbytes+len(raw)*spec['bytes']+model['matrix'].nbytes+model['mean'].nbytes
        if centers is not None:resident+=centers.nbytes
        phase(ident+':cpu_timing')
        timings=[] if spec['diagnostic'] else _timings(model,raw,queries,roles,budgets,index,cfg,check)
        products.json(prefix+'/cpu_timings.json',timings)
        # Match independent timed call outputs to the audited quality candidates.
        _verify_timed_quality(timings,quality)
        points=_timing_points(ident+':cpu_flat',quality,timings,resident)
        all_points+=points
        products.json(prefix+'/cpu_points.json',points)
        phase(ident+':cpu_one_thread_sensitivity')
        single=[] if spec['diagnostic'] else _timings(model,raw,queries,roles,budgets,index,cfg,check,threads=1,subset=sensitivity)
        products.json(prefix+'/cpu_one_thread_timings.json',single)
        if not spec['diagnostic'] and codes is None:
            phase(ident+':cuda_quality')
            dv=torch.as_tensor(values,device='cuda',dtype=torch.float32)
            gpu_candidates=np.asarray([rep.search_backend(model,q,max(cfg['budgets']),device_values=dv,torch=torch) for q in queries])
            products.array(prefix+'/cuda_candidates.npy',gpu_candidates.astype(np.int32))
            gpu_quality=_evaluate(raw,queries,truth,cutoffs,gpu_candidates,cfg['budgets'],cfg['k'],cfg['epsilon'],roles,ident+':cuda',reference)
            products.json(prefix+'/cuda_quality.json',gpu_quality);all_quality[ident+':cuda']=gpu_quality
            phase(ident+':cuda_timing')
            gpu_times=_timings(model,raw,queries,roles,_selected_budgets(gpu_quality,cfg),None,cfg,check,torch,dv)
            products.json(prefix+'/cuda_timings.json',gpu_times);_verify_timed_quality(gpu_times,gpu_quality)
            all_points+=_timing_points(ident+':cuda',gpu_quality,gpu_times,resident)
            del dv;torch.cuda.empty_cache()
        if spec['diagnostic']:
            # Seed diagnostics are not eligible for selection or latency claims.
            all_points=[p for p in all_points if not p['pipeline'].startswith(ident+':')]
        offline=sample_seconds*(spec['kind'] in ('pca','hybrid','opq') or 'pq_bytes' in spec)+pca_seconds*(spec['kind'] in ('pca','hybrid'))+fit_seconds+transform_seconds+build_seconds+model_write_seconds
        cost=dict(id=ident,fit_seconds=fit_seconds,transform_encode_seconds=transform_seconds,
                  flat_build_seconds=build_seconds,reference_audit_seconds=reference_seconds,
                  shared_sample_pca_seconds=shared_fit_seconds,offline_seconds=offline,
                  raw_bytes=raw.nbytes,payload_bytes=len(raw)*spec['bytes'],resident_bytes=resident,
                  representation_fit_seconds=model['fit_costs']['representation_seconds'],
                  pq_fit_seconds=model['fit_costs']['pq_seconds'],model_write_seconds=model_write_seconds,
                  sample_seconds=sample_seconds,pca_seconds=pca_seconds,input_load_sha_seconds=input_load_sha_seconds,
                  finite_precision_reference=True,roundoff_certified=False,
                  hardware_profile='CPU14 Faiss / BLAS1; separate CUDA linear scans',
                  cold_load_seconds_in_global_input_phase=True)
        ledger.append(cost);products.json(prefix+'/costs.json',cost)
        runtime_map[ident]=dict(model=models[ident],resident=resident,offline=offline)
        completed.append(ident);products.json(prefix+'/complete.json',dict(id=ident,completed=True))
        del index,model,values,codes,centers,reference,candidate;gc.collect();torch.cuda.empty_cache()
    products.json('representation_ledger.json',ledger)
    products.json('linear_frontiers.json',frontiers(all_points,cfg))
    # Freeze up to seven IVF identities using discovery only, before check tuning.
    selected=select_ivf_identities(all_points,configs,cfg)
    products.json('ivf_selection.json',dict(ids=selected,role='geometry_discovery',check_never_tunes=True))
    for ident in selected:
        spec=configs[ident];model=runtime_map[ident]['model'];prefix='ivf/'+ident
        phase(ident+':ivf_train_encode',component=True);tick=time.monotonic()
        index=rep.make_ivf(model,sample,raw,cfg,check)
        build=time.monotonic()-tick
        write_start=time.monotonic()
        # Persist coded IVF. Flat IVF persists coarse centroids/list row IDs only;
        # its stored floats are reproducible from the unchanged source + matrix.
        index_path=Path(root)/(prefix.replace('/','-')+'.faiss')
        if 'pq_bytes' in spec:
            size=int(index.ntotal)*(spec['pq_bytes']+8)+cfg['ivf']['nlist']*spec['dim']*4+2**22
            products.reserve(size);faiss.write_index(index,str(index_path))
        coarse=index.quantizer.reconstruct_n(0,index.nlist)
        products.array(prefix+'/coarse.npy',coarse)
        lists=[];offsets=[0]
        for list_id in range(index.nlist):
            size=index.invlists.list_size(list_id)
            ptr=index.invlists.get_ids(list_id)
            try:lists.append(faiss.rev_swig_ptr(ptr,size).copy().astype(np.int32))
            finally:index.invlists.release_ids(list_id,ptr)
            offsets.append(offsets[-1]+size)
        products.array(prefix+'/list_rows.npy',np.concatenate(lists))
        products.array(prefix+'/list_offsets.npy',np.asarray(offsets,np.int64))
        del lists
        index_write_seconds=time.monotonic()-write_start
        phase(ident+':ivf_same_code_fullprobe')
        transformed_q=np.asarray([rep.transform_batch(model,q[None])[0] for q in queries])
        reference=rep.exact_representation_search(rep.IVFDecodedView(index),transformed_q,max(cfg['budgets']),
            model['metric'],torch=torch,check=check,batch=cfg['query_batch'],block=cfg['block'])
        fullprobe=rep.ivf_same_code_reference(index,transformed_q,max(cfg['budgets']))
        products.array(prefix+'/fp32_fullprobe_candidates.npy',fullprobe.astype(np.int32))
        products.array(prefix+'/same_code_fullprobe_candidates.npy',reference.astype(np.int32))
        products.json(prefix+'/same_code_fullprobe_quality.json',_evaluate(raw,queries,truth,cutoffs,reference,cfg['budgets'],cfg['k'],cfg['epsilon'],roles,ident+':ivf_fullprobe'))
        for nprobe in cfg['ivf']['nprobes']:
            index.nprobe=nprobe;pipeline=ident+':ivf'+str(nprobe)
            phase(pipeline+':quality')
            candidates=np.asarray([rep.faiss_search(index,q,max(cfg['budgets']),model['metric']) for q in transformed_q])
            products.array(prefix+'/'+str(nprobe)+'_candidates.npy',candidates.astype(np.int32))
            quality=_evaluate(raw,queries,truth,cutoffs,candidates,cfg['budgets'],cfg['k'],cfg['epsilon'],roles,pipeline,reference)
            products.json(prefix+'/'+str(nprobe)+'_quality.json',quality);all_quality[pipeline]=quality
            phase(pipeline+':timing')
            timings=_timings(model,raw,queries,roles,_selected_budgets(quality,cfg),index,cfg,check)
            products.json(prefix+'/'+str(nprobe)+'_timings.json',timings);_verify_timed_quality(timings,quality)
            ivf_points=_timing_points(pipeline,quality,timings,runtime_map[ident]['resident']+8*len(raw))
            for p in ivf_points:
                p['ivf_build_seconds']=build
                p['ivf_write_seconds']=index_write_seconds
            all_points+=ivf_points
        products.json(prefix+'/costs.json',dict(train_encode_seconds=build,write_seconds=index_write_seconds,raw_bytes=raw.nbytes,
            actual_serialized_bytes=index_path.stat().st_size if index_path.exists() else None,
            flat_float_payload_not_persisted='pq_bytes' not in spec,
            residual_codebook_is_own=True))
        del index;gc.collect()
    products.json('all_frontiers.json',frontiers(all_points,cfg));products.json('all_points.json',all_points)
    products.json('same_byte_frontiers.json',same_byte_frontiers(all_points,configs,cfg))
    products.json('same_dimension_frontiers.json',{str(d):frontiers([p for p in all_points if configs[p['pipeline'].split(':')[0]]['dim']==d],cfg) for d in (128,256)})
    # Only fixed CPU pipelines are used for matched-hardware oracle coefficients.
    oracle_pipelines=select_gaussian_pipelines(all_points,configs,cfg)
    products.json('oracle_selection.json',dict(pipelines=oracle_pipelines,role='geometry_discovery',fixed_representation_and_backend=True))
    solver_spent=0.;oracle_summaries=[]
    for pipeline in oracle_pipelines:
        ident,backend=pipeline.split(':');spec=configs[ident]
        phase(pipeline+':oracle_rebuild',component=True)
        model=rep.fit_representation(spec,sample,pca,cfg,check)
        if backend.startswith('ivf'):
            index_path=Path(root)/('ivf-'+ident+'.faiss')
            if index_path.exists():index=faiss.read_index(str(index_path))
            else:
                index=rep.make_ivf(model,sample,raw,cfg,check)
                np.testing.assert_array_equal(index.quantizer.reconstruct_n(0,index.nlist),
                    np.load(Path(root)/('ivf/'+ident+'/coarse.npy'),allow_pickle=False))
            index.nprobe=int(backend[3:])
        else:
            values=None
            if 'pq_bytes' in spec:rep.encode_pq(model,raw,cfg['block'],check)
            else:
                values=np.empty((len(raw),spec['dim']),np.float32)
                for start in range(0,len(raw),cfg['block']):
                    check();values[start:start+cfg['block']]=rep.transform_batch(model,raw[start:start+cfg['block']])
            index=rep.make_flat(model,values)
        phase(pipeline+':oracle_all_M_timing')
        timings=_timings(model,raw,queries,roles,cfg['budgets'],index,cfg,check)
        quality=all_quality[pipeline];_verify_timed_quality(timings,quality)
        prefix='oracle/'+ident+'-'+backend
        products.json(prefix+'/timings.json',timings)
        for role in ('geometry_discovery','geometry_check'):
            chosen=[i for i,r in enumerate(roles) if r['role']==role]
            a={metric:np.asarray([[next(r[metric] for r in quality if r['query_index']==qi and r['budget']==m)
                 for m in cfg['budgets']] for qi in chosen]) for metric in ('recall','strict_id')}
            times=np.asarray([[np.mean([r['seconds'] for r in timings if r['query_index']==qi and r['budget']==m])
                 for m in cfg['budgets']] for qi in chosen])
            products.json(prefix+'/'+role+'_coefficients.json',dict(query_ids=[roles[i]['id'] for i in chosen],
                budgets=cfg['budgets'],seconds=times.tolist(),quality={k:v.tolist() for k,v in a.items()},
                no_pilot_no_policy_cost=True,search_transform_transfer_rerank_paid=True))
            for metric,target in itertools.product(('recall','strict_id'),cfg['targets']):
                check();phase(pipeline+':oracle_solver')
                remaining=cfg['limits']['solver_total_seconds']-solver_spent
                if remaining<=0:
                    result=dict(status='unresolved_solver_total_cap')
                else:
                    # Three optimizers share the remaining aggregate allowance.
                    result=oracle_budget_bound(a[metric],times,target,min(cfg['limits']['solver_seconds'],remaining/3))
                    solver_spent+=result['solver_seconds']
                selected_control=select_fixed([p for p in all_points if ':cuda' not in p['pipeline']],target,metric)
                best=next((p for p in all_points if selected_control and p['role']==role and
                    p['metric']==metric and p['id']==selected_control['id']),None)
                control_passes=best is not None and best['quality']>=target
                lower=result.get('lp',{}).get('objective_seconds') if result.get('lp',{}).get('optimal') else None
                headroom=(best['seconds']-lower) if control_passes and lower is not None else None
                record=dict(pipeline=pipeline,role=role,metric=metric,target=target,result=result,
                    best_matched_fixed=best,frozen_control_meets_role_target=control_passes,controller_seconds_headroom=headroom,
                    robust_gain_claim=False,less_than_5pct_is_small_or_unresolved=True)
                oracle_summaries.append(record)
                products.json(prefix+'/'+role+'_'+metric+'_'+str(target)+'.json',record)
        del model,index;gc.collect()
    products.json('oracle_summary.json',oracle_summaries)
    products.json('lifecycle.json',lifecycle_frontier(all_points,ledger))
    phase('final_source_full_sha')
    if rep.sha(cfg['source']['path'],check)!=cfg['source']['sha256']:raise RuntimeError('source changed during S0')
    products.json('summary.json',dict(**seal_report(completed,[s['id'] for s in cfg['representations']]),
        elapsed_scientific_seconds=time.monotonic()-started,ivf_indices=selected,
        oracle_pipelines=oracle_pipelines,environment=environment,formal_RAG_or_Tri_claim=False))


def _verify_timed_quality(timings,quality):
    expected={(r['query_index'],r['budget']):r['returned'] for r in quality}
    for record in timings:
        if record['returned']!=expected[(record['query_index'],record['budget'])]:
            raise RuntimeError('independent timed search differs from quality candidates')


def frontiers(points,cfg):
    rows=[]
    for metric,target,hardware in itertools.product(('recall','strict_id'),cfg['targets'],('cpu14','cuda')):
        subset=[p for p in points if (':cuda' in p['pipeline'])==(hardware=='cuda')]
        selected=select_fixed(subset,target,metric,'geometry_discovery')
        audit=next((p for p in subset if selected and p['id']==selected['id'] and
                    p['role']=='geometry_check' and p['metric']==metric),None)
        rows.append(dict(metric=metric,target=target,hardware=hardware,
            discovery_selected=selected,frozen_check_audit=audit,check_reselection=False))
    return rows


def same_byte_frontiers(points,configs,cfg):
    return {str(b):frontiers([p for p in points if configs[p['pipeline'].split(':')[0]]['bytes']==b],cfg)
            for b in (32,64)}


def _best_config(points,configs,predicate,cfg):
    eligible=[p for p in points if p['role']=='geometry_discovery' and p['metric']=='recall' and
              ':cuda' not in p['pipeline'] and p['quality']>=min(cfg['targets']) and predicate(configs[p['pipeline'].split(':')[0]])]
    return min(eligible,key=lambda p:(p['seconds'],p['resident_bytes'],p['id']))['pipeline'].split(':')[0] if eligible else None


def select_ivf_identities(points,configs,cfg):
    ids=['raw640']
    predicates=[lambda s:s['kind']=='pca' and 'pq_bytes' not in s]
    predicates += [lambda s,b=b:s.get('pq_bytes')==b and s['kind'] not in ('gaussian','hybrid') for b in (32,64)]
    predicates += [lambda s:s['kind']=='gaussian' and 'pq_bytes' not in s,
                   lambda s:s['kind']=='hybrid' and 'pq_bytes' not in s,
                   lambda s:s['kind'] in ('gaussian','hybrid') and 'pq_bytes' in s]
    for predicate in predicates:
        selected=_best_config(points,configs,predicate,cfg)
        if selected and selected not in ids:ids.append(selected)
    return ids


def select_gaussian_pipelines(points,configs,cfg):
    selected=[]
    for predicate in (lambda s:s['kind']=='gaussian' and 'pq_bytes' not in s,
                      lambda s:s['kind']=='hybrid' and 'pq_bytes' not in s,
                      lambda s:s['kind'] in ('gaussian','hybrid') and 'pq_bytes' in s):
        eligible=[p for p in points if p['role']=='geometry_discovery' and p['metric']=='recall' and
                  ':cuda' not in p['pipeline'] and p['quality']>=min(cfg['targets']) and
                  predicate(configs[p['pipeline'].split(':')[0]])]
        if eligible:selected.append(min(eligible,key=lambda p:(p['seconds'],p['resident_bytes'],p['id']))['pipeline'])
    return selected


def lifecycle_frontier(points,ledger):
    costs={r['id']:r for r in ledger};rows=[]
    for p in points:
        ident=p['pipeline'].split(':')[0];c=costs[ident]
        rows.append(dict(id=p['id'],role=p['role'],metric=p['metric'],quality=p['quality'],
            cold_input_sha_load_seconds=c['input_load_sha_seconds'],
            cold_start_input_verification_reported_separately=True,
            **lifecycle_cost((c['offline_seconds']-c['pq_fit_seconds']-c['transform_encode_seconds']-c['flat_build_seconds']+p['ivf_build_seconds']+p['ivf_write_seconds']) if 'ivf_build_seconds' in p else c['offline_seconds'],p['seconds'],c['raw_bytes'],c['payload_bytes'],
                             max(0,p['resident_bytes']-c['raw_bytes']-c['payload_bytes']))))
    cross=[]
    eligible=[r for r in rows if r['role']=='geometry_discovery' and r['metric']=='recall' and r['quality']>=.95]
    for a,b in itertools.combinations(eligible,2):
        q=break_even(a['offline_seconds'],a['query_seconds'],b['offline_seconds'],b['query_seconds'])
        if q is not None:cross.append(dict(a=a['id'],b=b['id'],queries=q))
    for r in rows:
        r['cold_horizons']={key:value+r['cold_input_sha_load_seconds'] for key,value in r['horizons'].items()}
    return dict(costs=rows,break_even=cross,source_acquisition_and_B0='historical sunk, not zero new S0 cost',
                reference_audit_is_not_online_deployment=True)
