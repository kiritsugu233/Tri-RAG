"""S0 v2 staged screen: all quality first, paired cohorts and actual cold costs."""
import gc
import itertools
import json
from pathlib import Path
import stat
import time

import numpy as np

from mechanism import tls_rag_step5_route6_representation_v2 as rep
from mechanism import tls_rag_step5_route6_representation_screen_v1 as frozen

benchmark_recall = frozen.benchmark_recall
loss_transitions = frozen.loss_transitions
measure_call = frozen.measure_call
oracle_budget_bound = frozen.oracle_budget_bound
select_fixed = frozen.select_fixed
frontiers = frozen.frontiers
same_byte_frontiers = frozen.same_byte_frontiers
select_ivf_identities = frozen.select_ivf_identities
seal_report = frozen.seal_report
class Products(frozen.Products):
    def __init__(self,root,limit,check=lambda:None,enforce_budget=True):
        super().__init__(root,limit,check);self.enforce_budget=enforce_budget

    def validate_path(self,name):
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('product path outside run')
        path=self.root
        if path.is_symlink():raise RuntimeError('run directory symlink')
        for part in relative.parts:
            path=path/part
            if path.is_symlink():raise RuntimeError('run directory symlink')

    def json(self,name,value):
        self.validate_path(name);return super().json(name,value)

    def array(self,name,value):
        self.validate_path(name);return super().array(name,value)

    def reserve(self,size):
        self.check()
        if size<0:raise ValueError('negative product size')
        if not self.enforce_budget:return
        used=0
        for path in self.root.rglob('*'):
            try:info=path.lstat()
            except FileNotFoundError:continue  # owned atomic telemetry rename
            if stat.S_ISLNK(info.st_mode):raise RuntimeError('run directory symlink')
            if stat.S_ISREG(info.st_mode):used+=info.st_size
        if size<0 or used+size>self.limit:raise RuntimeError('artifact write budget')


def products_for(root,cfg,check=lambda:None):
    return Products(root,cfg['limits']['artifact_bytes'],check,
        enforce_budget=cfg.get('supervision',{}).get('resource_budget_enforced',True))


def timing_cohort(roles, cfg, hardware):
    if hardware == 'cuda':
        return list(range(len(roles)))
    if hardware != 'cpu14': raise ValueError('unknown hardware cohort')
    selected = []
    for role in ('geometry_discovery', 'geometry_check'):
        indices = [i for i, r in enumerate(roles) if r['role'] == role]
        selected += sorted(indices, key=lambda i: rep.fingerprint(
            [cfg['timing']['cohort_seed'], roles[i]['id']]))[:cfg['timing']['cpu_queries_per_role']]
    return selected


def evaluate(raw, queries, truth, cutoffs, candidates, cfg, roles, pipeline, reference=None):
    return frozen._evaluate(raw, queries, truth, cutoffs, candidates, cfg['budgets'], cfg['k'],
                            cfg['epsilon'], roles, pipeline, reference)


def selected_budgets(quality, cfg):
    return frozen._selected_budgets(quality, cfg) or [cfg['budgets'][0], cfg['budgets'][-1]]


def timings(model, raw, queries, roles, budgets, index, cfg, check, torch=None,
            cuda_index=None, subset=None, threads=None):
    import faiss
    faiss.omp_set_num_threads(threads or cfg['timing']['faiss_threads'])
    hardware = 'cuda' if cuda_index is not None else 'cpu14'
    chosen = timing_cohort(roles, cfg, hardware) if subset is None else list(subset)
    schedule = list(itertools.product(chosen, budgets))
    np.random.Generator(np.random.PCG64(cfg['seeds']['schedule'])).shuffle(schedule)
    rows = []; sync = torch.cuda.synchronize if cuda_index is not None else lambda: None
    try:
        for qi, m in schedule:
            for repeat in range(-cfg['timing']['warmup'], cfg['timing']['repeats']):
                check()
                def call():
                    tick = time.perf_counter()
                    ids = rep.search_backend(model, queries[qi], m, index=index,
                                             cuda_index=cuda_index, torch=torch)
                    search = time.perf_counter()-tick; tick = time.perf_counter()
                    returned, distances = rep.rerank(raw, queries[qi], ids, cfg['k'])
                    return ids, returned, distances, search, time.perf_counter()-tick
                (ids, returned, distances, search, rerank), seconds = measure_call(call, sync)
                if repeat >= 0:
                    rows.append(dict(query_index=int(qi), query_id=roles[qi]['id'], role=roles[qi]['role'],
                        budget=int(m), repeat=repeat, seconds=seconds, search_transform_transfer_seconds=search,
                        rerank_seconds=rerank, candidate_sha256=rep.fingerprint(ids.tolist()),
                        returned=returned.tolist(), distances=distances.tolist(), hardware_profile=hardware))
    finally:
        faiss.omp_set_num_threads(cfg['timing']['faiss_threads'])
    return rows


def timing_points(pipeline, quality, measured, resident):
    """Full-role quality controls selection; paired-cohort quality is explicit."""
    points = frozen._timing_points(pipeline, quality, measured, resident)
    for point in points:
        indices = {r['query_index'] for r in measured if r['role']==point['role'] and r['budget']==point['budget']}
        paired = [r[point['metric']] for r in quality if r['role']==point['role']
                  and r['budget']==point['budget'] and r['query_index'] in indices]
        point.update(timing_cohort_quality=float(np.mean(paired)), timing_queries=len(indices),
                     quality_queries=len({r['query_index'] for r in quality if r['role']==point['role']}),
                     quality_is_full_role=True, cohort_identity=rep.fingerprint(sorted(indices)))
    return points


def verify_timings(measured, quality):
    frozen._verify_timed_quality(measured, quality)


def save_model(products, prefix, model, cfg):
    import faiss
    tick = time.monotonic()
    products.array(prefix+'/matrix.npy', model['matrix'])
    products.array(prefix+'/mean.npy', model['mean'])
    root = products.root/prefix
    meta = dict(namespace=cfg['namespace'], spec=model['spec'], metric=model['metric'],
                fingerprint=model['fingerprint'], fit_costs=model['fit_costs'],
                matrix_file_sha256=rep.sha(root/'matrix.npy'), mean_file_sha256=rep.sha(root/'mean.npy'))
    if 'index' in model:
        products.array(prefix+'/centroids.npy', faiss.vector_to_array(model['index'].pq.centroids))
        meta['centroids_file_sha256'] = rep.sha(root/'centroids.npy')
    products.json(prefix+'/model.json', meta)
    return time.monotonic()-tick


def prepared(root, cfg):
    root = Path(root); receipt = json.loads((root/'prepared.json').read_text())
    if receipt['source_sha256'] != cfg['source']['sha256'] or receipt['roles_sha256'] != cfg['roles_sha256']:
        raise ValueError('prepared input identity')
    for name, digest in receipt['files'].items():
        if rep.sha(root/name) != digest: raise ValueError('prepared product mutated: '+name)
    raw = rep.source_mapping(cfg)
    arrays = [np.load(root/name, allow_pickle=False) for name in
              ('queries.npy','published_truth.npy','published_cutoffs.npy','fit_rows.npy')]
    roles = json.loads((root/'roles.json').read_text())['queries']
    return (raw, *arrays, roles, receipt)


def prepare(cfg, root, torch, check, phase):
    products = products_for(root,cfg,check); root = Path(root)
    raw, queries, truth, roles, inputs = rep.load_screen_inputs(cfg, check, phase)
    products.json('inputs.json', inputs); products.json('roles.json', dict(queries=roles))
    products.array('queries.npy', queries); products.array('published_truth.npy', truth)
    products.array('corpus_id_map/rows.npy', np.arange(len(raw), dtype=np.int32))
    products.json('corpus_id_map/identity.json', dict(source_sha256=cfg['source']['sha256'],dataset='train',
        row_map='rows.npy', stable_id_format='vibe:sha256:<source>:train:%08d',
        candidate_arrays_are_explicit_train_row_map_references=True))
    products.array('published_cutoffs.npy', np.asarray([rep.original_distances(raw[t],q).max() for t,q in zip(truth,queries)]))
    phase('validate_closed_fp64_reference'); products.array('prior_fp64_top100.npy', rep.load_prior_fp64(cfg, roles, check))
    phase('fit_corpus_sample'); tick = time.monotonic()
    fit_rows = rep.sample_fit_rows(len(raw), cfg['fit_rows'], cfg['seeds']['fit'])
    products.array('fit_rows.npy', fit_rows); sample = np.array(raw[fit_rows], copy=True)
    sample_seconds = time.monotonic()-tick
    phase('fit_sample_pca', component=True); tick = time.monotonic(); pca = rep.fit_pca(sample)
    pca_seconds = time.monotonic()-tick
    for key,value in pca.items(): products.array('shared_pca/'+key+'.npy', value)
    reuse = cfg['reuse_v1']; old = Path(reuse['path']); phase('verify_closed_v1_reuse')
    if rep.sha(old/'delivery_index.json', check) != reuse['index_sha256']:
        raise ValueError('old S0 index identity')
    entries = json.loads((old/'delivery_index.json').read_text())['files']
    for entry in entries:
        check(); path = old/entry['path']
        if old.resolve() not in path.resolve().parents or path.is_symlink(): raise ValueError('unsafe v1 path')
        if path.stat().st_size != entry['bytes'] or rep.sha(path, check) != entry['sha256']:
            raise ValueError('old S0 product mutated')
    old_freeze = json.loads((old/'source_freeze.json').read_text())
    if old_freeze['commit'] != reuse['scientific_commit']: raise ValueError('old S0 commit')
    for key in ('representations','seeds','training','roles_sha256','epsilon','budgets','fit_rows'):
        if old_freeze['config'][key] != cfg[key]: raise ValueError('v1 reuse protocol mismatch: '+key)
    if rep.sha(old/'fit_rows.npy') != rep.sha(root/'fit_rows.npy'): raise ValueError('v1 fit rows mismatch')
    # Governance/test-membership bytes changed with explicit v2 authorization;
    # all original scientific dependencies remain the same frozen bytes.
    repo = Path(__file__).resolve().parents[1]
    for name,digest in old_freeze['sources'].items():
        if name not in ('scripts/run_tests.sh','docs/code_protection.json') and rep.sha(repo/name) != digest:
            raise ValueError('old scientific dependency changed: '+name)
    products.json('v1_reuse_verification.json', dict(index_sha256=reuse['index_sha256'],
        verified_products=len(entries), original_run_preserved=True, prior_times_are_not_v2_times=True))
    rest = list(cfg['representations'][1:]); np.random.Generator(np.random.PCG64(cfg['seeds']['schedule'])).shuffle(rest)
    schedule = dict(configuration_order=[cfg['representations'][0]['id']]+[s['id'] for s in rest],
        cpu_timing_indices=timing_cohort(roles,cfg,'cpu14'), cuda_timing_indices=timing_cohort(roles,cfg,'cuda'),
        cpu_one_thread_indices=sorted(range(cfg['discovery_queries']),
            key=lambda i:rep.fingerprint([cfg['seeds']['sensitivity'],roles[i]['id']]))[:cfg['timing']['sensitivity_queries']],
        seed=cfg['seeds']['schedule'], all_quality_before_any_main_timing=True)
    products.json('schedule.json', schedule)
    names=['inputs.json','roles.json','queries.npy','published_truth.npy','published_cutoffs.npy','fit_rows.npy',
           'prior_fp64_top100.npy','schedule.json','v1_reuse_verification.json']
    names += ['shared_pca/'+key+'.npy' for key in pca]
    products.json('prepared.json',dict(source_sha256=cfg['source']['sha256'], roles_sha256=cfg['roles_sha256'],
        files={name:rep.sha(root/name) for name in names}, sample_seconds=sample_seconds,pca_seconds=pca_seconds,
        input_audit_seconds=sum(inputs[k] for k in ('source_roles_sha_seconds','corpus_norm_audit_seconds','query_truth_load_seconds'))))


def _pca(root):
    return {key:np.load(Path(root)/('shared_pca/'+key+'.npy'),allow_pickle=False) for key in ('mean','basis','eigenvalues')}


def _values(model, raw, cfg, check):
    if model['spec']['kind']=='raw': return raw
    values = np.empty((len(raw),model['spec']['dim']),np.float32)
    for start in range(0,len(raw),cfg['block']):
        check(); values[start:start+cfg['block']] = rep.transform_batch(model,raw[start:start+cfg['block']])
    return values


def _reuse_pq64(cfg, root, spec, products):
    """Only the fully closed, identical v1 PQ64 has reusable measured fit costs."""
    import faiss
    old=Path(cfg['reuse_v1']['path'])/'representations'/spec['id']
    if spec['id']!='pq64' or not (old/'complete.json').exists(): return None
    meta=json.loads((old/'model.json').read_text()); costs=json.loads((old/'costs.json').read_text())
    if meta['spec']!=spec: raise ValueError('v1 PQ spec mismatch')
    model=dict(spec=spec,metric=meta['metric'],matrix=np.load(old/'matrix.npy'),mean=np.load(old/'mean.npy'),
        fit_costs=dict(representation_seconds=costs['representation_fit_seconds'],pq_seconds=costs['pq_fit_seconds']))
    model['fingerprint']=rep.model_fingerprint(model,cfg)
    model['source_sha256']=cfg['source']['sha256']
    index=faiss.IndexPQ(spec['dim'],spec['pq_bytes'],8,faiss.METRIC_INNER_PRODUCT)
    faiss.copy_array_to_vector(np.load(old/'centroids.npy').reshape(-1),index.pq.centroids); index.is_trained=True
    model['index']=index
    return model,costs,np.load(old/'codes.npy',mmap_mode='r')


def quality(cfg, root, ident, torch, check, phase):
    import faiss
    root=Path(root);products=products_for(root,cfg,check)
    raw,queries,truth,cutoffs,fit_rows,roles,receipt=prepared(root,cfg)
    spec=next(s for s in cfg['representations'] if s['id']==ident); prefix='representations/'+ident
    sample=np.array(raw[fit_rows],copy=True);phase(ident+':fit',component=True);tick=time.monotonic()
    reuse=_reuse_pq64(cfg,root,spec,products)
    if reuse:
        model,old_costs,codes=reuse; historical_fit=old_costs['fit_seconds']
    else:
        model=rep.fit_representation(spec,sample,_pca(root),cfg,check); codes=None;historical_fit=None
    actual_fit=time.monotonic()-tick
    write_seconds=save_model(products,prefix,model,cfg)
    phase(ident+':encode_transform',component=True); tick=time.monotonic(); centers=None
    if 'pq_bytes' in spec:
        if reuse:
            faiss.copy_array_to_vector(np.asarray(codes).reshape(-1),model['index'].codes);model['index'].ntotal=len(codes)
            centers=faiss.vector_to_array(model['index'].pq.centroids).reshape(spec['pq_bytes'],256,-1)
        else: codes,centers=rep.encode_pq(model,raw,cfg['block'],check)
        encode_seconds=time.monotonic()-tick;tick=time.monotonic();products.array(prefix+'/codes.npy',codes)
        code_write=time.monotonic()-tick
        decoded=rep.decode_codes(codes[:min(257,len(codes))],centers)
        q=rep.transform_batch(model,queries[:1])[0]
        np.testing.assert_allclose(rep.adc_scores(q,codes[:len(decoded)],centers,model['metric']),
            -decoded.astype(float)@q.astype(float) if model['metric']=='ip' else np.sum((decoded.astype(float)-q.astype(float))**2,axis=1),rtol=2e-12,atol=2e-12)
        values=None
    else:
        values=_values(model,raw,cfg,check);encode_seconds=time.monotonic()-tick;code_write=0.
    transformed=rep.query_coordinates(model,queries)
    phase(ident+':fp64_reference');tick=time.monotonic()
    reference=rep.exact_representation_search(values,transformed,max(cfg['budgets']),model['metric'],
        codes,centers,torch,check,cfg['query_batch'],cfg['block'])
    audit_seconds=time.monotonic()-tick
    products.array(prefix+'/fp64_candidates.npy',reference.astype(np.int32))
    products.json(prefix+'/fp64_quality.json',evaluate(raw,queries,truth,cutoffs,reference,cfg,roles,ident+':fp64'))
    if 'pq_bytes' in spec and model['metric']=='ip':
        phase(ident+':same_code_l2')
        l2=rep.exact_representation_search(None,transformed,max(cfg['budgets']),'l2',codes,centers,torch,check,cfg['query_batch'],cfg['block'])
        products.array(prefix+'/same_code_l2_candidates.npy',l2.astype(np.int32))
        products.json(prefix+'/same_code_l2_quality.json',evaluate(raw,queries,truth,cutoffs,l2,cfg,roles,ident+':same_code_l2',reference))
    phase(ident+':flat_build',component=True);tick=time.monotonic()
    index=rep.make_flat(model,values,cfg['timing']['faiss_threads']);build=time.monotonic()-tick
    phase(ident+':faiss_quality')
    candidates=np.asarray([rep.faiss_search(index,q,max(cfg['budgets']),model['metric']) for q in transformed])
    products.array(prefix+'/faiss_candidates.npy',candidates.astype(np.int32))
    products.json(prefix+'/faiss_quality.json',evaluate(raw,queries,truth,cutoffs,candidates,cfg,roles,ident+':cpu_flat',reference))
    if ident=='raw640':
        native=np.asarray([rep.faiss_search(index,q,cfg['k'],'ip') for q in queries])
        prior=np.load(root/'prior_fp64_top100.npy')
        delta=np.asarray([abs(float(rep.original_distances(raw[t],q).max())-float(c)) for t,q,c in zip(native,queries,cutoffs)])
        old_delta=np.asarray([abs(float(rep.original_distances(raw[t],q).max())-float(c)) for t,q,c in zip(prior,queries,cutoffs)])
        products.json('truth_gate.json',dict(passed=bool(max(delta.max(),old_delta.max())<=cfg['epsilon']),
            cutoff_differences=delta.tolist(),prior_cutoff_differences=old_delta.tolist(),
            native_published_overlap=[len(set(a)&set(b)) for a,b in zip(native,truth)],formal_test=False))
        products.array('native_fp32_top100.npy',native)
        if max(delta.max(),old_delta.max())>cfg['epsilon']:raise ValueError('native published truth cutoff gate')
    gpu_preparation=None
    if not spec['diagnostic'] and codes is None:
        del index;gc.collect();phase(ident+':cuda_prepare',component=True)
        cache=rep.prepare_cuda_index(values,model,cfg,torch,check)
        values=None
        gpu_preparation={k:cache[k] for k in ('h2d_seconds','norm_prepare_seconds','payload_bytes','norm_bytes','binding','fingerprint')}
        phase(ident+':cuda_quality')
        gpu=np.asarray([rep.search_backend(model,q,max(cfg['budgets']),cuda_index=cache,torch=torch) for q in queries])
        products.array(prefix+'/cuda_candidates.npy',gpu.astype(np.int32))
        products.json(prefix+'/cuda_quality.json',evaluate(raw,queries,truth,cutoffs,gpu,cfg,roles,ident+':cuda',reference))
        del cache;gc.collect();torch.cuda.empty_cache()
    products.json(prefix+'/quality_costs.json',dict(actual_model_fit_or_load_seconds=actual_fit,
        training_seconds=historical_fit if reuse else actual_fit, reused_v1=bool(reuse),
        representation_fit_seconds=model['fit_costs']['representation_seconds'],pq_fit_seconds=model['fit_costs']['pq_seconds'],
        historical_transform_encode_seconds=old_costs['transform_encode_seconds'] if reuse else None,
        transform_encode_seconds=encode_seconds,model_write_seconds=write_seconds,code_write_seconds=code_write,
        flat_build_seconds=build,reference_audit_seconds=audit_seconds,gpu_prepare=gpu_preparation,
        fit_rows_sha256=rep.sha(root/'fit_rows.npy'),source_sha256=cfg['source']['sha256']))
    if codes is not None:products.json(prefix+'/codes_identity.json',dict(sha256=rep.sha(root/(prefix+'/codes.npy'),check)))
    products.json(prefix+'/quality_complete.json',dict(id=ident,quality_queries=len(queries),diagnostic=spec['diagnostic'],
        all_competitors=len(raw),latency_complete=False,roundoff_certified=False))


def backend_state(cfg,root,ident,check,phase,payload=True):
    root=Path(root);spec=next(s for s in cfg['representations'] if s['id']==ident)
    tick=time.monotonic();model=rep.model_from_products(root/'representations'/ident,spec,cfg,restore_pq=payload)
    load=time.monotonic()-tick;raw,queries,truth,cutoffs,fit_rows,roles,receipt=prepared(root,cfg)
    phase(ident+':restore_payload',component=True);tick=time.monotonic()
    if not payload:values=None
    elif 'pq_bytes' in spec:
        path=root/'representations'/ident/'codes.npy'
        identity=json.loads((path.parent/'codes_identity.json').read_text())
        rep.install_codes(model,path,identity['sha256'],check);values=None
    else: values=_values(model,raw,cfg,check)
    payload_seconds=time.monotonic()-tick
    return model,raw,queries,truth,cutoffs,roles,receipt,values,dict(model_load_seconds=load,payload_restore_seconds=payload_seconds)


def lifecycle_cost(cost, backend, raw_bytes, payload_bytes, model_bytes, norm_bytes=0):
    """Each actual deployment path pays only its own measured preparation."""
    if min(raw_bytes,payload_bytes,model_bytes,norm_bytes)<0 or raw_bytes<=0:raise ValueError('retained raw cost required')
    shared=cost['sample_seconds']+cost['pca_seconds']+cost['training_seconds']
    common=shared+cost['model_write_seconds']+cost['code_write_seconds']+cost['model_load_seconds']
    if backend=='cuda':
        offline=common+cost['transform_encode_seconds']+cost['h2d_seconds']+cost['norm_prepare_seconds']
        duplicate_cpu_payload=0
    elif backend=='cpu14':
        offline=common+cost['transform_encode_seconds']+cost['flat_build_seconds']+cost['payload_restore_seconds']
        duplicate_cpu_payload=0
    else:raise ValueError('hardware cost ledger')
    return dict(hardware=backend,offline_seconds=float(offline),raw_bytes=int(raw_bytes),payload_bytes=int(payload_bytes),
        model_bytes=int(model_bytes),norm_bytes=int(norm_bytes),resident_bytes=int(raw_bytes+payload_bytes+model_bytes+norm_bytes),
        duplicate_cpu_payload_bytes=duplicate_cpu_payload,input_verification_seconds=cost['input_audit_seconds'],
        cpu_flat_float_payload_serialized=False,cold_index_is_rebuilt_from_retained_source=True,
        full_raw_source_file_bytes=cost['source_file_bytes'],measured_components=cost,
        gpu_does_not_inherit_cpu_flat_build=True,actual_physical_peak_is_separate=True)


def linear_timing(cfg,root,ident,torch,check,phase,all_budgets=False,backend=None):
    root=Path(root);products=products_for(root,cfg,check)
    model,raw,queries,truth,cutoffs,roles,receipt,values,restore=backend_state(cfg,root,ident,check,phase)
    spec=model['spec'];prefix='representations/'+ident
    if spec['diagnostic']:raise ValueError('seed diagnostic cannot enter timing')
    qcost=json.loads((root/prefix/'quality_costs.json').read_text())
    model_bytes=sum((root/prefix/name).stat().st_size for name in ('matrix.npy','mean.npy','model.json'))
    if 'pq_bytes' in spec:model_bytes+=(root/prefix/'centroids.npy').stat().st_size
    targets=['cpu14']+(['cuda'] if values is not None else []) if backend is None else [backend]
    for hardware in targets:
        check();is_gpu=hardware=='cuda';cache=None;index=None
        quality=json.loads((root/prefix/('cuda_quality.json' if is_gpu else 'faiss_quality.json')).read_text())
        prepare_cost=dict(h2d_seconds=0.,norm_prepare_seconds=0.,norm_bytes=0)
        phase(ident+':'+hardware+':build',component=True);tick=time.monotonic()
        if is_gpu:
            if values is None:
                restore_tick=time.monotonic();values=_values(model,raw,cfg,check)
                restore['payload_restore_seconds']=time.monotonic()-restore_tick
            cache=rep.prepare_cuda_index(values,model,cfg,torch,check)
            values=None  # release the CPU transformation workspace before warm GPU calls
            prepare_cost={k:cache[k] for k in ('h2d_seconds','norm_prepare_seconds','norm_bytes')};build=0.
        else:
            index=rep.make_flat(model,values,cfg['timing']['faiss_threads']);build=time.monotonic()-tick
            values=None  # the Faiss index owns the single search-payload copy
        phase(ident+':'+hardware+':timing')
        budgets=cfg['budgets'] if all_budgets else selected_budgets(quality,cfg)
        measured=timings(model,raw,queries,roles,budgets,index,cfg,check,torch,cache)
        verify_timings(measured,quality)
        p=('oracle/'+ident+'-'+hardware) if all_budgets else prefix+'/'+hardware
        products.json(p+'/timings.json',measured)
        cost=dict(sample_seconds=receipt['sample_seconds'] if (spec['kind'] in ('pca','hybrid','opq') or 'pq_bytes' in spec) else 0.,
            pca_seconds=receipt['pca_seconds'] if spec['kind'] in ('pca','hybrid') else 0.,training_seconds=qcost['training_seconds'],
            model_write_seconds=qcost['model_write_seconds'],code_write_seconds=qcost['code_write_seconds'],
            model_load_seconds=restore['model_load_seconds'],payload_restore_seconds=restore['payload_restore_seconds'] if 'pq_bytes' in spec else 0.,
            transform_encode_seconds=(qcost['historical_transform_encode_seconds'] if qcost['reused_v1'] else qcost['transform_encode_seconds']) if 'pq_bytes' in spec else restore['payload_restore_seconds'],
            flat_build_seconds=build,input_audit_seconds=receipt['input_audit_seconds'],source_file_bytes=cfg['source']['bytes'],**prepare_cost)
        ledger=lifecycle_cost(cost,hardware,raw.nbytes,len(raw)*spec['bytes'],model_bytes,prepare_cost['norm_bytes'])
        points=timing_points(ident+(':'+hardware if is_gpu else ':cpu_flat'),quality,measured,ledger['resident_bytes'])
        products.json(p+'/points.json',points);products.json(p+'/costs.json',ledger)
        if not all_budgets and not is_gpu:
            subset=json.loads((root/'schedule.json').read_text())['cpu_one_thread_indices']
            phase(ident+':cpu_one_thread_sensitivity')
            single=timings(model,raw,queries,roles,budgets,index,cfg,check,threads=1,subset=subset)
            verify_timings(single,quality);products.json(p+'/cpu_one_thread_timings.json',single)
        if cache is not None:del cache
        del index;gc.collect();torch.cuda.empty_cache()
    if not all_budgets:products.json(prefix+'/timing_complete.json',dict(id=ident,backends=targets,
        cpu_queries=len(timing_cohort(roles,cfg,'cpu14')) if 'cpu14' in targets else 0,
        gpu_queries=len(roles) if 'cuda' in targets else 0))


def collect_points(root):
    points=[]
    for p in sorted(Path(root).glob('representations/*/*/points.json')):points+=json.loads(p.read_text())
    for p in sorted(Path(root).glob('ivf/*/*/points.json')):points+=json.loads(p.read_text())
    return points


def select_gaussian_pipelines(points,configs,cfg):
    """At most3 families; rank discovery cost relative to same-hardware control."""
    selected=[]
    predicates=(lambda s:s['kind']=='gaussian' and 'pq_bytes' not in s,
                lambda s:s['kind']=='hybrid' and 'pq_bytes' not in s,
                lambda s:s['kind'] in ('gaussian','hybrid') and 'pq_bytes' in s)
    for predicate in predicates:
        candidates=[]
        for p in points:
            spec=configs[p['pipeline'].split(':')[0]]
            if p['role']!='geometry_discovery' or p['metric']!='recall' or spec['diagnostic'] or not predicate(spec):continue
            for target in cfg['targets']:
                if p['quality']<target:continue
                controls=[c for c in points if c['role']=='geometry_discovery' and c['metric']=='recall'
                    and c['hardware_profile']==p['hardware_profile'] and c['quality']>=target
                    and configs[c['pipeline'].split(':')[0]]['kind'] not in ('gaussian','hybrid')]
                if controls:
                    base=min(c['seconds'] for c in controls)
                    candidates.append((p['seconds']/base,p['seconds'],p['resident_bytes'],p['pipeline']))
        if candidates:selected.append(min(candidates)[-1])
    return selected


def write_selections(cfg,root,check=lambda:None):
    root=Path(root);products=products_for(root,cfg,check)
    points=collect_points(root);configs={s['id']:s for s in cfg['representations']}
    products.json('all_points.json',points);products.json('all_frontiers.json',frontiers(points,cfg))
    products.json('same_byte_frontiers.json',same_byte_frontiers(points,configs,cfg))
    products.json('same_dimension_frontiers.json',{str(d):frontiers([p for p in points if configs[p['pipeline'].split(':')[0]]['dim']==d],cfg) for d in (128,256)})
    products.json('oracle_selection.json',dict(pipelines=select_gaussian_pipelines(points,configs,cfg),role='geometry_discovery',
        score='relative to best measured non-Gaussian control on same hardware and quality target'))
    products.json('lifecycle.json',lifecycle_tables(cfg,root,points))


def lifecycle_tables(cfg,root,points):
    root=Path(root);rows=[];cold_input=json.loads((root/'prepared.json').read_text())['input_audit_seconds']
    for point in points:
        ident,backend=point['pipeline'].split(':')
        if backend.startswith('ivf'):
            cost=json.loads((root/'ivf'/ident/'costs.json').read_text())
            offline=cost['offline_seconds'];resident=cost['resident_bytes'];components=cost
        else:
            hardware='cuda' if backend=='cuda' else 'cpu14'
            components=json.loads((root/'representations'/ident/hardware/'costs.json').read_text())
            offline=components['offline_seconds'];resident=components['resident_bytes']
        rows.append(dict(id=point['id'],pipeline=point['pipeline'],hardware=point['hardware_profile'],role=point['role'],
            metric=point['metric'],quality=point['quality'],timing_cohort_quality=point['timing_cohort_quality'],
            timing_queries=point['timing_queries'],offline_seconds=offline,query_seconds=point['seconds'],
            resident_bytes=resident,raw_bytes=cfg['source']['rows']*cfg['source']['dimension']*4,
            horizons={str(n):offline+n*point['seconds'] for n in (1,1000,100000,1000000)},
            cold_verified_horizons={str(n):cold_input+offline+n*point['seconds'] for n in (1,1000,100000,1000000)},
            input_audit_seconds=cold_input,components=components))
    cross=[]
    eligible=[r for r in rows if r['role']=='geometry_discovery' and r['metric']=='recall' and r['quality']>=.95]
    for a,b in itertools.combinations(eligible,2):
        if a['hardware']!=b['hardware']:continue
        n=frozen.break_even(a['offline_seconds'],a['query_seconds'],b['offline_seconds'],b['query_seconds'])
        if n is not None:cross.append(dict(a=a['id'],b=b['id'],hardware=a['hardware'],queries=n))
    return dict(costs=rows,break_even=cross,source_acquisition_and_B0='historical sunk, shown separately',
        new_audit_reference_and_duplicate_staging_work_is_not_a_single_deployment=True)


def solve_oracle(cfg,root,pipeline,products,check,phase,solver_spent):
    root=Path(root);ident,backend=pipeline.split(':');gpu=backend=='cuda'
    p=root/'oracle'/(ident+'-'+('cuda' if gpu else 'cpu14') if backend in ('cuda','cpu_flat') else ident+'-'+backend)
    measured=json.loads((p/'timings.json').read_text())
    quality_path=root/'representations'/ident/('cuda_quality.json' if gpu else 'faiss_quality.json')
    if backend.startswith('ivf'):quality_path=root/'ivf'/ident/backend[3:]/'quality.json'
    quality=json.loads(quality_path.read_text());points=collect_points(root)
    for role in ('geometry_discovery','geometry_check'):
        chosen=sorted({r['query_index'] for r in measured if r['role']==role})
        expected=cfg['queries']-cfg['discovery_queries'] if gpu else cfg['timing']['cpu_queries_per_role']
        if len(chosen)!=expected:raise ValueError('oracle paired cohort size')
        a={metric:np.asarray([[next(r[metric] for r in quality if r['query_index']==qi and r['budget']==m)
             for m in cfg['budgets']] for qi in chosen]) for metric in ('recall','strict_id')}
        seconds=np.asarray([[np.mean([r['seconds'] for r in measured if r['query_index']==qi and r['budget']==m])
             for m in cfg['budgets']] for qi in chosen])
        products.json(str(p.relative_to(root))+'/'+role+'_coefficients.json',dict(query_indices=chosen,budgets=cfg['budgets'],
            seconds=seconds.tolist(),quality={k:v.tolist() for k,v in a.items()},no_pilot_no_policy_cost=True))
        for metric,target in itertools.product(('recall','strict_id'),cfg['targets']):
            phase(pipeline+':oracle_solver');check();remaining=cfg['limits']['solver_total_seconds']-solver_spent
            result=oracle_budget_bound(a[metric],seconds,target,min(cfg['limits']['solver_seconds'],remaining/3)) if remaining>0 else dict(status='unresolved_solver_total_cap')
            solver_spent+=result.get('solver_seconds',0.)
            controls=[c for c in points if c['role']=='geometry_discovery' and c['metric']==metric
                and c['hardware_profile']==('cuda' if gpu else 'cpu14') and c['timing_cohort_quality']>=target]
            fixed=min(controls,key=lambda c:(c['seconds'],c['resident_bytes'],c['id'])) if controls else None
            audit=next((c for c in points if fixed and c['id']==fixed['id'] and c['role']==role and c['metric']==metric),None)
            lower=result.get('lp',{}).get('objective_seconds') if result.get('lp',{}).get('optimal') else None
            passes=audit is not None and audit['timing_cohort_quality']>=target
            products.json(str(p.relative_to(root))+'/'+role+'_'+metric+'_'+str(target)+'.json',dict(
                pipeline=pipeline,role=role,metric=metric,target=target,result=result,best_matched_fixed=audit,
                frozen_control_meets_role_target=passes,controller_seconds_headroom=audit['seconds']-lower if passes and lower is not None else None,
                hardware_matched=True,paired_query_indices=chosen,robust_gain_claim=False,less_than_5pct_is_small_or_unresolved=True))
    return solver_spent


def ivf(cfg,root,ident,torch,check,phase):
    import faiss
    root=Path(root);products=products_for(root,cfg,check)
    model,raw,queries,truth,cutoffs,roles,receipt,values,restore=backend_state(cfg,root,ident,check,phase,payload=False)
    del values
    sample=raw[np.load(root/'fit_rows.npy')].copy();spec=model['spec'];prefix='ivf/'+ident
    phase(ident+':ivf_train_encode',component=True);tick=time.monotonic();index=rep.make_ivf(model,sample,raw,cfg,check)
    build=time.monotonic()-tick;tick=time.monotonic()
    products.array(prefix+'/matrix.npy',model['matrix']);products.array(prefix+'/mean.npy',model['mean'])
    products.json(prefix+'/model_identity.json',dict(fingerprint=model['fingerprint'],source=cfg['source']['sha256'],spec=spec))
    # Quantized indexes fit the8GiB output cap. Float indexes are rebuilt from
    # the retained source; only coarse/list identities are serialized.
    if 'pq_bytes' in spec:
        products.reserve(int(index.ntotal)*(spec['pq_bytes']+8)+2**24)
        faiss.write_index(index,str(root/(prefix+'.faiss')))
    products.array(prefix+'/coarse.npy',index.quantizer.reconstruct_n(0,index.nlist))
    lists=[];offsets=[0]
    for i in range(index.nlist):
        size=index.invlists.list_size(i);ptr=index.invlists.get_ids(i)
        try:lists.append(faiss.rev_swig_ptr(ptr,size).copy().astype(np.int32))
        finally:index.invlists.release_ids(i,ptr)
        offsets.append(offsets[-1]+size)
    products.array(prefix+'/list_rows.npy',np.concatenate(lists));products.array(prefix+'/list_offsets.npy',np.asarray(offsets,np.int64))
    write=time.monotonic()-tick;del lists
    transformed=rep.query_coordinates(model,queries);phase(ident+':ivf_same_code_reference')
    reference=rep.exact_representation_search(rep.IVFDecodedView(index),transformed,max(cfg['budgets']),model['metric'],
        torch=torch,check=check,batch=cfg['query_batch'],block=cfg['block'])
    full=rep.ivf_same_code_reference(index,transformed,max(cfg['budgets']))
    products.array(prefix+'/fp64_same_code_candidates.npy',reference.astype(np.int32))
    products.array(prefix+'/fp32_fullprobe_candidates.npy',full.astype(np.int32))
    products.json(prefix+'/same_code_quality.json',evaluate(raw,queries,truth,cutoffs,reference,cfg,roles,ident+':ivf_reference'))
    for nprobe in cfg['ivf']['nprobes']:
        check();index.nprobe=nprobe;pipeline=ident+':ivf'+str(nprobe);p=prefix+'/'+str(nprobe)
        phase(pipeline+':quality')
        candidate=np.asarray([rep.faiss_search(index,q,max(cfg['budgets']),model['metric']) for q in transformed])
        quality=evaluate(raw,queries,truth,cutoffs,candidate,cfg,roles,pipeline,reference)
        products.array(p+'/candidates.npy',candidate.astype(np.int32));products.json(p+'/quality.json',quality)
        phase(pipeline+':timing');measured=timings(model,raw,queries,roles,selected_budgets(quality,cfg),index,cfg,check)
        verify_timings(measured,quality);products.json(p+'/timings.json',measured)
        extra=int(index.quantizer.ntotal*spec['dim']*4+model['matrix'].nbytes+model['mean'].nbytes)
        if hasattr(index,'pq'):extra+=int(index.pq.M*index.pq.ksub*index.pq.dsub*4+index.precomputed_table.nbytes())
        resident=int(raw.nbytes+len(raw)*(spec['bytes']+8)+extra)
        points=timing_points(pipeline,quality,measured,resident)
        for point in points:point.update(ivf_build_seconds=build,ivf_write_seconds=write)
        products.json(p+'/points.json',points)
    qcost=json.loads((root/'representations'/ident/'quality_costs.json').read_text())
    sample_seconds=receipt['sample_seconds'] if (spec['kind'] in ('pca','hybrid','opq') or 'pq_bytes' in spec) else 0.
    pca_seconds=receipt['pca_seconds'] if spec['kind'] in ('pca','hybrid') else 0.
    offline=sample_seconds+pca_seconds+qcost['representation_fit_seconds']+restore['model_load_seconds']+build+write
    products.json(prefix+'/costs.json',dict(train_encode_seconds=build,write_seconds=write,raw_bytes=int(raw.nbytes),
        payload_bytes=int(len(raw)*spec['bytes']),id_bytes=int(8*len(raw)),residual_codebook_is_own=True,
        float_payload_not_serialized='pq_bytes' not in spec,model_load_seconds=restore['model_load_seconds'],
        sample_seconds=sample_seconds,pca_seconds=pca_seconds,representation_fit_seconds=qcost['representation_fit_seconds'],
        global_pq_fit_not_charged_seconds=qcost['pq_fit_seconds'],offline_seconds=offline,resident_bytes=resident,
        measured_model_and_coarse_codebook_precomputed_bytes=extra,ivf_does_not_inherit_unused_global_pq_fit=True))
    products.json(prefix+'/complete.json',dict(id=ident,complete=True))


def oracle_ivf_timing(cfg,root,pipeline,torch,check,phase):
    import faiss
    root=Path(root);ident,backend=pipeline.split(':')
    model,raw,queries,truth,cutoffs,roles,receipt,values,restore=backend_state(cfg,root,ident,check,phase,payload=False);del values
    phase(pipeline+':oracle_rebuild',component=True);path=root/'ivf'/(ident+'.faiss')
    if path.exists():index=faiss.read_index(str(path))
    else:
        index=rep.make_ivf(model,raw[np.load(root/'fit_rows.npy')].copy(),raw,cfg,check)
        np.testing.assert_array_equal(index.quantizer.reconstruct_n(0,index.nlist),np.load(root/'ivf'/ident/'coarse.npy'))
    index.nprobe=int(backend[3:]);phase(pipeline+':oracle_all_M')
    measured=timings(model,raw,queries,roles,cfg['budgets'],index,cfg,check)
    quality=json.loads((root/'ivf'/ident/backend[3:]/'quality.json').read_text());verify_timings(measured,quality)
    products_for(root,cfg,check).json('oracle/'+ident+'-'+backend+'/timings.json',measured)


def finalize(cfg,root,torch,check,phase):
    root=Path(root);products=products_for(root,cfg,check)
    phase('final_source_full_sha')
    if rep.sha(cfg['source']['path'],check)!=cfg['source']['sha256']:raise ValueError('source changed during v2')
    products.json('source_post_sha.json',dict(sha256=cfg['source']['sha256'],matched=True))
    quality_done=[s['id'] for s in cfg['representations'] if (root/'representations'/s['id']/'quality_complete.json').exists()]
    timing_done=[s['id'] for s in cfg['representations'] if (root/'representations'/s['id']/'timing_complete.json').exists()]
    products.json('summary.json',dict(**seal_report(quality_done,[s['id'] for s in cfg['representations']]),
        quality_complete=quality_done,timing_complete=timing_done,all_quality_first=True,
        formal_test=False,formal_RAG_or_Tri_claim=False,old_v1_preserved=True))
