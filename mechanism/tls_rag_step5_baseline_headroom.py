"""Approved real-input, local CPU baseline/headroom screen; no controller."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import resource
import shutil
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from mechanism.tls_rag_step5_audit import CONFIG
from mechanism.tls_rag_step5_reuse_probe import ExactDistanceSession, run_case
from scripts.analyze_tls_rag_step5_action_costs import solve, mixture
from tri_rag_harness.projection import dense_gaussian_projection
from tri_rag_harness.tls_rag_step5_ann import HnswCandidateIndex, CandidateSession
from tri_rag_harness.tls_rag_step5_protocol import source_binding, load_protocol
from tri_rag_harness.utils import fingerprint

ROOT=Path(__file__).resolve().parents[1]
PARENT_SAMPLING_SHA='52e22ee02d6da7746f202a430a34c0624a3bbf23cf841db97099029b8b85d6ae'
GRID=(32,64,128,256)
METHODS=('original_exact','original_ann64')+tuple('projected_fixed:'+str(m) for m in GRID)
SPEC=dict(schema='step5-baseline-headroom-v1',corpus_size=1000000,dimension=384,
    m_prime=128,projection_seed=55117,block_rows=8192,k_gt=10,k_ctx=5,
    timing_queries=64,repeats=3,warmup_queries=4,order_seed=55279,sampling_seed=55273,
    bootstrap_seed=55283,bootstrap_replicates=2000,target_retention=.90,target_hit_delta=-.02,
    rss_gib=8,seconds=2700,minimum_free_gib=12,maximum_new_gib=8)
SOURCES=('mechanism/tls_rag_step5_baseline_headroom.py',
    'tests/test_tls_rag_step5_baseline_headroom.py',
    'scripts/run_tls_rag_step5_baseline_headroom.py',
    'docs/TLS_RAG_STEP5_BASELINE_HEADROOM_PROTOCOL.md',
    'mechanism/tls_rag_step5_reuse_probe.py','mechanism/tls_rag_step5_exact_policy.py',
    'mechanism/tls_rag_step5_audit.py','src/tri_rag_harness/projection.py',
    'src/tri_rag_harness/tls_rag_step5_ann.py','scripts/analyze_tls_rag_step5_action_costs.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,data):
    with Path(path).open('x') as f:
        json.dump(data,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def sources():return {p:sha(ROOT/p) for p in SOURCES}


def make_request(sampling,bundle,archive):
    if sha(sampling)!=PARENT_SAMPLING_SHA:raise ValueError('parent sampling SHA mismatch')
    groups=json.loads(Path(sampling).read_text())['groups']
    selected={k:groups[k] for k in ('selection','evaluation')}
    if len(selected['selection'])!=256 or len(selected['evaluation'])!=512:
        raise ValueError('parent group sizes changed')
    return dict(schema=SPEC['schema'],spec=SPEC,groups=selected,bundle=bundle,archive=archive,
        parent_config=json.loads(CONFIG.read_text()),legacy_sources=source_binding(),
        legacy_protocol=fingerprint(load_protocol()),hnsw=load_protocol()['hnsw'],
        sampling_sha256=PARENT_SAMPLING_SHA)


def extract_observed_subset(request):
    """Standalone read-only remote exporter, serialized verbatim by the runner.

    Imports and helpers are local so this function needs no server code checkout.
    It only prints a small derived JSON package, never writes remote files.
    """
    import hashlib,json
    from pathlib import Path
    import numpy as np
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    def fp(data):
        return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),
            ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    bundle,archive=Path(request['bundle']).resolve(),Path(request['archive']).resolve()
    cfg=request['parent_config']
    bm=json.loads((bundle/'artifact_manifest.json').read_text())
    if fp(bm)!=cfg['bundle_binding']:raise ValueError('bundle manifest mismatch')
    maps={}
    for role in ('indexes','tune'):
        p=archive/role/'artifact_manifest.json'
        if digest(p)!=cfg['archive_manifests'][role]:raise ValueError('archive manifest mismatch')
        maps[role]=json.loads(p.read_text())
    def read_bundle(name):
        p=bundle/name
        if digest(p)!=bm[name]:raise ValueError('bundle checksum: '+name)
        return p
    def read_archive(role,name):
        p=archive/role/name
        if digest(p)!=maps[role][name]:raise ValueError('archive checksum: '+role+'/'+name)
        return p
    metadata=json.loads(read_bundle('metadata.json').read_text())
    if metadata['sources']!=request['legacy_sources'] or metadata['protocol']!=request['legacy_protocol']:
        raise ValueError('legacy scientific metadata mismatch')
    setup=json.loads(read_archive('indexes','setup.json').read_text())
    if setup['binding']!=cfg['archive_binding']:raise ValueError('index binding mismatch')
    for name in ('reservation.json','labels_opened.json'):
        receipt=json.loads(read_archive('tune',name).read_text())
        if receipt['stage']!='tune' or receipt['binding']!=cfg['archive_binding']:
            raise ValueError('tune not opened/bound')
    ids=json.loads(read_bundle('tune.ids.json').read_text())
    if fp(ids)!=metadata['roles']['tune']['ids']:raise ValueError('tune IDs mismatch')
    rows={qid:i for i,qid in enumerate(ids)}
    wanted=[q for group in request['groups'].values() for q in group]
    if len(set(wanted))!=len(wanted) or not set(wanted)<=set(rows):
        raise ValueError('overlap or missing observed queries')
    queries=np.load(read_bundle('tune.npy'),mmap_mode='r',allow_pickle=False)
    labels=json.loads(read_bundle('labels/tune.json').read_text())
    truth={}
    with read_archive('tune','truth.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['query_id'] in wanted:
                if r['query_id'] in truth:raise ValueError('duplicate truth')
                truth[r['query_id']]=r
    if set(truth)!=set(wanted):raise ValueError('missing truth')
    groups={g:[{'query_id':q,'query':queries[rows[q]].tolist(),
        'truth':truth[q]['exact_top_k'],'archived_exact_hit':truth[q]['exact_hit'],
        'qrels':labels[q]} for q in qids] for g,qids in request['groups'].items()}
    files={}
    for name,root,relative,h in [('corpus.npy',bundle,'corpus.npy',bm['corpus.npy']),
        ('ids.json',bundle,'corpus_ids.json',bm['corpus_ids.json']),
        ('projected.npy',archive/'indexes','projected.npy',maps['indexes']['projected.npy']),
        ('original.faiss',archive/'indexes','original.faiss',maps['indexes']['original.faiss'])]:
        p=root/relative
        files[name]={'path':str(p),'sha256':h,'bytes':p.stat().st_size}
    return {'schema':request['schema'],'request':request,'groups':groups,'files':files,
        'parent_metadata_fingerprint':fp(metadata),'opened_roles':['tune'],
        'forbidden_roles_opened':False}


def remote_source(request):
    return inspect.getsource(extract_observed_subset)+'\nimport json\nprint(json.dumps(extract_observed_subset('+repr(request)+'),allow_nan=False))\n'


class ObservedInputs:
    def __init__(self,root,expected_request):
        root=Path(root)
        pack=json.loads((root/'observed.json').read_text())
        if pack['request']!=expected_request or pack['schema']!=SPEC['schema']:
            raise ValueError('derived input request mismatch')
        if pack['forbidden_roles_opened'] or pack['opened_roles']!=['tune']:
            raise ValueError('invalid input access scope')
        for name,entry in pack['files'].items():
            if name not in ('corpus.npy','ids.json','projected.npy','original.faiss'):
                raise ValueError('unlisted input')
            p=root/name
            if p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:
                raise ValueError('download checksum mismatch: '+name)
        if set(pack['files'])!={'corpus.npy','ids.json','projected.npy','original.faiss'}:
            raise ValueError('missing input')
        ids=json.loads((root/'ids.json').read_text())
        if len(ids)!=SPEC['corpus_size'] or len(set(ids))!=len(ids) or any(type(i)!=str for i in ids):
            raise ValueError('corpus ID mismatch')
        corpus=np.load(root/'corpus.npy',mmap_mode='r',allow_pickle=False)
        projected=np.load(root/'projected.npy',mmap_mode='r',allow_pickle=False)
        if (corpus.shape!=(len(ids),SPEC['dimension']) or projected.shape!=(len(ids),SPEC['m_prime'])
                or corpus.dtype!=np.float64 or projected.dtype!=np.float64):
            raise ValueError('vector shape/dtype mismatch')
        for start in range(0,len(ids),SPEC['block_rows']):
            block=corpus[start:start+SPEC['block_rows']]
            if not np.all(np.isfinite(block)) or not np.allclose(np.linalg.norm(block,axis=1),1,rtol=0,atol=1e-12):
                raise ValueError('corpus normalization mismatch')
            if not np.all(np.isfinite(projected[start:start+SPEC['block_rows']])):
                raise ValueError('nonfinite projection')
        self.reader=SimpleNamespace(corpus=corpus,projected=projected,id_array=np.asarray(ids),
            row_by_id={i:n for n,i in enumerate(ids)})
        self.matrix=dense_gaussian_projection(SPEC['m_prime'],SPEC['dimension'],SPEC['projection_seed'])
        self.groups=pack['groups'];seen=set()
        for g,rows in self.groups.items():
            if [r['query_id'] for r in rows]!=expected_request['groups'][g]:raise ValueError('group IDs changed')
            for r in rows:
                q=np.asarray(r['query'],dtype=np.float64)
                if q.shape!=(SPEC['dimension'],) or not np.all(np.isfinite(q)) or abs(np.linalg.norm(q)-1)>1e-12:
                    raise ValueError('query geometry mismatch')
                if r['query_id'] in seen or len(r['truth'])!=10 or len(set(r['truth']))!=10:
                    raise ValueError('duplicate query/truth')
                if any(i not in self.reader.row_by_id for i in r['truth']):
                    raise ValueError('unknown truth ID')
                if any(type(i)!=str for i in r['qrels']) or r['archived_exact_hit'] not in (0,1):
                    raise ValueError('invalid qrels/exact Hit')
                seen.add(r['query_id'])
        import faiss
        if faiss.__version__!='1.8.0':raise ValueError('FAISS version changed')
        faiss.omp_set_num_threads(1)
        self.ann=HnswCandidateIndex.__new__(HnswCandidateIndex)
        self.ann.ids,self.ann.config=tuple(ids),expected_request['hnsw']
        self.ann.index=faiss.read_index(str(root/'original.faiss'))
        if self.ann.index.d!=SPEC['dimension'] or self.ann.index.ntotal!=len(ids):
            raise ValueError('ANN index geometry mismatch')
        self.input_binding=fingerprint(pack)
        self._eval_seal=None

    def records(self,group):
        if group not in ('selection','evaluation') or (group=='evaluation' and self._eval_seal is None):
            raise ValueError('evaluation not selected/sealed')
        return self.groups[group]

    def authorize_evaluation(self,seal,binding):
        content=json.loads(Path(seal).read_text())
        if fingerprint(content)!=binding or content['input_binding']!=self.input_binding or content['sources']!=sources():
            raise ValueError('selection seal mismatch')
        self._eval_seal=binding


def quality(candidate,context,record):
    return {'retention':len(set(candidate)&set(record['truth']))/10,
        'hit':float(bool(set(context)&set(record['qrels']))),
        'hit_delta':float(bool(set(context)&set(record['qrels'])))-record['archived_exact_hit']}


def original_exact(reader,query):
    at=perf_counter()
    cache=ExactDistanceSession(reader.id_array,reader.corpus,query,SPEC['block_rows'])
    rows,dd=cache.expand(10)
    candidate=reader.id_array[rows].tolist();context=candidate[:5]
    del cache
    elapsed=(perf_counter()-at)*1000
    return {'pipeline_ms':elapsed,'candidate_ids':candidate,'context_ids':context,
        'original_distances':len(reader.id_array),'projected_distances':0}


def projected_fixed(reader,query,matrix,budget):
    output={}
    def capture(prefixes,records,originals):
        output.update(candidate_ids=records[-1]['candidate_ids'],context_ids=records[-1]['context_ids'])
    result=run_case(reader,query,matrix,'cache_reselect_fixed',budget,validate=capture)
    return {**result,**output}


def original_ann(reader,query,index):
    at=perf_counter()
    session=CandidateSession(index.ids,reader.corpus,reader.corpus,query,query,index,5,reader.row_by_id)
    rec=session.expand(64)
    del session
    return {'pipeline_ms':(perf_counter()-at)*1000,'candidate_ids':rec['candidate_ids'],
        'context_ids':rec['context_ids'],'original_distances':64,'projected_distances':0}


def execute(inputs,record,method):
    q=np.asarray(record['query'],dtype=np.float64)
    if method=='original_exact':result=original_exact(inputs.reader,q)
    elif method=='original_ann64':result=original_ann(inputs.reader,q,inputs.ann)
    elif method.startswith('projected_fixed:'):
        result=projected_fixed(inputs.reader,q,inputs.matrix,int(method.split(':')[1]))
    else:raise ValueError('unknown method')
    result['metrics']=quality(result['candidate_ids'],result['context_ids'],record)
    if method=='original_exact' and (result['candidate_ids']!=record['truth'] or result['metrics']['hit_delta']!=0):
        raise ValueError('archived original truth/Hit mismatch: '+json.dumps({
            'query_id':record['query_id'],'expected':record['truth'],
            'actual':result['candidate_ids'],'metrics':result['metrics']},sort_keys=True))
    return result


def timing_ids(rows,group):
    return sorted(rows,key=lambda r:(fingerprint([SPEC['schema'],SPEC['sampling_seed'],group,r['query_id']]),
                                   r['query_id']))[:SPEC['timing_queries']]


def select_fixed(quality_rows,costs):
    eligible=[]
    for m in GRID:
        name='projected_fixed:'+str(m)
        r=[x['methods'][name]['metrics'] for x in quality_rows]
        if np.mean([x['retention'] for x in r])>=SPEC['target_retention']-1e-12 and np.mean([x['hit_delta'] for x in r])>=SPEC['target_hit_delta']-1e-12:
            eligible.append(m)
    return min(eligible,key=lambda m:(costs['projected_fixed:'+str(m)],m)) if eligible else None


def seal_selection(path,inputs,choice,quality_rows,costs):
    seal={'sources':sources(),'input_binding':inputs.input_binding,'fixed_budget':choice,
        'quality_binding':fingerprint(quality_rows),'selection_costs':costs,'spec':SPEC}
    save(path,seal)
    return fingerprint(seal)


def perfect_information_bound(quality_rows,costs,match_ann=False):
    r=np.array([[x['methods']['projected_fixed:'+str(m)]['metrics']['retention'] for m in GRID] for x in quality_rows])
    h=np.array([[x['methods']['projected_fixed:'+str(m)]['metrics']['hit'] for m in GRID] for x in quality_rows])
    c=np.array([costs['projected_fixed:'+str(m)] for m in GRID])
    exact_h=np.mean([x['methods']['original_exact']['metrics']['hit'] for x in quality_rows])
    ann_r=np.mean([x['methods']['original_ann64']['metrics']['retention'] for x in quality_rows])
    ann_h=np.mean([x['methods']['original_ann64']['metrics']['hit'] for x in quality_rows])
    target_r,target_h=(ann_r,ann_h) if match_ann else (SPEC['target_retention'],exact_h+SPEC['target_hit_delta'])
    oracle=solve(r,h,GRID,c,target_r,target_h,False)
    controls={}
    for name in METHODS:
        rr=np.mean([x['methods'][name]['metrics']['retention'] for x in quality_rows])
        hh=np.mean([x['methods'][name]['metrics']['hit'] for x in quality_rows])
        if rr>=target_r-1e-12 and hh>=target_h-1e-12:controls[name]=dict(mean_ms=costs[name],retention=rr,hit=hh)
    mix=mixture(r,h,c,target_r,target_h)
    if mix['feasible']:controls['query_independent_mixture']={'mean_ms':mix['model_mean_ms'],**mix}
    if oracle['optimal']:
        oracle['headroom_by_control']={name:{'gross_ms':v['mean_ms']-oracle['model_mean_ms'],
            'gross_fraction':1-oracle['model_mean_ms']/v['mean_ms'],
            'extra_cost_budget_ms':{str(f):v['mean_ms']*(1-f)-oracle['model_mean_ms'] for f in (0.,.05,.10,.20)}}
            for name,v in controls.items()}
    return {'target':{'retention':target_r,'hit':target_h},'oracle':oracle,'controls':controls,
        'interpretation':'group-mean single-call costs; free before-search information; diagnostic only'}


def check_resources(started,root,inputs_root):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform=='darwin' else 1024**2)
    if rss>SPEC['rss_gib'] or perf_counter()-started>SPEC['seconds']:
        raise RuntimeError('resource/time ceiling; preserve run, no automatic retry')
    if shutil.disk_usage(root).free<SPEC['minimum_free_gib']*1024**3:
        raise RuntimeError('disk reserve breached')
    size=sum(p.stat().st_size for base in (Path(root),Path(inputs_root)) for p in base.rglob('*') if p.is_file())
    if size>SPEC['maximum_new_gib']*1024**3:raise RuntimeError('new disk quota breached')


def measure_baselines(inputs,group,output,started,input_root):
    rows=inputs.records(group)
    qualities=[]
    with (output/(group+'.quality.jsonl')).open('x') as f:
        for pos,record in enumerate(rows):
            check_resources(started,output,input_root)
            results={name:execute(inputs,record,name) for name in METHODS}
            for value in results.values():value.pop('pipeline_ms',None)
            result={'query_id':record['query_id'],'methods':results}
            qualities.append(result);f.write(json.dumps(result,allow_nan=False)+'\n');f.flush()
            if (pos+1)%64==0:print(group+' quality '+str(pos+1)+'/'+str(len(rows)),flush=True)
    lookup={r['query_id']:r for r in qualities}
    sample=timing_ids(rows,group)
    # Hot both spaces and index storage outside timing, equally for all methods.
    float(np.sum(inputs.reader.corpus));float(np.sum(inputs.reader.projected))
    for record in sample[:SPEC['warmup_queries']]:
        for method in METHODS:execute(inputs,record,method)
    rng=np.random.default_rng(SPEC['order_seed']+(1 if group=='evaluation' else 0))
    timings=[]
    with (output/(group+'.timing.jsonl')).open('x') as f:
        for pos,record in enumerate(sample):
            for repeat in range(SPEC['repeats']):
                for method in rng.permutation(METHODS):
                    check_resources(started,output,input_root)
                    result=execute(inputs,record,str(method))
                    expected=lookup[record['query_id']]['methods'][method]
                    for key in ('candidate_ids','context_ids','metrics'):
                        if result[key]!=expected[key]:raise ValueError('timed quality/candidate mismatch: '+str(method))
                    result={k:v for k,v in result.items() if k not in ('candidate_ids','context_ids')}
                    result.update(method=str(method),query_id=record['query_id'],repeat=repeat)
                    timings.append(result);f.write(json.dumps(result,allow_nan=False)+'\n');f.flush()
            if (pos+1)%16==0:print(group+' timing '+str(pos+1)+'/'+str(len(sample)),flush=True)
    stats={}
    for method in METHODS:
        tt=np.array([r['pipeline_ms'] for r in timings if r['method']==method])
        stats[method]={'mean_ms':float(tt.mean()),'median_ms':float(np.median(tt)),
            'p95_ms':float(np.quantile(tt,.95)),'samples':len(tt)}
    return qualities,timings,stats
