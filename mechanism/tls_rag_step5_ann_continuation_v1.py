"""Approved route2 first trial: persistent original-space search, no predictor/Tri."""
from __future__ import annotations
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
from scipy.optimize import linprog
from tri_rag_harness.utils import fingerprint
from mechanism.tls_rag_step5_ann_enhancement import sha,save,ordered,rerank,metrics,check_resources,paired_ratio

ROOT=Path(__file__).resolve().parents[1]
NEW_PATHS=('mechanism/tls_rag_step5_ann_continuation_v1.py','mechanism/tls_rag_step5_ann_continuation_v1.cpp',
    'configs/tls_rag_step5_ann_continuation_v1.json','tests/test_tls_rag_step5_ann_continuation_v1.py',
    'scripts/run_tls_rag_step5_ann_continuation_v1.py')
BIND_PATHS=NEW_PATHS+('mechanism/tls_rag_step5_ann_enhancement.py','src/tri_rag_harness/utils.py')
STAT_FIELDS=('full_distances','upper_distances','expanded','neighbor_slots','result_updates','discovered',
    'advance_calls','core_ms','batch4_calls','scalar_calls','done','termination_reason','valid_results',
    'frontier_size','result_threshold','frontier_min')


def config():
    s=json.loads((ROOT/NEW_PATHS[2]).read_text())
    if s['active_routes']!=[2] or any(s[k] for k in ['predictor_training','adapter_training','paid_tri','scale_1m']):
        raise ValueError('only route2 first trial is executable')
    return s


def sources():return {p:sha(ROOT/p) for p in BIND_PATHS}


class OpenedContinuationInputs:
    def __init__(self,s):
        self.parent=ROOT/s['parent_root'];p=self.parent/'archive_verified.json'
        if sha(p)!=s['parent_archive_sha256']:raise ValueError('parent archive changed')
        files=json.loads(p.read_text())['files']
        for name in ['subset_corpus.npy','subset_ids.json','subset_original.faiss','input_binding.json','query_records.jsonl']:
            p=self.parent/name;v=files[name]
            if p.is_symlink() or p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise ValueError('parent file changed: '+name)
        root=Path(s['opened_input_root'])
        for name,key in [('observed.json','observed_sha256'),('request.json','request_sha256')]:
            if sha(root/name)!=s[key]:raise ValueError('opened packet changed')
        packet=json.loads((root/'observed.json').read_text());request=json.loads((root/'request.json').read_text())
        if packet['request']!=request or packet['opened_roles']!=['tune'] or packet['forbidden_roles_opened']:
            raise ValueError('opened role boundary changed')
        self.groups=json.loads((self.parent/'input_binding.json').read_text())['groups']
        if {k:len(v) for k,v in self.groups.items()}!=dict(fit=128,cal=64,select=64,report=256):raise ValueError('parent roles changed')
        if len(set(q for v in self.groups.values() for q in v))!=512:raise ValueError('role overlap')
        if set(self.groups['fit']+self.groups['cal']+self.groups['select'])!=set(request['groups']['selection']):
            raise ValueError('development identity changed')
        if not set(self.groups['report'])<=set(request['groups']['evaluation']):raise ValueError('report identity changed')
        self._rows={r['query_id']:r for g in packet['groups'].values() for r in g}
        self._truth={}
        with (self.parent/'query_records.jsonl').open() as stream:
            for line in stream:
                r=json.loads(line)
                if r['method']=='equivalence':self._truth[r['query_id']]=r['truth']
        for group in self.groups.values():
            for qid in group:
                q=np.asarray(self._rows[qid]['query'])
                if qid not in self._truth or q.shape!=(s['dimension'],) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-12:
                    raise ValueError('query geometry/truth changed')
        self.binding=fingerprint(dict(parent=s['parent_archive_sha256'],packet=s['observed_sha256'],groups=self.groups))
        self._seal=None;self.paths=dict(index=str(self.parent/'subset_original.faiss'),
            index_sha=files['subset_original.faiss']['sha256'],corpus=str(self.parent/'subset_corpus.npy'),
            ids=str(self.parent/'subset_ids.json'))

    def records(self,group):
        if group not in self.groups or group=='report' and self._seal is None:raise ValueError('report remains closed')
        return [dict(self._rows[q],truth=self._truth[q]) for q in self.groups[group]]

    def open_report(self,path,binding):
        v=json.loads(Path(path).read_text())
        if fingerprint(v)!=binding or v['inputs']!=self.binding or v['sources']!=sources():raise ValueError('selection seal changed')
        self._seal=binding


def seal_selection(path,inputs,choice):
    v=dict(inputs=inputs.binding,sources=sources(),choice=choice);save(path,v);return fingerprint(v)


def native_build(headers,omp_include,output):
    headers,omp_include,output=map(Path,(headers,omp_include,output));s=config()
    if output.exists():raise FileExistsError(output)
    if sys.platform!='darwin':raise ValueError('validated native builder is macOS only')
    for p,key in [(headers/'faiss/impl/HNSW.h','HNSW.h'),(omp_include/'omp.h','omp.h')]:
        if sha(p)!=s['native_headers'][key]:raise ValueError('fixed headers changed')
    cmd=['clang++','-std=c++17','-O3','-DNDEBUG','-dynamiclib','-undefined','dynamic_lookup',
        '-I'+str(headers),'-I'+str(omp_include),str(ROOT/NEW_PATHS[1]),'-o',str(output)]
    v=subprocess.run(cmd,text=True,capture_output=True)
    if v.returncode:raise RuntimeError(v.stderr)
    return dict(command=cmd,compiler=subprocess.check_output(['clang++','--version'],text=True),library_sha256=sha(output))


class Native:
    def __init__(self,path,s):
        import faiss
        if faiss.__version__!=s['faiss_version']:raise ValueError('FAISS version changed')
        faiss.omp_set_num_threads(s['threads']);self.s=s;self.path=str(path)
        module=next(Path(faiss.__file__).parent.glob('_swigfaiss*.so'))
        self.global_faiss=C.CDLL(str(module),mode=C.RTLD_GLOBAL);self.lib=C.CDLL(str(path))
        self.create=self.lib.continuation_create;self.create.restype=C.c_void_p
        self.create.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_void_p,C.c_int]
        self.advance=self.lib.continuation_advance;self.advance.restype=C.c_int
        self.advance.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int]
        self.snapshot=self.lib.continuation_snapshot;self.snapshot.restype=C.c_int
        self.snapshot.argtypes=[C.c_void_p]*6+[C.c_int]
        self.destroy=self.lib.continuation_destroy;self.destroy.restype=None;self.destroy.argtypes=[C.c_void_p]
        self.receipt=dict(library_sha256=sha(path),faiss_module_sha256=sha(module))


def ptr(x):return C.c_void_p(x.ctypes.data)


class Session:
    def __init__(self,native,index,query,debug=False,k=None,beam=None):
        self.state=None;self.n=native;self.index=index;self.k=native.s['return_pool'] if k is None else k
        beam=native.s['ef_search'] if beam is None else beam
        q=np.ascontiguousarray(query,dtype=np.float32)
        if (not hasattr(index,'hnsw') or index.metric_type!=1 or q.shape!=(index.d,) or not np.isfinite(q).all()
            or not isinstance(self.k,int) or self.k<1 or self.k>index.ntotal or beam<self.k):raise ValueError('invalid session inputs')
        e=C.create_string_buffer(1024)
        self.state=native.create(C.c_void_p(int(index.this)),ptr(q),self.k,beam,int(debug),e,len(e))
        if not self.state:raise RuntimeError(e.value.decode())

    def _arrays(self):return np.zeros(16),np.zeros(2,np.uint64),C.create_string_buffer(1024)

    @staticmethod
    def _stats(s,t):
        return dict({name:float(s[j]) if np.isfinite(s[j]) else None for j,name in enumerate(STAT_FIELDS)},
            distance_sequence=int(t[0]),state_hash=int(t[1]))

    def advance(self,target=0):
        if self.state is None:raise ValueError('closed session')
        if not isinstance(target,int) or isinstance(target,bool) or target<0 or target>=2**63:raise ValueError('invalid absolute work target')
        s,t,e=self._arrays();code=self.n.advance(self.state,target,ptr(s),ptr(t),e,len(e))
        if code:raise RuntimeError(e.value.decode())
        return self._stats(s,t)

    def snapshot(self):
        if self.state is None:raise ValueError('closed session')
        d=np.empty(self.k,np.float32);i=np.empty(self.k,np.int64);s,t,e=self._arrays()
        code=self.n.snapshot(self.state,ptr(d),ptr(i),ptr(s),ptr(t),e,len(e))
        if code:raise RuntimeError(e.value.decode())
        return d,i,self._stats(s,t)

    def close(self):
        if getattr(self,'state',None) is not None:self.n.destroy(self.state);self.state=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):self.close()


def gap_policy(stats,previous_updates,threshold,max_updates,minimum):
    """Only current, already-computed state; labels/future are not arguments."""
    a,b=stats['frontier_min'],stats['result_threshold']
    if stats['done'] or stats['full_distances']<minimum or a is None or b is None or b<=0:return False
    return (a-b)/b>=threshold and stats['result_updates']-previous_updates<=max_updates


def gap_methods(s):
    return [f'gap:{g}:{u}:{n}' for g in s['gap_thresholds'] for u in s['stagnation_updates'] for n in s['minimum_work']]


class Runtime:
    def __init__(self,paths,native,s):
        import faiss
        if sha(paths['index'])!=paths['index_sha']:raise ValueError('index SHA changed')
        self.index=faiss.read_index(paths['index']);self.x=np.load(paths['corpus'],mmap_mode='r',allow_pickle=False)
        self.ids=json.loads(Path(paths['ids']).read_text());self.n=native;self.s=s
        if self.index.ntotal!=len(self.x) or len(self.ids)!=len(self.x) or self.index.d!=self.x.shape[1]:raise ValueError('input geometry mismatch')
        for j in np.linspace(0,len(self.x)-1,17,dtype=int):
            if not np.array_equal(self.index.reconstruct(int(j)),self.x[j].astype(np.float32)):raise ValueError('index coordinate mismatch')

    def call(self,query,method):
        q=np.asarray(query,dtype=np.float64);q32=np.ascontiguousarray(q,dtype=np.float32);stats={};checks=0;control_ms=0
        if method.startswith('original:'):
            if int(method.split(':')[1]) not in self.s['fixed_ef']:raise ValueError('unsupported ef control')
            self.index.hnsw.efSearch=int(method.split(':')[1]);t=perf_counter()
            _,i=self.index.search(q32[None],self.s['return_pool']);rows=i[0];stats['core_ms']=(perf_counter()-t)*1000
        else:
            with Session(self.n,self.index,q32) as session:
                if method=='native':stats=session.advance()
                elif method.startswith('fixed:'):
                    target=int(method.split(':')[1])
                    if target not in self.s['checkpoints']:raise ValueError('unsupported fixed target')
                    stats=session.advance(target)
                else:
                    if method.startswith('gap:'):
                        if method not in gap_methods(self.s):raise ValueError('unsupported gap rule')
                        _,g,u,n=method.split(':');g=float(g);u=int(u);n=int(n)
                    elif method!='checkpoints' and not method.startswith('path:'):raise ValueError('unknown policy')
                    target=0 if method in ('checkpoints',) or method.startswith('gap:') else int(method.split(':')[1])
                    if target and target not in self.s['checkpoints']:raise ValueError('unsupported path target')
                    previous_updates=0;stats=None
                    for budget in self.s['checkpoints']:
                        if target and budget>target:break
                        # Crossing several targets triggers one observed decision.
                        if stats is not None and stats['full_distances']>=budget:continue
                        stats=session.advance(budget);checks+=1
                        if stats['done']:break
                        stop=False
                        if method.startswith('gap:'):
                            t=perf_counter();stop=gap_policy(stats,previous_updates,g,u,n);control_ms+=(perf_counter()-t)*1000
                        previous_updates=stats['result_updates']
                        if stop:break
                    else:budget=None
                    if stats is None or not stats['done'] and (method=='checkpoints' or method.startswith('gap:') and not stop or target==0 and not method.startswith('gap:')):
                        stats=session.advance()
                _,rows,stats=session.snapshot()
        t=perf_counter();pool,ctx,_=rerank(self.x,self.ids,rows,q,self.s['k_gt'],self.s['k_ctx'])
        stats.update(rerank_ms=(perf_counter()-t)*1000,checks=checks,control_ms=control_ms)
        return dict(candidate_ids=pool,context_ids=ctx,stats=stats)

    def trace(self,query):
        q=np.asarray(query,dtype=np.float64);out=[]
        with Session(self.n,self.index,q) as session:
            for target in self.s['checkpoints']+[0]:
                stats=session.advance(target);d,i,stats=session.snapshot()
                pool,ctx,_=rerank(self.x,self.ids,i,q,self.s['k_gt'],self.s['k_ctx'])
                out.append(dict(target=target,distances=d.tolist(),rows=i.tolist(),candidate_ids=pool,context_ids=ctx,stats=stats))
        return out

    def close(self):
        self.index=None
        if hasattr(self,'x'):self.x._mmap.close();del self.x


def stopping_oracle(retention,hit,cost,baseline,s):
    """Offline only: cohort prices, never per-query fastest timing."""
    a,b,c=np.asarray(retention),np.asarray(hit),np.asarray(cost)
    if a.ndim!=2 or b.shape!=a.shape or c.shape!=(a.shape[1],) or not np.isfinite([*a.flat,*b.flat,*c]).all() or (c<=0).any():
        raise ValueError('invalid oracle cohort')
    nq,ns=a.shape;prices=np.maximum.accumulate(c);tol=s['quality_tolerance']
    target=np.maximum(0,np.asarray(baseline)-[tol['retention_drop'],tol['hit_drop']])
    # Per-query constraints sum weights to1; two aggregate quality constraints.
    eq=np.zeros((nq,nq*ns))
    for q in range(nq):eq[q,q*ns:(q+1)*ns]=1
    lp=linprog(np.tile(prices,nq)/nq,A_ub=-np.stack([a.ravel(),b.ravel()])/nq,b_ub=-target,
        A_eq=eq,b_eq=np.ones(nq),bounds=(0,1),method='highs')
    mix=linprog(prices,A_ub=-np.stack([a.mean(0),b.mean(0)]),b_ub=-target,A_eq=np.ones((1,ns)),b_eq=[1],bounds=(0,1),method='highs')
    if not lp.success or not mix.success:raise ValueError('oracle quality infeasible')
    perquery=[]
    for j in range(nq):
        eligible=np.flatnonzero((a[j]>=a[j,-1]-1e-12)&(b[j]>=b[j,-1]))
        perquery.append(int(eligible[np.argmin(prices[eligible])]))
    return dict(prices_raw=c.tolist(),prices_monotone=prices.tolist(),target=target.tolist(),
        oracle_ms=float(lp.fun),oracle_weights=lp.x.reshape(a.shape).tolist(),
        mixture_ms=float(mix.fun),mixture_weights=mix.x.tolist(),
        same_endpoint_ms=float(np.mean(prices[perquery])),same_endpoint_stages=perquery,
        fractional_diagnostic=True,deployed=False)
