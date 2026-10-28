"""Approved route-1 development screen; routes 2/3 deliberately not executable."""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
from scipy.stats import chi2, f
from tri_rag_harness.projection import dense_gaussian_projection
from tri_rag_harness.utils import canonical_json_bytes, fingerprint

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/tls_rag_step5_ann_enhancement_v1.json'
SOURCE_PATHS=('mechanism/tls_rag_step5_ann_enhancement.py',
    'mechanism/tls_rag_step5_ann_enhancement.cpp',str(CONFIG.relative_to(ROOT)),
    'tests/test_tls_rag_step5_ann_enhancement.py','scripts/run_tls_rag_step5_ann_enhancement.py',
    'src/tri_rag_harness/projection.py','src/tri_rag_harness/tri_law.py')
EVENT_FIELDS=('row','pivot','reference','distance','threshold','projected','reference_projected',
    'rho','safe_local_skip','predicted_skip','audited','full_count','result_threshold','frontier_threshold')
STAT_FIELDS=('full_distances','upper_distances','projected_distances','expanded','neighbor_slots',
    'gate_calls','predicted','oracle_skipped','deferred','locally_redundant','audited','pending',
    'events','oracle_rounding_differences')
RULES={'chi':0,'pair':1,'margin':2,'tri':3,'angle':4}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def save(path,data):
    with Path(path).open('x') as stream:
        json.dump(data,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def sources():return {p:sha(ROOT/p) for p in SOURCE_PATHS}


def config():
    value=json.loads(CONFIG.read_text())
    if value['active_routes']!=[1] or value['deferred_routes']!=[2,3]:
        raise ValueError('only route1 is authorized for execution')
    return value


def ordered(ids,tag,seed):
    return sorted(ids,key=lambda q:(hashlib.sha256(canonical_json_bytes([tag,seed,q])).hexdigest(),q))


def split_groups(selection,evaluation,spec):
    if len(selection)!=256 or len(evaluation)!=512 or len(set(selection+evaluation))!=768:
        raise ValueError('opened parent identity mismatch')
    a=ordered(selection,'ann-enhancement-v1',spec['seeds']['split'])
    b=ordered(evaluation,'ann-enhancement-v1',spec['seeds']['report'])[:256]
    return dict(fit=a[:128],cal=a[128:192],select=a[192:],report=b)


class ObservedInputs:
    """Only the already exported opened-role packet; no legacy role loader."""
    def __init__(self,root,spec):
        root=Path(root).resolve();self.spec=spec
        for name,key in [('request.json','input_request_sha256'),('observed.json','input_observed_sha256')]:
            if sha(root/name)!=spec[key]:raise ValueError('packet identity changed: '+name)
        packet=json.loads((root/'observed.json').read_text())
        request=json.loads((root/'request.json').read_text())
        if packet['request']!=request or packet['forbidden_roles_opened'] or packet['opened_roles']!=['tune']:
            raise ValueError('input boundary mismatch')
        if set(packet['files'])!={'corpus.npy','ids.json','projected.npy','original.faiss'}:
            raise ValueError('input file allowlist changed')
        for name,r in packet['files'].items():
            path=root/name
            if path.is_symlink() or path.stat().st_size!=r['bytes'] or sha(path)!=r['sha256']:
                raise ValueError('input SHA/size mismatch: '+name)
        self.ids=json.loads((root/'ids.json').read_text())
        if len(self.ids)!=spec['corpus_size'] or len(set(self.ids))!=len(self.ids):
            raise ValueError('corpus identity mismatch')
        self.corpus=np.load(root/'corpus.npy',mmap_mode='r',allow_pickle=False)
        if self.corpus.shape!=(spec['corpus_size'],spec['dimension']) or self.corpus.dtype!=np.float64:
            raise ValueError('corpus geometry changed')
        for start in range(0,len(self.corpus),8192):
            block=np.asarray(self.corpus[start:start+8192])
            if not np.isfinite(block).all() or np.max(np.abs(np.linalg.norm(block,axis=1)-1))>1e-12:
                raise ValueError('corpus not normalized/finite')
        self.groups=split_groups(request['groups']['selection'],request['groups']['evaluation'],spec)
        self._rows={r['query_id']:r for g in packet['groups'].values() for r in g}
        if set(self._rows)!={q for group in request['groups'].values() for q in group}:
            raise ValueError('query identities changed')
        for r in self._rows.values():
            q=np.asarray(r['query'],dtype=np.float64)
            if q.shape!=(spec['dimension'],) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-12:
                raise ValueError('query not normalized')
        self.binding=fingerprint({'packet':spec['input_observed_sha256'],'groups':self.groups})
        self._report_seal=None;self.root=root

    def records(self,group):
        if group not in self.groups or (group=='report' and self._report_seal is None):
            raise ValueError('report is closed until selection is sealed')
        return [self._rows[q] for q in self.groups[group]]

    def open_report(self,path,binding):
        seal=json.loads(Path(path).read_text())
        if (fingerprint(seal)!=binding or seal['inputs']!=self.binding or seal['sources']!=sources()):
            raise ValueError('selection seal mismatch')
        self._report_seal=binding


def seal_selection(path,inputs,choice,extra=None):
    record=dict(inputs=inputs.binding,sources=sources(),choice=choice,extra=extra or {})
    save(path,record);return fingerprint(record)


def critical_beta(fcritical,rho):
    rho=np.asarray(rho,dtype=np.float64)
    if fcritical<=1 or not np.isfinite(rho).all() or (np.abs(rho)>1).any():
        raise ValueError('invalid critical ratio geometry')
    c=np.sqrt((1-rho)*(1+rho))*(np.sqrt(fcritical)-1/np.sqrt(fcritical))
    return ((np.hypot(c,2)+c)/2)**2


def native_build(headers,omp_include,output):
    headers,omp_include,output=map(Path,(headers,omp_include,output))
    if output.exists():raise FileExistsError(output)
    if sys.platform!='darwin':raise RuntimeError('this build recipe is validated for macOS only')
    for path in (headers/'faiss/impl/HNSW.h',omp_include/'omp.h'):
        if not path.is_file():raise ValueError('missing native header: '+str(path))
    expected=config()['native_headers']
    if sha(headers/'faiss/impl/HNSW.h')!=expected['HNSW.h'] or sha(omp_include/'omp.h')!=expected['omp.h']:
        raise ValueError('fixed native headers changed')
    command=['clang++','-std=c++17','-O3','-DNDEBUG','-dynamiclib','-undefined','dynamic_lookup',
        '-I'+str(headers),'-I'+str(omp_include),str(ROOT/'mechanism/tls_rag_step5_ann_enhancement.cpp'),
        '-o',str(output)]
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    return dict(command=command,compiler=subprocess.check_output(['clang++','--version'],text=True),
        headers={'HNSW.h':sha(headers/'faiss/impl/HNSW.h'),'omp.h':sha(omp_include/'omp.h')},
        library_sha256=sha(output))


class Native:
    def __init__(self,path,spec):
        import faiss
        if faiss.__version__!=spec['faiss_version']:raise ValueError('FAISS version changed')
        faiss.omp_set_num_threads(spec['threads'])
        library=next(Path(faiss.__file__).parent.glob('_swigfaiss*.so'))
        self.faiss_global=C.CDLL(str(library),mode=C.RTLD_GLOBAL)
        self.lib=C.CDLL(str(path));self.fn=self.lib.route1_search
        self.fn.restype=C.c_int
        self.fn.argtypes=[C.c_void_p]*4+[C.c_int]*5+[C.c_double]*2+[C.c_int]+[C.c_void_p]*2+[
            C.c_int]+[C.c_void_p]*4+[C.c_int]
        self.spec=spec
        self.thresholds={a:(float(f.isf(a,spec['m_prime'],spec['m_prime'])),
            float(chi2.isf(a,spec['m_prime'])/spec['m_prime'])) for a in spec['alphas']}
        self.receipt={'faiss_module_sha256':sha(library),'native_sha256':sha(path)}

    def search(self,index,query,projected=None,qp=None,mode=0,rule='pair',alpha=.01,
               oracle=None,events=False,audit=0,k=None,ef=None):
        q=np.ascontiguousarray(query,dtype=np.float32);k=k or self.spec['return_pool'];ef=ef or self.spec['ef_search']
        if getattr(index,'metric_type',None)!=1 or not hasattr(index,'hnsw'):
            raise ValueError('only original-space L2 HNSW is supported')
        if q.shape!=(index.d,) or not np.isfinite(q).all() or index.ntotal<k or ef<k:
            raise ValueError('invalid query/search geometry')
        if mode not in (0,1,2,3,4) or rule not in RULES or not 0<alpha<.5:
            raise ValueError('invalid comparator')
        if mode in (1,2,4):
            if (not isinstance(projected,np.ndarray) or projected.dtype!=np.float32 or
                not projected.flags.c_contiguous or projected.shape!=(index.ntotal,self.spec['m_prime'])):
                raise ValueError('invalid projected storage')
            qp=np.ascontiguousarray(qp,dtype=np.float32)
            if qp.shape!=(self.spec['m_prime'],) or not np.isfinite(qp).all():raise ValueError('invalid projected query')
        if mode==3:
            if oracle is None:raise ValueError('oracle information required in diagnostic mode only')
            oracle=np.ascontiguousarray(oracle,dtype=np.float32)
            if oracle.shape!=(index.ntotal,):raise ValueError('oracle geometry')
        elif oracle is not None:raise ValueError('oracle information forbidden in deployable mode')
        D=np.empty(k,np.float32);I=np.empty(k,np.int64);stats=np.zeros(20)
        log=np.empty((self.spec['limits']['events_per_query'],len(EVENT_FIELDS))) if events else None
        error=C.create_string_buffer(1024)
        def ptr(x):return None if x is None else C.c_void_p(x.ctypes.data)
        if mode in (1,2,4):
            if alpha not in self.thresholds:raise ValueError('alpha outside frozen table')
            fc,cc=self.thresholds[alpha]
        else:fc=cc=2.
        code=self.fn(C.c_void_p(int(index.this)),ptr(q),ptr(qp),ptr(projected),self.spec['m_prime'],
            k,ef,mode,RULES[rule],fc,cc,audit,ptr(oracle),
            ptr(log),0 if log is None else len(log),ptr(D),ptr(I),ptr(stats),error,len(error))
        if code:raise RuntimeError(error.value.decode())
        if np.any(I<0) or len(set(I))!=k:raise RuntimeError('missing/duplicate native candidates')
        return D,I,{name:float(stats[i]) for i,name in enumerate(STAT_FIELDS)},None if log is None else log[:int(stats[12])].copy()


def rerank(corpus,ids,rows,query,k_gt=10,k_ctx=5):
    if len(set(map(int,rows)))!=len(rows):raise ValueError('duplicate candidates')
    diff=np.asarray(corpus[rows],dtype=np.float64)-query
    dist=np.einsum('ij,ij->i',diff,diff)
    order=sorted(range(len(rows)),key=lambda j:(float(dist[j]),ids[int(rows[j])]))
    pool=[ids[int(rows[j])] for j in order]
    return pool,pool[:k_ctx],pool[:k_gt]


def metrics(pool,context,truth,qrels):
    return dict(retention=len(set(pool)&set(truth))/len(truth),hit=int(bool(set(context)&set(qrels))))


def exact_truth(corpus,ids,query,k=10,block=8192):
    keep=[]
    for start in range(0,len(ids),block):
        diff=np.asarray(corpus[start:start+block])-query
        dd=np.einsum('ij,ij->i',diff,diff)
        merged=keep+[(float(d),ids[start+j]) for j,d in enumerate(dd)]
        keep=sorted(merged)[:k]
    return [i for d,i in keep]


def check_resources(started,output,spec):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    size=sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())
    if perf_counter()-started>spec['limits']['seconds']:raise RuntimeError('resource_stop: wall time')
    if rss>spec['limits']['rss_gib']*1024**3:raise RuntimeError('resource_stop: RSS')
    log_size=sum(p.stat().st_size for p in (Path(output)/'events').glob('*') if p.is_file())
    if log_size>spec['limits']['log_gib']*1024**3:raise RuntimeError('resource_stop: event logs')
    if size>spec['limits']['new_gib']*1024**3:raise RuntimeError('resource_stop: disk')
    if shutil.disk_usage(output).free<spec['limits']['free_gib']*1024**3:raise RuntimeError('resource_stop: free disk')
    return dict(rss_highwater_bytes=rss,new_bytes=size,seconds=perf_counter()-started)


def paired_ratio(rows,numerator,denominator,spec):
    qids=sorted({r['query_id'] for r in rows})
    values={m:np.array([np.mean([r['ms'] for r in rows if r['method']==m and r['query_id']==q])
        for q in qids]) for m in (numerator,denominator)}
    a,b=values[numerator],values[denominator]
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (b<=0).any():raise ValueError('unpaired timing')
    rng=np.random.default_rng(spec['seeds']['bootstrap'])
    ii=rng.integers(len(qids),size=(spec['bootstrap_replicates'],len(qids)))
    ratios=a[ii].mean(1)/b[ii].mean(1)
    return dict(ratio=float(a.mean()/b.mean()),interval95=np.quantile(ratios,[.025,.975]).tolist())
