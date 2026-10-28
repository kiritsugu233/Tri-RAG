"""Approved route1 v2: direct Tri-F and related Gaussian distance comparators.

All probabilities are fixed-pair projection laws, not adaptive ANN certificates.
Old v1 scientific sources, results and roles are never modified by this module.
"""
from __future__ import annotations

import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from scipy.stats import chi2,f,norm,nct
from tri_rag_harness.utils import fingerprint
from mechanism.tls_rag_step5_ann_enhancement import (sha,save,ordered,rerank,metrics,
    exact_truth,check_resources,paired_ratio)

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/tls_rag_step5_ann_comparators_v2.json'
NEW_PATHS=('mechanism/tls_rag_step5_ann_comparators_v2.py',
    'mechanism/tls_rag_step5_ann_comparators_v2.cpp',str(CONFIG.relative_to(ROOT)),
    'tests/test_tls_rag_step5_ann_comparators_v2.py','scripts/run_tls_rag_step5_ann_comparators_v2.py')
BIND_PATHS=NEW_PATHS+('mechanism/tls_rag_step5_ann_enhancement.py',
    'src/tri_rag_harness/projection.py','src/tri_rag_harness/tri_law.py')
RULES={'chi':0,'polar':1,'normal':2,'student':3,'corr':4}
STAT_FIELDS=('full_distances','upper_distances','dot_calls','dot_dimensions','expanded',
    'neighbor_slots','gated','predicted','skipped','batch_skipped','partial_batches','fallback',
    'stage_first','stage_second','stage_third','norm_reads','events')
EVENT_FIELDS=('row','pivot','threshold','projected_dot','cutoff','predicted','full_distance','skipped')


def config():
    s=json.loads(CONFIG.read_text())
    if s['active_routes']!=[1] or s['deferred_routes']!=[2,3] or s['adapter_training']:
        raise ValueError('only approved route1 comparators are executable')
    return s


def sources():return {p:sha(ROOT/p) for p in BIND_PATHS}


def _validate(m,alpha,*values):
    if not isinstance(m,int) or isinstance(m,bool) or m<1 or not np.isfinite(alpha) or not 0<alpha<.5:
        raise ValueError('invalid fixed-pair distribution')
    if not np.isfinite(values).all():raise ValueError('nonfinite geometry')


def polarization_threshold(T,Sq,Sx,m,alpha):
    _validate(m,alpha,T,Sq,Sx)
    if not 0<T<4 or min(Sq,Sx)<=0:raise ValueError('invalid unit-pair geometry')
    c=float(f.isf(alpha,m,m))*T/(4-T)
    return (1-c)/(2*(1+c))*(Sq+Sx)


def conditional_normal_threshold(h,Sq,m,alpha):
    _validate(m,alpha,h,Sq)
    if not 0<=h<1 or Sq<=0:raise ValueError('normal rule outside proven domain')
    return h*Sq-float(norm.isf(alpha))*np.sqrt((1-h*h)*Sq/m)


def conditional_student_table(m,alpha,spec):
    _validate(m,alpha)
    if m<2:raise ValueError('Student df must be positive')
    lo,hi,step=(spec[k] for k in ('student_delta_min','student_delta_max','student_delta_step'))
    if not np.isfinite([lo,hi,step]).all() or step<=0 or hi<=lo:raise ValueError('invalid table domain')
    delta=lo+step*np.arange(int(round((hi-lo)/step))+1)
    t=nct.ppf(alpha,m-1,delta)
    cuts=t/np.hypot(t,np.sqrt(m-1))
    if not np.isfinite(cuts).all() or np.any(np.diff(cuts)<0):raise ValueError('nonmonotone/nonfinite Student table')
    return np.asarray(cuts,dtype=np.float64)


def table_lookup(table,delta,spec):
    if not np.isfinite(delta):raise ValueError('nonfinite delta')
    if not spec['student_delta_min']<=delta<=spec['student_delta_max']:return None
    i=int(np.floor((delta-spec['student_delta_min'])/spec['student_delta_step']))
    if i<0 or i>=len(table):return None
    return float(table[i])


def unit_boundary(nq,nmin,nmax,T):
    if not np.isfinite([nq,nmin,nmax,T]).all() or min(nq,nmin)<=0 or nmin>nmax or T<0:
        raise ValueError('invalid original norm range')
    nx=np.clip(nq-T,nmin,nmax)
    return float((nq+nx-T)/(2*np.sqrt(nq*nx)))


class OpenedInputs:
    """Read only the sealed parent subset and already exported tune-role packet."""
    def __init__(self,spec):
        self.spec=spec;self.parent=ROOT/spec['parent_root']
        path=self.parent/'archive_verified.json'
        if sha(path)!=spec['parent_archive_sha256']:raise ValueError('parent archive identity changed')
        archive=json.loads(path.read_text())['files']
        needed=('subset_corpus.npy','subset_ids.json','subset_original.faiss','projection.npy',
            'input_binding.json','query_records.jsonl','result.json')
        for name in needed:
            p=self.parent/name;r=archive[name]
            if p.is_symlink() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:
                raise ValueError('parent SHA/size changed: '+name)
        self.index_sha=archive['subset_original.faiss']['sha256']
        self.x=np.load(self.parent/'subset_corpus.npy',mmap_mode='r',allow_pickle=False)
        self.ids=json.loads((self.parent/'subset_ids.json').read_text())
        self.base_matrix=np.load(self.parent/'projection.npy',allow_pickle=False)
        if (self.x.shape!=(spec['subset_size'],spec['dimension']) or self.x.dtype!=np.float64 or
            len(set(self.ids))!=len(self.ids) or len(self.ids)!=len(self.x) or
            self.base_matrix.shape!=(128,spec['dimension'])):raise ValueError('parent geometry changed')
        for st in range(0,len(self.x),8192):
            x=self.x[st:st+8192]
            if not np.isfinite(x).all() or np.max(np.abs(np.linalg.norm(x,axis=1)-1))>1e-12:
                raise ValueError('parent normalization changed')
        root=Path(spec['opened_input_root']);self.full_root=root
        for name,key in [('observed.json','observed_sha256'),('request.json','request_sha256')]:
            if sha(root/name)!=spec[key]:raise ValueError('opened packet changed')
        packet=json.loads((root/'observed.json').read_text());request=json.loads((root/'request.json').read_text())
        if packet['request']!=request or packet['opened_roles']!=['tune'] or packet['forbidden_roles_opened']:
            raise ValueError('opened role boundary changed')
        parent_groups=json.loads((self.parent/'input_binding.json').read_text())['groups']
        development=ordered(request['groups']['selection'],'ann-comparators-v2',spec['seeds']['development'])
        report=ordered(list(set(request['groups']['evaluation'])-set(parent_groups['report'])),
            'ann-comparators-v2',spec['seeds']['report'])
        if len(development)!=256 or len(report)!=256 or set(development)&set(report) or set(report)&set(parent_groups['report']):
            raise ValueError('new report must exclude old route1 report')
        self.groups=dict(development=development,report=report)
        self._rows={r['query_id']:r for g in packet['groups'].values() for r in g}
        for qid in development+report:
            q=np.asarray(self._rows[qid]['query'])
            if q.shape!=(spec['dimension'],) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-12:
                raise ValueError('opened query geometry changed')
        self.truth={}
        with (self.parent/'query_records.jsonl').open() as stream:
            for line in stream:
                r=json.loads(line)
                if r['method']=='equivalence' and r['query_id'] in development:self.truth[r['query_id']]=r['truth']
        if set(self.truth)!=set(development):raise ValueError('parent development truth missing')
        self.binding=fingerprint(dict(parent=spec['parent_archive_sha256'],packet=spec['observed_sha256'],groups=self.groups))
        self._seal=None;self.full_descriptors=packet['files']

    def records(self,group):
        if group not in self.groups or (group=='report' and self._seal is None):raise ValueError('report remains closed')
        return [self._rows[q] for q in self.groups[group]]

    def open_report(self,path,binding):
        value=json.loads(Path(path).read_text())
        if fingerprint(value)!=binding or value['inputs']!=self.binding or value['sources']!=sources():
            raise ValueError('selection/source seal changed')
        self._seal=binding


def seal_selection(path,inputs,choice):
    value=dict(inputs=inputs.binding,sources=sources(),choice=choice)
    save(path,value);return fingerprint(value)


def alpha_values(spec):
    return sorted(set(spec['alphas']+[spec['strict_alpha']]+[a/3 for a in spec['alphas']]))


def table_name(m,a):return 'student_m'+str(m)+'_a'+format(a,'.12g')+'.npy'


def prepare_cache(corpus,base_matrix,output,spec,index_sha):
    output=Path(output);output.mkdir(exist_ok=False);n=len(corpus);receipts={};norm_lo=np.inf;norm_hi=0
    for m in spec['dimensions']:
        matrix=np.asarray(base_matrix[:m]*np.sqrt(128/m),dtype=np.float64)
        np.save(output/('matrix_'+str(m)+'.npy'),matrix)
        xp=np.lib.format.open_memmap(output/('points_'+str(m)+'.npy'),mode='w+',dtype=np.float32,shape=(n,m))
        meta=np.lib.format.open_memmap(output/('norms_'+str(m)+'.npy'),mode='w+',dtype=np.float32,shape=(n,1))
        for st in range(0,n,4096):
            # Normalize actual index float32 coordinates only for the auxiliary
            # representation, then apply dense Gaussian. Index data stay intact.
            x=np.asarray(corpus[st:st+4096],dtype=np.float32).astype(np.float64)
            ns=np.einsum('ij,ij->i',x,x)
            if not np.isfinite(ns).all() or (ns<=0).any():raise ValueError('invalid auxiliary source vectors')
            norm_lo=min(norm_lo,float(ns.min()));norm_hi=max(norm_hi,float(ns.max()))
            xp[st:st+len(x)]=np.asarray((x/np.sqrt(ns)[:,None])@matrix.T,dtype=np.float32)
            stored=xp[st:st+len(x)].astype(np.float64)
            meta[st:st+len(x),0]=np.sqrt(np.einsum('ij,ij->i',stored,stored))
        xp.flush();meta.flush();del xp,meta
    xp=np.load(output/'points_64.npy',mmap_mode='r')
    meta=np.lib.format.open_memmap(output/'norms_progressive.npy',mode='w+',dtype=np.float32,shape=(n,3))
    for st in range(0,n,4096):
        x=xp[st:st+4096].astype(np.float64)
        for j,m in enumerate(spec['progressive_stages']):meta[st:st+len(x),j]=np.sqrt(np.sum(x[:,:m]**2,axis=1)*64/m)
    meta.flush();del xp,meta
    for m in spec['dimensions']:
        for a in alpha_values(spec):np.save(output/table_name(m,a),conditional_student_table(m,a,spec))
    for p in output.iterdir():
        if p.is_file():receipts[p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
    value=dict(schema='aux-comparator-cache-v2',sources=sources(),index_sha256=index_sha,rows=n,
        norm_min=norm_lo,norm_max=norm_hi,post_projection_normalized=False,
        auxiliary_original_normalized=True,files=receipts)
    value['fingerprint']=fingerprint(value);save(output/'cache.json',value);return value


class Context:
    def __init__(self,path,m,spec,index_sha,progressive=False):
        self.path=Path(path);value=json.loads((self.path/'cache.json').read_text())
        payload={k:v for k,v in value.items() if k!='fingerprint'}
        if fingerprint(payload)!=value['fingerprint'] or value['sources']!=sources() or value['index_sha256']!=index_sha:
            raise ValueError('auxiliary source/index/fingerprint mismatch')
        if m not in spec['dimensions'] or progressive and m!=64:raise ValueError('invalid cached dimension')
        names=['points_'+str(m)+'.npy','matrix_'+str(m)+'.npy','norms_progressive.npy' if progressive else 'norms_'+str(m)+'.npy']
        for name in names:
            p=self.path/name;r=value['files'][name]
            if p.is_symlink() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('auxiliary file changed')
        self.xp=np.load(self.path/names[0],mmap_mode='r');self.matrix=np.load(self.path/names[1])
        self.norms=np.load(self.path/names[2],mmap_mode='r');self.value=value;self.m=m;self.progressive=progressive
        if self.xp.dtype!=np.float32 or not self.xp.flags.c_contiguous or self.xp.shape!=(value['rows'],m) or self.norms.shape!=(value['rows'],3 if progressive else 1):
            raise ValueError('auxiliary shape/dtype changed')


def native_build(headers,omp_include,output):
    headers,omp_include,output=map(Path,(headers,omp_include,output));s=config()
    if output.exists():raise FileExistsError(output)
    if sys.platform!='darwin':raise RuntimeError('validated builder is macOS only')
    for p,name in [(headers/'faiss/impl/HNSW.h','HNSW.h'),(omp_include/'omp.h','omp.h')]:
        if sha(p)!=s['native_headers'][name]:raise ValueError('fixed native header changed')
    command=['clang++','-std=c++17','-O3','-DNDEBUG','-dynamiclib','-undefined','dynamic_lookup',
        '-I'+str(headers),'-I'+str(omp_include),str(ROOT/NEW_PATHS[1]),'-o',str(output)]
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    return dict(command=command,compiler=subprocess.check_output(['clang++','--version'],text=True),library_sha256=sha(output))


class Native:
    def __init__(self,path,spec):
        import faiss
        if faiss.__version__!=spec['faiss_version']:raise ValueError('FAISS version changed')
        faiss.omp_set_num_threads(spec['threads']);self.s=spec
        library=next(Path(faiss.__file__).parent.glob('_swigfaiss*.so'))
        self.faiss_global=C.CDLL(str(library),mode=C.RTLD_GLOBAL);self.lib=C.CDLL(str(path))
        self.fn=self.lib.comparators_v2_search;self.fn.restype=C.c_int
        self.fn.argtypes=[C.c_void_p]*12+[C.c_int,C.c_void_p,C.c_int]
        self.critical={(m,a):(float(f.isf(a,m,m)),float(chi2.isf(a,m)/m),float(norm.isf(a)))
            for m in spec['dimensions'] for a in alpha_values(spec)}
        self.tables={};self.table_blocks={};self.receipt=dict(native_sha256=sha(path),faiss_module_sha256=sha(library))

    def search(self,index,query,context=None,rule='chi',alpha=.05,mode=0,skip='point',events=False,k=None,ef=None):
        t0=perf_counter();q=np.ascontiguousarray(query,dtype=np.float32)
        k=self.s['return_pool'] if k is None else k;ef=self.s['ef_search'] if ef is None else ef
        if (not hasattr(index,'hnsw') or index.metric_type!=1 or q.shape!=(index.d,) or not np.isfinite(q).all()
            or k<1 or k>index.ntotal or ef<k or mode not in (0,1,2) or skip not in ('point','batch') or rule not in RULES):
            raise ValueError('unsupported query/index/comparator')
        stages=1;m=16;table=np.zeros(1);params=np.zeros(12);qp=xp=metadata=None
        if mode:
            if context is None or context.xp.shape[0]!=index.ntotal or alpha not in self.s['alphas']+[self.s['strict_alpha']]:
                raise ValueError('invalid context/alpha')
            m=context.m;stages=3 if context.progressive else 1
            q64=q.astype(np.float64);nq=float(q64@q64)
            if nq<=0:raise ValueError('zero auxiliary query')
            qp=np.ascontiguousarray((q64/np.sqrt(nq))@context.matrix.T,dtype=np.float32)
            xp=context.xp;metadata=context.norms
            params=np.zeros(8+4*stages);params[:8]=[nq,context.value['norm_min'],context.value['norm_max'],
                self.s['dot_guard'],self.s['original_distance_guard_relative'],self.s['original_distance_guard_absolute'],
                self.s['student_delta_min'],self.s['student_delta_step']]
            dims=self.s['progressive_stages'] if stages==3 else [m];a=alpha/3 if stages==3 else alpha
            tables=[]
            for j,d in enumerate(dims):
                sq=float(qp[:d].astype(np.float64)@qp[:d].astype(np.float64))*(m/d)
                params[8+4*j:12+4*j]=[sq,*self.critical[d,a]]
                if rule=='student':
                    key=(str(context.path),d,a)
                    if key not in self.tables:
                        name=table_name(d,a);path=context.path/name
                        if sha(path)!=context.value['files'][name]['sha256']:raise ValueError('Student table changed')
                        self.tables[key]=np.load(path)
                    tables.append(self.tables[key])
            key=(str(context.path),m,alpha,stages,rule)
            if key not in self.table_blocks:
                self.table_blocks[key]=np.ascontiguousarray(np.stack(tables) if tables else np.zeros((stages,1)),dtype=np.float64)
            table=self.table_blocks[key]
        opt=np.array([m,k,ef,mode,RULES[rule],int(skip=='batch'),stages,table.shape[-1]],dtype=np.int32)
        D=np.empty(k,np.float32);I=np.empty(k,np.int64);stats=np.zeros(20)
        log=np.empty((self.s['limits']['events_per_query'],len(EVENT_FIELDS))) if events else None
        error=C.create_string_buffer(1024)
        def ptr(x):return None if x is None else C.c_void_p(x.ctypes.data)
        prep=(perf_counter()-t0)*1000;t=perf_counter()
        code=self.fn(C.c_void_p(int(index.this)),ptr(q),ptr(qp),ptr(xp),ptr(metadata),ptr(table),ptr(opt),ptr(params),
            ptr(D),ptr(I),ptr(stats),ptr(log),0 if log is None else len(log),error,len(error))
        native_ms=(perf_counter()-t)*1000
        if code:raise RuntimeError(error.value.decode())
        if np.any(I<0):raise RuntimeError('incomplete native results')
        values={name:float(stats[j]) for j,name in enumerate(STAT_FIELDS)}
        values.update(preparation_ms=prep,native_ms=native_ms)
        return D,I,values,None if log is None else log[:int(stats[16])].copy()
