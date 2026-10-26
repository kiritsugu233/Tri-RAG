#!/usr/bin/env python3
"""Execute the approved synthetic-only CPU reuse probe, preserving raw receipts."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
import traceback
import tracemalloc
from types import SimpleNamespace

# Apply before NumPy import, identically in the main and memory-worker processes.
THREAD_ENV = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
for key in THREAD_ENV:
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'src')]

import numpy as np

from mechanism.tls_rag_step5_audit import ExactScan
from mechanism.tls_rag_step5_reuse_probe import SPEC, CASES, GRID, run_case
from tri_rag_harness.projection import dense_gaussian_projection, project_rows

SOURCE_PATHS = ('mechanism/tls_rag_step5_reuse_probe.py',
    'tests/test_tls_rag_step5_reuse_probe.py','scripts/profile_tls_rag_step5_reuse_probe.py',
    'docs/TLS_RAG_STEP5_REUSE_PROBE_PROTOCOL.md','mechanism/tls_rag_step5_audit.py',
    'mechanism/tls_rag_step5_exact_policy.py','src/tri_rag_harness/projection.py',
    'src/tri_rag_harness/tls_rag_step5_ann.py')


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def sources():return {p:sha(ROOT/p) for p in SOURCE_PATHS}


def prepare(root):
    root.mkdir()  # refuse any existing output, including a failed earlier probe
    inputs=root/'inputs';inputs.mkdir()
    n,d,mp=SPEC['corpus_size'],SPEC['dimension'],SPEC['m_prime']
    matrix=dense_gaussian_projection(mp,d,SPEC['projection_seed'])
    np.save(inputs/'matrix.npy',matrix)
    corpus=np.lib.format.open_memmap(inputs/'corpus.npy',mode='w+',dtype=np.float64,shape=(n,d))
    projected=np.lib.format.open_memmap(inputs/'projected.npy',mode='w+',dtype=np.float64,shape=(n,mp))
    rng=np.random.default_rng(SPEC['corpus_seed'])
    for start in range(0,n,SPEC['block_rows']):
        end=min(n,start+SPEC['block_rows'])
        block=rng.normal(size=(end-start,d))
        block/=np.linalg.norm(block,axis=1)[:,None]
        corpus[start:end]=block
        projected[start:end]=project_rows(block,matrix)
    corpus.flush();projected.flush();del corpus,projected
    rng=np.random.default_rng(SPEC['query_seed'])
    queries=rng.normal(size=(SPEC['queries'],d))
    queries/=np.linalg.norm(queries,axis=1)[:,None]
    np.save(inputs/'queries.npy',queries)
    np.save(inputs/'ids.npy',np.array(['d%07d'%i for i in range(n)]))
    save(root/'manifest.json',{'spec':SPEC,'sources':sources(),
        'input_sha256':{p.name:sha(p) for p in sorted(inputs.iterdir())},
        'input_bytes':sum(p.stat().st_size for p in inputs.iterdir())})


def load(root):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['spec']!=SPEC or manifest['sources']!=sources():
        raise ValueError('configuration/source binding mismatch')
    for name,digest in manifest['input_sha256'].items():
        if sha(root/'inputs'/name)!=digest:raise ValueError('input SHA mismatch: '+name)
    def array(name):return np.load(root/'inputs'/(name+'.npy'),mmap_mode='r',allow_pickle=False)
    ids=array('ids')
    if len(set(ids))!=len(ids):raise ValueError('nonunique IDs')
    reader=SimpleNamespace(id_array=ids,corpus=array('corpus'),projected=array('projected'),
        row_by_id={item:i for i,item in enumerate(ids)})
    return reader,array('queries'),array('matrix')


def validator(reader,query,matrix,counter):
    qp=project_rows(query[None],matrix)[0]
    rr,dd,_=ExactScan(reader.id_array,reader.projected,SPEC['block_rows']).search(qp,GRID[-1])
    difference=np.asarray(reader.corpus[rr])-query
    od=np.einsum('ij,ij->i',difference,difference)
    originals=dict(zip(reader.id_array[rr],od))
    def check(prefixes,records,observed):
        for (rows,distances),record in zip(prefixes,records):
            k=len(rows)
            if (not np.array_equal(rows,rr[:k]) or
                    not np.array_equal(distances.view(np.uint64),dd[:k].view(np.uint64))):
                raise AssertionError('candidate or projected-distance bit mismatch')
            ids=reader.id_array[rows].tolist()
            if record['candidate_ids']!=ids:raise AssertionError('candidate interface mismatch')
            rank=sorted(ids,key=lambda i:(originals[i],i))
            if record['reranked_ids']!=rank or record['context_ids']!=rank[:SPEC['k_ctx']]:
                raise AssertionError('reranking/context mismatch')
            counter['prefixes']+=1
        final_ids=set(records[-1]['candidate_ids'])
        if set(observed)!=final_ids:raise AssertionError('future/missing original scores')
        for item,d in observed.items():
            if np.float64(d).view(np.uint64)!=originals[item].view(np.uint64):
                raise AssertionError('original-distance bit mismatch')
        counter['calls']+=1
    return check


def bootstrap_ratio(a,b):
    # Paired query blocks, preserving all three repeats within each query.
    rng=np.random.default_rng(SPEC['bootstrap_seed'])
    idx=rng.integers(len(a),size=(SPEC['bootstrap_replicates'],len(a)))
    values=np.mean(a[idx],axis=1)/np.mean(b[idx],axis=1)
    return {'ratio_of_means':float(np.mean(a)/np.mean(b)),
        'query_block_interval95':np.quantile(values,[.025,.975]).tolist()}


def summarize(rows):
    stats={};means={}
    for case in CASES:
        for m in GRID:
            selected=[r for r in rows if r['case']==case and r['final_m']==m]
            t=np.array([r['pipeline_ms'] for r in selected])
            key=case+':'+str(m)
            stats[key]={'samples':len(t),'mean_ms':float(t.mean()),
                'median_ms':float(np.median(t)),'p95_ms':float(np.quantile(t,.95)),
                'component_mean_ms':{k:float(np.mean([r[k] for r in selected]))
                    for k in ('projection_ms','scan_ms','selection_ms','gather_score_ms','rerank_ms')},
                'retained_buffer_bytes':selected[0]['retained_buffer_bytes']}
            means[key]=np.array([np.mean([r['pipeline_ms'] for r in selected if r['query']==q])
                                 for q in range(SPEC['queries'])])
    pairs={}
    comparisons=[('cache_reselect_lazy','cache_reselect_fixed'),
        ('cache_partition_lazy','cache_partition_fixed'),('cache_max','cache_reselect_fixed'),
        ('cache_reselect_fixed','stream_fixed'),('cache_partition_fixed','stream_fixed'),
        ('cache_reselect_lazy','stream_max'),('cache_partition_lazy','stream_max'),
        ('cache_reselect_lazy','cache_max'),('cache_partition_lazy','cache_max'),
        ('cache_partition_fixed','cache_reselect_fixed')]
    for a,b in comparisons:
        for m in GRID:pairs[a+'/'+b+':'+str(m)]=bootstrap_ratio(means[a+':'+str(m)],means[b+':'+str(m)])
    return stats,pairs


def memory_worker(root,case):
    reader,queries,matrix=load(root)
    float(np.sum(reader.projected))  # shared projected input resident before measurement
    check=validator(reader,queries[0],matrix,dict(prefixes=0,calls=0))
    check  # reference also touches the same original rows, outside memory tracing
    gc.collect()
    tracemalloc.start()
    calibration=np.empty(1000000,dtype=np.float64)
    calibration.fill(1.)
    calibrated=tracemalloc.get_traced_memory()[1]
    tracemalloc.stop();del calibration;gc.collect()
    if calibrated<8000000:raise RuntimeError('NumPy allocations are not traced')
    baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()
    result=run_case(reader,queries[0],matrix,case,GRID[-1],block_rows=SPEC['block_rows'])
    live,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    end=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale=1 if sys.platform=='darwin' else 1024
    return {'case':case,'endpoint':GRID[-1],'numpy_trace_calibration_bytes':calibrated,
        'new_traced_peak_bytes':peak,'live_traced_after_call_bytes':live,
        'baseline_process_peak_rss_bytes':int(baseline*scale),
        'process_peak_rss_bytes':int(end*scale),'increase_rss_highwater_bytes':int((end-baseline)*scale),
        'retained_buffer_bytes':result['retained_buffer_bytes'],
        'note':'separate process with tracing; not a latency sample; highwater delta is imperfect'}


def run(root):
    started=perf_counter();prepare(root)
    print('Prepared synthetic inputs and source binding',flush=True)
    reader,queries,matrix=load(root)
    float(np.sum(reader.projected))
    rng=np.random.default_rng(SPEC['order_seed'])
    cases=[(c,m) for c in CASES for m in GRID]
    counter=dict(calls=0,prefixes=0)
    check=validator(reader,queries[0],matrix,counter)
    for case,m in cases:
        for _ in range(SPEC['warmups']):run_case(reader,queries[0],matrix,case,m,
            block_rows=SPEC['block_rows'],validate=check)
    rows=[]
    with (root/'timing.jsonl').open('x') as log:
        for q,query in enumerate(queries):
            check=validator(reader,query,matrix,counter)
            for repeat in range(SPEC['repeats']):
                for pos in rng.permutation(len(cases)):
                    case,m=cases[pos]
                    result=run_case(reader,query,matrix,case,m,
                        block_rows=SPEC['block_rows'],validate=check)
                    if (result['projected_distances']!=SPEC['corpus_size'] or result['scans']!=1
                            or result['original_distances']!=m):
                        raise AssertionError('distance/scan accounting mismatch')
                    result.update(query=q,repeat=repeat)
                    rows.append(result);log.write(json.dumps(result,allow_nan=False)+'\n');log.flush()
            print('Timed and checked query %d/%d (%d samples)'%(q+1,len(queries),len(rows)),flush=True)
    stats,pairs=summarize(rows)
    # Release main-process mappings before fresh memory workers to reduce pressure.
    del reader,queries,matrix,check;gc.collect()
    memory=[]
    for case in CASES:
        child=subprocess.run([sys.executable,str(Path(__file__).resolve()),
            '--memory-worker',str(root),'--case',case],capture_output=True,text=True)
        (root/('memory-'+case+'.stderr.txt')).write_text(child.stderr)
        if child.returncode:
            raise RuntimeError('memory worker failed: '+case+' '+child.stderr)
        record=json.loads(child.stdout);memory.append(record)
        save(root/('memory-'+case+'.json'),record)
        print('Measured memory: '+case,flush=True)
    receipt={'schema':'step5-reuse-probe-receipt-v1','spec':SPEC,
        'environment':{'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
            'thread_env':{k:os.environ[k] for k in THREAD_ENV}},
        'manifest':json.loads((root/'manifest.json').read_text()),'validation':counter,
        'summary':stats,'paired_ratios':pairs,'memory':memory,'timings':rows,
        'elapsed_seconds':perf_counter()-started,'formal_certification':False,
        'limitations':['synthetic Mac hot-memory CPU only','scripted endpoints; no quality claims',
            'no predictor/Tri','small-sample empirical p95','no independent evaluation'],
        'artifacts_sha256':{p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file()}}
    save(root/'receipt.json',receipt)
    print('COMPLETE '+str(root/'receipt.json'),flush=True)


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output',type=Path);group.add_argument('--memory-worker',type=Path)
    parser.add_argument('--case',choices=CASES)
    args=parser.parse_args()
    if args.memory_worker:
        if args.case is None:parser.error('--case required for memory worker')
        print(json.dumps(memory_worker(args.memory_worker,args.case),allow_nan=False));return
    try:run(args.output)
    except Exception:
        if args.output.is_dir() and not (args.output/'failure.txt').exists():
            (args.output/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':main()
