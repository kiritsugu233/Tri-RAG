#!/usr/bin/env python3
"""Local CPU-only runner and read-only remote export request; no credentials."""
from __future__ import annotations
import argparse
import gc
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

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from mechanism.tls_rag_step5_baseline_headroom import (SPEC,METHODS,ObservedInputs,
    make_request,remote_source,save,sources,sha,fingerprint,execute,measure_baselines,
    select_fixed,seal_selection,perfect_information_bound,check_resources)


def paired_summary(timings):
    qids=sorted({r['query_id'] for r in timings})
    values={name:np.array([np.mean([r['pipeline_ms'] for r in timings
        if r['method']==name and r['query_id']==q]) for q in qids]) for name in METHODS}
    rng=np.random.default_rng(SPEC['bootstrap_seed'])
    ii=rng.integers(len(qids),size=(SPEC['bootstrap_replicates'],len(qids)))
    result={}
    for a in METHODS:
        for b in ('original_exact','original_ann64'):
            if a==b:continue
            aa,bb=values[a],values[b]
            ratio=np.mean(aa[ii],axis=1)/np.mean(bb[ii],axis=1)
            result[a+'/'+b]={'ratio_of_means':float(aa.mean()/bb.mean()),
                'query_block_interval95':np.quantile(ratio,[.025,.975]).tolist()}
    return result


def memory(input_root,request,method):
    inputs=ObservedInputs(input_root,request)
    float(np.sum(inputs.reader.corpus));float(np.sum(inputs.reader.projected))
    rec=inputs.records('selection')[0]
    execute(inputs,rec,method);gc.collect()
    tracemalloc.start();cal=np.empty(1000000);cal.fill(1.)
    traced=tracemalloc.get_traced_memory()[1]
    tracemalloc.stop();del cal;gc.collect()
    if traced<8000000:raise RuntimeError('NumPy tracing calibration failed')
    baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start();execute(inputs,rec,method)
    live,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    end=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale=1 if sys.platform=='darwin' else 1024
    if end*scale>SPEC['rss_gib']*1024**3:raise RuntimeError('memory ceiling breached')
    return {'method':method,'traced_peak_bytes':peak,'traced_live_bytes':live,
        'numpy_calibration_bytes':traced,'baseline_rss_highwater_bytes':int(baseline*scale),
        'rss_highwater_bytes':int(end*scale),'rss_highwater_increase_bytes':int((end-baseline)*scale),
        'limitations':'includes common resident inputs/index; tracing perturbs RSS; highwater difference is imperfect'}


def run(input_root,request,output):
    output.mkdir()
    started=perf_counter()
    save(output/'reservation.json',{'sources':sources(),'spec':SPEC,
        'request_binding':fingerprint(request),'input_root':str(input_root.resolve())})
    memory_rows=[]
    for method in ('original_exact','original_ann64','projected_fixed:256'):
        child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--memory',method,
            '--inputs',str(input_root),'--request',str(input_root/'request.json')],capture_output=True,text=True)
        (output/('memory-'+method.replace(':','-')+'.stderr.txt')).write_text(child.stderr)
        if child.returncode:raise RuntimeError('memory worker failed: '+child.stderr)
        row=json.loads(child.stdout);memory_rows.append(row)
        save(output/('memory-'+method.replace(':','-')+'.json'),row)
        print('Memory checked '+method,flush=True)
    inputs=ObservedInputs(input_root,request)
    setup_seconds=perf_counter()-started
    reports={}
    choice=None
    for group in ('selection','evaluation'):
        quality_rows,timings,stats=measure_baselines(inputs,group,output,started,input_root)
        costs={k:v['mean_ms'] for k,v in stats.items()}
        if group=='selection':
            choice=select_fixed(quality_rows,costs)
            binding=seal_selection(output/'selection_seal.json',inputs,choice,quality_rows,costs)
        means={method:{key:float(np.mean([r['methods'][method]['metrics'][key] for r in quality_rows]))
            for key in ('retention','hit','hit_delta')} for method in METHODS}
        reports[group]={'quality_mean':means,'timing':stats,'paired_ratios':paired_summary(timings),
            'oracle_main':perfect_information_bound(quality_rows,costs),
            'oracle_match_ann':perfect_information_bound(quality_rows,costs,True),
            'fixed_selection_budget':choice,'selected_fixed_quality':means.get('projected_fixed:'+str(choice)),
            'quality_compact':[{'query_id':r['query_id'],
                'metrics':{m:r['methods'][m]['metrics'] for m in METHODS}} for r in quality_rows],
            'timing_raw':timings}
        save(output/(group+'.summary.json'),reports[group])
        if group=='selection':inputs.authorize_evaluation(output/'selection_seal.json',binding)
    check_resources(started,output,input_root)
    receipt={'schema':SPEC['schema'],'sources':sources(),'spec':SPEC,
        'input_binding':inputs.input_binding,'input_pack_sha256':sha(input_root/'observed.json'),
        'environment':{'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
            'thread_env':{k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}},
        'setup_and_memory_seconds':setup_seconds,'elapsed_seconds':perf_counter()-started,
        'reports':reports,'memory':memory_rows,'formal_certification':False,
        'history_exclusion_established':False,'semantic_independence_established':False,
        'protected_roles_opened':False,
        'artifacts_sha256':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    save(output/'receipt.json',receipt)
    print('COMPLETE '+str(output/'receipt.json'),flush=True)


def main():
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',type=Path);group.add_argument('--run',type=Path)
    group.add_argument('--memory',choices=METHODS)
    p.add_argument('--sampling',type=Path);p.add_argument('--inputs',type=Path)
    p.add_argument('--request',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--bundle',default='/root/autodl-fs/tls-rag-step5-input-dPpTfj/preparation-ov283f6o/bundle')
    p.add_argument('--archive',default='/root/autodl-fs/tls-rag-step5-input-dPpTfj/experiment-aynq9z39/run_archive')
    args=p.parse_args()
    if args.prepare:
        if args.sampling is None:p.error('--sampling required')
        request=make_request(args.sampling,args.bundle,args.archive)
        args.prepare.mkdir();save(args.prepare/'request.json',request)
        (args.prepare/'export_readonly.py').write_text(remote_source(request))
        print(args.prepare);return
    if args.memory:
        if args.inputs is None or args.request is None:p.error('--inputs and --request required')
        print(json.dumps(memory(args.inputs,json.loads(args.request.read_text()),args.memory),allow_nan=False));return
    if args.inputs is None:p.error('--inputs required')
    request=json.loads((args.inputs/'request.json').read_text())
    existed=args.run.exists()
    try:run(args.inputs,request,args.run)
    except Exception:
        if not existed and args.run.is_dir() and not (args.run/'failure.txt').exists():
            (args.run/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':main()
