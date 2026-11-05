#!/usr/bin/env python3
"""Gate2 local tiny, CPU readiness check, one manually supervised server run."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
# Unchanged previously audited ownership/stop/closure machinery. This does not
# invoke C1 science, alter its gate flag, or modify a closed C1 directory.
spec=importlib.util.spec_from_file_location('frozen_c1_supervision',ROOT/'scripts/run_tls_rag_step5_pq_refinement_cost_v1.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


def local_tiny(root):
    from mechanism import tls_rag_step5_pq_refinement_geometry_v1 as g
    import numpy as np
    import faiss
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    g.cost.unique_json(root/'source_freeze.json',dict(namespace='synthetic-gate2-tiny-only',
        sources=g.source_map(g.config()),actual_gate2_started=False,seed=2026110519,
        original_vectors_or_query_roles_read=False))
    faiss.omp_set_num_threads(1)
    rng=np.random.default_rng(2026110519);d=8
    raw=rng.normal(size=(1024,d)).astype(np.float32);raw/=np.linalg.norm(raw,axis=1)[:,None]
    A=np.diag(np.linspace(.97,1.03,d)).astype(np.float32)
    quantizer=faiss.IndexFlatIP(d);index=faiss.IndexIVFPQ(quantizer,d,4,2,8,faiss.METRIC_INNER_PRODUCT)
    index.cp.niter=2;index.pq.cp.niter=2;index.pq.cp.min_points_per_centroid=1
    index.train(np.ascontiguousarray(raw@A.T));index.add(np.ascontiguousarray(raw@A.T));index.nprobe=4
    ids=np.arange(len(raw),dtype=np.int64);book=g.extract_actual_ivf_codes(index,ids)
    c=dict(g.config(),k=5,dimensions=[2,4],sketch_seeds=[2026110401],side_pq_bytes=[2],
        side_pq_seeds=[2026110501],side_pq_training=dict(niter=2,nredo=1),scan_block=32,build_block=256)
    ctx=g.reconstruction_context(book,A,np.zeros(d),c)
    summaries,errors,offline=g.build_residual_summaries(raw,book,ctx,np.arange(1024),c,phase=g.progress_callback(root))
    q=rng.normal(size=d).astype(np.float32);q/=np.linalg.norm(q)
    model=dict(spec=dict(kind='opq'),metric='ip',matrix=A,mean=np.zeros(d))
    search=lambda query:g.cost.stored.search_backend(model,query,256,index=index)
    ids=search(q);pos=np.searchsorted(book['rows'],ids)
    rr,dd=g.cost.native.rerank(raw,q,ids,c['k'])
    checks=[]
    for name,s in summaries.items():
        r=g.scan_and_refine(raw,q,ids,book,pos,ctx,s,c)
        er,ed=g.cost.native.rerank(raw,q,r['used'],c['k'])
        if not np.array_equal(er,r['returned']) or not np.array_equal(ed,r['distances']):
            raise RuntimeError('tiny original-score subset mismatch')
        checks.append(dict(name=name,read_rows=len(r['used']),
            retained=len(set(rr)&set(r['returned']))/c['k'],bound_audit=r['bound_audit']))
    full=g.scan_with_bounds(raw,q,ids,np.full(len(ids),-np.inf),c['k'],32)
    if not np.array_equal(rr,full['returned']) or not np.array_equal(dd,full['distances']):
        raise RuntimeError('tiny forced-full-read mismatch')
    callback_checks=[]
    for mode in ['baseline','fullread','prefix_64']+list(summaries)+['oracle:'+n for n in summaries]:
        r=g.paired_geometry_replay(raw,q,search,ids,book,pos,ctx,summaries,c,mode,rr)
        er,ed=g.cost.native.rerank(raw,q,r['used'],c['k'])
        if not np.array_equal(er,r['returned']) or not np.array_equal(ed,r['distances']):
            raise RuntimeError('tiny searched complete callback mismatch')
        callback_checks.append(dict(mode=mode,seconds=r['seconds'],aux_seconds=r['aux_seconds'],
            actual_search_performed=True,unique_read_rows=len(r['used'])))
    tiny_source=dict(c['source'],rows=len(raw),dimension=d)
    tiny_c=dict(c,source=tiny_source)
    g.save_summaries(root,book,ctx,summaries,offline,tiny_c)
    _,geometry=g.geometry_audit(raw,q,ids,pos,dict(book=book,ctx=ctx,model=model),errors,summaries,c,root/'geometry.npz')
    controller.write(root/'tiny.json',dict(status='passed',synthetic_only=True,
        real_gate2_started=False,checks=checks,offline_costs=offline,seed=2026110519,
        actual_IVF_search=True,callback_checks=callback_checks,geometry=geometry,
        nonorthogonal_matrix=True,formal_certification=False))
    controller.close_evidence(root)
    return dict(status='passed',synthetic_only=True,real_gate2_started=False,root=str(root))


def preflight():
    from mechanism import tls_rag_step5_pq_refinement_geometry_v1 as g
    import faiss,h5py,numpy,scipy
    c=g.config();g.source_map(c)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise RuntimeError('CPU preflight requires a clean reviewed checkout')
    source=Path(c['source']['path']);old=Path(c['source_run'])
    versions=dict(numpy=numpy.__version__,faiss=faiss.__version__,scipy=scipy.__version__,h5py=h5py.__version__)
    if versions!=c['versions']:raise RuntimeError('fixed CPU dependency mismatch')
    if not source.is_file() or source.stat().st_size!=c['source']['bytes']:raise RuntimeError('original source missing/size mismatch')
    if g.cost.sha(old/'delivery_index.json')!=c['source_index_sha256']:raise RuntimeError('closed source index mismatch')
    for e in c['products']:g.cost.verify_product(old,e)
    return dict(status='ready_for_reviewed_CPU_run',hardware=g.cost.cpu_identity(),versions=versions,
        no_science_started=True,GPU_not_queried=True,source_full_SHA_deferred_to_one_real_run=True,
        scientific_sources=g.source_map(c),free_bytes={str(p):os.statvfs(p).f_bavail*os.statvfs(p).f_frsize for p in [old,source.parent]})


def worker(root):
    from mechanism import tls_rag_step5_pq_refinement_geometry_v1 as g
    c=g.config();g.cost.unique_json(root/'source_freeze.json',dict(namespace=c['namespace'],config=c,
        sources=g.source_map(c),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        started_unix=time.time(),GPU_not_used=True,actual_consent=c['authorization']))
    try:g.cost.unique_json(root/'result.json',g.run_science(c,root))
    except Exception as e:
        g.cost.unique_json(root/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['tiny','preflight','run','worker','status','stop'])
    p.add_argument('--root');a=p.parse_args()
    # Set thread environment BEFORE importing any numerical package.
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if a.action=='preflight':print(json.dumps(preflight()));return 0
    if not a.root:p.error('--root is required for this action')
    original=Path(a.root).absolute()
    if original.is_symlink():raise ValueError('run root symlink')
    root=original.resolve()
    if a.action=='tiny':print(json.dumps(local_tiny(root)))
    elif a.action=='worker':worker(root)
    elif a.action=='status':print((root/'status.json').read_text())
    elif a.action=='stop':controller.request_stop(root)
    else:
        if sys.platform!='linux':raise RuntimeError('real gate2 requires reviewed Linux CPU; use tiny locally')
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
            raise RuntimeError('real gate2 requires a clean reviewed checkout')
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--root',str(root)]
        result=controller.supervise(command,root,os.environ.copy(),interval=300)
        return 0 if result['status']=='complete' else 1
    return 0


if __name__=='__main__':sys.exit(main())
