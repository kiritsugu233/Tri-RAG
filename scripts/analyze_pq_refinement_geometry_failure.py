#!/usr/bin/env python3
"""Read-only failed-G2/G1 record audit; no original vectors or model execution."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

COMMIT='2cb1c62c546fbee215562b0ce29dd59cbb93a89a'
INDEX='9f386b7e86b152b44ace3a60e03cb50bf8c8d1bbc8bcd9b9be31abee430a981e'
REPO=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def audit(recovery,s0,output):
    recovery=Path(recovery).resolve();s0=Path(s0).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    receipt=read(recovery/'recovery-verification.json');root=Path(receipt['scientific_directory'])
    if not root.is_relative_to(recovery/'received'):raise ValueError('unverified input location')
    if sha(root/'archive_index.json')!=INDEX:raise ValueError('different closed failure index')
    entries=read(root/'archive_index.json')['files']
    for r in entries:
        q=root/r['path']
        if q.stat().st_size!=r['bytes'] or sha(q)!=r['sha256']:raise ValueError(r['path'])
    freeze=read(root/'source_freeze.json');c=freeze['config'];status=read(root/'status.json');failure=read(root/'failure.json')
    if freeze['commit']!=COMMIT or status['status']!='compute_failed' or status['returncode']!=1:
        raise ValueError('not the specified failed G2')
    if (root/'calls.jsonl').exists() or (root/'summary.json').exists():raise ValueError('unexpected timed products')
    for path,h in freeze['sources'].items():
        blob=subprocess.check_output(['git','show',COMMIT+':'+path],cwd=REPO)
        if hashlib.sha256(blob).hexdigest()!=h:raise ValueError('executed-source identity')
    if sha(s0/'delivery_index.json')!=c['source_index_sha256']:raise ValueError('original S0 index')
    old={r['path']:r for r in read(s0/'delivery_index.json')['files']}
    for r in c['products']:
        if old[r['path']]!=r:raise ValueError('G2 configuration changed source products')
    consumed=[]
    def original(path):
        r=old[path];q=s0/path
        if q.stat().st_size!=r['bytes'] or sha(q)!=r['sha256']:raise ValueError('original local data changed')
        consumed.append(r);return q
    roles=read(original('roles.json'))['queries'];queries=np.load(original('queries.npy'),allow_pickle=False)
    old_freeze=read(original('source_freeze.json'));inp=read(root/'input_receipt.json')
    native=old_freeze['native_binary_sha256']
    if native['numpy']!=inp['binary_sha256']['numpy']:raise ValueError('original NumPy native SHA')
    if any(native.get(k)!=v for k,v in inp['binary_sha256']['faiss'].items()):raise ValueError('original FAISS native SHA')
    rows=[];samples={};g1=read(root/'G1_summary.json')
    if len(g1)!=64:raise ValueError('incomplete G1 cohort')
    raw_distance_changes=0;raw_distance_max=0.;max_algebra_error=0.
    for probe in c['nprobe']:
        candidates=np.load(original('ivf/opq64/{}/candidates.npy'.format(probe)),allow_pickle=False)
        timed=read(original('ivf/opq64/{}/timings.json'.format(probe)))
        for qi in c['query_indices']:
            r=next(x for x in g1 if x['nprobe']==probe and x['query_index']==qi)
            if r['role']!=roles[qi]['role']:raise ValueError('role mismatch')
            p=root/'G1_geometry'/('{}-{}.npz'.format(probe,qi))
            with np.load(p,allow_pickle=False) as arrays:a={k:arrays[k] for k in arrays.files}
            ids=a['ids'];d=a['native_distances'];radius=a['residual_radius'];lower=a['lower_norm']
            if len(ids)!=6400 or not np.array_equal(ids,candidates[qi]):raise ValueError('candidate sequence')
            pos=np.lexsort((ids,d))[:100];top=ids[pos];tau=float(np.sort(d)[99])
            if not np.array_equal(top,a['native_topk']):raise ValueError('independent stable full-C order')
            ref=[x for x in timed if x['query_index']==qi and x['budget']==6400]
            if len(ref)!=3 or any(x['returned']!=top.tolist() for x in ref):raise ValueError('S0 true candidate-top100 identity')
            # Record, rather than hide behind a tolerance, full-batch vs final
            # 100-row FP32 rescoring differences. Raw vectors are NOT reread.
            expected=np.asarray(ref[0]['distances'],np.float32)
            raw_distance_changes+=int(np.count_nonzero(d[pos]!=expected))
            raw_distance_max=max(raw_distance_max,float(np.max(abs(d[pos]-expected))))
            qnorm=float(np.linalg.norm(queries[qi].astype(float)))
            if not np.allclose(a['signed_native_gap'],tau-d.astype(float),rtol=0,atol=1e-12):raise ValueError('saved signed gap')
            gap=np.divide(tau-d.astype(float),qnorm*radius,out=np.zeros(len(d)),where=qnorm*radius>0)
            if not np.allclose(gap,a['normalized_gap'],rtol=1e-12,atol=1e-12):raise ValueError('saved normalized-gap arithmetic')
            algebra=np.max(abs(a['original_reconstruction_dot']+a['residual_correction']-a['real_raw_dot']))
            max_algebra_error=max(max_algebra_error,float(algebra))
            if algebra>1e-12:raise ValueError('saved correction decomposition')
            if not np.isfinite(d).all() or not np.isfinite(radius).all() or np.any(radius<0):raise ValueError('invalid stored geometry')
            violations=int(np.count_nonzero(lower>d))
            skip=(lower>tau)&~np.isin(ids,top);skips=int(skip.sum())
            b=r['bounds'][0]
            if b['name']!='norm' or b['candidate_bound_violations']!=violations or b['beneficial_threshold_skip']!=skips:
                raise ValueError('saved norm diagnostic')
            for key,values in [('radius_quantiles',radius),('absolute_normalized_gap_quantiles',abs(gap))]:
                if not np.allclose(r[key],np.quantile(values,[0,.5,.9,.99,1]),rtol=1e-12,atol=1e-12):raise ValueError('saved quantiles')
            if r['native_cutoff']!=tau:raise ValueError('saved native threshold')
            band_counts=[]
            for band in r['boundary_bands']:
                count=int(np.count_nonzero(abs(gap)<=band['normalized_band']))
                if count!=band['rows']:raise ValueError('saved boundary band count')
                band_counts.append(count)
            row=dict(nprobe=probe,query_index=qi,role=r['role'],candidates=6400,
                radius_median=float(np.median(radius)),radius_p99=float(np.quantile(radius,.99)),
                abs_normalized_gap_median=float(np.median(abs(gap))),abs_normalized_gap_p99=float(np.quantile(abs(gap),.99)),
                norm_bound_violations=violations,norm_favorable_threshold_skips=skips,
                norm_favorable_threshold_skip_fraction=skips/6400,
                norm_favorable_threshold_required_rows=6400-skips,
                normalized_boundary_0_01_rows=band_counts[0],normalized_boundary_0_1_rows=band_counts[1],
                ADC_reconstruction_difference_max=float(np.max(abs(a['actual_ADC_reconstructed_dot']-a['original_reconstruction_dot']))))
            rows.append(row);samples.setdefault((probe,r['role']),[]).append((radius,abs(gap),skips))
    groups=[]
    for (probe,role),parts in sorted(samples.items()):
        radius=np.concatenate([x[0] for x in parts]);gap=np.concatenate([x[1] for x in parts]);skips=np.array([x[2] for x in parts])
        if len(parts)!=16:raise ValueError('role cohort')
        groups.append(dict(nprobe=probe,role=role,queries=16,candidate_occurrences=len(radius),
            pooled_radius_median=float(np.median(radius)),pooled_radius_p99=float(np.quantile(radius,.99)),
            pooled_abs_normalized_gap_median=float(np.median(gap)),pooled_abs_normalized_gap_p99=float(np.quantile(gap,.99)),
            norm_favorable_threshold_skip_mean=float(skips.mean()),norm_favorable_threshold_skip_max=int(skips.max()),
            norm_favorable_threshold_fraction_mean=float(skips.mean()/6400),
            pooled_normalized_boundary_0_1_fraction=float(np.mean(gap<=.1))))
    # Static Python-binding reproducer, no geometry or retraining. Both build
    # sites have the same conflicting keyword; fixing just the first is unsafe.
    source=subprocess.check_output(['git','show',COMMIT+':mechanism/tls_rag_step5_pq_refinement_geometry_v1.py'],cwd=REPO,text=True)
    tree=ast.parse(source);callbacks=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='phase']
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='phase' and any(k.arg=='name' for k in n.keywords)]
    if len(callbacks)!=1 or callbacks[0].args.args[0].arg!='name' or len(calls)!=2:raise ValueError('different callback defect')
    def phase(name,**fields):return dict(phase=name,**fields)
    try:phase('sketch_build',name='demonstration_only')
    except TypeError as error:binding_error=str(error)
    else:raise AssertionError('binding error not reproduced')
    output.mkdir(parents=True)
    table(output/'G1_per_query.csv',rows);table(output/'G1_groups.csv',groups)
    result=dict(source_commit=COMMIT,source_index_sha256=INDEX,closed_files_verified=len(entries),
        G1_pairs_verified=len(rows),candidate_occurrences_verified=6400*len(rows),
        stable_top100_matches_original=64,norm_bound_violations=sum(r['norm_bound_violations'] for r in rows),
        norm_favorable_threshold_total_skips=sum(r['norm_favorable_threshold_skips'] for r in rows),
        fullbatch_vs_final100_native_distance_differences=raw_distance_changes,
        fullbatch_vs_final100_native_distance_max_difference=raw_distance_max,
        maximum_saved_decomposition_error=max_algebra_error,
        independent_arithmetic_check_atol=1e-12,arithmetic_check_is_not_a_new_retrieval_tolerance=True,
        static_callback_signature_line=callbacks[0].lineno,static_conflicting_call_lines=[n.lineno for n in calls],
        binding_error_reproduced=binding_error,elapsed_seconds=status['elapsed_seconds'],
        source_verify_seconds=inp['source_verify_seconds'],decode_checks=inp['decode_checks'],
        native_binary_identity_matches_S0=True,original_vector_distances_recomputed=False,
        formal_calls=0,side_PQ_not_started=True,no_closed_Gaussian_quality_or_cost_result=True,
        formal_certification=False,gate2_complete=False,no_retry_started=True,consumed_original_files=consumed,
        script_sha256=sha(__file__))
    write(output/'audit.json',result);write(output/'groups.json',groups)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recovery',required=True);p.add_argument('--s0',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.recovery,a.s0,a.output)
