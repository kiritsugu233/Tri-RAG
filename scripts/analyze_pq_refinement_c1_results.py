#!/usr/bin/env python3
"""Read-only C1 record audit. No original vectors, search or policy execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCIENCE='2b7d2a934c784b38a19b03ea6044f5b061a90457'
INDEX='ae587cb09bc1f9c1d69b6d29e64c8fef63547e69214f70daf822f5ff1b3c5d57'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=True,indent=2,allow_nan=False)+'\n')


def table(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def audit(recovery, s0, output):
    recovery=Path(recovery).resolve();s0=Path(s0).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    receipt=read(recovery/'recovery-verification.json');root=Path(receipt['scientific_directory'])
    if not root.is_relative_to(recovery/'received'):raise ValueError('unverified partial transfer')
    if sha(root/'archive_index.json')!=INDEX:raise ValueError('closed C1 index changed')
    entries=read(root/'archive_index.json')['files']
    for e in entries:
        p=root/e['path']
        if p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:raise ValueError(e['path'])
    freeze=read(root/'source_freeze.json');cfg=freeze['config'];status=read(root/'status.json');result=read(root/'result.json')
    if freeze['commit']!=SCIENCE or status['status']!='complete' or result['status']!='complete':
        raise ValueError('different/unfinished scientific run')
    for p,digest in freeze['sources'].items():
        blob=subprocess.check_output(['git','show',SCIENCE+':'+p],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('executed source identity')
    original=read(s0/'delivery_index.json')
    if sha(s0/'delivery_index.json')!=cfg['source_index_sha256']:raise ValueError('S0 index')
    old_entries={e['path']:e for e in original['files']};old_consumed={}
    def old_path(p):
        e=old_entries[p];path=s0/p
        if sha(path)!=e['sha256'] or path.stat().st_size!=e['bytes']:raise ValueError('S0 file changed: '+p)
        old_consumed[p]=e;return path
    roles=read(old_path('roles.json'))['queries']
    old_freeze=read(old_path('source_freeze.json'));native=read(root/'input_receipt.json')['binary_sha256']
    if native['numpy']!=old_freeze['native_binary_sha256']['numpy']:raise ValueError('NumPy binary differs')
    for name,digest in native['faiss'].items():
        if old_freeze['native_binary_sha256'].get(name)!=digest:raise ValueError('Faiss binary differs')
    candidates={};expected={}
    for probe in cfg['nprobe']:
        candidates[probe]=np.load(old_path('ivf/opq64/{}/candidates.npy'.format(probe)),mmap_mode='r')
        rs=read(old_path('ivf/opq64/{}/timings.json'.format(probe)))
        expected[probe]={r['query_index']:r for r in rs if r['budget']==6400}
    calls=[json.loads(l) for l in (root/'calls.jsonl').read_text().splitlines()]
    if len(calls)!=5824 or result['conformance_calls']!=7552:raise ValueError('incomplete scientific coverage')
    by={};used_cache={};full_checks=0;page_checks=0
    for r in calls:
        probe,qi,mode=r['nprobe'],r['query_index'],r['mode']
        if probe not in cfg['nprobe'] or qi not in cfg['query_indices'] or mode not in cfg['modes']:
            raise ValueError('unexpected probe/query/mode')
        if r['role']!=roles[qi]['role'] or r['query_id']!=roles[qi]['id']:raise ValueError('query identity')
        if (r['seconds']<0 or r['rerank_seconds']<0 or r['search_seconds']<0
                or abs(r['seconds']-r['search_seconds']-r['rerank_seconds'])>1e-11):raise ValueError('clock partition')
        ids=candidates[probe][qi];key=probe,qi,mode
        if key not in used_cache:
            used=ids
            if mode.startswith('subset_'):
                _,pattern,count=mode.split('_');count=int(count)
                if pattern=='prefix':positions=np.arange(count)
                elif pattern=='spread':positions=np.arange(count,dtype=np.int64)*6400//count
                elif pattern=='hash':positions=np.sort(sorted(range(6400),key=lambda j:(fp([cfg['subset_seed'],int(ids[j])]),j))[:count])
                else:raise ValueError('unknown mask')
                used=ids[positions]
            used_cache[key]=used
        used=used_cache[key]
        if r['full_candidate_sha256']!=fp(ids.tolist()) or r['used_candidate_sha256']!=fp(used.tolist()):
            raise ValueError('actual candidate address sequence')
        a=r['address'];starts=6144+used.astype(np.int64)*2560
        if a['logical_payload_bytes']!=(len(used)+100)*2560 or a['gather_batches']!=2:
            raise ValueError('logical requests/payload')
        for size in [4096,65536]:
            union=set((starts//size).tolist()) | set(((starts+2559)//size).tolist())
            if len(union)!=a['unique_virtual_pages_'+str(size)]:raise ValueError('virtual page union')
            page_checks+=1
        if any(a[k] is not None for k in ['actual_dram_bytes','actual_ssd_bytes','physical_requests']):
            raise ValueError('unsupported physical traffic claim')
        if not mode.startswith('subset_'):
            old=expected[probe][qi]
            if r['returned_sha256']!=fp(old['returned']) or r['distance_sha256']!=hashlib.sha256(
                    np.asarray(old['distances'],np.float32).tobytes()).hexdigest():raise ValueError('original FP32 output')
            full_checks+=1
        if mode=='observed':
            parts=r['parts'];keys=['id_seconds','gather_copy_seconds','distance_clip_seconds','selection_seconds','output_gather_seconds','output_score_seconds']
            if any(parts[k]<0 for k in keys) or abs(sum(parts[k] for k in keys)-parts['observed_inner_seconds'])>1e-11:
                raise ValueError('overlapping/missing phase clocks')
            if parts['observed_inner_seconds']>r['rerank_seconds']+1e-11:raise ValueError('phase outside callback')
        by.setdefault((probe,r['role'],mode,qi),[]).append(r)
    saved=read(root/'summary.json');stored_q={(r['nprobe'],r['role'],r['mode'],r['query_index']):r for r in saved['query_rows']}
    qrows=[]
    for key,rs in sorted(by.items()):
        if sorted(r['repeat'] for r in rs)!=list(range(7)):raise ValueError('repeat count')
        if len({(r['returned_sha256'],r['distance_sha256']) for r in rs})!=1:raise ValueError('repeat output changed')
        probe,role,mode,qi=key
        q=dict(nprobe=probe,role=role,mode=mode,query_index=qi,query_id=rs[0]['query_id'],calls=7)
        for field in ['seconds','rerank_seconds','search_seconds']:
            q[field]=float(np.mean([r[field] for r in rs]))
            if abs(q[field]-stored_q[key][field])>1e-12:raise ValueError('saved query mean')
        if mode=='observed':
            for field in rs[0]['parts']:
                q[field]=float(np.mean([r['parts'][field] for r in rs]))
                if abs(q[field]-stored_q[key]['parts'][field])>1e-10:raise ValueError('saved phase mean')
        qrows.append(q)
    if len(qrows)!=832:raise ValueError('query group coverage')
    lookup={(q['nprobe'],q['role'],q['mode'],q['query_index']):q for q in qrows}
    gs={(g['nprobe'],g['role'],g['mode']):g for g in saved['groups']};summary=[];fees=[];parts=[]
    rng=np.random.Generator(np.random.PCG64(cfg['bootstrap_seed']))
    for probe in cfg['nprobe']:
        for role in ['geometry_discovery','geometry_check']:
            indices=sorted(qi for qi in cfg['query_indices'] if roles[qi]['role']==role)
            if len(indices)!=16:raise ValueError('role cohort')
            draws=rng.integers(0,16,(10000,16))
            base=np.array([lookup[probe,role,'baseline',qi]['seconds'] for qi in indices]);bt=base[draws].mean(1)
            for mode in cfg['modes']:
                vals=np.array([lookup[probe,role,mode,qi]['seconds'] for qi in indices]);vt=vals[draws].mean(1)
                delta=base-vals;g=gs[probe,role,mode];ci=np.quantile((bt-vt)/bt,[.025,.975])
                if abs(g['mean_seconds']-vals.mean())>1e-12 or not np.allclose(ci,g['saving_fraction_ci95'],atol=1e-12,rtol=0):
                    raise ValueError('saved aggregate/paired interval')
                summary.append(dict(nprobe=probe,role=role,mode=mode,queries=16,mean_ms=1000*vals.mean(),
                    rerank_ms=1000*np.mean([lookup[probe,role,mode,qi]['rerank_seconds'] for qi in indices]),
                    median_query_ms=1000*np.median(vals),p95_query_ms=1000*np.quantile(vals,.95),
                    saving_ms=1000*delta.mean(),saving_fraction=float(delta.mean()/base.mean()),
                    saving_ci95_low=float(ci[0]),saving_ci95_high=float(ci[1])))
                for gamma in cfg['gammas']:
                    aci=np.quantile(bt-vt-gamma*bt,[.025,.975]);mean=float(np.mean(delta-gamma*base))
                    old=next(a for a in g['aux_allowance'] if a['gamma']==gamma)
                    if abs(mean-old['mean_seconds'])>1e-12 or not np.allclose(aci,old['ci95'],rtol=0,atol=1e-12):
                        raise ValueError('saved added-fee window')
                    fees.append(dict(nprobe=probe,role=role,mode=mode,gamma=gamma,mean_aux_allowance_ms=1000*mean,
                        ci95_low_ms=1000*aci[0],ci95_high_ms=1000*aci[1]))
            obs=[lookup[probe,role,'observed',qi] for qi in indices]
            part=dict(nprobe=probe,role=role,baseline_total_ms=float(1000*base.mean()),
                baseline_rerank_ms=float(1000*np.mean([lookup[probe,role,'baseline',qi]['rerank_seconds'] for qi in indices])))
            for key in ['id_seconds','gather_copy_seconds','distance_clip_seconds','selection_seconds','output_gather_seconds','output_score_seconds','observed_inner_seconds']:
                part[key.replace('_seconds','_ms')]=float(1000*np.mean([q[key] for q in obs]))
            parts.append(part)
    prep=read(root/'preparations.json');observations=[read(p)['after'] for p in (root/'query_observations').glob('*.json')]
    paid=float(np.mean([p['paid_prepare_seconds'] for p in prep]))
    audit_receipt=dict(source_commit=SCIENCE,index_sha256=INDEX,closed_files_verified=len(entries),calls_verified=len(calls),
        virtual_page_unions_verified=page_checks,query_groups_verified=len(qrows),aggregate_groups_verified=len(summary),
        fee_intervals_verified=len(fees),full_original_output_hashes_verified=full_checks,
        diagnostic_subset_repeats_verified=len(calls)-full_checks,source_binary_matches_S0=True,
        old_consumed_files=list(old_consumed.values()),elapsed_seconds=status['elapsed_seconds'],
        source_verify_seconds=read(root/'input_receipt.json')['source_verify_seconds'],
        paid_oracle_and_mask_preparation_mean_ms=paid*1000,
        maximum_worker_rss_observed_bytes=max(o['rss_highwater_bytes'] for o in observations),
        full_original_vector_distances_recomputed_by_this_audit=False,
        actual_dram_bytes=None,actual_ssd_bytes=None,no_fresh_certification=True,
        audit_script_sha256=sha(__file__))
    decision=dict(first_gate_measurements_complete=True,
        cost_verdict='conditional cost window supported for predeclared OPQ64/IVF64 primary interface',
        deployed_net_gain='not established: quality-preserving selection and its full added fee unmeasured',
        high_quality_control='read/copy-only and half-survivor diagnostics do not support 10%; quarter-survivor intervals overlap target',
        mandatory_limit='prepaid/mask preparation is separately paid; do not deploy this diagnostic as a gain',
        physical_traffic_reduction='unmeasured; logical payload is not DRAM/SSD bytes',
        first_gate_formal_certification=False,gate2_started=False,tri_implemented=False)
    output.mkdir(parents=True)
    for name,rows in [('per_query.csv',qrows),('summary.csv',summary),('fee_windows.csv',fees),('parts.csv',parts)]:table(output/name,rows)
    save(output/'audit.json',audit_receipt);save(output/'decision.json',decision)
    save(output/'tables.json',dict(summary=summary,fees=fees,parts=parts))
    print(json.dumps(audit_receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery',required=True);p.add_argument('--s0',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.recovery,a.s0,a.output)
