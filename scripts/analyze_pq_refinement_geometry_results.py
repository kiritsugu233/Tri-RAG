#!/usr/bin/env python3
"""Read-only closed G2 audit. Never imports retrieval, reads raw vectors or fits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

REPO=Path(__file__).resolve().parents[1]
COMMIT='3f82b40ac6812c72bbab0a54ee46911c6b1ae74b'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def table(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def audit(recovery,s0,output):
    recovery=Path(recovery).resolve();s0=Path(s0).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    receipt=read(recovery/'recovery-verification.json');root=Path(receipt['scientific_directory'])
    if not root.is_relative_to(recovery/'received'):raise ValueError('unverified result root')
    if sha(root/'archive_index.json')!=receipt['scientific_index_sha256']:raise ValueError('closed index changed')
    entries=read(root/'archive_index.json')['files']
    cloud_only={e['path']:e for e in receipt.get('scientific_files_on_cloud_only',[])}
    locally_verified=0
    for e in entries:
        p=root/e['path']
        if not p.exists():
            if cloud_only.get(e['path'])!=e or not e['path'].endswith('.npz'):
                raise ValueError('missing required local record: '+e['path'])
            continue
        if p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:raise ValueError(e['path'])
        locally_verified+=1
    freeze=read(root/'source_freeze.json');c=freeze['config'];state=read(root/'status.json')
    if freeze['commit']!=COMMIT or state['status']!='complete' or state['returncode']!=0:
        raise ValueError('this auditor requires the completed fixed-source G2; retain failures separately')
    for p,digest in freeze['sources'].items():
        if hashlib.sha256(subprocess.check_output(['git','show',COMMIT+':'+p],cwd=REPO)).hexdigest()!=digest:
            raise ValueError('executed scientific source')
    if sha(s0/'delivery_index.json')!=c['source_index_sha256']:raise ValueError('original S0 index')
    old={r['path']:r for r in read(s0/'delivery_index.json')['files']};consumed={}
    def original(name):
        e=old[name];p=s0/name
        if p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:raise ValueError('original artifact changed')
        consumed[name]=e;return p
    for e in c['products']:
        if old[e['path']]!=e:raise ValueError('original product identity')
    roles=read(original('roles.json'))['queries'];queries=np.load(original('queries.npy'),allow_pickle=False)
    truth=np.load(original('published_truth.npy'),allow_pickle=False)
    cutoffs=np.load(original('published_cutoffs.npy'),allow_pickle=False)
    global_gt=np.load(original('native_fp32_top100.npy'),allow_pickle=False)
    source=read(original('source_freeze.json'));inp=read(root/'input_receipt.json')
    binaries=source['native_binary_sha256']
    if inp['binary_sha256']['numpy']!=binaries['numpy']:raise ValueError('native NumPy identity')
    if any(binaries.get(k)!=v for k,v in inp['binary_sha256']['faiss'].items()):raise ValueError('native FAISS identity')
    candidates={};references={};geometry={};bound_events=[]
    closed_geometry={(r['nprobe'],r['query_index']):r for r in read(root/'geometry_summary.json')}
    for probe in c['nprobe']:
        candidates[probe]=np.load(original('ivf/opq64/{}/candidates.npy'.format(probe)),allow_pickle=False)
        refs=read(original('ivf/opq64/{}/timings.json'.format(probe)))
        references[probe]={r['query_index']:r for r in refs if r['budget']==6400}
        for qi in c['query_indices']:
            geometry_path=root/'geometry'/('{}-{}.npz'.format(probe,qi))
            if geometry_path.exists():
                with np.load(geometry_path,allow_pickle=False) as z:geometry[probe,qi]={k:z[k] for k in z.files}
            else:
                geometry[probe,qi]=dict(ids=candidates[probe][qi],native_topk=np.asarray(references[probe][qi]['returned'],np.int64),native_distances=None)
                for b in closed_geometry[probe,qi]['bounds']:
                    bound_events.append(dict(nprobe=probe,role=roles[qi]['role'],query_index=qi,
                        method=b['name'],candidate_violations=b['candidate_bound_violations'],topk_violations=b['true_topk_bound_violations'],
                        favorable_threshold_skip_rows=b['beneficial_threshold_skip'],locally_recomputed=False))
                continue
            a=geometry[probe,qi];ids=a['ids'];d=a['native_distances']
            if not np.array_equal(ids,candidates[probe][qi]):raise ValueError('geometry candidates')
            full=ids[np.lexsort((ids,d))[:c['k']]]
            if full.tolist()!=references[probe][qi]['returned'] or not np.array_equal(full,a['native_topk']):
                raise ValueError('original candidate-top100 reference')
            tau=float(np.partition(d,c['k']-1)[c['k']-1]);top=np.isin(ids,full)
            for key,lower in a.items():
                if key.startswith('lower_'):
                    bad=lower>d
                    bound_events.append(dict(nprobe=probe,role=roles[qi]['role'],query_index=qi,
                        method=key[6:],candidate_violations=int(bad.sum()),topk_violations=int((bad&top).sum()),
                        favorable_threshold_skip_rows=int(((lower>tau)&~top).sum())))
    records=[json.loads(s) for s in (root/'calls.jsonl').read_text().splitlines()]
    if len(records)!=16576 or read(root/'result.json')['formal_calls']!=16576:raise ValueError('incomplete formal calls')
    by={};traces={};quality_rows=[];quality_map={};distance_differences=0;trace_checks=0
    score_rerank_unavailable=0;bound_replay_unavailable=0
    for r in records:
        probe,qi,mode=r['nprobe'],r['query_index'],r['mode'];key=probe,qi,mode
        if probe not in c['nprobe'] or qi not in c['query_indices'] or mode not in c['modes']:raise ValueError('unexpected call identity')
        if roles[qi]['id']!=r['query_id'] or roles[qi]['role']!=r['role']:raise ValueError('role or query mismatch')
        a=geometry[probe,qi];ids=a['ids'];rr=np.asarray(r['returned'],np.int64);dd=np.asarray(r['distances'],np.float32)
        if fp(ids.tolist())!=r['candidate_sha256']:raise ValueError('candidate hash')
        if len(rr)!=100 or len(set(rr.tolist()))!=100 or not np.isfinite(dd).all():raise ValueError('returned schema')
        if abs(r['seconds']-r['search_seconds']-r['rerank_seconds'])>1e-11 or min(r['seconds'],r['rerank_seconds'],r['search_seconds'])<0:
            raise ValueError('clock partition')
        if any(r[k] is not None for k in ['actual_dram_bytes','actual_ssd_bytes','physical_requests']):raise ValueError('unsupported physical counters')
        if key not in traces:
            tr=read(root/'traces'/('{}-{}-{}.json'.format(probe,qi,mode.replace(':','_'))))
            pos=np.asarray(tr['read_positions'],np.int64)
            if len(pos)<100 or np.any(np.diff(pos)<=0) or pos[0]<0 or pos[-1]>=6400:raise ValueError('read positions')
            used=ids[pos]
            if not np.isin(rr,used).all():raise ValueError('returned point was not read')
            if a['native_distances'] is not None:
                selected=used[np.lexsort((used,a['native_distances'][pos]))[:100]]
                if not np.array_equal(selected,rr):raise ValueError('saved full-score subset stable top100 differs: '+str(key))
            else:score_rerank_unavailable+=1
            if mode in ('baseline','fullread'):
                ref=references[probe][qi]
                if rr.tolist()!=ref['returned'] or dd.tolist()!=ref['distances']:raise ValueError('original full-C output')
            if tr['signature']!=fp([pos.tolist(),rr.tolist(),dd.tolist()]):raise ValueError('closed trace signature')
            if tr['trace'] is None:
                count=6400 if mode=='baseline' else int(mode.split('_')[1])
                if not np.array_equal(pos,np.arange(count)):raise ValueError('fixed prefix trace')
                batches=2
            else:
                actual=tr['trace'];seen=np.zeros(6400,bool)
                pilot=np.asarray(actual[0]['positions'],np.int64)
                expected=np.flatnonzero(np.isin(ids,a['native_topk'])) if mode.startswith('oracle:') else np.arange(100)
                if not np.array_equal(pilot,expected):raise ValueError('pilot truth isolation/order')
                seen[pilot]=True;best=pilot.copy()
                lower=(np.full(6400,-np.inf) if mode=='fullread' else a.get('lower_'+mode.removeprefix('oracle:')))
                block=6400 if mode.startswith('oracle:') else c['scan_block']
                if len(actual)!=1+len(range(0,6400,block)):raise ValueError('block trace count')
                previous=float(actual[0]['threshold'])
                for event,lo in zip(actual[1:],range(0,6400,block)):
                    tau=float(event['threshold']);available=np.arange(lo,min(lo+block,6400))
                    if not np.isfinite(tau) or not 0<=tau<=2 or tau>previous:raise ValueError('native threshold monotonicity')
                    if lower is not None:
                        take=available[(~seen[available])&(lower[available]<=tau)]
                        if take.tolist()!=event['positions']:raise ValueError('independent bound decisions: '+str(key))
                    else:
                        take=np.asarray(event['positions'],np.int64)
                        if len(np.unique(take))!=len(take) or not np.isin(take,available).all() or seen[take].any():
                            raise ValueError('actual block read addresses')
                    seen[take]=True;previous=tau
                if lower is None:bound_replay_unavailable+=1
                if not np.array_equal(np.flatnonzero(seen),pos):raise ValueError('read union')
                batches=sum(bool(e['positions']) for e in actual)+1;trace_checks+=1
            traces[key]=(tr,pos,batches,rr.copy(),dd.copy())
            q=dict(nprobe=probe,role=r['role'],query_index=qi,mode=mode,read_rows=len(pos),gather_batches=batches,
                recall=float(np.count_nonzero(dd<=cutoffs[qi]+.001)/100),strict_id=len(set(rr)&set(truth[qi]))/100,
                candidate_topk_retention=len(set(rr)&set(a['native_topk']))/100,
                global_native_retention=len(set(rr)&set(global_gt[qi]))/100,
                preserves_candidate_topk=bool(np.array_equal(rr,a['native_topk'])))
            quality_rows.append(q)
            quality_map[key]=q
            lookup={int(row):i for i,row in enumerate(ids)}
            if a['native_distances'] is not None:
                distance_differences+=int(np.count_nonzero(dd!=a['native_distances'][[lookup[int(row)] for row in rr]]))
        tr,pos,batches,expected_rr,expected_dd=traces[key]
        if (fp(pos.tolist())!=r['read_positions_sha256'] or fp(tr['trace'])!=r['read_trace_sha256']
                or not np.array_equal(rr,expected_rr) or not np.array_equal(dd,expected_dd)):
            raise ValueError('repeat changed read/output trace')
        if r['unique_read_rows']!=len(pos) or r['gather_batches']!=batches or r['logical_original_payload_bytes']!=(len(pos)+100)*640*4:
            raise ValueError('logical work accounting')
        q=quality_map[key]
        if any(r['quality'][k]!=q[k] for k in r['quality']):raise ValueError('quality recomputation')
        by.setdefault((probe,r['role'],mode,qi),[]).append(r)
    qrows=[]
    for key,rs in sorted(by.items()):
        if sorted(r['repeat'] for r in rs)!=list(range(7)):raise ValueError('repeat coverage')
        q=dict(nprobe=key[0],role=key[1],mode=key[2],query_index=key[3])
        for field in ['seconds','rerank_seconds','search_seconds','aux_seconds']:
            q[field]=float(np.mean([r[field] for r in rs]))
        qrows.append(q)
    saved=read(root/'summary.json');savedq={(r['nprobe'],r['role'],r['mode'],r['query_index']):r for r in saved['query_rows']}
    for q in qrows:
        oldq=savedq[q['nprobe'],q['role'],q['mode'],q['query_index']]
        if any(abs(q[f]-oldq[f])>1e-12 for f in ['seconds','rerank_seconds','search_seconds']):raise ValueError('saved query means')
    lookup={(q['nprobe'],q['role'],q['mode'],q['query_index']):q for q in qrows};gs={(g['nprobe'],g['role'],g['mode']):g for g in saved['groups']}
    summary=[];fees=[];rng=np.random.Generator(np.random.PCG64(c['bootstrap_seed']))
    for probe in c['nprobe']:
        for role in ['geometry_discovery','geometry_check']:
            indices=sorted(qi for qi in c['query_indices'] if roles[qi]['role']==role)
            if len(indices)!=16:raise ValueError('query cohort')
            draws=rng.integers(0,16,(10000,16));base=np.array([lookup[probe,role,'baseline',i]['seconds'] for i in indices]);bt=base[draws].mean(1)
            for mode in c['modes']:
                values=np.array([lookup[probe,role,mode,i]['seconds'] for i in indices]);vt=values[draws].mean(1);g=gs[probe,role,mode]
                ci=np.quantile((bt-vt)/bt,[.025,.975])
                if abs(values.mean()-g['mean_seconds'])>1e-12 or not np.allclose(ci,g['saving_fraction_ci95'],rtol=0,atol=1e-12):raise ValueError('paired statistic')
                qs=[q for q in quality_rows if (q['nprobe'],q['role'],q['mode'])==(probe,role,mode)]
                summary.append(dict(nprobe=probe,role=role,mode=mode,queries=16,mean_ms=float(values.mean()*1000),
                    rerank_ms=float(np.mean([lookup[probe,role,mode,i]['rerank_seconds'] for i in indices])*1000),
                    bound_only_aux_ms=float(np.mean([lookup[probe,role,mode,i]['aux_seconds'] for i in indices])*1000),
                    saving_fraction=float((base-values).mean()/base.mean()),saving_low=float(ci[0]),saving_high=float(ci[1]),
                    mean_read_rows=float(np.mean([q['read_rows'] for q in qs])),
                    candidate_retention=float(np.mean([q['candidate_topk_retention'] for q in qs])),
                    queries_losing_candidate_topk=sum(not q['preserves_candidate_topk'] for q in qs),
                    recall=float(np.mean([q['recall'] for q in qs])),strict_id=float(np.mean([q['strict_id'] for q in qs]))))
                for gamma in c['gammas']:
                    interval=np.quantile(bt-vt-gamma*bt,[.025,.975]);mean=float(np.mean(base-values-gamma*base));old=next(x for x in g['aux_allowance'] if x['gamma']==gamma)
                    if abs(mean-old['mean_seconds'])>1e-12 or not np.allclose(interval,old['ci95'],rtol=0,atol=1e-12):raise ValueError('fee interval')
                    fees.append(dict(nprobe=probe,role=role,mode=mode,gamma=gamma,mean_ms=mean*1000,low_ms=float(interval[0]*1000),high_ms=float(interval[1]*1000)))
    if len(qrows)!=2368 or len(summary)!=148 or len(fees)!=444:raise ValueError('aggregate coverage')
    output.mkdir(parents=True)
    for name,rows in [('per_query.csv',qrows),('quality_reads.csv',quality_rows),('summary.csv',summary),('fee_windows.csv',fees),('bound_events.csv',bound_events)]:table(output/name,rows)
    receipt=dict(source_commit=COMMIT,index_sha256=sha(root/'archive_index.json'),closed_cloud_files=len(entries),
        local_scientific_payloads_verified=locally_verified,scientific_NPZ_files_cloud_only=len(cloud_only),
        calls_verified=len(records),query_groups_verified=len(qrows),read_trace_address_groups_replayed=trace_checks,
        local_fullscore_rerank_unavailable_groups=score_rerank_unavailable,
        local_numeric_bound_replay_unavailable_groups=bound_replay_unavailable,
        bound_event_counts_are_node_closed_outputs_not_locally_recomputed=bool(bound_replay_unavailable),
        quality_read_groups=len(quality_rows),paired_groups=148,fee_intervals=444,
        saved_subset_vs_fullbatch_distance_differences=(None if score_rerank_unavailable else distance_differences),
        original_S0_full_output_groups_verified=128,
        native_binary_matches_original=True,raw_vector_distances_recomputed=False,
        formal_certification=False,full_N_deployment_measured=False,physical_traffic_measured=False,
        elapsed_seconds=state['elapsed_seconds'],input_receipt=inp,consumed_original_files=list(consumed.values()),script_sha256=sha(__file__))
    write(output/'audit.json',receipt);write(output/'tables.json',dict(summary=summary,fees=fees,bound_events=bound_events))
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recovery',required=True);p.add_argument('--s0',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.recovery,a.s0,a.output)
