#!/usr/bin/env python3
"""Section38 A -> B, isolated workers, immutable decisions and bounded science."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
from mechanism import tls_rag_step5_route4_scenarios_v1 as core
from mechanism import tls_rag_step5_route4_advisor_v1 as advisor
from scripts.run_tls_rag_step5_route4_screen_v1 import descendant_processes, terminate_owned_tree

GUARD_SOURCE = 'scripts/run_tls_rag_step5_route4_screen_v1.py'


def source_map():
    return dict(core.sources(), **{GUARD_SOURCE:core.sha(core.ROOT/GUARD_SOURCE)})


def event(root,stage,**value):
    v = dict(stage=stage,elapsed_seconds=time.monotonic()-START,**value)
    with (root/'events.jsonl').open('a') as f:
        f.write(json.dumps(v,sort_keys=True)+'\n')
    print(json.dumps(v,sort_keys=True),flush=True)


def assert_frozen(root):
    f = json.loads((root/'source_freeze.json').read_text())
    if f['sources'] != source_map() or f['config'] != core.config():
        raise ValueError('scientific source/config changed')
    if time.time() >= f['deadline_unix']:
        raise RuntimeError('cumulative scientific deadline')
    if (root/'input_manifest.json').exists():
        seal = json.loads((root/'input_seal.json').read_text())
        core.checked(root/'input_manifest.json',seal['manifest_sha256'])
        core.checked(root/'source_freeze.json',seal['source_freeze_sha256'])


def bank_inputs(root,name):
    m = json.loads((root/'input_manifest.json').read_text())
    bank = m['banks'][name]
    for key in ('vectors','ids'):
        core.checked(root/bank[key]['path'],bank[key]['sha256'])
    X = np.load(root/bank['vectors']['path'],mmap_mode='r',allow_pickle=False)
    ids = json.loads((root/bank['ids']['path']).read_text())
    if len(X)!=len(ids) or len(ids)!=len(set(ids)):
        raise ValueError('bank ID map')
    return X,ids


def queries(root,role,phase):
    if role not in ('advice','select'):
        raise ValueError('closed role')
    if role=='select':
        required = 'A_selection_seal.json' if phase=='A' else 'B_selection_seal.json'
        if not (root/required).exists():
            raise ValueError('select before decision seal')
        decision_seal=json.loads((root/required).read_text())
        if decision_seal['sources']!=source_map() or decision_seal['input_manifest_sha256']!=core.sha(root/'input_manifest.json'):
            raise ValueError('decision seal source/input mismatch')
        for n,h in decision_seal['sealed_files'].items():
            core.checked(root/n,h)
        seal = json.loads((root/'select_input_seal.json').read_text())
        core.checked(root/'A_selection_seal.json',seal['A_selection_sha256'])
        core.checked(root/'inputs/select_queries.json',seal['sha256'])
    else:
        m = json.loads((root/'input_manifest.json').read_text())
        core.checked(root/'inputs/advice_queries.json',m['advice_sha256'])
    return json.loads((root/'inputs'/f'{role}_queries.json').read_text())


def job_call(root,name,job):
    assert_frozen(root)
    file = root/'jobs'/f'{name}.json'
    core.save(file,job)
    out = root/'workers'/f'{name}.json'
    err = root/'workers'/f'{name}.stderr'
    err.parent.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    with err.open('xb') as f:
        subprocess.run([sys.executable,__file__,'worker','--root',str(root),'--job',str(file),'--output',str(out)],
                       check=True,stdout=f,stderr=subprocess.STDOUT)
    value = json.loads(out.read_text())
    core.save(root/'workers'/f'{name}.orchestration.json',dict(seconds=time.perf_counter()-start,
              includes_interpreter_launch=True,application_work_seconds=value['application_work_seconds']))
    return value


def model_path(root,bank,spec,rep=0):
    return root/'models'/bank/core.base_key(spec)/f'rep{rep}'


def deployment_seed(spec,s):
    return s['production_seed'] if spec['kind']=='gaussian' else (s['pq_seed'] if spec['kind']=='pq' else 0)


def build_worker(root,job,s):
    start = time.perf_counter()
    X,ids = bank_inputs(root,job['bank'])
    recs = queries(root,'advice','A')
    loaded = time.perf_counter()
    spec,rep = job['spec'],job['repeat']
    t = time.perf_counter()
    model = core.build_method(X,ids,spec,s)
    built = time.perf_counter()
    directory = model_path(root,job['bank'],spec,rep)
    core.store_model(model,directory)
    stored = time.perf_counter()
    del model
    before_load = time.perf_counter()
    model = core.load_model(X,ids,directory,spec,deployment_seed(spec,s))
    reloaded = time.perf_counter()
    B = core.budgets(spec,s)
    firstM = B[rep%len(B)]
    first = core.query_method(model,np.asarray(recs[0]['query']),firstM,s['k'])
    deploy_end = time.perf_counter()
    measurements = {str(M):core.measure_queries(model,recs,M,s) for M in B} if rep==0 else {}
    end = time.perf_counter()
    value = dict(spec=spec,repeat=rep,common_input_seconds=loaded-start,
                 build_seconds=built-t,store_seconds=stored-built,load_seconds=reloaded-before_load,
                 teardown_seconds=before_load-stored,deploy_prequery_seconds=reloaded-start,
                 first_M=firstM,first_call=first,first_query_id=recs[0]['query_id'],
                 first_query_outer_seconds=deploy_end-reloaded,
                 advice_measurement_seconds=end-deploy_end,advice=measurements,
                 model_directory=str(directory.relative_to(root)),
                 stored_bytes=sum(p.stat().st_size for p in directory.iterdir()),scan_bytes=model.scan_bytes,
                 maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 cold_ssd_claim=False)
    return value


def family_worker(root,job,s):
    start = time.perf_counter()
    X,ids = bank_inputs(root,job['bank'])
    recs = queries(root,'advice','A')
    G = core.truth(X,ids,recs,s['k'])
    truth_done = time.perf_counter()
    family = job['family']
    x32 = np.ascontiguousarray(X,dtype=np.float32)
    basis = core.family_basis(X,family,s)
    specs = [v for v in core.base_specs(s) if v['kind']==family]
    records,good,models = {},[],{}
    for spec in specs:
        model = core.build_method(X,ids,spec,s,basis=basis,shared32=x32)
        models[core.base_key(spec)] = model
        for M in core.budgets(spec,s):
            measured = core.measure_queries(model,recs,M,s)
            q = core.quality(measured['records'],G,s['k'])
            key = core.recipe_key(spec,M)
            records[key] = dict(spec=spec,M=M,quality=q,mean_ns=measured['mean_ns'],
                                measurements=measured)
            if core.eligible(q,s):
                good.append((measured['mean_ns'],key))
    selected = records[min(good)[1]] if good else None
    # A final delivery is paid; no false assumption that the fitted index is
    # already persisted when the service begins. All choices precede select.
    if selected:
        spec,M = selected['spec'],selected['M']
    else:
        spec,M = dict(kind='original64'),s['k']
    model = models[core.base_key(spec)] if selected else core.build_method(X,ids,spec,s)
    directory = root/'family_models'/job['bank']/family
    core.store_model(model,directory)
    model = core.load_model(X,ids,directory,spec,deployment_seed(spec,s))
    first = core.query_method(model,np.asarray(recs[0]['query']),M,s['k'])
    return dict(family=family,selected=dict(spec=spec,M=M,key=core.recipe_key(spec,M)),
                records=records,model_directory=str(directory.relative_to(root)),
                common_and_truth_seconds=truth_done-start,total_selection_seconds=time.perf_counter()-start,
                first_call=first,fitting_shared_within_family=True,select_opened=False)


def evaluate_worker(root,job,s):
    X,ids = bank_inputs(root,job['bank'])
    recs = queries(root,'select',job['phase'])
    model = core.load_model(X,ids,root/job['model_directory'],job['spec'],deployment_seed(job['spec'],s))
    G = core.truth(X,ids,recs,s['k'])
    B = job['budgets']
    measured = {str(M):core.measure_queries(model,recs,M,s) for M in B}
    result = {str(M):dict(quality=core.quality(v['records'],G,s['k']),measurements=v) for M,v in [(M,measured[str(M)]) for M in B]}
    return dict(spec=job['spec'],G=G,results=result,
                precision=core.precision_probe(model,np.asarray(recs[0]['query']),max(B),s),
                precision_query_id=recs[0]['query_id'],precision_query_count=1)


def confirmation_worker(root,job,s):
    X,ids = bank_inputs(root,job['bank'])
    roles = {r:queries(root,r,'B') for r in ('advice','select')}
    G = {r:core.truth(X,ids,recs,s['k']) for r,recs in roles.items()}
    results = []
    for seed in s['confirmation_seeds']:
        for m in s['gaussian_dimensions']:
            spec = dict(kind='gaussian',m=m)
            model = core.build_method(X,ids,spec,s,seed=seed)
            ideal = core.Model(X,ids,spec)
            ideal.seed=seed
            ideal.P=core.gaussian_projection(X.shape[1],m,seed,maximum=max(s['gaussian_dimensions']))
            ideal.Y=X@ideal.P.T
            for arithmetic,current in [('deployed_fp32',model),('ideal_gaussian_fp64',ideal)]:
                for role,recs in roles.items():
                    measured = core.probe_model(current,recs,s['budgets'],s)
                    for M in s['budgets']:
                        results.append(dict(role=role,seed=seed,spec=spec,M=M,arithmetic=arithmetic,
                                       quality=core.quality(measured[str(M)],G[role],s['k']),records=measured[str(M)]))
    for seed in s['pq_confirmation_seeds']:
        for bits in s['pq_bits']:
            spec = dict(kind='pq',bits=bits)
            model = core.build_method(X,ids,spec,s,seed=seed)
            for role,recs in roles.items():
                measured = core.probe_model(model,recs,s['budgets'],s)
                for M in s['budgets']:
                    results.append(dict(role=role,seed=seed,spec=spec,M=M,arithmetic='deployed_fp32',
                                   quality=core.quality(measured[str(M)],G[role],s['k']),records=measured[str(M)]))
    return dict(results=results,G=G,no_policy_updated=True,formal_certification=False)


def worker(root,jobfile,output):
    assert_frozen(root)
    start = time.perf_counter()
    s = core.config()
    core.check_versions(s)
    job = json.loads(jobfile.read_text())
    mode = job['mode']
    if mode=='build':
        value = build_worker(root,job,s)
    elif mode=='family':
        value = family_worker(root,job,s)
    elif mode=='evaluate':
        value = evaluate_worker(root,job,s)
    elif mode=='advisor':
        X,ids = bank_inputs(root,job['bank'])
        recs = queries(root,'advice','B')
        value = advisor.run_probe_policy(X,ids,recs,s,job['policy'],root/'advisors'/job['bank']/job['policy'])
    elif mode=='confirmation':
        value = confirmation_worker(root,job,s)
    else:
        raise ValueError('unknown worker mode')
    value['application_work_seconds'] = time.perf_counter()-start
    value['maximum_worker_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS ru_maxrss is bytes (this approved run is local macOS); portable readout.
    if sys.platform!='darwin':
        value['maximum_worker_rss_bytes'] *= 1024
    core.save(output,value)
    assert_frozen(root)


def prepare(root,s):
    start = time.perf_counter()
    X,ids,recs = core.load_observed_inputs(s)
    core.save(root/'inputs/advice_queries.json',recs)
    manifest = dict(schema=s['schema'],advice_sha256=core.sha(root/'inputs/advice_queries.json'),banks={})
    for name,rows in core.make_banks(ids,s).items():
        directory = root/'banks'/name
        directory.mkdir(parents=True)
        with (directory/'corpus64.npy').open('xb') as f:
            np.save(f,np.asarray(X[rows]),allow_pickle=False)
        core.save(directory/'ids.json',[ids[j] for j in rows])
        core.save(directory/'parent_rows.json',rows)
        manifest['banks'][name] = {key:dict(path=str((directory/n).relative_to(root)),sha256=core.sha(directory/n))
                                  for key,n in [('vectors','corpus64.npy'),('ids','ids.json')]}
    core.save(root/'input_manifest.json',manifest)
    core.save(root/'input_seal.json',dict(manifest_sha256=core.sha(root/'input_manifest.json'),
              source_freeze_sha256=core.sha(root/'source_freeze.json')))
    event(root,'seven_banks_prepared',banks=list(manifest['banks']),seconds=time.perf_counter()-start,
          query_roles_opened=['advice'],report_opened=False)
    return list(manifest['banks'])


def seal(root,name,entries):
    names=['A_builds.json','A_families.json'] if name.startswith('A_') else ['B_decisions.json']
    value = dict(entries=entries,sources=source_map(),input_manifest_sha256=core.sha(root/'input_manifest.json'),
                 sealed_files={n:core.sha(root/n) for n in names},
                 select_inspected_for_choice=False)
    core.save(root/name,value)
    return core.sha(root/name)


def scenario_A(root,banks,s):
    builds,families,choices = {},{},{}
    specs = core.base_specs(s)
    for name in banks:
        builds[name] = {}
        jobs = [(spec,r) for spec in specs for r in range(s['build_repeats'])]
        rng = np.random.Generator(np.random.PCG64(s['schedule_seed']))
        for i in rng.permutation(len(jobs)):
            spec,rep = jobs[i]
            key = core.base_key(spec)
            v = job_call(root,f'A-build-{name}-{key}-{rep}',dict(mode='build',bank=name,spec=spec,repeat=rep))
            builds[name].setdefault(key,{})[str(rep)] = v
        families[name] = {}
        for family in ['gaussian','pca','pq']:
            v = job_call(root,f'A-family-{name}-{family}',dict(mode='family',bank=name,family=family))
            families[name][family] = v
        choices[name] = {f:v['selected'] for f,v in families[name].items()}
        event(root,'A_advice_bank_complete',bank=name)
    core.save(root/'A_builds.json',builds)
    core.save(root/'A_families.json',families)
    a_seal = seal(root,'A_selection_seal.json',choices)
    _,_,select = core.load_observed_inputs(s,'select',dict(path=str(root/'A_selection_seal.json'),sha256=a_seal))
    advice_ids = {r['query_id'] for r in queries(root,'advice','A')}
    if advice_ids & {r['query_id'] for r in select}:
        raise ValueError('advice/select overlap')
    core.save(root/'inputs/select_queries.json',select)
    core.save(root/'select_input_seal.json',dict(sha256=core.sha(root/'inputs/select_queries.json'),A_selection_sha256=a_seal))
    evaluations,family_evaluations = {},{}
    for name in banks:
        evaluations[name] = {}
        for spec in specs:
            key = core.base_key(spec)
            modeldir = builds[name][key]['0']['model_directory']
            evaluations[name][key] = job_call(root,f'A-eval-{name}-{key}',
                dict(mode='evaluate',phase='A',bank=name,spec=spec,budgets=core.budgets(spec,s),model_directory=modeldir))
        family_evaluations[name]={}
        for family,v in families[name].items():
            chosen=v['selected']
            family_evaluations[name][family]=job_call(root,f'A-family-eval-{name}-{family}',
                dict(mode='evaluate',phase='A',bank=name,spec=chosen['spec'],budgets=[chosen['M']],model_directory=v['model_directory']))
        event(root,'A_select_bank_complete',bank=name)
    core.save(root/'A_evaluations.json',evaluations)
    core.save(root/'A_family_evaluations.json',family_evaluations)
    summary = lifecycle_summary(builds,families,evaluations,family_evaluations,s)
    core.save(root/'A_summary.json',summary)
    event(root,'A_complete',qualified_gaussian_bank_recipes=summary['qualified_gaussian_bank_recipes'])


def lifecycle_summary(builds,families,evaluations,family_evaluations,s):
    table,selected = {},{}
    for bank,allmodels in builds.items():
        table[bank] = {}
        for key,reps in allmodels.items():
            r0 = reps['0']
            pre = float(np.mean([r['deploy_prequery_seconds'] for r in reps.values()]))
            for M in core.budgets(r0['spec'],s):
                ev = evaluations[bank][key]['results'][str(M)]
                cold = next((r['first_query_outer_seconds']*1e9 for r in reps.values() if r['first_M']==M),reps['0']['first_query_outer_seconds']*1e9)
                name = core.recipe_key(r0['spec'],M)
                table[bank][name] = dict(spec=r0['spec'],M=M,quality=ev['quality'],quality_pass=core.eligible(ev['quality'],s),
                    mean_ns=ev['measurements']['mean_ns'],p95_ns=ev['measurements']['p95_ns'],
                    build_prequery_seconds=pre,cold_ns=cold,
                    lifetime_seconds={str(Q):core.lifecycle_seconds(pre,cold,ev['measurements']['mean_ns'],Q) for Q in s['lifetime_queries']})
        selected[bank] = {}
        for family,v in families[bank].items():
            measured=family_evaluations[bank][family]['results'][str(v['selected']['M'])]
            item=dict(quality=measured['quality'],quality_pass=core.eligible(measured['quality'],s),
                      mean_ns=measured['measurements']['mean_ns'])
            selected[bank][family] = dict(selected=v['selected'],quality=item['quality'],quality_pass=item['quality_pass'],
                paid_selection_seconds=v['total_selection_seconds'],
                lifetime_seconds={str(Q):float(v['total_selection_seconds']+Q*item['mean_ns']/1e9) for Q in s['lifetime_queries']})
    winners = {bank:{str(Q):min((v['lifetime_seconds'][str(Q)],key) for key,v in rows.items() if v['quality_pass'])[1]
                     for Q in s['lifetime_queries']} for bank,rows in table.items()}
    count = sum(v['quality_pass'] and v['spec']['kind']=='gaussian' for rows in table.values() for v in rows.values())
    return dict(fixed_recipe_table=table,paid_family_selection=selected,descriptive_expost_winners=winners,
                winners_are_not_deployment_selection=True,qualified_gaussian_bank_recipes=int(count),
                extrapolation_above_observed_query_count=True,embedding_and_LLM_costs_not_measured=True,
                setup_mode='per-bank fresh fitting; no external pretrained codebook',formal_certification=False)


def scenario_B(root,banks,s):
    decisions={}
    for name in banks:
        decisions[name]={}
        for policy in s['advisors']:
            v=job_call(root,f'B-advisor-{name}-{policy}',dict(mode='advisor',bank=name,policy=policy))
            decisions[name][policy]=v
        event(root,'B_advice_bank_complete',bank=name)
    core.save(root/'B_decisions.json',decisions)
    seal(root,'B_selection_seal.json',{n:{p:core.fingerprint(v) for p,v in vv.items()} for n,vv in decisions.items()})
    evaluations,confirmations={},{}
    for name in banks:
        evaluations[name]={}
        for policy,v in decisions[name].items():
            selected=v['selected']
            modeldir=str((root/'advisors'/name/policy/'deployment_model').relative_to(root))
            ev=job_call(root,f'B-eval-{name}-{policy}',dict(mode='evaluate',phase='B',bank=name,
                spec=selected['spec'],budgets=[selected['M']],model_directory=modeldir))
            measures=ev['results'][str(selected['M'])]['measurements']
            evaluations[name][policy]=advisor.evaluate_decisions(v,measures,ev['G'],s)
            evaluations[name][policy]['application_work_seconds']=v['application_work_seconds']
        confirmations[name]=job_call(root,f'B-confirm-{name}',dict(mode='confirmation',bank=name))
        event(root,'B_confirmation_bank_complete',bank=name)
    core.save(root/'B_evaluations.json',evaluations)
    core.save(root/'B_confirmation.json',confirmations)
    core.save(root/'B_summary.json',risk_summary(decisions,evaluations,confirmations,s))
    event(root,'B_complete')


def risk_summary(decisions,evaluations,confirmations,s):
    tables={}
    for bank,policies in decisions.items():
        rows={}
        for policy,v in policies.items():
            states=v['states']
            row=dict(actual_model_builds=sum(not x.get('reused_model',False) for x in v['attempted']),
                logical_probe_actions=sum(len(x['budgets']) for x in v['attempted']),
                states={name:sum(x['state']==name for x in states.values()) for name in ['low_expected_risk','high_expected_risk','unknown']},
                advisor_total_seconds=v['total_seconds'],application_work_seconds=v['application_work_seconds'],
                evaluation=evaluations[bank][policy])
            if states:
                mismatch=[]
                for key,state in states.items():
                    items=[r for r in confirmations[bank]['results'] if r['role']=='advice' and r['arithmetic']=='ideal_gaussian_fp64' and core.recipe_key(r['spec'],r['M'])==key]
                    a=np.asarray([r['quality']['missed'] for r in items])/s['k']
                    rng=np.random.Generator(np.random.PCG64(s['bootstrap_seed']))
                    boot=np.asarray([a[rng.integers(0,len(a),len(a))][:,rng.integers(0,a.shape[1],a.shape[1])].mean()
                                     for _ in range(s['bootstrap_repeats'])])
                    mismatch.append(dict(key=key,**state,observed_seed_means=a.mean(axis=1).tolist(),
                        observed_mean=float(a.mean()),query_seed_descriptive_interval=np.quantile(boot,[.025,.975]).tolist(),
                        seed_individual_passes=sum(core.eligible(r['quality'],s) for r in items)))
                row['risk_comparison']=mismatch
            rows[policy]=row
        tables[bank]=rows
    return dict(banks=tables,expectation_intervals_are_not_fixed_seed_guarantees=True,
                all_negative_accuracy_is_not_value=True,formal_certification=False,fresh_independence=False)


def science(root):
    s=core.config()
    core.check_versions(s)
    assert_frozen(root)
    banks=prepare(root,s)
    scenario_A(root,banks,s)
    # B remains independent and runs even if A has no lifecycle advantage.
    scenario_B(root,banks,s)
    assert_frozen(root)
    core.save(root/'result.json',dict(status='completed',stages=['A','B'],
              elapsed_seconds=time.monotonic()-START,report_opened=False,
              formal_certification=False,fresh_independence=False))


def supervisor(root):
    s=core.config()
    core.check_versions(s)
    descendant_processes(os.getpid())  # real preflight before spawn/directory
    root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    core.save(root/'source_freeze.json',dict(sources=source_map(),config=s,
              git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=core.ROOT,text=True).strip(),
              user_consent='批准第38节全部范围，按A→B执行',proposal_commit='ecd85c7',
              supervisor_pid=os.getpid(),deadline_unix=time.time()+s['limits']['seconds']))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
             MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    maximum_rss=maximum_bytes=0
    reason=None
    proc=None
    if shutil.disk_usage(root).free<s['limits']['free_bytes']:
        reason='insufficient_free_space'
    else:
        with (root/'science.log').open('xb') as log:
            proc=subprocess.Popen([sys.executable,__file__,'science','--root',str(root)],env=env,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            last_disk=-10.
            try:
                while proc.poll() is None:
                    elapsed=time.monotonic()-start
                    rss=sum(v for _,v in descendant_processes(os.getpid()))
                    maximum_rss=max(maximum_rss,rss)
                    if elapsed-last_disk>=5:
                        maximum_bytes=max(maximum_bytes,sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
                        last_disk=elapsed
                    if elapsed>=s['limits']['seconds']:
                        reason='cumulative_wall_limit'
                    elif rss>s['limits']['rss_bytes']:
                        reason='aggregate_RSS_limit'
                    elif maximum_bytes>s['limits']['artifact_bytes']:
                        reason='artifact_bytes_limit'
                    elif shutil.disk_usage(root).free<s['limits']['free_bytes']:
                        reason='free_space_limit'
                    if reason:
                        terminate_owned_tree(proc)
                        break
                    time.sleep(.5)
            except Exception:
                reason='monitor_failure'
                core.save(root/'monitor_failure.json',dict(traceback=traceback.format_exc()))
                try:
                    terminate_owned_tree(proc)
                except Exception:
                    proc.kill()
                    proc.wait(timeout=5)
    state=('execution_failure' if reason=='monitor_failure' else 'resource_stop') if reason else ('completed' if proc.returncode==0 else 'execution_failure')
    core.save(root/'status.json',dict(status=state,reason=reason,exit_code=None if proc is None else proc.returncode,
              supervision_seconds=time.monotonic()-start,maximum_aggregate_rss_bytes=maximum_rss,
              maximum_observed_artifact_bytes=maximum_bytes,formal_certification=False,fresh_independence=False))
    entries={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=core.sha(p)) for p in sorted(root.rglob('*')) if p.is_file()}
    core.save(root/'archive_index.json',dict(files=entries))
    for n,v in entries.items():
        core.checked(root/n,v['sha256'])
    core.save(root/'archive_verified.json',dict(index_sha256=core.sha(root/'archive_index.json'),
              file_count=len(entries),bytes=sum(v['bytes'] for v in entries.values()),verified=True))
    print(json.dumps(dict(status=state,root=str(root),reason=reason)),flush=True)
    if state!='completed':
        raise SystemExit(2)


START=time.monotonic()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['run','science','worker'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--job',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    root=a.root.resolve()
    if a.mode=='run':
        supervisor(root)
    elif a.mode=='science':
        try:
            science(root)
        except Exception:
            core.save(root/'failure.json',dict(traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START))
            raise
    else:
        worker(root,a.job,a.output)


if __name__=='__main__':
    main()
