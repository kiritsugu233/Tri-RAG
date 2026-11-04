"""Stdlib v2 controller: independent GPU telemetry, one heavy worker at a time."""
import time
ENTRY_UNIX = time.time()
ENTRY_MONOTONIC = time.clock_gettime(time.CLOCK_MONOTONIC)
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts import run_tls_rag_step5_route6_representation_v1 as frozen

CONFIG = ROOT/'configs/tls_rag_step5_route6_representation_v2.json'
THREADS = frozen.THREADS
sha = frozen.sha
write_json = frozen.write_json
allocated_resources = frozen.allocated_resources
process_tree = frozen.process_tree
terminate_owned = frozen.terminate_owned
close_index = frozen.close_index
verify_index = frozen.verify_index
SOURCES = list(frozen.SOURCES) + [
    'mechanism/tls_rag_step5_route6_representation_v2.py',
    'mechanism/tls_rag_step5_route6_representation_screen_v2.py',
    'configs/tls_rag_step5_route6_representation_v2.json',
    'tests/test_tls_rag_step5_route6_representation_v2.py',
    'docs/TLS_RAG_STEP5_ROUTE6_REPRESENTATION_PROTOCOL_V2.md',
    'scripts/run_tls_rag_step5_route6_representation_v2.py']


def monotonic():
    """System-wide POSIX clock; Python3.9/mac monotonic has per-process origin."""
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def source_map():
    return {name:sha(ROOT/name) for name in SOURCES}


def config():
    cfg = json.loads(CONFIG.read_text())
    if (cfg['namespace'] != 'tls-rag-route6-representation-screen-v2' or len(cfg['representations']) != 24
            or cfg['queries'] != 128 or cfg['query_dataset'] != 'learn' or cfg['k'] != 100
            or cfg['new_mc_samples'] != 0 or cfg['tri_stage'] or cfg['timing']['cpu_queries_per_role'] != 16
            or cfg['timing']['gpu_queries'] != 128 or cfg['monitor']['gpu_max_age_seconds'] != 3.):
        raise ValueError('frozen v2 scope')
    return cfg


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.pending')
    with temporary.open('w') as stream:
        json.dump(value, stream, allow_nan=False); stream.write('\n')
    os.replace(temporary, path)


def storage_sizes(root):
    size=0
    for path in Path(root).rglob('*'):
        try:info=path.lstat()
        except FileNotFoundError:continue  # reporter atomic rename, not a zero sample
        if stat.S_ISLNK(info.st_mode):raise RuntimeError('run directory symlink')
        if stat.S_ISREG(info.st_mode):size+=info.st_size
    return size,size


def parse_gpu(text):
    rows = {}
    for line in text.splitlines():
        if not line.strip(): continue
        parts = [s.strip() for s in line.split(',')]
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise ValueError('invalid GPU PID/memory telemetry')
        pid, mib = map(int, parts)
        if pid <= 0 or pid in rows: raise ValueError('duplicate/nonpositive GPU PID')
        rows[pid] = mib*2**20
    return rows


def reporter(root, nonce, owner_pid, settings, query_command=None):
    root = Path(root)
    command = query_command or ['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits']
    sequence = 0
    with (root/'telemetry.log').open('x') as log:
        while os.getppid() == owner_pid:
            started = monotonic()
            try:
                result = subprocess.run(command,capture_output=True,text=True,check=True,
                                        timeout=settings['command_seconds'])
                values = parse_gpu(result.stdout)
                sequence += 1
                sample = dict(nonce=nonce, reporter_pid=os.getpid(), owner_pid=owner_pid,
                    sequence=sequence, started_monotonic=started, completed_monotonic=monotonic(),
                    memory_bytes={str(p):v for p,v in values.items()})
                atomic_json(root/'telemetry.json',sample)
                log.write(json.dumps(dict(sequence=sequence,query_seconds=monotonic()-started,valid=True))+'\n')
            except Exception as exc:
                # Retain the last valid sample, never fabricate a zero sample.
                log.write(json.dumps(dict(sequence=sequence,valid=False,error=type(exc).__name__+': '+str(exc)))+'\n')
            log.flush()
            time.sleep(settings['report_seconds'])


def telemetry_memory(root, nonce, reporter_pid, owned, settings, now=None, retired_owned=()):
    value = json.loads((Path(root)/'telemetry.json').read_text())
    now = monotonic() if now is None else now
    if (value['nonce'] != nonce or value['reporter_pid'] != reporter_pid or value['owner_pid'] != os.getpid()
            or not isinstance(value['sequence'],int) or value['sequence'] < 1
            or not 0 <= now-value['started_monotonic'] <= settings['gpu_max_age_seconds']
            or not value['started_monotonic'] <= value['completed_monotonic'] <= now):
        raise RuntimeError('GPU telemetry identity/staleness')
    memory = {}
    for pid, amount in value['memory_bytes'].items():
        if not pid.isdigit() or type(amount) is not int or amount < 0:
            raise RuntimeError('invalid GPU memory sample')
        memory[int(pid)] = amount
    # A fresh sample can still describe our just-reaped worker. Accept only
    # previously observed owned PIDs that no longer exist; live external PIDs
    # (including any reused PID) remain forbidden.
    allowed=set(owned)
    for pid in retired_owned:
        try:os.kill(pid,0)
        except ProcessLookupError:allowed.add(pid)
    if set(memory)-allowed: raise RuntimeError('unrelated GPU compute process appeared')
    return sum(memory.values())


def environment(cfg):
    versions = {name:importlib.metadata.version(name) for name in cfg['dependencies']}
    if any(versions[n].split('+')[0] != v for n,v in cfg['dependencies'].items()):
        raise RuntimeError('dependency version lock')
    rows = subprocess.run(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],
        capture_output=True,text=True,check=True,timeout=cfg['monitor']['command_seconds']).stdout.strip().splitlines()
    if len(rows)!=1 or 'A800' not in rows[0]: raise RuntimeError('one A800 required')
    if any(os.environ.get(n)!='1' for n in THREADS): raise RuntimeError('five BLAS thread limits must be1')
    return dict(versions=versions,physical_identity=rows[0],python=sys.version,executable=sys.executable,
        threads={n:os.environ[n] for n in THREADS},
        package_binaries={n:sha(importlib.util.find_spec(n).origin) for n in ('numpy','scipy','h5py','faiss','torch')})


class ResourceStop(RuntimeError):
    pass


class MonitorFailure(RuntimeError):
    pass


class Guard:
    def __init__(self, root, limits, settings, start=None, use_gpu=True, query_command=None):
        self.root=Path(root);self.limits=limits;self.settings=settings
        self.start=monotonic() if start is None else start
        self.scientific_start=None;self.report=None;self.nonce=secrets.token_hex(16)
        self.peak_rss=self.peak_gpu=0;self.use_gpu=use_gpu;self.query_command=query_command
        self.known_owned=set()
        self.first_sample_deadline=monotonic()+settings['gpu_max_age_seconds']

    def start_reporter(self):
        if not self.use_gpu:return
        command=[sys.executable,str(Path(__file__).resolve()),'reporter','--run-dir',str(self.root),
            '--nonce',self.nonce,'--owner-pid',str(os.getpid())]
        if self.query_command:command+=['--query-command-json',json.dumps(self.query_command)]
        self.report=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        write_json(self.root/'owned_reporter.json',dict(pid=self.report.pid,nonce=self.nonce,parent_pid=os.getpid()))

    def check(self, worker=None, reserve=0):
        now=monotonic()
        if now-self.start >= self.limits['total_seconds']-reserve:return 'total_seconds'
        if self.scientific_start is not None and now-self.scientific_start >= self.limits['scientific_seconds']-reserve:
            return 'scientific_seconds'
        tree=process_tree(os.getpid());self.known_owned.update(tree);rss=sum(tree.values());used=0;waiting=False
        if self.use_gpu:
            if self.report is None or self.report.poll() is not None:raise RuntimeError('GPU reporter exited')
            if not (self.root/'telemetry.json').exists() and now <= self.first_sample_deadline:
                waiting=True
            else:used=telemetry_memory(self.root,self.nonce,self.report.pid,set(tree),self.settings,retired_owned=self.known_owned)
        self.peak_rss=max(self.peak_rss,rss);self.peak_gpu=max(self.peak_gpu,used)
        disk,artifacts=storage_sizes(self.root)
        values=dict(rss_bytes=rss,gpu_bytes=used,disk_bytes=disk,artifact_bytes=artifacts)
        for key,value in values.items():
            if value>self.limits[key]:return key
        if shutil.disk_usage(self.root).free<self.limits['free_bytes']:return 'free_bytes'
        if waiting:return 'waiting_first_gpu_sample'
        if worker is not None and (self.root/'phase.json').exists():
            phase=json.loads((self.root/'phase.json').read_text())
            if phase.get('worker_pid')==worker.pid and phase.get('component_deadline_monotonic') is not None:
                if now>=phase['component_deadline_monotonic']:return 'component_seconds:'+phase['name']
        return None

    def ready(self):
        while True:
            try:result=self.check()
            except Exception as exc:raise MonitorFailure(str(exc)) from exc
            if result is None:return
            if result!='waiting_first_gpu_sample':raise ResourceStop(result)
            time.sleep(min(.05,self.settings['poll_seconds']))

    def close(self):
        if self.report is not None:terminate_owned(self.report)


def run_job(command, root, guard, name, env=None):
    """One owned session. Component stops are explicit; all other stops are fatal."""
    root=Path(root);stage=root/'stages'/name;stage.mkdir(parents=True,exist_ok=False)
    reason=failure=None;started=monotonic();guard.ready()
    with (stage/'worker.log').open('xb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        guard.known_owned.add(child.pid)
        if guard.scientific_start is None:guard.scientific_start=monotonic()
        write_json(stage/'owned_worker.json',dict(pid=child.pid,parent_pid=os.getpid()))
        try:
            while child.poll() is None:
                reason=guard.check(child,reserve=min(60.,guard.limits['total_seconds']/10))
                if reason:break
                time.sleep(guard.settings['poll_seconds'])
        except BaseException as exc:failure=type(exc).__name__+': '+str(exc)
        finally:terminate_owned(child);code=child.wait(timeout=2.)
    status=('monitor_failed' if failure else 'component_stop' if reason and reason.startswith('component_seconds:')
            else 'resource_stop' if reason else 'complete' if code==0 else 'worker_failed')
    result=dict(name=name,status=status,reason=reason,monitor_error=failure,returncode=code,
        seconds=monotonic()-started,worker_pid=child.pid,peak_rss_bytes=guard.peak_rss,peak_gpu_bytes=guard.peak_gpu)
    write_json(stage/'execution.json',result)
    return result


def tiny(cfg,root,torch,check):
    import numpy as np
    from mechanism import tls_rag_step5_route6_representation_v2 as rep
    rng=np.random.Generator(np.random.PCG64(23));x=rng.normal(size=(64,16)).astype(np.float32)
    queries=rng.normal(size=(4,16)).astype(np.float32);pca=rep.fit_pca(x);outputs=[]
    for kind in ('raw','gaussian','pca','hybrid'):
        spec=dict(id=kind,kind=kind,dim=16 if kind=='raw' else 8,seed=cfg['seeds']['gaussian'],retained=4)
        model=rep.fit_representation(spec,x,pca,cfg,check);values=rep.transform_batch(model,x);q=rep.transform_batch(model,queries)
        expected=rep.exact_representation_search(values,q,8,model['metric'])
        reference=rep.exact_representation_search(values,q,8,model['metric'],torch=torch,check=check,block=13)
        np.testing.assert_array_equal(expected,reference)
        cache=rep.prepare_cuda_index(values,model,dict(cfg,block=13),torch,check)
        actual=np.asarray([rep.search_backend(model,z,8,cuda_index=cache,torch=torch) for z in queries])
        np.testing.assert_array_equal(expected,actual)
        if model['metric']=='l2':np.testing.assert_allclose(cache['norms'].cpu().numpy(),np.sum(values*values,axis=1),rtol=2e-7)
        outputs.append(dict(kind=kind,cpu_cuda_exact=True,norm_bytes=cache['norm_bytes'],
            h2d_seconds=cache['h2d_seconds'],norm_prepare_seconds=cache['norm_prepare_seconds']))
        del cache;torch.cuda.empty_cache()
    freeze=json.loads((Path(root)/'source_freeze.json').read_text())
    write_json(Path(root)/'device_receipt.json',dict(status='passed',sources=freeze['sources'],config=cfg,
        environment=freeze['environment'],checks=outputs))


def worker(root,cfg,stage,ident=None,pipeline=None):
    root=Path(root);freeze=json.loads((root/'source_freeze.json').read_text())
    if source_map()!=freeze['sources'] or cfg!=freeze['config'] or os.getppid()!=freeze['parent_pid']:
        raise RuntimeError('worker source/config/owner lock')
    last=[0.]
    def check():
        if os.getppid()!=freeze['parent_pid'] or monotonic()>=freeze['deadline_monotonic']-5:
            raise TimeoutError('parent/global deadline')
        if monotonic()-last[0]>=10:
            if source_map()!=freeze['sources']:raise RuntimeError('source changed during worker')
            last[0]=monotonic()
    def phase(name,component=False):
        check();value=dict(name=name,worker_pid=os.getpid(),time_unix=time.time(),
            component_deadline_monotonic=monotonic()+cfg['limits']['component_seconds'] if component else None)
        atomic_json(root/'phase.json',value);print(json.dumps(value),flush=True)
    try:
        from mechanism.tls_rag_step5_route6_a800_v1 import torch_backend
        from mechanism import tls_rag_step5_route6_representation_screen_v2 as screen
        import faiss
        torch,precision=torch_backend();torch.set_num_threads(1);faiss.omp_set_num_threads(cfg['timing']['faiss_threads'])
        check()
        if stage=='tiny':tiny(cfg,root,torch,check)
        elif stage=='prepare':screen.prepare(cfg,root,torch,check,phase)
        elif stage=='quality':screen.quality(cfg,root,ident,torch,check,phase)
        elif stage=='timing':screen.linear_timing(cfg,root,ident,torch,check,phase)
        elif stage=='ivf':screen.ivf(cfg,root,ident,torch,check,phase)
        elif stage=='oracle-timing':
            if pipeline.split(':')[1].startswith('ivf'):screen.oracle_ivf_timing(cfg,root,pipeline,torch,check,phase)
            else:screen.linear_timing(cfg,root,pipeline.split(':')[0],torch,check,phase,all_budgets=True,
                backend='cuda' if pipeline.endswith(':cuda') else 'cpu14')
        elif stage=='oracle-solve':
            spent=0.;products=screen.Products(root,cfg['limits']['artifact_bytes'],check)
            for p in json.loads((root/'oracle_selection.json').read_text())['pipelines']:
                ident,backend=p.split(':');folder=root/'oracle'/(ident+'-'+('cpu14' if backend=='cpu_flat' else backend))
                if (folder/'timings.json').exists():spent=screen.solve_oracle(cfg,root,p,products,check,phase,spent)
            products.json('oracle_solver_spent.json',dict(seconds=spent))
        elif stage=='select-ivf':
            points=screen.collect_points(root);configs={s['id']:s for s in cfg['representations']}
            screen.Products(root,cfg['limits']['artifact_bytes'],check).json('ivf_selection.json',dict(
                ids=screen.select_ivf_identities(points,configs,cfg),role='geometry_discovery'))
        elif stage=='select-oracle':screen.write_selections(cfg,root,check)
        elif stage=='finalize':screen.finalize(cfg,root,torch,check,phase)
        else:raise ValueError('unknown scientific stage')
        check();return 0
    except BaseException as exc:
        destination=root/'stages'/freeze_stage_name(stage,ident,pipeline)/'failure.json'
        write_json(destination,dict(type=type(exc).__name__,detail=str(exc),traceback=traceback.format_exc(),
            result_is_incomplete=True,automatic_retry=False))
        return 2 if isinstance(exc,TimeoutError) else 1


def freeze_stage_name(stage,ident=None,pipeline=None):
    return stage+('-'+ident if ident else '')+('-'+pipeline.replace(':','-') if pipeline else '')


def close_run(root,result,start,total_seconds):
    """Charge hashing and its independent verification before final metadata."""
    root=Path(root);tick=monotonic()
    def check():
        if monotonic()-start>=total_seconds-1:raise TimeoutError('global deadline during closure')
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            while True:
                check();block=stream.read(8*1024*1024)
                if not block:break
                h.update(block)
        return h.hexdigest()
    try:
        write_json(root/'execution.json',dict(result,sealing_in_progress=True))
        files=[]
        for path in sorted(root.rglob('*')):
            if path.is_symlink():raise ValueError('closed product symlink')
            if path.is_file() and path.name!='delivery_index.json':
                files.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=digest(path)))
        for item in files:
            path=root/item['path']
            if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
                raise ValueError('closed product changed during independent verification')
        result.update(total_seconds=monotonic()-start,closure_hash_and_verify_seconds=monotonic()-tick,
                      all_worker_and_reporter_streams_closed=True,sealing_in_progress=False)
        atomic_json(root/'execution.json',result)
        execution=next(item for item in files if item['path']=='execution.json')
        execution.update(bytes=(root/'execution.json').stat().st_size,sha256=digest(root/'execution.json'))
        check()
        write_json(root/'delivery_index.json',dict(files=files,closed_products=True,
            exclusions=['delivery_index.json itself; no logs remain open'],
            total_seconds_at_final_index=monotonic()-start,primary_products_independently_hashed_twice=True))
    except BaseException as exc:
        result.update(status='resource_stop' if isinstance(exc,TimeoutError) else 'closure_failed',
            closure_error=type(exc).__name__+': '+str(exc),total_seconds=monotonic()-start,
            closed_delivery_verified=False,sealing_in_progress=False)
        atomic_json(root/'execution.json',result)
        write_json(root/'closure_failure.json',dict(detail=str(exc),full_index_not_claimed=True))
    return result


def supervise(root,cfg,receipt=None,device_only=False):
    if sys.platform!='linux':raise RuntimeError('real screen requires Linux node')
    root=Path(root).resolve();root.mkdir(exist_ok=False)
    limits=dict(cfg['limits'])
    if device_only:limits.update(scientific_seconds=120,total_seconds=180)
    guard=Guard(root,limits,cfg['monitor'],start=ENTRY_MONOTONIC);results=[];failure=None;status='preflight_failed'
    childenv=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        guard.start_reporter();guard.ready()
        subprocess.run([sys.executable,str(ROOT/'scripts/check_code_protection.py')],cwd=ROOT,check=True,timeout=30)
        if subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,capture_output=True,text=True,check=True,timeout=5).stdout.strip():
            raise RuntimeError('tracked checkout dirty')
        env=environment(cfg);sources=source_map();allocation=allocated_resources()
        if allocation['cpu_threads']<limits['cpu_threads'] or allocation['memory_bytes']<limits['rss_bytes']:
            raise RuntimeError('cgroup/affinity allocation')
        if telemetry_memory(root,guard.nonce,guard.report.pid,set(process_tree(os.getpid())),cfg['monitor'])!=0:
            raise RuntimeError('GPU busy before science')
        if not device_only:
            saved=json.loads(Path(receipt).read_text())
            if saved['status']!='passed' or saved['sources']!=sources or saved['config']!=cfg or saved['environment']!=env:
                raise RuntimeError('missing/mismatched successful tiny CUDA receipt')
        write_json(root/'source_freeze.json',dict(config=cfg,sources=sources,environment=env,allocation=allocation,
            parent_pid=os.getpid(),global_start_unix=ENTRY_UNIX,global_start_monotonic=ENTRY_MONOTONIC,
            deadline_monotonic=ENTRY_MONOTONIC+limits['total_seconds'],device_receipt_sha256=sha(receipt) if receipt else None,
            commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True,timeout=5).stdout.strip(),
            mode='tiny only' if device_only else 'one approved S0 v2'))
        def job(stage,ident=None,pipeline=None,allow_component=False):
            name=freeze_stage_name(stage,ident,pipeline)
            command=[sys.executable,str(Path(__file__).resolve()),'worker','--run-dir',str(root),'--stage',stage]
            if ident:command+=['--ident',ident]
            if pipeline:command+=['--pipeline',pipeline]
            result=run_job(command,root,guard,name,childenv);results.append(result)
            if result['status']=='complete':return True
            if allow_component and result['status']=='component_stop':return False
            raise RuntimeError('stage stopped: '+json.dumps(result))
        if device_only:job('tiny')
        else:
            job('prepare');order=json.loads((root/'schedule.json').read_text())['configuration_order']
            for ident in order:
                good=job('quality',ident,allow_component=True)
                if ident=='raw640' and not good:raise RuntimeError('raw native truth gate unavailable')
            write_json(root/'quality_stage_closed.json',dict(all_configurations_attempted=order,
                complete=[i for i in order if (root/'representations'/i/'quality_complete.json').exists()],
                no_main_timing_before_quality_stage_closed=True))
            specs={s['id']:s for s in cfg['representations']}
            for ident in order:
                if specs[ident]['diagnostic'] or not (root/'representations'/ident/'quality_complete.json').exists():continue
                job('timing',ident,allow_component=True)
            job('select-ivf')
            for ident in json.loads((root/'ivf_selection.json').read_text())['ids']:
                job('ivf',ident,allow_component=True)
            job('select-oracle')
            for pipeline in json.loads((root/'oracle_selection.json').read_text())['pipelines']:
                job('oracle-timing',pipeline=pipeline,allow_component=True)
            job('oracle-solve');job('finalize')
        status='incomplete' if any(r['status']!='complete' for r in results) else 'complete'
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        if isinstance(exc,ResourceStop):status='resource_stop'
        elif isinstance(exc,MonitorFailure):status='monitor_failed'
        elif results:status=results[-1]['status'] if results[-1]['status']!='complete' else 'worker_failed'
        write_json(root/'controller_failure.json',dict(detail=failure,traceback=traceback.format_exc(),
            science_started=bool(results),automatic_retry=False))
    finally:guard.close()
    disk,artifacts=storage_sizes(root)
    result=dict(status=status,monitor_error=failure,total_seconds=monotonic()-ENTRY_MONOTONIC,
        scientific_elapsed_seconds=0. if guard.scientific_start is None else monotonic()-guard.scientific_start,
        peak_rss_bytes=guard.peak_rss,peak_gpu_bytes=guard.peak_gpu,artifact_bytes=artifacts,disk_bytes=disk,
        stages=results,sampling_and_shutdown_overshoot_possible=True,automatic_retry=False)
    return close_run(root,result,ENTRY_MONOTONIC,limits['total_seconds'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('device-check','run','worker','reporter','verify'))
    parser.add_argument('--run-dir',required=True);parser.add_argument('--device-receipt');parser.add_argument('--stage')
    parser.add_argument('--ident');parser.add_argument('--pipeline');parser.add_argument('--nonce');parser.add_argument('--owner-pid',type=int)
    parser.add_argument('--query-command-json')
    args=parser.parse_args();cfg=config()
    if args.action=='verify':print(verify_index(args.run_dir));return 0
    if args.action=='reporter':
        reporter(args.run_dir,args.nonce,args.owner_pid,cfg['monitor'],json.loads(args.query_command_json) if args.query_command_json else None);return 0
    if args.action=='worker':return worker(args.run_dir,cfg,args.stage,args.ident,args.pipeline)
    if args.action=='run' and not args.device_receipt:parser.error('run requires tiny --device-receipt')
    result=supervise(args.run_dir,cfg,args.device_receipt,args.action=='device-check');print(json.dumps(result))
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
