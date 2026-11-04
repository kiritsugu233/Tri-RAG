"""Stdlib v2 controller: independent GPU telemetry, one heavy worker at a time."""
import time
ENTRY_UNIX = time.time()
ENTRY_MONOTONIC = time.clock_gettime(time.CLOCK_MONOTONIC)
import argparse
import ctypes
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
    if (cfg['namespace'] != 'tls-rag-route6-representation-screen-v2-manual-r2' or len(cfg['representations']) != 24
            or cfg['queries'] != 128 or cfg['query_dataset'] != 'learn' or cfg['k'] != 100
            or cfg['new_mc_samples'] != 0 or cfg['tri_stage'] or cfg['timing']['cpu_queries_per_role'] != 16
            or cfg['timing']['gpu_queries'] != 128):
        raise ValueError('frozen v2 scope')
    if (cfg['monitor'] != MANUAL_MONITOR or cfg['supervision'] != MANUAL_POLICY
            or cfg['device_gate'] != dict(independent_cuda_workers=3,scientific_seconds=None,total_seconds=None)):
        raise ValueError('frozen manual r2 monitor/gate')
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


# Official nvmlProcessInfo_t ABI for nvmlDeviceGetComputeRunningProcesses_v3.
# Pin the installed driver; unsupported/unknown/NA never means zero memory.
MONITOR = dict(poll_seconds=.25,command_seconds=3.,gpu_max_age_seconds=3.,report_seconds=.1,
    backend='persistent-nvml-v3',library_path='/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.82.07',
    library_sha256='008cc04afc61b8317ed4ddccfa2d1bd367ceb6e71334eb352dc846e0ac450340',
    driver_version='580.82.07',nvml_version='13.580.82.07',
    device_uuid='GPU-0c08388b-2980-4723-8c2e-2318f276cd0c',max_processes=1024)


MANUAL_MONITOR=dict(MONITOR,poll_seconds=300.,report_seconds=300.,gpu_max_age_seconds=None,command_seconds=30.)
MANUAL_POLICY=dict(mode='manual',automatic_stop=False,resource_budget_enforced=False,report_interval_seconds=300)


class NVMLProcess(ctypes.Structure):
    _fields_=[('pid',ctypes.c_uint),('usedGpuMemory',ctypes.c_ulonglong),
              ('gpuInstanceId',ctypes.c_uint),('computeInstanceId',ctypes.c_uint)]


def provider_identity(settings):
    return {key:settings[key] for key in ('backend','library_path','library_sha256',
        'driver_version','nvml_version','device_uuid')}


class NVMLProvider:
    def __init__(self,settings):
        self.settings=settings;self.active=False;self.handle=ctypes.c_void_p()
        if (ctypes.sizeof(NVMLProcess)!=24 or ctypes.alignment(NVMLProcess)!=8
                or [getattr(NVMLProcess,n).offset for n,_ in NVMLProcess._fields_]!=[0,8,16,20]):
            raise RuntimeError('unsupported NVML process ABI')
        if sha(settings['library_path'])!=settings['library_sha256']:
            raise RuntimeError('NVML library SHA mismatch')
        self.lib=ctypes.CDLL(settings['library_path'])
        signatures={
            'nvmlInit_v2':[], 'nvmlShutdown':[],
            'nvmlSystemGetDriverVersion':[ctypes.c_void_p,ctypes.c_uint],
            'nvmlSystemGetNVMLVersion':[ctypes.c_void_p,ctypes.c_uint],
            'nvmlDeviceGetCount_v2':[ctypes.POINTER(ctypes.c_uint)],
            'nvmlDeviceGetHandleByIndex_v2':[ctypes.c_uint,ctypes.POINTER(ctypes.c_void_p)],
            'nvmlDeviceGetUUID':[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint],
            'nvmlDeviceGetComputeRunningProcesses_v3':[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(NVMLProcess)]}
        for name,args in signatures.items():
            function=getattr(self.lib,name);function.argtypes=args;function.restype=ctypes.c_int
        self.ok(self.lib.nvmlInit_v2(),'init');self.active=True
        try:
            self.ok(self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle)),'handle')
            self.validate_identity();self.last_hash=monotonic()
        except BaseException:
            self.close();raise

    @staticmethod
    def ok(code,operation):
        if code!=0:raise RuntimeError('NVML '+operation+' return code '+str(code))

    def text(self,function,*args):
        buffer=ctypes.create_string_buffer(96)
        self.ok(function(*args,buffer,len(buffer)),'identity')
        return buffer.value.decode('ascii',errors='strict')

    def validate_identity(self):
        count=ctypes.c_uint()
        self.ok(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count)),'device count')
        if (count.value!=1 or self.text(self.lib.nvmlSystemGetDriverVersion)!=self.settings['driver_version']
                or self.text(self.lib.nvmlSystemGetNVMLVersion)!=self.settings['nvml_version']
                or self.text(self.lib.nvmlDeviceGetUUID,self.handle)!=self.settings['device_uuid']):
            raise RuntimeError('NVML device/version identity mismatch')
        return provider_identity(self.settings)

    def query(self):
        self.validate_identity()
        if monotonic()-self.last_hash>=10:
            if sha(self.settings['library_path'])!=self.settings['library_sha256']:
                raise RuntimeError('NVML library changed')
            self.last_hash=monotonic()
        count=ctypes.c_uint(0);function=self.lib.nvmlDeviceGetComputeRunningProcesses_v3
        code=function(self.handle,ctypes.byref(count),None)
        if code==0 and count.value==0:return {}
        if code!=7:raise RuntimeError('NVML process count return code '+str(code))
        capacity=count.value
        for attempt in range(2):
            if not 0<capacity<=self.settings['max_processes']:raise RuntimeError('NVML process capacity limit')
            capacity=min(self.settings['max_processes'],max(16,capacity+8))
            rows=(NVMLProcess*capacity)();count=ctypes.c_uint(capacity)
            code=function(self.handle,ctypes.byref(count),rows)
            if code==7:
                capacity=max(count.value,capacity+1);continue
            self.ok(code,'process query')
            if count.value>capacity:raise RuntimeError('NVML invalid returned count')
            result={}
            for row in rows[:count.value]:
                if row.pid<=0 or row.pid in result or row.usedGpuMemory==2**64-1:
                    raise RuntimeError('NVML unknown/duplicate PID or NA memory')
                result[row.pid]=int(row.usedGpuMemory)  # bytes, not MiB
            return result
        raise RuntimeError('NVML process capacity changed twice')

    def close(self):
        if self.active:
            self.active=False;self.ok(self.lib.nvmlShutdown(),'shutdown')


def reporter(root, nonce, owner_pid, settings, query_command=None):
    root=Path(root);sequence=0;provider=None;identity=provider_identity(settings)
    # The explicit command adapter is CPU-fixture-only. Production always uses
    # the pinned persistent native provider, with no CLI retry/fallback.
    if query_command:identity=dict(backend='cpu-command-fixture')
    with (root/'telemetry.log').open('x') as log:
        try:
            if not query_command:provider=NVMLProvider(settings)
            while os.getppid()==owner_pid:
                started=monotonic()
                event=dict(sequence=sequence,started_monotonic=started,provider=identity)
                try:
                    if query_command:
                        result=subprocess.run(query_command,capture_output=True,text=True,check=True,
                                              timeout=settings['command_seconds'])
                        values=parse_gpu(result.stdout)
                    else:values=provider.query()
                    completed=monotonic();sequence+=1
                    sample=dict(nonce=nonce,reporter_pid=os.getpid(),owner_pid=owner_pid,provider=identity,
                        sequence=sequence,started_monotonic=started,completed_monotonic=completed,
                        memory_bytes={str(p):v for p,v in values.items()})
                    atomic_json(root/'telemetry.json',sample)
                    event.update(sequence=sequence,completed_monotonic=completed,query_seconds=completed-started,valid=True)
                except Exception as exc:
                    event.update(completed_monotonic=monotonic(),valid=False,error=type(exc).__name__+': '+str(exc))
                    log.write(json.dumps(event)+'\n');log.flush()
                    raise  # fail closed; never reuse a sample after an explicit API error
                log.write(json.dumps(event)+'\n');log.flush()
                time.sleep(max(0.,settings['report_seconds']-(monotonic()-started)))
        except BaseException as exc:
            log.write(json.dumps(dict(valid=False,provider=identity,error=type(exc).__name__+': '+str(exc),
                started_monotonic=monotonic(),completed_monotonic=monotonic(),sequence=sequence))+'\n');log.flush()
            raise
        finally:
            if provider:provider.close()


def telemetry_memory(root, nonce, reporter_pid, owned, settings, now=None, retired_owned=(), expected_provider=None, return_sample=False):
    value = json.loads((Path(root)/'telemetry.json').read_text())
    now = monotonic() if now is None else now
    expected_provider=provider_identity(settings) if expected_provider is None else expected_provider
    if (value.get('provider')!=expected_provider or value['nonce'] != nonce or value['reporter_pid'] != reporter_pid or value['owner_pid'] != os.getpid()
            or type(value['sequence']) is not int or value['sequence'] < 1
            or not 0 <= now-value['started_monotonic'] <= settings['gpu_max_age_seconds']
            or not value['started_monotonic'] <= value['completed_monotonic'] <= now):
        raise RuntimeError('GPU telemetry identity/staleness: '+json.dumps(dict(age_seconds=now-value['started_monotonic'],sequence=value['sequence'],provider=value.get('provider'),expected_provider=expected_provider)))
    memory = {}
    for pid, amount in value['memory_bytes'].items():
        if not pid.isdigit() or int(pid)<=0 or str(int(pid))!=pid or type(amount) is not int or amount < 0:
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
    return (sum(memory.values()),value) if return_sample else sum(memory.values())


def environment(cfg):
    versions = {name:importlib.metadata.version(name) for name in cfg['dependencies']}
    if any(versions[n].split('+')[0] != v for n,v in cfg['dependencies'].items()):
        raise RuntimeError('dependency version lock')
    identity_error=None;physical=None
    try:
        rows=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],
            capture_output=True,text=True,check=True,timeout=cfg['monitor']['command_seconds']).stdout.strip().splitlines()
        if len(rows)!=1 or 'A800' not in rows[0]:raise ValueError('one A800 required')
        physical=rows[0]
    except (subprocess.SubprocessError,OSError) as exc:
        if not manual_policy(cfg):raise
        identity_error=type(exc).__name__+': '+str(exc)
    if any(os.environ.get(n)!='1' for n in THREADS): raise RuntimeError('five BLAS thread limits must be1')
    return dict(versions=versions,physical_identity=physical,gpu_identity_observation_error=identity_error,python=sys.version,executable=sys.executable,
        threads={n:os.environ[n] for n in THREADS},
        package_binaries={n:sha(importlib.util.find_spec(n).origin) for n in ('numpy','scipy','h5py','faiss','torch')})


def native_binary_identity(torch):
    """Bind loaded native extensions as well as distribution/init metadata."""
    import numpy as np
    paths=dict(numpy=np.core._multiarray_umath.__file__,torch=torch._C.__file__)
    paths.update({name:module.__file__ for name,module in sys.modules.copy().items()
        if name.startswith('faiss._swigfaiss') and hasattr(module,'__file__')})
    if not any(name.startswith('faiss.') for name in paths):raise RuntimeError('loaded Faiss native binary missing')
    return {name:sha(path) for name,path in paths.items()}


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
        self.provider=dict(backend='cpu-command-fixture') if query_command else provider_identity(settings)
        self.first_sample_deadline=monotonic()+settings['gpu_max_age_seconds']

    def start_reporter(self):
        if not self.use_gpu:return
        command=[sys.executable,str(Path(__file__).resolve()),'legacy-reporter','--run-dir',str(self.root),
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
            else:used=telemetry_memory(self.root,self.nonce,self.report.pid,set(tree),self.settings,retired_owned=self.known_owned,expected_provider=self.provider)
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
        seconds=monotonic()-started,exited_monotonic=monotonic(),worker_pid=child.pid,peak_rss_bytes=guard.peak_rss,peak_gpu_bytes=guard.peak_gpu)
    write_json(stage/'execution.json',result)
    return result


def tiny(cfg,root,torch,check,stage_name):
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
    native=json.loads((Path(root)/'stages'/stage_name/'binary_identity.json').read_text())
    write_json(Path(root)/'stages'/stage_name/'device_receipt.json',dict(status='passed',worker_pid=os.getpid(),provider=provider_identity(cfg['monitor']),sources=freeze['sources'],config=cfg,
        environment=freeze['environment'],checks=outputs,binary_sha256=native['binary_sha256'],precision=native['precision']))


def worker(root,cfg,stage,ident=None,pipeline=None):
    root=Path(root);freeze=json.loads((root/'source_freeze.json').read_text())
    if source_map()!=freeze['sources'] or cfg!=freeze['config'] or os.getppid()!=freeze['parent_pid']:
        raise RuntimeError('worker source/config/owner lock')
    last=[0.]
    def check():worker_check(freeze,last,cfg)
    def phase(name,component=False):
        check();value=dict(name=name,worker_pid=os.getpid(),time_unix=time.time(),
            component_deadline_monotonic=monotonic()+cfg['limits']['component_seconds'] if component and not manual_policy(cfg) else None)
        atomic_json(root/'phase.json',value);print(json.dumps(value),flush=True)
    try:
        from mechanism.tls_rag_step5_route6_a800_v1 import torch_backend
        from mechanism import tls_rag_step5_route6_representation_screen_v2 as screen
        import faiss
        torch,precision=torch_backend();torch.set_num_threads(1);faiss.omp_set_num_threads(cfg['timing']['faiss_threads'])
        binaries=native_binary_identity(torch)
        if freeze.get('native_binary_sha256') is not None and (binaries!=freeze['native_binary_sha256'] or precision!=freeze['precision']):
            raise RuntimeError('native binary/precision differs from successful tiny receipt')
        write_json(root/'stages'/freeze_stage_name(stage,ident,pipeline)/'binary_identity.json',
                   dict(binary_sha256=binaries,precision=precision))
        check()
        if stage=='tiny':tiny(cfg,root,torch,check,freeze_stage_name(stage,ident,pipeline))
        elif stage=='prepare':screen.prepare(cfg,root,torch,check,phase)
        elif stage=='quality':screen.quality(cfg,root,ident,torch,check,phase)
        elif stage=='timing':screen.linear_timing(cfg,root,ident,torch,check,phase)
        elif stage=='ivf':screen.ivf(cfg,root,ident,torch,check,phase)
        elif stage=='oracle-timing':
            if pipeline.split(':')[1].startswith('ivf'):screen.oracle_ivf_timing(cfg,root,pipeline,torch,check,phase)
            else:screen.linear_timing(cfg,root,pipeline.split(':')[0],torch,check,phase,all_budgets=True,
                backend='cuda' if pipeline.endswith(':cuda') else 'cpu14')
        elif stage=='oracle-solve':
            spent=0.;products=screen.products_for(root,cfg,check)
            for p in json.loads((root/'oracle_selection.json').read_text())['pipelines']:
                ident,backend=p.split(':');folder=root/'oracle'/(ident+'-'+('cpu14' if backend=='cpu_flat' else backend))
                if (folder/'timings.json').exists():spent=screen.solve_oracle(cfg,root,p,products,check,phase,spent)
            products.json('oracle_solver_spent.json',dict(seconds=spent))
        elif stage=='select-ivf':
            points=screen.collect_points(root);configs={s['id']:s for s in cfg['representations']}
            screen.products_for(root,cfg,check).json('ivf_selection.json',dict(
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


def await_gpu_idle(root,guard,exited):
    deadline=monotonic()+guard.settings['gpu_max_age_seconds']
    while True:
        guard.ready()
        memory,sample=telemetry_memory(root,guard.nonce,guard.report.pid,set(process_tree(os.getpid())),
            guard.settings,retired_owned=guard.known_owned,return_sample=True)
        if memory==0 and sample['started_monotonic']>=exited:return
        if monotonic()>=deadline:raise MonitorFailure('no post-exit empty GPU sample')
        time.sleep(guard.settings['poll_seconds'])


def seal_device_gate(root,cfg,results,provider):
    """All three exited startups, never the best one, authorize the real run."""
    root=Path(root);freeze=json.loads((root/'source_freeze.json').read_text());cases=[];pids=set()
    if len(results)!=cfg['device_gate']['independent_cuda_workers']:raise RuntimeError('three startups required')
    for number,result in enumerate(results,1):
        name='tiny-startup'+str(number)
        if result['name']!=name or result['status']!='complete':raise RuntimeError('startup gate incomplete')
        receipt=json.loads((root/'stages'/name/'device_receipt.json').read_text())
        pid=receipt['worker_pid']
        if pid in pids or pid!=result['worker_pid']:raise RuntimeError('independent startup PID mismatch')
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise RuntimeError('startup worker still alive')
        pids.add(pid)
        if (receipt['status']!='passed' or receipt['sources']!=freeze['sources'] or receipt['config']!=cfg
                or receipt['environment']!=freeze['environment'] or receipt['provider']!=provider
                or [c['kind'] for c in receipt['checks']]!=['raw','gaussian','pca','hybrid']
                or not all(c['cpu_cuda_exact'] for c in receipt['checks'])):
            raise RuntimeError('startup receipt identity/check mismatch')
        if cases and any(receipt[k]!=cases[0][k] for k in ('binary_sha256','precision','provider')):
            raise RuntimeError('startup native/precision/provider changed')
        cases.append(receipt)
    saved=dict(cases[0],independent_cuda_workers=len(cases),worker_pids=sorted(pids),
        startup_receipt_sha256=[sha(root/'stages'/r['name']/'device_receipt.json') for r in results])
    saved.pop('worker_pid');write_json(root/'device_receipt.json',saved)
    return saved


def close_run(root,result,start,total_seconds):
    """Charge hashing and its independent verification before final metadata."""
    root=Path(root);tick=monotonic()
    def check():
        if total_seconds is not None and monotonic()-start>=total_seconds-1:raise TimeoutError('global deadline during closure')
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


class UserStopped(BaseException):
    """Only an explicit operator signal requests worker termination."""


def manual_policy(cfg):
    return cfg.get('supervision',{}).get('mode')=='manual'


def next_observation(started,completed,interval):
    # Missed slots never cause a burst of catch-up probes.
    return started+interval if started+interval>completed else completed+interval


def process_identity(pid):
    """Linux start ticks reject PID reuse; portable CPU fixtures also bind argv."""
    if sys.platform=='linux':
        folder=Path('/proc')/str(pid);tail=(folder/'stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=pid,start_identity=tail[19],process_group=int(tail[2]),session=int(tail[3]),
            uid=folder.stat().st_uid,commandline=(folder/'cmdline').read_bytes().decode().split('\0')[:-1])
    output=subprocess.check_output(['ps','-p',str(pid),'-o','pgid=,lstart=,command='],text=True).strip()
    if not output:raise ProcessLookupError(pid)
    fields=output.split(None,6)
    return dict(pid=pid,process_group=int(fields[0]),start_identity=' '.join(fields[1:6]),
        commandline=fields[6],uid=os.getuid())


def install_manual_control(root):
    root=Path(root).resolve();identity=process_identity(os.getpid())
    control=dict(run_dir=str(root),controller=identity,nonce=secrets.token_hex(16),signal='SIGUSR1')
    def stopped(signum,frame):raise UserStopped('operator signal '+signal.Signals(signum).name)
    previous={sig:signal.signal(sig,stopped) for sig in (signal.SIGUSR1,signal.SIGINT,signal.SIGTERM)}
    try:write_json(root/'manual_control.json',dict(control,state='running'))
    except BaseException:
        for sig,handler in previous.items():signal.signal(sig,handler)
        raise
    return previous


def request_stop(root):
    root=Path(root).resolve()
    if (root/'delivery_index.json').exists():raise ValueError('run is already closed')
    path=root/'manual_control.json'
    if path.is_symlink():raise ValueError('control symlink')
    control=json.loads(path.read_text());saved=control['controller']
    current=process_identity(saved['pid'])
    argv=current['commandline']
    args=argv if isinstance(argv,list) else argv.split()
    bound=any(arg.startswith('/') and Path(arg).resolve()==root for arg in args)
    if control.get('state')!='running':raise ValueError('compute finished; product index is closing')
    if (control['run_dir']!=str(root) or current!=saved or current['uid']!=os.getuid() or not bound):
        raise ValueError('controller PID/start/argv/owner mismatch')
    request=dict(controller_pid=saved['pid'],nonce=control['nonce'],time_unix=time.time(),
                 action='explicit_user_stop',run_dir=str(root))
    write_json(root/'manual_stop_request.json',request)
    os.kill(saved['pid'],signal.SIGUSR1)
    return request


def manual_reporter(root,nonce,owner_pid,settings,query_command=None):
    """One observation per300s; a slow/failed probe never controls science."""
    root=Path(root);provider=None;sequence=0
    identity=dict(backend='cpu-command-fixture') if query_command else provider_identity(settings)
    with (root/'telemetry.log').open('x') as log:
        try:
            while os.getppid()==owner_pid:
                sequence+=1;started=monotonic()
                event=dict(sequence=sequence,started_monotonic=started,provider=identity,
                    nonce=nonce,owner_pid=owner_pid,reporter_pid=os.getpid(),status='pending')
                atomic_json(root/'telemetry_status.json',event)
                try:
                    if query_command:
                        value=subprocess.run(query_command,capture_output=True,text=True,check=True,
                            timeout=settings['command_seconds'])
                        values=parse_gpu(value.stdout)
                    else:
                        if provider is None:provider=NVMLProvider(settings)
                        values=provider.query()
                    completed=monotonic()
                    sample=dict(event,status='ok',completed_monotonic=completed,
                        memory_bytes={str(p):v for p,v in values.items()})
                    atomic_json(root/'telemetry.json',sample)
                    event.update(status='ok',valid=True,completed_monotonic=completed,query_seconds=completed-started)
                except Exception as exc:
                    event.update(status='error',valid=False,completed_monotonic=monotonic(),
                        error=type(exc).__name__+': '+str(exc))
                atomic_json(root/'telemetry_status.json',event)
                log.write(json.dumps(event)+'\n');log.flush()
                due=next_observation(started,monotonic(),settings['report_seconds'])
                time.sleep(max(0.,due-monotonic()))
        finally:
            if provider:provider.close()


def manual_gpu_observation(root,nonce,reporter_pid,owned,settings,reporter_alive,retired_owned=()):
    result=dict(status='unknown',sampled_bytes=None,last_known_bytes=None,sample_age_seconds=None,
                reporter_alive=reporter_alive,unrelated_compute_pids=[])
    try:
        value=json.loads((Path(root)/'telemetry.json').read_text());now=monotonic()
        expected=dict(backend='cpu-command-fixture') if settings.get('fixture_command') else provider_identity(settings)
        if (value['nonce']!=nonce or value['owner_pid']!=os.getpid() or value['reporter_pid']!=reporter_pid
                or value['provider']!=expected or type(value['sequence']) is not int or value['sequence']<1
                or not 0<=value['started_monotonic']<=value['completed_monotonic']<=now):
            raise ValueError('telemetry identity/timestamp')
        memory={}
        for pid,amount in value['memory_bytes'].items():
            if not pid.isdigit() or int(pid)<=0 or str(int(pid))!=pid or type(amount) is not int or amount<0:
                raise ValueError('unknown/malformed GPU memory')
            memory[int(pid)]=amount
        allowed=set(owned)
        for pid in retired_owned:
            try:os.kill(pid,0)
            except ProcessLookupError:allowed.add(pid)
        result.update(last_known_bytes=sum(memory.values()),sample_age_seconds=now-value['started_monotonic'],
            unrelated_compute_pids=sorted(set(memory)-allowed),sample_sequence=value['sequence'],process_memory_bytes=memory,
            last_known_owned_bytes=sum(v for p,v in memory.items() if p in allowed))
        latest=json.loads((Path(root)/'telemetry_status.json').read_text())
        if (latest.get('nonce')!=nonce or latest.get('owner_pid')!=os.getpid()
                or latest.get('reporter_pid')!=reporter_pid or latest.get('provider')!=expected):
            raise ValueError('reporter status identity')
        result['latest_query_status']=latest['status']
        if reporter_alive and latest['status']=='ok' and latest['sequence']==value['sequence']:
            result.update(status='sampled',sampled_bytes=sum(memory.values()))
        else:result['detail']=latest.get('error','pending query or exited reporter')
    except Exception as exc:result['detail']=type(exc).__name__+': '+str(exc)
    return result


def launch_owned(command,publish,**kwargs):
    # Publish the Popen handle before delivering a stop signal received during
    # fork/exec. The caller can then always reap its own session in finally.
    previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGUSR1,signal.SIGINT,signal.SIGTERM})
    try:
        child=subprocess.Popen(command,start_new_session=True,
            preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,previous),**kwargs)
        publish(child)
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous)
    return child


class ManualSupervisor:
    """Observe on schedule; neither time nor resource values terminate work."""
    def __init__(self,root,limits,settings,start=None,use_gpu=True,query_command=None):
        self.root=Path(root);self.limits=limits;self.settings=settings
        self.start=monotonic() if start is None else start;self.scientific_start=None
        self.nonce=secrets.token_hex(16);self.report=None;self.use_gpu=use_gpu;self.query_command=query_command
        self.provider=dict(backend='cpu-command-fixture') if query_command else provider_identity(settings)
        self.next_sample=self.start;self.last_observation=None;self.active=None
        self.sampled_max_rss=None;self.sampled_max_gpu=None;self.known_owned=set()

    def start_reporter(self):
        if not self.use_gpu:return
        command=[sys.executable,str(Path(__file__).resolve()),'reporter','--run-dir',str(self.root),
            '--nonce',self.nonce,'--owner-pid',str(os.getpid())]
        if self.query_command:command+=['--query-command-json',json.dumps(self.query_command)]
        launch_owned(command,lambda child:setattr(self,'report',child),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        write_json(self.root/'owned_reporter.json',dict(pid=self.report.pid,nonce=self.nonce,parent_pid=os.getpid()))

    def observe(self):
        started=monotonic()
        if started<self.next_sample:return self.last_observation
        snapshot=dict(time_unix=time.time(),started_monotonic=started,elapsed_seconds=started-self.start,
            sampling_interval_seconds=self.settings['report_seconds'],active_stage=self.active,
            rss_bytes=None,disk_bytes=None,artifact_bytes=None,free_bytes=None,advisories=[],errors={},
            automatic_stop=False,continuous_resource_compliance_claim=False,reference_thresholds=self.limits)
        owned=set()
        try:
            tree=process_tree(os.getpid());owned=set(tree);self.known_owned.update(owned);snapshot['rss_bytes']=sum(tree.values())
            self.sampled_max_rss=max(self.sampled_max_rss or 0,snapshot['rss_bytes'])
        except Exception as exc:snapshot['errors']['rss']=str(exc)
        if self.use_gpu:
            snapshot['gpu']=manual_gpu_observation(self.root,self.nonce,self.report.pid if self.report else None,
                owned,dict(self.settings,fixture_command=bool(self.query_command)),
                self.report is not None and self.report.poll() is None,retired_owned=self.known_owned)
            amount=snapshot['gpu']['sampled_bytes']
            if amount is not None:self.sampled_max_gpu=max(self.sampled_max_gpu or 0,amount)
            if snapshot['gpu']['status']!='sampled':snapshot['advisories'].append('GPU observation unknown; user decision')
            if snapshot['gpu']['unrelated_compute_pids']:snapshot['advisories'].append('unrelated GPU compute observed; user decision')
        else:snapshot['gpu']=dict(status='not_requested',sampled_bytes=None)
        try:snapshot['disk_bytes'],snapshot['artifact_bytes']=storage_sizes(self.root)
        except Exception as exc:snapshot['errors']['storage']=str(exc)
        try:snapshot['free_bytes']=shutil.disk_usage(self.root).free
        except Exception as exc:snapshot['errors']['free_bytes']=str(exc)
        for key in ('rss_bytes','disk_bytes','artifact_bytes','free_bytes'):
            value=snapshot[key]
            if value is not None and (value<self.limits[key] if key=='free_bytes' else value>self.limits[key]):
                snapshot['advisories'].append(key+' crossed reference threshold; user decision')
        gpu=snapshot['gpu']['sampled_bytes']
        if gpu is not None and gpu>self.limits['gpu_bytes']:snapshot['advisories'].append('gpu_bytes crossed reference threshold; user decision')
        phase=self.root/'phase.json'
        if phase.exists():
            try:
                value=json.loads(phase.read_text())
                snapshot['phase']=value if self.active and value.get('worker_pid')==self.active['worker_pid'] else None
            except Exception as exc:snapshot['errors']['phase']=str(exc)
        snapshot['completed_quality']=[p.parent.name for p in (self.root/'representations').glob('*/quality_complete.json')]
        snapshot['completed_stages']=len(list((self.root/'stages').glob('*/execution.json')))
        try:
            atomic_json(self.root/'observation.json',snapshot)
            with (self.root/'observations.jsonl').open('a') as log:log.write(json.dumps(snapshot)+'\n')
        except Exception as exc:print('observation write failed: '+str(exc),file=sys.stderr,flush=True)
        self.last_observation=snapshot
        self.next_sample=next_observation(started,monotonic(),self.settings['report_seconds'])
        return snapshot

    def safe_observe(self):
        try:return self.observe()
        except Exception as exc:
            # A monitor implementation/IO failure is advisory, never a worker kill.
            self.next_sample=monotonic()+self.settings['report_seconds']
            print('observation unavailable: '+str(exc),file=sys.stderr,flush=True)
            return None

    def close(self):
        if self.report is not None:terminate_owned(self.report)


def manual_run_job(command,root,guard,name,env=None):
    root=Path(root);stage=root/'stages'/name;stage.mkdir(parents=True,exist_ok=False)
    started=monotonic();failure=None;status=None;owned=[None];code=None
    with (stage/'worker.log').open('xb') as log:
        try:
            launch_owned(command,lambda child:owned.__setitem__(0,child),stdout=log,stderr=subprocess.STDOUT,env=env)
            child=owned[0];guard.known_owned.add(child.pid);guard.active=dict(name=name,worker_pid=child.pid)
            atomic_json(root/'progress.json',dict(active_stage=guard.active,time_unix=time.time()))
            if guard.scientific_start is None:guard.scientific_start=monotonic()
            write_json(stage/'owned_worker.json',dict(pid=child.pid,parent_pid=os.getpid()))
            while True:
                guard.safe_observe()
                try:
                    child.wait(timeout=max(.01,guard.next_sample-monotonic()))
                    break
                except subprocess.TimeoutExpired:continue  # reporting cadence, never termination
        except UserStopped as exc:status='user_stopped';failure=str(exc)
        except BaseException as exc:status='controller_failed';failure=type(exc).__name__+': '+str(exc)
        finally:
            child=owned[0]
            if child is not None:
                if child.poll() is None:terminate_owned(child)
                code=child.wait(timeout=2.)
    result=dict(name=name,status=status or ('complete' if code==0 else 'worker_failed'),reason=failure,
        returncode=code,seconds=monotonic()-started,worker_pid=child.pid if child else None,exited_monotonic=monotonic(),
        automatic_stop=False)
    write_json(stage/'execution.json',result);guard.active=None
    atomic_json(root/'progress.json',dict(active_stage=None,last_stage=result,time_unix=time.time()))
    return result


def worker_check(freeze,last,cfg):
    if os.getppid()!=freeze['parent_pid']:raise RuntimeError('worker parent identity lost')
    if not manual_policy(cfg) and monotonic()>=freeze['deadline_monotonic']-5:
        raise TimeoutError('parent/global deadline')
    interval=cfg['monitor']['report_seconds'] if manual_policy(cfg) else 10.
    if monotonic()-last[0]>=interval:
        if source_map()!=freeze['sources']:raise RuntimeError('source changed during worker')
        last[0]=monotonic()


def status_snapshot(root):
    """Read saved progress only; never launch an extra NVML/resource probe."""
    root=Path(root);result=dict(run_dir=str(root.resolve()),read_only=True)
    for name in ('observation.json','progress.json','execution.json'):
        path=root/name
        if path.exists():result[name]=json.loads(path.read_text())
    value=result.get('observation.json')
    if value:
        age=max(0.,time.time()-value['time_unix']);result['snapshot_age_seconds']=age
        gpu_age=value.get('gpu',{}).get('sample_age_seconds')
        result['last_gpu_sample_age_seconds_now']=None if gpu_age is None else age+gpu_age
    result['completed_quality']=[p.parent.name for p in (root/'representations').glob('*/quality_complete.json')]
    active=result.get('progress.json',{}).get('active_stage')
    phase=root/'phase.json'
    if phase.exists():
        current=json.loads(phase.read_text())
        result['current_phase']=current if active and current.get('worker_pid')==active['worker_pid'] else None
    return result


def same_environment(saved,current):
    # Telemetry availability is not a scientific identity. Known identities must
    # still agree; dependency/native/precision/A800 checks remain independent.
    ignored={'physical_identity','gpu_identity_observation_error'}
    if {k:v for k,v in saved.items() if k not in ignored}!={k:v for k,v in current.items() if k not in ignored}:return False
    return not (saved.get('physical_identity') and current.get('physical_identity')
        and saved['physical_identity']!=current['physical_identity'])


def supervise(root,cfg,receipt=None,device_only=False):
    if sys.platform!='linux':raise RuntimeError('real screen requires Linux node')
    root=Path(root).resolve();root.mkdir(exist_ok=False)
    limits=dict(cfg['limits'])
    guard=ManualSupervisor(root,limits,cfg['monitor'],start=ENTRY_MONOTONIC);results=[];failure=None;status='preflight_failed'
    previous=install_manual_control(root)
    childenv=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        guard.start_reporter();guard.safe_observe()
        subprocess.run([sys.executable,str(ROOT/'scripts/check_code_protection.py')],cwd=ROOT,check=True)
        if subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip():
            raise RuntimeError('tracked checkout dirty')
        env=environment(cfg);sources=source_map()
        try:allocation=allocated_resources()
        except Exception as exc:allocation=dict(observation_error=str(exc))
        if not device_only:
            saved=json.loads(Path(receipt).read_text())
            if saved['status']!='passed' or saved['sources']!=sources or saved['config']!=cfg or not same_environment(saved['environment'],env) or saved.get('provider')!=guard.provider or saved.get('independent_cuda_workers')!=3 or len(saved.get('startup_receipt_sha256',[]))!=3 or len(set(saved.get('worker_pids',[])))!=3:
                raise RuntimeError('missing/mismatched successful tiny CUDA receipt')
        write_json(root/'source_freeze.json',dict(config=cfg,sources=sources,environment=env,allocation=allocation,
            parent_pid=os.getpid(),global_start_unix=ENTRY_UNIX,global_start_monotonic=ENTRY_MONOTONIC,
            deadline_monotonic=None,device_receipt_sha256=sha(receipt) if receipt else None,
            native_binary_sha256=saved['binary_sha256'] if not device_only else None,
            precision=saved['precision'] if not device_only else None,
            commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip(),
            mode='manual r2 tiny' if device_only else 'manual r2 S0',automatic_stop=False))
        def job(stage,ident=None,pipeline=None,allow_component=False):
            name=freeze_stage_name(stage,ident,pipeline)
            command=[sys.executable,str(Path(__file__).resolve()),'worker','--run-dir',str(root),'--stage',stage]
            if ident:command+=['--ident',ident]
            if pipeline:command+=['--pipeline',pipeline]
            result=manual_run_job(command,root,guard,name,childenv);results.append(result)
            if result['status']=='complete':return True
            if result['status']=='user_stopped':raise UserStopped(result['reason'])
            raise RuntimeError('stage stopped: '+json.dumps(result))
        if device_only:
            for number in range(1,4):
                job('tiny',ident='startup'+str(number))
                # Worker exit and all numeric checks matter; no telemetry-age/idle deadline gate.
            seal_device_gate(root,cfg,results,guard.provider)
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
        if isinstance(exc,UserStopped):status='user_stopped'
        elif results:status=results[-1]['status'] if results[-1]['status']!='complete' else 'worker_failed'
        write_json(root/('user_stop.json' if isinstance(exc,UserStopped) else 'controller_failure.json'),dict(detail=failure,traceback=traceback.format_exc(),
            science_started=bool(results),automatic_retry=False))
    finally:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        control=json.loads((root/'manual_control.json').read_text());control['state']='closing'
        atomic_json(root/'manual_control.json',control)
        guard.close()
    disk,artifacts=storage_sizes(root)
    result=dict(status=status,monitor_error=failure,total_seconds=monotonic()-ENTRY_MONOTONIC,
        scientific_elapsed_seconds=0. if guard.scientific_start is None else monotonic()-guard.scientific_start,
        sampled_max_rss_bytes=guard.sampled_max_rss,sampled_max_gpu_bytes=guard.sampled_max_gpu,artifact_bytes=artifacts,disk_bytes=disk,
        stages=results,automatic_stop=False,report_interval_seconds=300,continuous_resource_compliance_claim=False,automatic_retry=False)
    try:return close_run(root,result,ENTRY_MONOTONIC,None)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('device-check','run','worker','reporter','legacy-reporter','verify','status','stop'))
    parser.add_argument('--run-dir',required=True);parser.add_argument('--device-receipt');parser.add_argument('--stage')
    parser.add_argument('--ident');parser.add_argument('--pipeline');parser.add_argument('--nonce');parser.add_argument('--owner-pid',type=int)
    parser.add_argument('--query-command-json')
    args=parser.parse_args()
    if args.action=='status':print(json.dumps(status_snapshot(args.run_dir)));return 0
    if args.action=='stop':print(json.dumps(request_stop(args.run_dir)));return 0
    cfg=config()
    if args.action=='verify':print(verify_index(args.run_dir));return 0
    if args.action in ('reporter','legacy-reporter'):
        report=manual_reporter if args.action=='reporter' else reporter
        report(args.run_dir,args.nonce,args.owner_pid,cfg['monitor'] if args.action=='reporter' else MONITOR,json.loads(args.query_command_json) if args.query_command_json else None);return 0
    if args.action=='worker':return worker(args.run_dir,cfg,args.stage,args.ident,args.pipeline)
    if args.action=='run' and not args.device_receipt:parser.error('run requires tiny --device-receipt')
    result=supervise(args.run_dir,cfg,args.device_receipt,args.action=='device-check');print(json.dumps(result))
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
