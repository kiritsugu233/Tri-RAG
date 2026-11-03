"""Stdlib-only parent for the one approved R6-S0, with owned-tree supervision."""
import time
ENTRY_UNIX=time.time()
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
CONFIG=ROOT/'configs/tls_rag_step5_route6_representation_v1.json'
THREADS=('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
SOURCES=['mechanism/tls_rag_step5_route6_representation_v1.py',
 'mechanism/tls_rag_step5_route6_representation_screen_v1.py',
 'configs/tls_rag_step5_route6_representation_v1.json',
 'tests/test_tls_rag_step5_route6_representation_v1.py',
 'docs/TLS_RAG_STEP5_ROUTE6_REPRESENTATION_PROTOCOL_V1.md',
 'scripts/run_tls_rag_step5_route6_representation_v1.py',
 'mechanism/tls_rag_step5_route6_a800_v1.py',
 'mechanism/tls_rag_step5_route6_real_audit_v1.py',
 'mechanism/tls_rag_step5_route6_real_geometry_v1.py',
 'mechanism/tls_rag_step5_route6_bridge_v1.py','src/tri_rag_harness/tri_law.py',
 'scripts/check_code_protection.py','scripts/run_tests.sh','docs/code_protection.json']


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):digest.update(data)
    return digest.hexdigest()


def write_json(path,value):
    data=(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
    with Path(path).open('xb') as stream:stream.write(data)


def source_map():return {name:sha(ROOT/name) for name in SOURCES}


def config():
    value=json.loads(CONFIG.read_text())
    if (value['namespace']!='tls-rag-route6-representation-screen-v1' or len(value['representations'])!=24
        or value['query_dataset']!='learn' or value['queries']!=128 or value['k']!=100
        or value['new_mc_samples']!=0 or value['tri_stage']):raise ValueError('frozen S0 scope')
    return value


def allocated_resources(cgroup_root='/sys/fs/cgroup'):
    """Read either cgroup generation without importing numerical libraries."""
    root=Path(cgroup_root)
    memory_path=root/'memory.max'
    if not memory_path.is_file():memory_path=root/'memory/memory.limit_in_bytes'
    memory=memory_path.read_text().strip()
    if not memory.isdigit() or int(memory)<=0:
        raise RuntimeError('finite positive cgroup memory required')
    if (root/'cpu.max').is_file():
        q,p=(root/'cpu.max').read_text().split()
        if q=='max':raise RuntimeError('finite positive cgroup CPU quota required')
    else:
        q=(root/'cpu/cpu.cfs_quota_us').read_text().strip()
        p=(root/'cpu/cpu.cfs_period_us').read_text().strip()
    if not q.isdigit() or not p.isdigit() or int(q)<=0 or int(p)<=0:
        raise RuntimeError('finite positive cgroup CPU allocation required')
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    return dict(memory_bytes=int(memory),cpu_threads=min(int(q)/int(p),affinity))


def process_tree(pid):
    """Only descendants of this parent; no name-based killing or unrelated RSS."""
    if sys.platform=='linux':
        inventory={}
        for item in Path('/proc').iterdir():
            if not item.name.isdigit():continue
            try:
                stat=(item/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
                inventory[int(item.name)]=(int(tail[1]),int(tail[21])*os.sysconf('SC_PAGE_SIZE'))
            except (FileNotFoundError,ProcessLookupError):continue
    else:
        result=subprocess.run(['ps','-axo','pid=,ppid=,rss='],capture_output=True,text=True,check=True,timeout=.75)
        inventory={int(p):(int(pp),int(r)*1024) for p,pp,r in (line.split() for line in result.stdout.splitlines())}
    owned={pid}
    while True:
        extra={p for p,(parent,_) in inventory.items() if parent in owned}
        new=owned|extra
        if new==owned:break
        owned=new
    if pid not in inventory:raise RuntimeError('process monitor lost parent')
    return {p:inventory[p][1] for p in owned if p in inventory}


def gpu_memory(owned):
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=.75)
    memory=0
    for line in result.stdout.splitlines():
        if not line.strip():continue
        pid,amount=[s.strip() for s in line.split(',')]
        if int(pid) not in owned:raise RuntimeError('unrelated GPU compute process appeared')
        memory+=int(amount)*2**20
    return memory


def storage_sizes(root):
    root=Path(root)
    if any(p.is_symlink() for p in root.rglob('*')):raise RuntimeError('run directory symlink')
    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    return size,size


def terminate_owned(process):
    if process.poll() is not None:return
    # A new session is created by monitor_worker; this PGID belongs to its child.
    if os.getpgid(process.pid)!=process.pid:raise RuntimeError('owned worker lost session')
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=1.)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=1.)


def close_index(root):
    root=Path(root)
    files=[]
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name!='delivery_index.json':
            files.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=sha(path)))
    write_json(root/'delivery_index.json',dict(files=files,closed_products=True,
        exclusions=['delivery_index.json itself; no logs remain open']))
    return files


def verify_index(root):
    root=Path(root);value=json.loads((root/'delivery_index.json').read_text())
    expected={r['path'] for r in value['files']}
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='delivery_index.json'}
    if actual!=expected or not value['closed_products']:raise ValueError('closed inventory mismatch')
    for item in value['files']:
        path=root/item['path']
        if root.resolve() not in path.resolve().parents or path.is_symlink():raise ValueError('unsafe index path')
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('closed hash mismatch')
    return len(expected)


def monitor_worker(command,root,limits,env=None,start_unix=None,use_gpu=True):
    root=Path(root);start=time.time() if start_unix is None else start_unix
    deadline=start+limits['total_seconds'];peak_rss=peak_gpu=0
    reason=failure=None
    with (root/'worker.log').open('xb') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        born=time.time()
        write_json(root/'owned_worker.json',dict(pid=process.pid,parent_pid=os.getpid(),started_unix=born))
        try:
            while process.poll() is None:
                tree=process_tree(os.getpid());rss=sum(tree.values())
                used=gpu_memory(set(tree)) if use_gpu else 0
                peak_rss=max(peak_rss,rss);peak_gpu=max(peak_gpu,used)
                disk,artifacts=storage_sizes(root)
                values=dict(rss_bytes=rss,gpu_bytes=used,disk_bytes=disk,artifact_bytes=artifacts)
                reason=next((key for key,value in values.items() if value>limits[key]),None)
                if time.time()>=deadline-5:reason='total_seconds'
                if time.time()-born>=limits['scientific_seconds']-5:reason='scientific_seconds'
                if shutil.disk_usage(root).free<limits['free_bytes']:reason='free_bytes'
                phase_path=root/'phase.json'
                if phase_path.exists():
                    phase=json.loads(phase_path.read_text())
                    if phase.get('component_deadline_unix') is not None and time.time()>=phase['component_deadline_unix']:
                        reason='component_seconds:'+phase['name']
                if reason:break
                time.sleep(.25)
        except BaseException as exc:
            failure=type(exc).__name__+': '+str(exc)
        finally:
            terminate_owned(process);code=process.wait(timeout=2.)
    disk,artifacts=storage_sizes(root)
    result=dict(status='monitor_failed' if failure else 'resource_stop' if reason or code==2 else 'complete' if code==0 else 'worker_failed',
        reason=reason,monitor_error=failure,returncode=code,total_seconds=time.time()-start,
        scientific_worker_seconds=time.time()-born,peak_rss_bytes=peak_rss,peak_gpu_bytes=peak_gpu,
        artifact_bytes=artifacts,disk_bytes=disk,gpu_memory_monitored=use_gpu,
        sampling_and_shutdown_overshoot_possible=True)
    write_json(root/'execution.json',result);close_index(root);verify_index(root)
    return result


def environment(cfg):
    versions={name:importlib.metadata.version(name) for name in cfg['dependencies']}
    if any(versions[name].split('+')[0]!=version for name,version in cfg['dependencies'].items()):
        raise RuntimeError('dependency version lock')
    rows=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],
        capture_output=True,text=True,check=True,timeout=.75).stdout.strip().splitlines()
    if len(rows)!=1 or 'A800' not in rows[0]:raise RuntimeError('one A800 required')
    if any(os.environ.get(name)!='1' for name in THREADS):raise RuntimeError('five BLAS thread limits must be1')
    return dict(versions=versions,physical_identity=rows[0],python=sys.version,
        executable=sys.executable,threads={name:os.environ[name] for name in THREADS},
        package_binaries={name:sha(importlib.util.find_spec(name).origin) for name in ('numpy','scipy','h5py','faiss','torch')})


def _worker(root,cfg,device_only=False):
    root=Path(root);freeze=json.loads((root/'source_freeze.json').read_text())
    if source_map()!=freeze['sources'] or cfg!=freeze['config'] or os.getppid()!=freeze['parent_pid']:
        raise RuntimeError('source/owner/config lock')
    last=[0.]
    def check():
        if os.getppid()!=freeze['parent_pid'] or time.time()>=freeze['deadline_unix']-10:
            raise TimeoutError('parent/global deadline')
        if time.monotonic()-last[0]>=10:
            if source_map()!=freeze['sources']:raise RuntimeError('source changed during worker')
            last[0]=time.monotonic()
    def phase(name,component=False):
        check();value=dict(name=name,time_unix=time.time(),
            component_deadline_unix=time.time()+cfg['limits']['component_seconds'] if component else None)
        temp=root/'phase.pending';temp.write_text(json.dumps(value));os.replace(temp,root/'phase.json')
        print(json.dumps(value),flush=True)
    try:
        from mechanism.tls_rag_step5_route6_a800_v1 import torch_backend
        from mechanism import tls_rag_step5_route6_representation_v1 as rep
        from mechanism import tls_rag_step5_route6_representation_screen_v1 as screen
        import numpy as np
        import faiss
        torch,precision=torch_backend();torch.set_num_threads(1)
        faiss.omp_set_num_threads(cfg['timing']['faiss_threads'])
        check()
        if environment(cfg)!=freeze['environment']:raise RuntimeError('worker environment changed')
        binary_paths=dict(numpy=np.core._multiarray_umath.__file__,torch=torch._C.__file__)
        binary_paths.update({name:module.__file__ for name,module in sys.modules.copy().items()
            if name.startswith('faiss._swigfaiss') and hasattr(module,'__file__')})
        binary_sha={key:sha(path) for key,path in binary_paths.items()}
        write_json(root/'binary_identity.json',dict(binary_sha256=binary_sha,precision=precision,
            faiss_gpu_available=faiss.get_num_gpus(),gpu_faiss_used=False))
        if device_only:
            rng=np.random.Generator(np.random.PCG64(23));x=rng.normal(size=(64,16)).astype(np.float32)
            queries=rng.normal(size=(4,16)).astype(np.float32);pca=rep.fit_pca(x)
            outputs=[]
            for kind in ('raw','gaussian','pca','hybrid'):
                spec=dict(id=kind,kind=kind,dim=16 if kind=='raw' else 8,seed=cfg['seeds']['gaussian'],retained=4)
                model=rep.fit_representation(spec,x,pca,cfg,check);v=rep.transform_batch(model,x);q=rep.transform_batch(model,queries)
                cpu=rep.exact_representation_search(v,q,8,model['metric'])
                gpu=rep.exact_representation_search(v,q,8,model['metric'],torch=torch,check=check)
                if not np.array_equal(cpu,gpu):raise RuntimeError('tiny CUDA FP64 ordering mismatch')
                dv=torch.as_tensor(v,device='cuda');native=np.asarray([rep.search_backend(model,z,8,device_values=dv,torch=torch) for z in queries])
                if not np.array_equal(cpu,native):raise RuntimeError('tiny CUDA FP32 ordering mismatch')
                outputs.append(dict(kind=kind,cpu_cuda_exact=True));del dv;torch.cuda.empty_cache()
            write_json(root/'device_receipt.json',dict(status='passed',sources=freeze['sources'],config=cfg,
                environment=freeze['environment'],precision=precision,checks=outputs,binary_sha256=binary_sha))
        else:
            screen.run_screen(cfg,root,torch,check,phase,freeze['environment'])
        check();return 0
    except BaseException as exc:
        write_json(root/'failure.json',dict(type=type(exc).__name__,detail=str(exc),traceback=traceback.format_exc(),
            result_is_incomplete=True,automatic_retry=False))
        return 2 if isinstance(exc,TimeoutError) else 1


def supervise(root,cfg,receipt=None,device_only=False):
    start=ENTRY_UNIX
    if sys.platform!='linux':raise RuntimeError('real S0 requires existing Linux node')
    root=Path(root).resolve();root.mkdir(exist_ok=False)
    try:
        subprocess.run([sys.executable,str(ROOT/'scripts/check_code_protection.py')],cwd=ROOT,check=True,timeout=30)
        status=subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,
            capture_output=True,text=True,check=True,timeout=5).stdout
        if status.strip():raise RuntimeError('tracked checkout dirty')
        env=environment(cfg);sources=source_map()
        if gpu_memory({os.getpid()})!=0:raise RuntimeError('GPU busy before S0')
        if shutil.disk_usage(root).free<cfg['limits']['free_bytes']:raise RuntimeError('free disk gate')
        allocation=allocated_resources()
        if allocation['cpu_threads']<cfg['limits']['cpu_threads']:raise RuntimeError('CPU quota/affinity')
        if allocation['memory_bytes']<cfg['limits']['rss_bytes']:raise RuntimeError('RSS allocation')
        if not device_only:
            saved=json.loads(Path(receipt).read_text())
            if saved['status']!='passed' or saved['sources']!=sources or saved['config']!=cfg or saved['environment']!=env:
                raise RuntimeError('missing/mismatched successful CUDA receipt')
        limits=dict(cfg['limits'])
        if device_only:limits.update(scientific_seconds=120,total_seconds=180)
        write_json(root/'source_freeze.json',dict(config=cfg,sources=sources,environment=env,parent_pid=os.getpid(),
            allocation=allocation,global_start_unix=start,deadline_unix=start+limits['total_seconds'],
            device_receipt_sha256=sha(receipt) if receipt else None,
            commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True,timeout=5).stdout.strip(),
            mode='tiny CUDA only' if device_only else 'one approved S0'))
        childenv=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        command=[sys.executable,str(Path(__file__).resolve()),'device-worker' if device_only else 'worker','--run-dir',str(root)]
        return monitor_worker(command,root,limits,childenv,start_unix=start)
    except BaseException as exc:
        write_json(root/'preflight_failure.json',dict(type=type(exc).__name__,detail=str(exc),
            traceback=traceback.format_exc(),elapsed_in_global_timer=time.time()-start,science_started=False))
        close_index(root);raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('device-check','device-worker','run','worker','verify'))
    parser.add_argument('--run-dir',required=True);parser.add_argument('--device-receipt')
    args=parser.parse_args();cfg=config()
    if args.action=='verify':print(verify_index(args.run_dir));return 0
    if args.action in ('worker','device-worker'):return _worker(args.run_dir,cfg,args.action=='device-worker')
    if args.action=='run' and not args.device_receipt:parser.error('run requires --device-receipt')
    result=supervise(args.run_dir,cfg,args.device_receipt,args.action=='device-check')
    print(json.dumps(result));return 0 if result['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
