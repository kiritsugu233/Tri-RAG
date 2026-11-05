#!/usr/bin/env python3
"""One approved C1 CPU replay with manual stop and closed SHA evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def pid_start(pid):
    """Linux kernel start tick; portable tests still own a live Popen handle."""
    try:
        fields=Path('/proc/{}/stat'.format(pid)).read_text().rsplit(')',1)[1].split()
        return fields[19]  # field22; fields starts at field3
    except (OSError,IndexError):
        return None


def request_stop(root):
    root=Path(root)
    owner=json.loads((root/'ownership.json').read_text())
    if owner['uid']!=os.getuid() or root.is_symlink():raise ValueError('unowned C1 root')
    if json.loads((root/'status.json').read_text()).get('status')!='running':
        raise ValueError('refuse to modify a closed C1 run')
    write(root/'stop_requested.json',dict(owner_token=owner['owner_token'],action='stop',requested_unix=time.time()))


def terminate_owned(child, owner, token):
    if (owner['child_pid']!=child.pid or owner['parent_pid']!=os.getpid()
            or owner['uid']!=os.getuid() or owner['owner_token']!=token):
        raise ValueError('refuse unowned process stop')
    if child.poll() is not None:return
    if owner['child_start'] is not None and pid_start(child.pid)!=owner['child_start']:
        raise ValueError('child identity changed')
    child.terminate()
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Only after explicit user stop of this exact live Popen child.
        child.kill();child.wait()


def close_evidence(root):
    root=Path(root)
    excluded={'archive_index.json','archive_verified.json'}
    entries=[]
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('C1 evidence symlink')
        if p.is_file() and p.name not in excluded:
            entries.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p)))
    write(root/'archive_index.json',dict(files=entries,streams_closed=True,old_products_modified=False))
    for e in entries:
        p=root/e['path']
        if p.stat().st_size!=e['bytes'] or digest(p)!=e['sha256']:raise RuntimeError('closed evidence changed')
    write(root/'archive_verified.json',dict(verified=True,files=len(entries),bytes=sum(e['bytes'] for e in entries),
        index_sha256=digest(root/'archive_index.json')))


def observation(root, child):
    free=os.statvfs(root);rss=None
    try:
        line=next(l for l in Path('/proc/{}/status'.format(child.pid)).read_text().splitlines() if l.startswith('VmRSS:'))
        rss=int(line.split()[1])*1024
    except (OSError,StopIteration,ValueError):pass
    return dict(time_unix=time.time(),child_pid=child.pid,rss_bytes=rss,
        free_bytes=free.f_bavail*free.f_frsize,resources_are_observation_only=True,gpu_not_queried=True)


def supervise(command, root, environment, interval=300, poll=.2):
    """No elapsed/resource/telemetry abort. Compute exit and user stop preserved."""
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    token=secrets.token_hex(24);start=time.monotonic();status=None
    with (root/'worker.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,
            cwd=ROOT,start_new_session=True)
        owner=dict(parent_pid=os.getpid(),child_pid=child.pid,child_start=pid_start(child.pid),
            uid=os.getuid(),argv=command,owner_token=token)
        write(root/'status.json',dict(status='running',child_pid=child.pid,automatic_budget_abort=False))
        write(root/'ownership.json',owner)
        last=-float('inf')
        try:
            while child.poll() is None:
                marker=root/'stop_requested.json'
                if marker.exists():
                    try:
                        if marker.is_symlink():raise ValueError('stop request symlink')
                        request=json.loads(marker.read_text())
                        if request.get('owner_token')!=token or request.get('action')!='stop':
                            raise ValueError('unowned stop request')
                    except (OSError,ValueError) as e:
                        write(root/'stop_request_error.json',dict(error=repr(e),worker_not_stopped=True))
                    else:
                        terminate_owned(child,owner,token);status='user_stopped';break
                if time.monotonic()-last>=interval:
                    try:value=observation(root,child)
                    except Exception as e:value=dict(observation_error=repr(e),not_a_stop_reason=True)
                    write(root/'observation.json',value);last=time.monotonic()
                    progress=None
                    try:progress=json.loads((root/'progress.json').read_text())
                    except (OSError,ValueError):pass
                    print(json.dumps(dict(stage='C1_running',progress=progress,observation=value)),flush=True)
                time.sleep(poll)
        except KeyboardInterrupt:
            terminate_owned(child,owner,token);status='user_stopped'
        except Exception as e:
            # Supervisor cannot leave an unobserved own compute worker behind.
            terminate_owned(child,owner,token);status='controller_error'
            write(root/'controller_error.json',dict(error=repr(e)))
        code=child.wait()
    if status is None:status='complete' if code==0 else 'compute_failed'
    value=dict(status=status,returncode=code,elapsed_seconds=time.monotonic()-start,
        automatic_budget_abort=False,worker_exited=True,formal_certification=False)
    write(root/'status.json',value)
    close_evidence(root)
    print(json.dumps(value),flush=True)
    return value


def worker(root):
    from mechanism import tls_rag_step5_pq_refinement_cost_v1 as cost
    cfg=cost.config();freeze=dict(sources=cost.source_map(cfg),config=cfg,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        started_unix=time.time(),manual_supervision=True,gpu_not_used=True)
    cost.unique_json(root/'source_freeze.json',freeze)
    try:
        result=cost.run_science(cfg,root)
        cost.unique_json(root/'result.json',result)
    except Exception as e:
        cost.unique_json(root/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),no_automatic_retry=True))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','worker','status','stop'])
    p.add_argument('--root',required=True)
    args=p.parse_args();root=Path(args.root).resolve()
    if args.action=='worker':worker(root)
    elif args.action=='stop':request_stop(root)
    elif args.action=='status':print((root/'status.json').read_text())
    else:
        if sys.platform!='linux':raise RuntimeError('real C1 is Linux server CPU only; use synthetic tests locally')
        if root.exists():raise FileExistsError(root)
        env=os.environ.copy()
        for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[name]='1'
        env['PYTHONDONTWRITEBYTECODE']='1'
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--root',str(root)]
        result=supervise(command,root,env)
        return 0 if result['status']=='complete' else 1
    return 0


if __name__=='__main__':sys.exit(main())
