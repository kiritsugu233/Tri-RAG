"""CPU-only v2 fixtures: source layout, cold costs, paired roles and real stops."""
import copy
import ctypes
import struct
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import faiss
import h5py
import numpy as np

from mechanism import tls_rag_step5_route6_representation_v2 as rep
from mechanism import tls_rag_step5_route6_representation_screen_v2 as screen
from scripts import run_tls_rag_step5_route6_representation_v2 as runner


class Buffer:
    def __init__(self,shape,dtype='torch.float32',device='cuda:0'):
        self.shape=shape;self.dtype=dtype;self.device=device
    def __len__(self):return self.shape[0]


class FakeNVML:
    """Independent packed C bytes; never constructs the production structure."""
    def __init__(self,steps=None):
        self.steps=list(steps or [(0,0,[]) ]);self.init_count=0;self.shutdown_count=0
        self.driver='580.82.07';self.version='13.580.82.07';self.count=1
        self.uuid='GPU-0c08388b-2980-4723-8c2e-2318f276cd0c';self.calls=0
        def uint(pointer,value):ctypes.cast(pointer,ctypes.POINTER(ctypes.c_uint))[0]=value
        def string(buffer,size,value):ctypes.memmove(buffer,value.encode()+b'\0',len(value)+1);return 0
        def init():self.init_count+=1;return 0
        def shutdown():self.shutdown_count+=1;return 0
        def handle(index,pointer):ctypes.cast(pointer,ctypes.POINTER(ctypes.c_void_p))[0]=123;return 0
        def count(pointer):uint(pointer,self.count);return 0
        def query(handle,pointer,buffer):
            self.calls+=1;code,n,rows=self.steps.pop(0);uint(pointer,n)
            if buffer is not None and rows:
                packed=b''.join(struct.pack('=I4xQII',pid,memory,0xffffffff,0xffffffff) for pid,memory in rows)
                ctypes.memmove(buffer,packed,len(packed))
            return code
        functions=dict(nvmlInit_v2=init,nvmlShutdown=shutdown,nvmlDeviceGetCount_v2=count,
            nvmlDeviceGetHandleByIndex_v2=handle,nvmlSystemGetDriverVersion=lambda b,n:string(b,n,self.driver),
            nvmlSystemGetNVMLVersion=lambda b,n:string(b,n,self.version),
            nvmlDeviceGetUUID=lambda h,b,n:string(b,n,self.uuid),nvmlDeviceGetComputeRunningProcesses_v3=query)
        for name,function in functions.items():setattr(self,name,mock.Mock(side_effect=function))


class RepresentationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        faiss.omp_set_num_threads(1);cls.cfg=json.loads(runner.CONFIG.read_text());cls.legacy_monitor=dict(runner.MONITOR)

    def fixture(self,root):
        x=np.array([[1,0],[0,1],[-1,0],[0,-1]],np.float32);q=x[:2].copy()
        source=root/'source.h5'
        with h5py.File(source,'w') as f:
            f.attrs['distance']='normalized';f.attrs['dimension']=2
            f['train']=x;f['learn']=q;f['learn_neighbors']=np.array([[0,1],[1,0]],np.uint32)
            for name in ('test','neighbors','distances','avg_distances'):f[name]=[999]
        cfg=copy.deepcopy(self.cfg)
        with h5py.File(source,'r') as f:offset=f['train'].id.get_offset()
        digest=rep.sha(source);roles=[dict(row=i,id=rep.stable_id(digest,'learn',i),
            vector_sha256=rep.canonical_vector_hash(q[i]),role='geometry_discovery' if i==0 else 'geometry_check') for i in range(2)]
        path=root/'roles.json';path.write_text(json.dumps(dict(queries=roles)))
        cfg.update(queries=2,discovery_queries=1,k=2,block=2,roles_path=str(path),roles_sha256=rep.sha(path))
        cfg['source'].update(path=str(source),rows=4,dimension=2,bytes=source.stat().st_size,sha256=digest)
        cfg['source']['layout']['offset']=offset
        return cfg,x,q,roles

    def test_mmap_is_readonly_and_original_file_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,x,_,_=self.fixture(Path(tmp));before=rep.sha(cfg['source']['path'])
            values=rep.source_mapping(cfg,verify_hash=True)
            self.assertIsInstance(values,np.memmap);self.assertFalse(values.flags.writeable)
            np.testing.assert_array_equal(values,x)
            with self.assertRaises(ValueError):values[0,0]=0
            self.assertEqual(rep.sha(cfg['source']['path']),before)

    def test_layout_shape_offset_dtype_and_source_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,*_=self.fixture(Path(tmp))
            for key,value in [('offset',0),('dtype','>f4'),('chunks',[2,2]),('virtual',True)]:
                bad=copy.deepcopy(cfg);bad['source']['layout'][key]=value
                with self.assertRaises(ValueError):rep.source_mapping(bad)
            bad=copy.deepcopy(cfg);bad['source']['rows']=3
            with self.assertRaises(ValueError):rep.source_mapping(bad)
            bad=copy.deepcopy(cfg);bad['source']['sha256']='0'*64
            with self.assertRaises(ValueError):rep.source_mapping(bad,verify_hash=True)

    def test_only_original_query_and_truth_rows_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,x,q,roles=self.fixture(Path(tmp));calls=[];real=rep.read_payload
            def spy(f,name,row=None):calls.append((name,row));return real(f,name,row)
            with mock.patch.object(rep,'read_payload',side_effect=spy):actual=rep.load_screen_inputs(cfg)
            np.testing.assert_array_equal(actual[0],x);np.testing.assert_array_equal(actual[1],q)
            self.assertEqual(actual[3],roles);self.assertEqual(calls,[('learn',0),('learn_neighbors',0),('learn',1),('learn_neighbors',1)])
            self.assertFalse(actual[-1]['test_payload_read'])
            with self.assertRaises(ValueError):rep.read_payload({},'test',0)

    def test_cached_l2_matches_direct_nonunit_distances_and_ties(self):
        values=np.array([[2.,0.],[0.,3.],[2.,0.],[-1.,.5]],np.float32)
        norms=np.sum(values*values,axis=1)
        for q in [np.array([1.,0.],np.float32),np.array([.5,-2.],np.float32)]:
            actual=rep.cached_scores(values,q,norms,'l2')
            np.testing.assert_array_equal(actual,np.sum((values-q)**2,axis=1))
            np.testing.assert_array_equal(rep.stable_top(actual,3),np.lexsort((np.arange(4),np.sum((values-q)**2,axis=1)))[:3])
            np.testing.assert_array_equal(rep.cached_scores(values,q,None,'ip'),-values@q)
        with self.assertRaises(ValueError):rep.cached_scores(values,q,None,'l2')

    def cache(self):
        model=dict(fingerprint='model',metric='l2',source_sha256='source')
        binding=rep.cache_binding(model,'source',(3,2),'cuda:0')
        return model,dict(values=Buffer((3,2)),norms=Buffer((3,)),binding=binding,fingerprint=rep.fingerprint(binding))

    def test_cuda_binding_rejects_model_source_shape_dtype_device(self):
        model,cache=self.cache();rep.validate_cuda_cache(model,cache)
        for key,value in [('model','other'),('source','other'),('shape',[4,2]),('dtype','float64'),('device','cuda:1')]:
            bad=copy.deepcopy(cache);bad['binding'][key]=value;bad['fingerprint']=rep.fingerprint(bad['binding'])
            with self.assertRaises(ValueError):rep.validate_cuda_cache(model,bad)
        bad=copy.deepcopy(cache);bad['norms']=Buffer((2,))
        with self.assertRaises(ValueError):rep.validate_cuda_cache(model,bad)

    def test_saved_model_and_config_mutation_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);x=np.eye(4,dtype=np.float32);cfg=copy.deepcopy(self.cfg)
            spec=dict(id='g',kind='gaussian',dim=2,seed=4)
            model=rep.fit_representation(spec,x,rep.fit_pca(x),cfg)
            screen.save_model(screen.Products(root,10**6),'m',model,cfg)
            restored=rep.model_from_products(root/'m',spec,cfg)
            np.testing.assert_array_equal(rep.transform_batch(restored,x),rep.transform_batch(model,x))
            bad=copy.deepcopy(cfg);bad['seeds']['gaussian']+=1
            with self.assertRaises(ValueError):rep.model_from_products(root/'m',spec,bad)
            np.save(root/'m/matrix.npy',np.zeros((2,4),np.float32))
            with self.assertRaises(ValueError):rep.model_from_products(root/'m',spec,cfg)

    def test_pq_restores_exact_centers_codes_and_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);x=np.random.default_rng(3).normal(size=(512,8)).astype(np.float32)
            cfg=copy.deepcopy(self.cfg);cfg['training']['niter']=2
            spec=dict(id='pq',kind='raw',dim=8,pq_bytes=2)
            model=rep.fit_representation(spec,x,rep.fit_pca(x),cfg)
            codes,centers=rep.encode_pq(model,x,100);p=screen.Products(root,10**7)
            screen.save_model(p,'m',model,cfg);p.array('m/codes.npy',codes)
            restored=rep.model_from_products(root/'m',spec,cfg)
            rep.install_codes(restored,root/'m/codes.npy',rep.sha(root/'m/codes.npy'))
            for q in x[:3]:
                np.testing.assert_array_equal(rep.search_backend(model,q,10,index=model['index']),
                                              rep.search_backend(restored,q,10,index=restored['index']))
            with self.assertRaises(ValueError):rep.install_codes(restored,root/'m/codes.npy','0'*64)

    def test_cpu_cohort_fixed_per_role_not_by_latency(self):
        roles=[dict(id='query-%03d'%i,role='geometry_discovery' if i<64 else 'geometry_check') for i in range(128)]
        chosen=screen.timing_cohort(roles,self.cfg,'cpu14')
        self.assertEqual(len(chosen),32);self.assertEqual(sum(i<64 for i in chosen),16)
        self.assertEqual(chosen,screen.timing_cohort(roles,self.cfg,'cpu14'))
        changed=[dict(r,latency=999 if i in chosen else 0) for i,r in enumerate(roles)]
        self.assertEqual(chosen,screen.timing_cohort(changed,self.cfg,'cpu14'))
        self.assertEqual(screen.timing_cohort(roles,self.cfg,'cuda'),list(range(128)))

    def test_timing_points_keep_full_and_paired_quality_distinct(self):
        quality=[dict(query_index=i,role='geometry_discovery',budget=100,recall=float(i==0),strict_id=float(i==0)) for i in range(2)]
        measured=[dict(query_index=0,role='geometry_discovery',budget=100,seconds=.2)]
        points=screen.timing_points('a:cpu_flat',quality,measured,100)
        for p in points:
            self.assertEqual(p['quality'],.5);self.assertEqual(p['timing_cohort_quality'],1.)
            self.assertEqual(p['quality_queries'],2);self.assertEqual(p['timing_queries'],1)

    def test_lifecycle_uses_real_hardware_costs_and_retains_raw(self):
        cost=dict(sample_seconds=1,pca_seconds=2,training_seconds=3,model_write_seconds=4,code_write_seconds=5,
            model_load_seconds=6,transform_encode_seconds=7,h2d_seconds=8,norm_prepare_seconds=9,
            flat_build_seconds=100,payload_restore_seconds=10,input_audit_seconds=11,source_file_bytes=1024)
        gpu=screen.lifecycle_cost(cost,'cuda',100,20,3,4)
        cpu=screen.lifecycle_cost(cost,'cpu14',100,20,3)
        self.assertEqual(gpu['offline_seconds'],45);self.assertEqual(cpu['offline_seconds'],138)
        self.assertEqual(gpu['resident_bytes'],127);self.assertEqual(gpu['input_verification_seconds'],11)
        with self.assertRaises(ValueError):screen.lifecycle_cost(cost,'cuda',0,20,3)

    def test_gaussian_selection_can_choose_cuda_by_matched_control(self):
        configs={'raw':dict(kind='raw',diagnostic=False),'g':dict(kind='gaussian',diagnostic=False),'h':dict(kind='hybrid',diagnostic=False)}
        rows=[]
        for pipeline,hardware,seconds in [('raw:cpu_flat','cpu14',10),('raw:cuda','cuda',1),('g:cpu_flat','cpu14',8),('g:cuda','cuda',.5)]:
            rows.append(dict(pipeline=pipeline,hardware_profile=hardware,seconds=seconds,role='geometry_discovery',metric='recall',quality=.99,resident_bytes=10))
        self.assertEqual(screen.select_gaussian_pipelines(rows,configs,self.cfg),['g:cuda'])
        rows.append(dict(rows[-1],pipeline='h:cuda',role='geometry_check',seconds=.001))
        self.assertEqual(screen.select_gaussian_pipelines(rows,configs,self.cfg),['g:cuda'])

    def test_search_is_label_free(self):
        self.assertFalse(set(inspect.signature(rep.search_backend).parameters)&{'truth','recall','labels','oracle','cutoff'})

    def test_quality_query_path_matches_deployed_search_at_numeric_boundary(self):
        x=np.random.default_rng(742).normal(size=(128,640)).astype(np.float32)
        x/=np.linalg.norm(x,axis=1)[:,None]
        model=rep.fit_representation(dict(id='g',kind='gaussian',dim=256,seed=2026110353),x,None,self.cfg)
        batch=rep.transform_batch(model,x)[0];single=rep.transform_batch(model,x[:1])[0]
        # Two boundary competitors expose batch-vs-row rounding where the
        # installed BLAS paths differ; canonical ties also work if they agree.
        index=faiss.IndexFlatL2(256);index.add(np.stack([batch,single]))
        expected=rep.search_backend(model,x[0],1,index=index)
        actual=rep.faiss_search(index,rep.query_coordinates(model,x)[0],1,'l2')
        np.testing.assert_array_equal(actual,expected)

    def test_cpu_staged_quality_restore_timing_and_lifecycle_smoke(self):
        # Exercise real CPU IO/metrics/index/cost paths. CUDA is independently
        # required on the node; this fixture substitutes only that device work.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/'input';inputs.mkdir();cfg,x,q,roles=self.fixture(inputs)
            cfg.update(fit_rows=4,budgets=[2,4],representations=[dict(id='raw640',kind='raw',family='raw',dim=2,bytes=8,diagnostic=False)])
            cfg['timing'].update(cpu_queries_per_role=1,warmup=0,repeats=1,sensitivity_queries=1,faiss_threads=1)
            cfg['ivf'].update(nlist=2,nprobes=[2])
            truth=np.array([[0,1],[1,0]]);products=screen.Products(root,10**7)
            products.json('roles.json',dict(queries=roles));products.array('queries.npy',q)
            products.array('published_truth.npy',truth);products.array('published_cutoffs.npy',np.array([1.,1.],np.float32))
            products.array('fit_rows.npy',np.arange(4));products.array('prior_fp64_top100.npy',truth)
            products.json('schedule.json',dict(cpu_one_thread_indices=[0]))
            for name,value in rep.fit_pca(x).items():products.array('shared_pca/'+name+'.npy',value)
            names=['roles.json','queries.npy','published_truth.npy','published_cutoffs.npy','fit_rows.npy','prior_fp64_top100.npy','schedule.json']
            products.json('prepared.json',dict(source_sha256=cfg['source']['sha256'],roles_sha256=cfg['roles_sha256'],
                files={name:rep.sha(root/name) for name in names},sample_seconds=.01,pca_seconds=.02,input_audit_seconds=.03))
            real_reference=rep.exact_representation_search;real_search=rep.search_backend
            def cpu_reference(*args,**kwargs):
                args=list(args)
                if args and isinstance(args[0],rep.IVFDecodedView):args[0]=args[0][:len(args[0])]
                if len(args)>6:args[6]=None
                else:kwargs['torch']=None
                return real_reference(*args,**kwargs)
            def cpu_search(model,query,count,index=None,cuda_index=None,torch=None):
                if index is None:index=rep.make_flat(model,cuda_index['values'],1)
                return real_search(model,query,count,index=index)
            def cpu_cache(values,model,*args):
                return dict(values=values,h2d_seconds=.01,norm_prepare_seconds=0.,payload_bytes=values.nbytes,norm_bytes=0,binding={},fingerprint='cpu-fixture')
            device=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None))
            with mock.patch.object(rep,'exact_representation_search',side_effect=cpu_reference),mock.patch.object(rep,'prepare_cuda_index',side_effect=cpu_cache),mock.patch.object(rep,'search_backend',side_effect=cpu_search):
                screen.quality(cfg,root,'raw640',device,lambda:None,lambda *a,**k:None)
            screen.linear_timing(cfg,root,'raw640',device,lambda:None,lambda *a,**k:None,backend='cpu14')
            completion=json.loads((root/'representations/raw640/timing_complete.json').read_text())
            self.assertEqual(completion['cpu_queries'],2);self.assertEqual(completion['gpu_queries'],0)
            points=screen.collect_points(root);self.assertEqual(len(points),4)
            lifecycle=screen.lifecycle_tables(cfg,root,points)
            self.assertEqual(len(lifecycle['costs']),4)
            self.assertTrue(all(p['quality']==1 for p in points))
            with mock.patch.object(rep,'exact_representation_search',side_effect=cpu_reference):
                screen.ivf(cfg,root,'raw640',device,lambda:None,lambda *a,**k:None)
            points=screen.collect_points(root);self.assertEqual(len(points),8)
            lifecycle=screen.lifecycle_tables(cfg,root,points)
            self.assertEqual(len(lifecycle['costs']),8)

    def sample(self,root,**changes):
        value=dict(provider=runner.provider_identity(self.legacy_monitor),nonce='n',reporter_pid=123,owner_pid=os.getpid(),sequence=1,
            started_monotonic=10.,completed_monotonic=10.1,memory_bytes={})
        value.update(changes);runner.atomic_json(Path(root)/'telemetry.json',value)

    def test_telemetry_rejects_stale_unknown_malformed_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.sample(tmp);self.assertEqual(runner.telemetry_memory(tmp,'n',123,[],self.legacy_monitor,now=11),0)
            with self.assertRaises(RuntimeError):runner.telemetry_memory(tmp,'n',123,[],self.legacy_monitor,now=13.01)
            for changes in [dict(nonce='bad'),dict(memory_bytes={'999999':1}),dict(memory_bytes={'1':'N/A'}),dict(completed_monotonic=20)]:
                self.sample(tmp,**changes)
                with self.assertRaises(RuntimeError):runner.telemetry_memory(tmp,'n',123,[],self.legacy_monitor,now=11)

    def test_gpu_parse_never_fills_unknown_with_zero(self):
        self.assertEqual(runner.parse_gpu('123, 10\n'),{123:10*2**20})
        for value in ['123, N/A','123, -1','0, 2','123, 1\n123, 2','NaN, 0']:
            with self.assertRaises(ValueError):runner.parse_gpu(value)

    def test_cross_process_clock_has_one_system_origin(self):
        before=time.clock_gettime(time.CLOCK_MONOTONIC)
        child=float(subprocess.check_output([sys.executable,'-c',
            'from scripts.run_tls_rag_step5_route6_representation_v2 import monotonic;print(monotonic())'],text=True))
        after=time.clock_gettime(time.CLOCK_MONOTONIC)
        self.assertLessEqual(before,child);self.assertLessEqual(child,after)

    def test_native_receipt_binds_loaded_extensions_and_mutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.so';path.write_bytes(b'first')
            stub=SimpleNamespace(_C=SimpleNamespace(__file__=str(path)))
            first=runner.native_binary_identity(stub)
            self.assertIn('numpy',first);self.assertIn('torch',first)
            self.assertTrue(any(n.startswith('faiss.') for n in first))
            path.write_bytes(b'second');second=runner.native_binary_identity(stub)
            self.assertNotEqual(first['torch'],second['torch']);self.assertEqual(first['numpy'],second['numpy'])

    def test_fresh_retired_owned_sample_does_not_authorize_live_external_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.sample(tmp,memory_bytes={str(os.getpid()):1024})
            with self.assertRaises(RuntimeError):
                runner.telemetry_memory(tmp,'n',123,[],self.legacy_monitor,now=11,retired_owned=[os.getpid()])

    def limits(self,**changes):
        value=dict(total_seconds=30,scientific_seconds=30,component_seconds=1200,rss_bytes=8*2**30,
            gpu_bytes=2**30,artifact_bytes=10**7,disk_bytes=10**7,free_bytes=0)
        value.update(changes);return value

    def job(self,root,guard,name,code):
        return runner.run_job([sys.executable,'-c',code],root,guard,name)

    def test_actual_component_stop_then_next_owned_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);guard=runner.Guard(root,self.limits(),self.legacy_monitor,use_gpu=False)
            code='import json,os,time;from pathlib import Path;Path('+repr(str(root/'phase.json'))+').write_text(json.dumps(dict(name="train",worker_pid=os.getpid(),component_deadline_monotonic=time.clock_gettime(time.CLOCK_MONOTONIC)+.1)));time.sleep(30)'
            first=self.job(root,guard,'train',code);self.assertEqual(first['status'],'component_stop')
            with self.assertRaises(ProcessLookupError):os.kill(first['worker_pid'],0)
            scientific_start=guard.scientific_start
            second=self.job(root,guard,'next','pass');self.assertEqual(second['status'],'complete')
            self.assertEqual(guard.scientific_start,scientific_start)

    def test_actual_rss_and_artifact_stops(self):
        for resource in ('rss_bytes','artifact_bytes'):
            with tempfile.TemporaryDirectory() as tmp:
                guard=runner.Guard(tmp,self.limits(**{resource:1}),self.legacy_monitor,use_gpu=False)
                # Exercise post-spawn enforcement separately from the preflight
                # gate, which would already reject this deliberately tiny cap.
                with mock.patch.object(guard,'ready'):
                    result=self.job(tmp,guard,'stop','import time;time.sleep(30)')
                self.assertEqual(result['status'],'resource_stop');self.assertEqual(result['reason'],resource)

    def test_actual_gpu_query_above_old_timeout_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            command=[sys.executable,'-c','import time;time.sleep(.9);print("")']
            guard=runner.Guard(tmp,self.limits(),self.legacy_monitor,query_command=command)
            try:
                guard.start_reporter();guard.ready()
                result=self.job(tmp,guard,'work','import time;time.sleep(1.1)')
                self.assertEqual(result['status'],'complete')
            finally:guard.close()

    def test_global_deadline_works_while_gpu_query_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);flag=root/'first'
            code='from pathlib import Path;import time;p=Path('+repr(str(flag))+');first=not p.exists();p.touch();time.sleep(0 if first else 10);print("")'
            guard=runner.Guard(root,self.limits(total_seconds=2),self.legacy_monitor,query_command=[sys.executable,'-c',code])
            try:
                guard.start_reporter();guard.ready();tick=time.monotonic()
                result=self.job(root,guard,'blocked','import time;time.sleep(30)')
                self.assertEqual(result['reason'],'total_seconds');self.assertLess(time.monotonic()-tick,3)
            finally:guard.close()

    def test_slow_valid_query_keeps_three_second_freshness_without_extra_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);count=root/'query_count'
            code='from pathlib import Path;import time;p=Path('+repr(str(count))+');n=int(p.read_text())+1 if p.exists() else 1;p.write_text(str(n));time.sleep(2.8 if n==2 else 0);print("")'
            settings=dict(self.legacy_monitor,poll_seconds=.02)
            guard=runner.Guard(root,self.limits(),settings,query_command=[sys.executable,'-c',code])
            try:
                guard.start_reporter();guard.ready()
                result=self.job(root,guard,'slow-valid','import time;time.sleep(3.6)')
                self.assertEqual(result['status'],'complete',result)
                samples=[json.loads(line) for line in (root/'telemetry.log').read_text().splitlines()]
                self.assertTrue(any(s.get('query_seconds',0)>=2.8 for s in samples))
                self.assertEqual(settings['gpu_max_age_seconds'],3.)
            finally:guard.close()

    def test_actual_reporter_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard=runner.Guard(tmp,self.limits(),self.legacy_monitor,query_command=[sys.executable,'-c','print("N/A")'])
            try:
                guard.start_reporter()
                with self.assertRaises(runner.MonitorFailure):guard.ready()
                self.assertFalse((Path(tmp)/'telemetry.json').exists())
            finally:guard.close()

    def test_monitor_exception_stops_owned_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard=runner.Guard(tmp,self.limits(),self.legacy_monitor,use_gpu=False)
            real=guard.check;calls=[0]
            def fail(*a,**kw):
                calls[0]+=1
                if calls[0]>1:raise RuntimeError('broken monitor')
                return real(*a,**kw)
            with mock.patch.object(guard,'check',side_effect=fail):result=self.job(tmp,guard,'fail','import time;time.sleep(30)')
            self.assertEqual(result['status'],'monitor_failed')
            with self.assertRaises(ProcessLookupError):os.kill(result['worker_pid'],0)

    def test_bounded_products_exclusive_and_closed_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=screen.Products(tmp,1000);p.json('a.json',dict(a=1))
            with self.assertRaises(FileExistsError):p.json('a.json',dict(a=2))
            with self.assertRaises(RuntimeError):p.array('large.npy',np.zeros(1000))
            runner.close_index(tmp);self.assertEqual(runner.verify_index(tmp),1)
            (Path(tmp)/'a.json').write_text('bad')
            with self.assertRaises(ValueError):runner.verify_index(tmp)

    def test_closure_counts_hash_verification_time_and_final_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'data').write_bytes(b'abc'*100)
            result=runner.close_run(tmp,dict(status='complete'),runner.monotonic(),10)
            self.assertEqual(result['status'],'complete');self.assertGreater(result['closure_hash_and_verify_seconds'],0)
            self.assertEqual(runner.verify_index(tmp),2)
            saved=json.loads((Path(tmp)/'execution.json').read_text());self.assertFalse(saved['sealing_in_progress'])

    def test_closure_deadline_does_not_claim_complete_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'data').write_bytes(b'abc')
            result=runner.close_run(tmp,dict(status='complete'),runner.monotonic()-10,5)
            self.assertEqual(result['status'],'resource_stop')
            self.assertFalse((Path(tmp)/'delivery_index.json').exists())


    def nvml(self,library):
        with mock.patch.object(runner,'sha',return_value=self.legacy_monitor['library_sha256']),mock.patch.object(runner.ctypes,'CDLL',return_value=library):
            return runner.NVMLProvider(self.legacy_monitor)

    def test_nvml_independent_abi_bytes_and_persistent_initialization(self):
        library=FakeNVML([(7,2,[]),(0,2,[(101,1234567),(202,0)]),(0,0,[])])
        provider=self.nvml(library)
        try:
            self.assertEqual(ctypes.sizeof(runner.NVMLProcess),len(struct.pack('=I4xQII',1,2,3,4)))
            self.assertEqual(provider.query(),{101:1234567,202:0});self.assertEqual(provider.query(),{})
            self.assertEqual(library.init_count,1);self.assertEqual(library.calls,3)
            self.assertEqual(len(library.nvmlDeviceGetComputeRunningProcesses_v3.argtypes),3)
        finally:provider.close()
        self.assertEqual(library.shutdown_count,1)

    def test_nvml_na_unknown_errors_counts_and_capacity_fail_closed(self):
        cases=[[(3,0,[])],[(0,1,[])],[(7,1025,[])],[(7,1,[]),(7,20,[]),(7,30,[])],
            [(7,1,[]),(0,1,[(12,2**64-1)])],[(7,2,[]),(0,2,[(12,1),(12,2)])],
            [(7,1,[]),(0,1,[(0,1)])],[(7,1,[]),(0,1,[])] ]
        for steps in cases:
            with self.subTest(steps=steps):
                library=FakeNVML(steps);provider=self.nvml(library)
                try:
                    with self.assertRaises(RuntimeError):provider.query()
                    self.assertLessEqual(library.calls,3)
                finally:provider.close()
        library=FakeNVML([(7,1,[]),(7,20,[]),(0,1,[(5,77)])]);provider=self.nvml(library)
        try:self.assertEqual(provider.query(),{5:77})
        finally:provider.close()

    def test_nvml_wrong_library_device_version_or_mutation_rejected(self):
        with mock.patch.object(runner,'sha',return_value='bad'),mock.patch.object(runner.ctypes,'CDLL') as loader:
            with self.assertRaises(RuntimeError):runner.NVMLProvider(self.legacy_monitor)
            loader.assert_not_called()
        for field,value in [('driver','wrong'),('version','wrong'),('uuid','wrong'),('count',2)]:
            library=FakeNVML();setattr(library,field,value)
            with self.assertRaises(RuntimeError):self.nvml(library)
            self.assertEqual(library.shutdown_count,1)
        library=FakeNVML();provider=self.nvml(library)
        try:
            library.driver='changed'
            with self.assertRaises(RuntimeError):provider.query()
            library.driver='580.82.07';provider.last_hash=runner.monotonic()-11
            with mock.patch.object(runner,'sha',return_value='changed'):
                with self.assertRaises(RuntimeError):provider.query()
        finally:provider.close()

    def test_provider_mismatch_and_stale_error_contains_actual_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.sample(tmp,provider=dict(backend='unapproved'))
            with self.assertRaisesRegex(RuntimeError,'age_seconds.*1.0'):
                runner.telemetry_memory(tmp,'n',123,[],self.legacy_monitor,now=11)
            self.sample(tmp,memory_bytes={'01':1})
            with self.assertRaises(RuntimeError):runner.telemetry_memory(tmp,'n',123,[],self.legacy_monitor,now=11)

    def test_actual_native_query_block_stops_parent_and_owned_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);guard=runner.Guard(root,self.limits(),self.legacy_monitor)
            # Native sleep is inside the reporter process, with no subprocess
            # timeout to rescue it. Only the parent's unchanged3s age can stop.
            code="""import ctypes,os
from scripts import run_tls_rag_step5_route6_representation_v2 as r
class Provider:
 def __init__(self,settings):self.n=0
 def query(self):
  self.n+=1
  if self.n>1:ctypes.CDLL(None).sleep(20)
  return {}
 def close(self):pass
r.NVMLProvider=Provider
r.reporter(ROOT,NONCE,OWNER,r.MONITOR)
""".replace('ROOT',repr(str(root))).replace('NONCE',repr(guard.nonce)).replace('OWNER',str(os.getpid()))
            guard.report=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            try:
                guard.ready();tick=time.monotonic();result=self.job(root,guard,'native-block','import time;time.sleep(30)')
                self.assertEqual(result['status'],'monitor_failed');self.assertIn('age_seconds',result['monitor_error'])
                self.assertLess(time.monotonic()-tick,4.5)
                with self.assertRaises(ProcessLookupError):os.kill(result['worker_pid'],0)
            finally:guard.close()
            self.assertIsNotNone(guard.report.poll())

    def gate_fixture(self,root):
        freeze=dict(sources={'a':'sha'},environment={'native':'same'})
        runner.write_json(root/'source_freeze.json',freeze);results=[]
        for i in range(1,4):
            name='tiny-startup'+str(i);folder=root/'stages'/name;folder.mkdir(parents=True)
            child=subprocess.Popen([sys.executable,'-c','pass']);child.wait()
            results.append(dict(name=name,status='complete',worker_pid=child.pid))
            receipt=dict(status='passed',worker_pid=child.pid,provider=runner.provider_identity(self.legacy_monitor),
                sources=freeze['sources'],config=self.cfg,environment=freeze['environment'],binary_sha256={'native':'same'},
                precision={'exact':True},checks=[dict(kind=k,cpu_cuda_exact=True) for k in ('raw','gaussian','pca','hybrid')])
            runner.write_json(folder/'device_receipt.json',receipt)
        return results

    def test_gate_requires_three_independent_exited_identical_startups(self):
        for defect in ('none','missing','failed','pid','native','provider','check'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);results=self.gate_fixture(root);path=root/'stages/tiny-startup3/device_receipt.json'
                receipt=json.loads(path.read_text())
                if defect=='missing':results.pop()
                if defect=='failed':results[-1]['status']='worker_failed'
                if defect=='pid':receipt['worker_pid']=results[0]['worker_pid']
                if defect=='native':receipt['binary_sha256']={'native':'changed'}
                if defect=='provider':receipt['provider']={}
                if defect=='check':receipt['checks'][0]['cpu_cuda_exact']=False
                runner.atomic_json(path,receipt)
                if defect=='none':
                    saved=runner.seal_device_gate(root,self.cfg,results,runner.provider_identity(self.legacy_monitor))
                    self.assertEqual(saved['independent_cuda_workers'],3);self.assertEqual(len(saved['startup_receipt_sha256']),3)
                else:
                    with self.assertRaises(RuntimeError):runner.seal_device_gate(root,self.cfg,results,runner.provider_identity(self.legacy_monitor))
                    self.assertFalse((root/'device_receipt.json').exists())

    def test_frozen_old_inventory_training_and_new_limits(self):
        old=json.loads((runner.ROOT/'configs/tls_rag_step5_route6_representation_v1.json').read_text())
        for key in ('representations','seeds','training','budgets','targets','limits','queries','roles_sha256'):
            self.assertEqual(self.cfg[key],old[key])
        self.assertEqual(self.cfg['timing']['cpu_queries_per_role'],16)
        self.assertEqual(self.legacy_monitor['gpu_max_age_seconds'],3)
        self.assertEqual(self.cfg['monitor']['report_seconds'],300)
        self.assertIsNone(self.cfg['monitor']['gpu_max_age_seconds'])
        self.assertFalse(self.cfg['supervision']['automatic_stop'])
        self.assertFalse(self.cfg['tri_stage']);self.assertEqual(self.cfg['new_mc_samples'],0)


class ManualSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cfg=json.loads(runner.CONFIG.read_text())

    def limits(self):
        return dict(self.cfg['limits'],scientific_seconds=-1,total_seconds=-1,component_seconds=-1,
            rss_bytes=1,gpu_bytes=1,artifact_bytes=1,disk_bytes=1,free_bytes=10**30)

    def test_exact_five_minute_config_and_no_deadline_gate(self):
        self.assertEqual(runner.config()['namespace'],'tls-rag-route6-representation-screen-v2-manual-r2')
        self.assertEqual(self.cfg['monitor']['report_seconds'],300)
        self.assertEqual(self.cfg['monitor']['poll_seconds'],300)
        self.assertIsNone(self.cfg['monitor']['gpu_max_age_seconds'])
        self.assertIsNone(self.cfg['device_gate']['total_seconds'])
        self.assertIsNone(self.cfg['device_gate']['scientific_seconds'])
        self.assertEqual(self.cfg['supervision'],dict(mode='manual',automatic_stop=False,
            resource_budget_enforced=False,report_interval_seconds=300))

    def test_observations_only_on_five_minute_schedule_and_no_catchup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);clock=[0.];guard=runner.ManualSupervisor(root,self.limits(),self.cfg['monitor'],start=0,use_gpu=False)
            with mock.patch.object(runner,'monotonic',side_effect=lambda:clock[0]),mock.patch.object(runner,'process_tree',return_value={os.getpid():999}) as scan:
                first=guard.observe()
                for now in (1,10,299.99):clock[0]=now;self.assertIs(guard.observe(),first)
                self.assertEqual(scan.call_count,1)
                clock[0]=300;guard.observe();self.assertEqual(scan.call_count,2)
                clock[0]=950;guard.observe();self.assertEqual(scan.call_count,3)
                self.assertEqual(guard.next_sample,1250)
                self.assertFalse(guard.last_observation['automatic_stop']);self.assertTrue(first['advisories'])
            self.assertEqual(len((root/'observations.jsonl').read_text().splitlines()),3)
        self.assertEqual(runner.next_observation(0,350,300),650)

    def test_unknown_stale_failed_or_unrelated_gpu_is_report_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings=self.cfg['monitor'];nonce='n';pid=123;now=runner.monotonic()
            sample=dict(nonce=nonce,reporter_pid=pid,owner_pid=os.getpid(),sequence=1,
                started_monotonic=now-10000,completed_monotonic=now-9999,
                provider=runner.provider_identity(settings),memory_bytes={'987654321':1024})
            runner.atomic_json(root/'telemetry.json',sample)
            status=dict(sample,status='ok');runner.atomic_json(root/'telemetry_status.json',status)
            value=runner.manual_gpu_observation(root,nonce,pid,[],settings,True)
            self.assertEqual(value['sampled_bytes'],1024);self.assertGreater(value['sample_age_seconds'],9999)
            self.assertEqual(value['unrelated_compute_pids'],[987654321])
            for state,alive in [('pending',True),('error',True),('ok',False)]:
                runner.atomic_json(root/'telemetry_status.json',dict(status,status=state))
                value=runner.manual_gpu_observation(root,nonce,pid,[],settings,alive)
                self.assertEqual(value['status'],'unknown');self.assertIsNone(value['sampled_bytes'])
                self.assertEqual(value['last_known_bytes'],1024)
            runner.atomic_json(root/'telemetry.json',dict(sample,memory_bytes={'1':'N/A'}))
            value=runner.manual_gpu_observation(root,nonce,pid,[],settings,True)
            self.assertIsNone(value['sampled_bytes']);self.assertIsNone(value['last_known_bytes'])

    def test_actual_blocked_native_probe_and_all_old_limits_do_not_kill_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);guard=runner.ManualSupervisor(root,self.limits(),self.cfg['monitor'])
            code="""import ctypes,os,sys
from scripts import run_tls_rag_step5_route6_representation_v2 as r
class Provider:
 def __init__(self,settings):pass
 def query(self):ctypes.CDLL(None).sleep(20);return {}
 def close(self):pass
r.NVMLProvider=Provider
r.manual_reporter(sys.argv[1],sys.argv[2],int(sys.argv[3]),r.config()['monitor'])
"""
            guard.report=subprocess.Popen([sys.executable,'-c',code,str(root),guard.nonce,str(os.getpid())],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            try:
                result=runner.manual_run_job([sys.executable,'-c','import time;time.sleep(4.1)'],root,guard,'past-all-old-limits')
                self.assertEqual(result['status'],'complete',result);self.assertEqual(result['returncode'],0)
                self.assertGreaterEqual(result['seconds'],4.1)
                self.assertEqual(guard.report.poll(),None)
                self.assertEqual(json.loads((root/'telemetry_status.json').read_text())['status'],'pending')
                self.assertTrue(guard.last_observation['advisories'])
            finally:guard.close()
            self.assertIsNotNone(guard.report.poll())

    def test_actual_monitor_exception_does_not_kill_and_worker_error_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard=runner.ManualSupervisor(tmp,self.limits(),self.cfg['monitor'],use_gpu=False)
            with mock.patch.object(guard,'observe',side_effect=RuntimeError('broken monitoring')):
                result=runner.manual_run_job([sys.executable,'-c','import time;time.sleep(.1)'],tmp,guard,'monitor-broken')
            self.assertEqual(result['status'],'complete')
            result=runner.manual_run_job([sys.executable,'-c','raise SystemExit(7)'],tmp,guard,'real-error')
            self.assertEqual(result['status'],'worker_failed');self.assertEqual(result['returncode'],7)

    def test_user_stop_is_not_swallowed_by_observation_error_handling(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard=runner.ManualSupervisor(tmp,self.limits(),self.cfg['monitor'],use_gpu=False)
            with mock.patch.object(guard,'observe',side_effect=runner.UserStopped('operator')):
                result=runner.manual_run_job([sys.executable,'-c','import time;time.sleep(30)'],tmp,guard,'explicit-stop')
            self.assertEqual(result['status'],'user_stopped')
            with self.assertRaises(ProcessLookupError):os.kill(result['worker_pid'],0)

    def test_spawn_preserves_worker_signal_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard=runner.ManualSupervisor(tmp,self.limits(),self.cfg['monitor'],use_gpu=False)
            code='import signal;assert not (set(signal.pthread_sigmask(signal.SIG_BLOCK,[])) & {signal.SIGUSR1,signal.SIGINT,signal.SIGTERM})'
            result=runner.manual_run_job([sys.executable,'-c',code],tmp,guard,'signals')
            self.assertEqual(result['status'],'complete')

    def test_worker_old_deadline_ignored_but_owner_and_source_locks_remain(self):
        freeze=dict(parent_pid=42,deadline_monotonic=-100000,sources={'s':'hash'});last=[0.]
        with mock.patch.object(runner.os,'getppid',return_value=42),mock.patch.object(runner,'monotonic',return_value=100000),mock.patch.object(runner,'source_map',return_value={'s':'hash'}):
            runner.worker_check(freeze,last,self.cfg)
        with mock.patch.object(runner.os,'getppid',return_value=43):
            with self.assertRaisesRegex(RuntimeError,'parent'):runner.worker_check(freeze,last,self.cfg)
        with mock.patch.object(runner.os,'getppid',return_value=42),mock.patch.object(runner,'monotonic',return_value=100301),mock.patch.object(runner,'source_map',return_value={'s':'changed'}):
            with self.assertRaisesRegex(RuntimeError,'source'):runner.worker_check(freeze,last,self.cfg)

    def test_manual_products_exceed_old_budget_but_remain_exclusive_and_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=copy.deepcopy(self.cfg);cfg['limits']['artifact_bytes']=1
            products=screen.products_for(tmp,cfg);products.array('larger.npy',np.zeros(1000))
            self.assertGreater((Path(tmp)/'larger.npy').stat().st_size,1)
            with self.assertRaises(FileExistsError):products.array('larger.npy',np.ones(1))
            with self.assertRaises(ValueError):products.json('bad.json',{'value':float('nan')})
            with self.assertRaises(ValueError):products.reserve(-1)
            outside=Path(tmp)/'outside';outside.mkdir();(Path(tmp)/'link').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(RuntimeError):products.array('link/escaped.npy',np.zeros(1))
            with self.assertRaises(ValueError):products.json('../escaped.json',{})
            self.assertFalse((outside/'escaped.npy').exists())

    def test_no_closure_deadline_and_status_launches_no_extra_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'payload').write_bytes(b'abc')
            value=dict(started_monotonic=runner.monotonic()-10000,time_unix=time.time()-10000,
                automatic_stop=False,gpu=dict(status='unknown',sampled_bytes=None))
            runner.atomic_json(root/'observation.json',value)
            with mock.patch.object(runner,'NVMLProvider') as provider,mock.patch.object(runner,'process_tree') as scan:
                status=runner.status_snapshot(root);self.assertGreater(status['snapshot_age_seconds'],9999)
                provider.assert_not_called();scan.assert_not_called()
            result=runner.close_run(root,dict(status='complete'),runner.monotonic()-100000,None)
            self.assertEqual(result['status'],'complete');self.assertEqual(runner.verify_index(root),3)

    def test_preflight_gpu_probe_timeout_is_advisory(self):
        with mock.patch.object(runner.importlib.metadata,'version',side_effect=lambda n:self.cfg['dependencies'][n]),mock.patch.object(runner.subprocess,'run',side_effect=subprocess.TimeoutExpired('nvidia-smi',30)),mock.patch.object(runner,'sha',return_value='hash'),mock.patch.object(runner.importlib.util,'find_spec',return_value=SimpleNamespace(origin='fixture')),mock.patch.dict(os.environ,{n:'1' for n in runner.THREADS}):
            value=runner.environment(self.cfg)
            self.assertIsNone(value['physical_identity']);self.assertIn('TimeoutExpired',value['gpu_identity_observation_error'])
        other=dict(value,physical_identity='known A800',gpu_identity_observation_error=None)
        self.assertTrue(runner.same_environment(other,value))
        self.assertFalse(runner.same_environment(other,dict(other,physical_identity='different A800')))
        self.assertFalse(runner.same_environment(other,dict(value,threads={})))

    def test_cpu_supervisor_gate_finishes_with_dead_reporter_and_expired_old_limits(self):
        # CPU stand-ins isolate the changed orchestration, not CUDA numerics.
        # Numeric CUDA conformance still requires a separately authorized node run.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'gate';cfg=copy.deepcopy(self.cfg)
            stage_worker="""import sys,os,json
from pathlib import Path
from scripts import run_tls_rag_step5_route6_representation_v2 as r
root=Path(sys.argv[1]);ident=sys.argv[2];freeze=json.loads((root/'source_freeze.json').read_text())
assert freeze['deadline_monotonic'] is None
r.write_json(root/'stages'/('tiny-'+ident)/'device_receipt.json',dict(status='passed',worker_pid=os.getpid(),
 provider=r.provider_identity(freeze['config']['monitor']),sources=freeze['sources'],config=freeze['config'],
 environment=freeze['environment'],checks=[dict(kind=k,cpu_cuda_exact=True) for k in ('raw','gaussian','pca','hybrid')],
 binary_sha256={'cpu_fixture':'same'},precision={'cpu_fixture':True}))
"""
            real_job=runner.manual_run_job
            def job(command,folder,guard,name,env=None):
                self.assertIsInstance(guard,runner.ManualSupervisor)
                ident=command[command.index('--ident')+1]
                return real_job([sys.executable,'-c',stage_worker,str(folder),ident],folder,guard,name,env)
            def fake_reporter(guard):guard.report=SimpleNamespace(pid=999999999,poll=lambda:1)
            def command(args,**kwargs):return SimpleNamespace(stdout='fixture' if args[:2]==['git','rev-parse'] else '')
            identity=dict(pid=os.getpid(),uid=os.getuid(),start_identity='cpu_fixture',commandline=[str(root.resolve())])
            with mock.patch.object(runner.sys,'platform','linux'),mock.patch.object(runner,'process_identity',return_value=identity),mock.patch.object(runner,'process_tree',return_value={os.getpid():100}),mock.patch.object(runner.ManualSupervisor,'start_reporter',fake_reporter),mock.patch.object(runner,'environment',return_value={'cpu_fixture':True}),mock.patch.object(runner,'allocated_resources',side_effect=RuntimeError('unavailable')),mock.patch.object(runner,'source_map',return_value={'cpu_fixture':'same'}),mock.patch.object(runner.subprocess,'run',side_effect=command),mock.patch.object(runner,'manual_run_job',side_effect=job),mock.patch.object(runner,'ENTRY_MONOTONIC',runner.monotonic()-100000):
                result=runner.supervise(root,cfg,device_only=True)
            self.assertEqual(result['status'],'complete',result)
            self.assertFalse(result['automatic_stop']);self.assertEqual(result['report_interval_seconds'],300)
            receipt=json.loads((root/'device_receipt.json').read_text())
            self.assertEqual(receipt['independent_cuda_workers'],3)
            self.assertEqual(len(set(receipt['worker_pids'])),3)
            self.assertEqual(result['sampled_max_gpu_bytes'],None)
            runner.verify_index(root)

    def test_actual_manual_stop_checks_start_identity_and_seals_only_owned_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=(root/'controller.log').open('xb')
            code="""import sys,signal,json
from pathlib import Path
from scripts import run_tls_rag_step5_route6_representation_v2 as r
root=Path(sys.argv[1]);cfg=r.config();previous=r.install_manual_control(root)
guard=r.ManualSupervisor(root,cfg['limits'],cfg['monitor'],use_gpu=False)
result=r.manual_run_job([sys.executable,'-c','import time;time.sleep(30)'],root,guard,'work')
for sig in previous:signal.signal(sig,signal.SIG_IGN)
control=json.loads((root/'manual_control.json').read_text());control['state']='closing';r.atomic_json(root/'manual_control.json',control)
r.close_run(root,dict(status=result['status']),r.monotonic(),None)
"""
            controller=subprocess.Popen([sys.executable,'-c',code,str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            original=None
            try:
                deadline=time.monotonic()+6
                while not (root/'stages/work/owned_worker.json').exists():
                    if controller.poll() is not None or time.monotonic()>deadline:self.fail((root/'controller.log').read_text())
                    time.sleep(.02)
                original=json.loads((root/'manual_control.json').read_text())
                forged=copy.deepcopy(original);forged['controller']['start_identity']='wrong'
                runner.atomic_json(root/'manual_control.json',forged)
                with self.assertRaises(ValueError):runner.request_stop(root)
                self.assertIsNone(controller.poll())
                runner.atomic_json(root/'manual_control.json',original)
                request=runner.request_stop(root);self.assertEqual(request['controller_pid'],controller.pid)
                controller.wait(timeout=5);self.assertEqual(controller.returncode,0,(root/'controller.log').read_text())
                self.assertEqual(json.loads((root/'execution.json').read_text())['status'],'user_stopped')
                worker=json.loads((root/'stages/work/owned_worker.json').read_text())['pid']
                with self.assertRaises(ProcessLookupError):os.kill(worker,0)
                runner.verify_index(root)
                with self.assertRaisesRegex(ValueError,'closed'):runner.request_stop(root)
            finally:
                if controller.poll() is None:
                    if original:runner.atomic_json(root/'manual_control.json',original)
                    try:runner.request_stop(root);controller.wait(timeout=3)
                    finally:
                        if controller.poll() is None:runner.terminate_owned(controller)
                log.close()

if __name__=='__main__':unittest.main()
