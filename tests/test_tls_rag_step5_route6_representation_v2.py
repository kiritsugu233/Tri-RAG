"""CPU-only v2 fixtures: source layout, cold costs, paired roles and real stops."""
import copy
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


class RepresentationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        faiss.omp_set_num_threads(1);cls.cfg=json.loads(runner.CONFIG.read_text())

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
        value=dict(nonce='n',reporter_pid=123,owner_pid=os.getpid(),sequence=1,
            started_monotonic=10.,completed_monotonic=10.1,memory_bytes={})
        value.update(changes);runner.atomic_json(Path(root)/'telemetry.json',value)

    def test_telemetry_rejects_stale_unknown_malformed_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.sample(tmp);self.assertEqual(runner.telemetry_memory(tmp,'n',123,[],self.cfg['monitor'],now=11),0)
            with self.assertRaises(RuntimeError):runner.telemetry_memory(tmp,'n',123,[],self.cfg['monitor'],now=13.01)
            for changes in [dict(nonce='bad'),dict(memory_bytes={'999999':1}),dict(memory_bytes={'1':'N/A'}),dict(completed_monotonic=20)]:
                self.sample(tmp,**changes)
                with self.assertRaises(RuntimeError):runner.telemetry_memory(tmp,'n',123,[],self.cfg['monitor'],now=11)

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

    def test_fresh_retired_owned_sample_does_not_authorize_live_external_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.sample(tmp,memory_bytes={str(os.getpid()):1024})
            with self.assertRaises(RuntimeError):
                runner.telemetry_memory(tmp,'n',123,[],self.cfg['monitor'],now=11,retired_owned=[os.getpid()])

    def limits(self,**changes):
        value=dict(total_seconds=30,scientific_seconds=30,component_seconds=1200,rss_bytes=8*2**30,
            gpu_bytes=2**30,artifact_bytes=10**7,disk_bytes=10**7,free_bytes=0)
        value.update(changes);return value

    def job(self,root,guard,name,code):
        return runner.run_job([sys.executable,'-c',code],root,guard,name)

    def test_actual_component_stop_then_next_owned_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);guard=runner.Guard(root,self.limits(),self.cfg['monitor'],use_gpu=False)
            code='import json,os,time;from pathlib import Path;Path('+repr(str(root/'phase.json'))+').write_text(json.dumps(dict(name="train",worker_pid=os.getpid(),component_deadline_monotonic=time.clock_gettime(time.CLOCK_MONOTONIC)+.1)));time.sleep(30)'
            first=self.job(root,guard,'train',code);self.assertEqual(first['status'],'component_stop')
            with self.assertRaises(ProcessLookupError):os.kill(first['worker_pid'],0)
            scientific_start=guard.scientific_start
            second=self.job(root,guard,'next','pass');self.assertEqual(second['status'],'complete')
            self.assertEqual(guard.scientific_start,scientific_start)

    def test_actual_rss_and_artifact_stops(self):
        for resource in ('rss_bytes','artifact_bytes'):
            with tempfile.TemporaryDirectory() as tmp:
                guard=runner.Guard(tmp,self.limits(**{resource:1}),self.cfg['monitor'],use_gpu=False)
                # Exercise post-spawn enforcement separately from the preflight
                # gate, which would already reject this deliberately tiny cap.
                with mock.patch.object(guard,'ready'):
                    result=self.job(tmp,guard,'stop','import time;time.sleep(30)')
                self.assertEqual(result['status'],'resource_stop');self.assertEqual(result['reason'],resource)

    def test_actual_gpu_query_above_old_timeout_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            command=[sys.executable,'-c','import time;time.sleep(.9);print("")']
            guard=runner.Guard(tmp,self.limits(),self.cfg['monitor'],query_command=command)
            try:
                guard.start_reporter();guard.ready()
                result=self.job(tmp,guard,'work','import time;time.sleep(1.1)')
                self.assertEqual(result['status'],'complete')
            finally:guard.close()

    def test_global_deadline_works_while_gpu_query_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);flag=root/'first'
            code='from pathlib import Path;import time;p=Path('+repr(str(flag))+');first=not p.exists();p.touch();time.sleep(0 if first else 10);print("")'
            guard=runner.Guard(root,self.limits(total_seconds=2),self.cfg['monitor'],query_command=[sys.executable,'-c',code])
            try:
                guard.start_reporter();guard.ready();tick=time.monotonic()
                result=self.job(root,guard,'blocked','import time;time.sleep(30)')
                self.assertEqual(result['reason'],'total_seconds');self.assertLess(time.monotonic()-tick,3)
            finally:guard.close()

    def test_actual_reporter_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard=runner.Guard(tmp,self.limits(),self.cfg['monitor'],query_command=[sys.executable,'-c','print("N/A")'])
            try:
                guard.start_reporter()
                with self.assertRaises(runner.MonitorFailure):guard.ready()
                self.assertFalse((Path(tmp)/'telemetry.json').exists())
            finally:guard.close()

    def test_monitor_exception_stops_owned_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard=runner.Guard(tmp,self.limits(),self.cfg['monitor'],use_gpu=False)
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

    def test_frozen_old_inventory_training_and_new_limits(self):
        old=json.loads((runner.ROOT/'configs/tls_rag_step5_route6_representation_v1.json').read_text())
        for key in ('representations','seeds','training','budgets','targets','limits','queries','roles_sha256'):
            self.assertEqual(self.cfg[key],old[key])
        self.assertEqual(self.cfg['timing']['cpu_queries_per_role'],16)
        self.assertEqual(self.cfg['monitor']['gpu_max_age_seconds'],3)
        self.assertFalse(self.cfg['tri_stage']);self.assertEqual(self.cfg['new_mc_samples'],0)


if __name__=='__main__':unittest.main()
