"""Independent CPU fixtures for the approved S0 contract; no network or CUDA."""
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import faiss
import h5py
import numpy as np

from mechanism import tls_rag_step5_route6_representation_v1 as rep
from mechanism import tls_rag_step5_route6_representation_screen_v1 as screen
from scripts import run_tls_rag_step5_route6_representation_v1 as runner


class RepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        faiss.omp_set_num_threads(1)
        cls.cfg=json.loads(runner.CONFIG.read_text())

    def fixture(self,root):
        x=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]],np.float32)
        q=np.array([[1.,0.],[0.,1.]],np.float32)
        source=root/'source.h5'
        with h5py.File(source,'w') as f:
            f.attrs['distance']='normalized';f.attrs['dimension']=2
            f['train']=x;f['learn']=q;f['learn_neighbors']=np.array([[0,1],[1,0]],np.uint32)
            for key in ('test','neighbors','distances','avg_distances'):f[key]=[999]
        source_sha=rep.sha(source)
        roles=[dict(row=i,id=rep.stable_id(source_sha,'learn',i),vector_sha256=rep.canonical_vector_hash(q[i]),
                    role='geometry_discovery' if i==0 else 'geometry_check') for i in range(2)]
        role_path=root/'roles.json';role_path.write_text(json.dumps(dict(queries=roles)))
        cfg=copy.deepcopy(self.cfg);cfg.update(queries=2,discovery_queries=1,k=2,block=2,roles_path=str(role_path),roles_sha256=rep.sha(role_path))
        cfg['source'].update(path=str(source),rows=4,dimension=2,sha256=source_sha,bytes=source.stat().st_size)
        return cfg,x,q,roles

    def test_original_metric_is_not_stored_l2(self):
        q=np.array([1.,0.],np.float32)
        x=np.array([[1.+1e-6,0.],[1.,0.]],np.float32)
        self.assertEqual(rep.stable_top(rep.original_distances(x,q),1).tolist(),[0])
        self.assertEqual(rep.stable_top(np.sum((x-q)**2,axis=1),1).tolist(),[1])
        np.testing.assert_array_equal(x[0],[np.float32(1.+1e-6),0.])

    def test_distance_threshold_and_strict_id_are_separate(self):
        result=screen.benchmark_recall([.4,.5009],.5,[0,1],[0,2],2)
        self.assertEqual(result,dict(recall=1.,strict_id=.5))
        self.assertEqual(screen.benchmark_recall([.4,.5011],.5,[0,1],[0,2],2)['recall'],.5)
        with self.assertRaises(ValueError):screen.benchmark_recall([.1,.2],.5,[0,1],[0,0],2)

    def test_partial_selection_cutoff_ties_canonical_ids(self):
        ids=np.array([7,2,8,4,9]);scores=np.array([1.,0.,1.,0.,2.])
        self.assertEqual(rep.stable_top(scores,3,ids).tolist(),[2,4,7])
        for count in range(1,len(scores)+1):
            self.assertEqual(rep.stable_top(scores,count,ids).tolist(),ids[np.lexsort((ids,scores))[:count]].tolist())
        with self.assertRaises(ValueError):rep.stable_top([float('nan')],1)

    def test_rerank_keeps_duplicate_vectors_with_distinct_ids(self):
        x=np.array([[1.,0.],[1.,0.],[0.,1.]],np.float32)
        result,distances=rep.rerank(x,x[0],[2,1,0],2)
        self.assertEqual(result.tolist(),[0,1]);self.assertEqual(distances.tolist(),[0,0])
        with self.assertRaises(ValueError):rep.rerank(x,x[0],[0,0],2)

    def test_only_approved_payload_rows_are_read(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg,x,q,roles=self.fixture(Path(temp));read=[]
            real=rep.read_payload
            def spy(f,name,row=None):read.append((name,row));return real(f,name,row)
            with mock.patch.object(rep,'read_payload',side_effect=spy):
                actual=rep.load_screen_inputs(cfg)
            np.testing.assert_array_equal(actual[0],x);np.testing.assert_array_equal(actual[1],q)
            self.assertEqual(actual[3],roles)
            self.assertEqual({name for name,_ in read},{'train','learn','learn_neighbors'})
            self.assertEqual([row for name,row in read if name=='learn_neighbors'],[0,1])
            with self.assertRaises(ValueError):rep.read_payload({},'test',0)
            with self.assertRaises(ValueError):rep.read_payload({},'learn')

    def test_source_roles_and_truth_mutations_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg,*_=self.fixture(Path(temp));bad=copy.deepcopy(cfg);bad['roles_sha256']='0'*64
            with self.assertRaises(ValueError):rep.load_screen_inputs(bad)
            with h5py.File(cfg['source']['path'],'r+') as f:f['learn_neighbors'][0]=[0,0]
            cfg['source']['sha256']=rep.sha(cfg['source']['path'])
            cfg['source']['bytes']=Path(cfg['source']['path']).stat().st_size
            # source identity changes also invalidate stable query IDs.
            with self.assertRaises(ValueError):rep.load_screen_inputs(cfg)

    def test_sample_is_corpus_only_sorted_seeded(self):
        rows=rep.sample_fit_rows(100,20,3)
        self.assertEqual(len(set(rows)),20);self.assertTrue(np.all(np.diff(rows)>0))
        np.testing.assert_array_equal(rows,rep.sample_fit_rows(100,20,3))
        with self.assertRaises(ValueError):rep.sample_fit_rows(3,4,3)

    def test_pca_centering_cancels_pair_differences(self):
        x=np.array([[1.,1.,0.],[4.,1.,0.],[3.,2.,0.],[3.,0.,0.]],np.float32)
        pca=rep.fit_pca(x);spec=dict(kind='pca',dim=2,id='fixture')
        model=rep.fit_representation(spec,x,pca,self.cfg)
        z=rep.transform_batch(model,x)
        np.testing.assert_allclose(z[0]-z[1],(x[0]-x[1])@pca['basis'][:,:2],atol=1e-7)
        np.testing.assert_allclose(z.mean(axis=0),0,atol=1e-7)
        self.assertFalse(np.allclose(np.linalg.norm(z,axis=1),1))

    def test_gaussian_scale_prefix_and_no_post_normalization(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(512,640)).astype(np.float32);pca=rep.fit_pca(x[:8])
        models=[rep.fit_representation(dict(kind='gaussian',id=str(m),dim=m,seed=2026110353),x,pca,self.cfg) for m in (128,256)]
        a,b=models
        self.assertAlmostEqual(float(np.var(a['matrix'])),1/128,delta=.0002)
        np.testing.assert_allclose(a['matrix']*np.sqrt(128),b['matrix'][:128]*16,atol=3e-7)
        self.assertFalse(np.allclose(np.linalg.norm(rep.transform_batch(a,x[:4]),axis=1),1))

    def test_hybrid_residual_annihilates_retained_space_and_counts_bytes(self):
        x=np.random.default_rng(8).normal(size=(64,16)).astype(np.float32);pca=rep.fit_pca(x)
        model=rep.fit_representation(dict(kind='hybrid',id='fixture',dim=8,retained=2,seed=7),x,pca,self.cfg)
        np.testing.assert_allclose(model['matrix'][2:]@pca['basis'][:,:2],0,atol=3e-8)
        self.assertEqual(model['matrix'].shape,(8,16));self.assertEqual(rep.transform_batch(model,x).nbytes,len(x)*8*4)

    def test_independent_adc_decode_hand_fixture(self):
        centers=np.zeros((2,256,2),np.float32)
        centers[0,1]=[1,2];centers[1,1]=[-1,3]
        codes=np.array([[0,0],[1,0],[0,1],[1,1]],np.uint8);q=np.array([.5,1.,-1.,1.])
        decoded=np.array([[0,0,0,0],[1,2,0,0],[0,0,-1,3],[1,2,-1,3]],float)
        np.testing.assert_array_equal(rep.decode_codes(codes,centers),decoded)
        np.testing.assert_array_equal(rep.adc_scores(q,codes,centers),np.sum((decoded-q)**2,axis=1))
        np.testing.assert_array_equal(rep.adc_scores(q,codes,centers,'ip'),-decoded@q)

    def test_faiss_boundary_ties_grow_beyond_m_plus_one(self):
        x=np.zeros((20,4),np.float32);idx=faiss.IndexFlatL2(4);idx.add(x)
        self.assertEqual(rep.faiss_search(idx,x[0],5,'l2').tolist(),list(range(5)))

    def test_fp64_reference_matches_independent_original_and_adc(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(20,4)).astype(np.float32);q=rng.normal(size=(3,4)).astype(np.float32)
        expected=np.array([np.lexsort((np.arange(len(x)),np.sum((x.astype(float)-z.astype(float))**2,axis=1)))[:5] for z in q])
        np.testing.assert_array_equal(rep.exact_representation_search(x,q,5),expected)

    def test_ivf_fullprobe_uses_same_codes_and_restores_probe(self):
        rng=np.random.default_rng(17);x=rng.normal(size=(512,8)).astype(np.float32)
        index=faiss.IndexIVFPQ(faiss.IndexFlatL2(8),8,4,2,8)
        index.cp.niter=2;index.pq.cp.niter=2;index.cp.seed=7;index.pq.cp.seed=9
        index.train(x);index.add(x);index.nprobe=1
        reference=rep.ivf_same_code_reference(index,x[:3],10)
        self.assertEqual(index.nprobe,1)
        decoded=rep.IVFDecodedView(index)[:len(x)]
        for qi in range(3):
            expected=np.argsort(np.sum((decoded-x[qi])**2,axis=1))[:10]
            self.assertEqual(set(reference[qi]),set(expected))
        view=rep.IVFDecodedView(index)
        np.testing.assert_allclose(view[0:4],index.reconstruct_n(0,4),atol=3e-7)
        with self.assertRaises(ValueError):view[1]

    def test_actual_fit_encode_and_opq_rotation(self):
        x=np.random.default_rng(2).normal(size=(512,16)).astype(np.float32)
        cfg=copy.deepcopy(self.cfg);cfg['fit_rows']=512
        pca=rep.fit_pca(x)
        model=rep.fit_representation(dict(id='tinyOPQ',kind='opq',dim=16,pq_bytes=2),x,pca,cfg)
        np.testing.assert_allclose(model['matrix']@model['matrix'].T,np.eye(16),atol=2e-6)
        codes,centers=rep.encode_pq(model,x,block=200)
        self.assertEqual(codes.shape,(512,2))
        q=rep.transform_batch(model,x[:1])[0]
        expected=rep.stable_top(rep.adc_scores(q,codes,centers,'ip'),10)
        self.assertEqual(set(rep.faiss_search(model['index'],q,10,'ip')),set(expected))

    def test_actual_make_ivf_matches_own_reference(self):
        x=np.random.default_rng(12).normal(size=(512,8)).astype(np.float32)
        cfg=copy.deepcopy(self.cfg);cfg['ivf']['nlist']=4
        cfg['training']['niter']=2;cfg['block']=200
        model=rep.fit_representation(dict(id='tiny',kind='raw',dim=8),x,rep.fit_pca(x),cfg)
        index=rep.make_ivf(model,x,x,cfg)
        result=rep.ivf_same_code_reference(index,x[:2],10)
        expected=rep.exact_representation_search(x,x[:2],10,'ip')
        np.testing.assert_array_equal(result,expected)

    def test_real_component_deadline_and_free_disk_stop(self):
        for kind in ('component','free'):
            with tempfile.TemporaryDirectory() as temp:
                if kind=='component':
                    (Path(temp)/'phase.json').write_text(json.dumps(dict(name='train',component_deadline_unix=time.time()-1)))
                    result=self.monitor(temp,'import time;time.sleep(30)')
                    self.assertEqual(result['reason'],'component_seconds:train')
                else:
                    result=self.monitor(temp,'import time;time.sleep(30)',free_bytes=10**30)
                    self.assertEqual(result['reason'],'free_bytes')

    def test_byte_matched_and_seed_diagnostics_selection_rules(self):
        configs={r['id']:r for r in self.cfg['representations']}
        self.assertEqual({s['dim'] for s in configs.values() if s.get('byte_matched')},{8,16})
        self.assertEqual(self.cfg['training'],dict(niter=25,nredo=1,opq_outer=50,opq_first=40,opq_later=4))
        self.assertEqual(self.cfg['ivf']['max_indices'],7)
        self.assertEqual(self.cfg['seeds']['hybrid_domain'],1)

    def test_search_interface_is_label_free(self):
        import inspect
        parameters=inspect.signature(rep.search_backend).parameters
        self.assertFalse(set(parameters)&{'truth','recall','oracle','cutoff','labels'})
        q=np.array([1.,0.],np.float32);x=np.array([[1.,0.],[0.,1.],[-1.,0.]],np.float32)
        m=dict(spec=dict(kind='raw'),metric='ip');idx=faiss.IndexFlatIP(2);idx.add(x)
        self.assertEqual(rep.search_backend(m,q,2,index=idx).tolist(),[0,1])

    def test_fingerprints_reject_cross_geometry_cache(self):
        a=rep.fingerprint(dict(seed=1,normalize=False));b=rep.fingerprint(dict(seed=2,normalize=False))
        with self.assertRaises(ValueError):rep.require_fingerprint(a,b)
        rep.require_fingerprint(a,a)

    def test_transitions_allow_rescue_and_signed_change(self):
        t=screen.loss_transitions([0,1,2],[0,3],[1,2])
        self.assertEqual(t['lost'],[0]);self.assertEqual(t['rescued'],[1,2])
        self.assertEqual(t['signed_retention_change'],1/3)
        self.assertEqual(screen.loss_transitions([0,1,2],[1,2],[0,3])['signed_retention_change'],-1/3)

    def test_lifecycle_keeps_raw_and_transfer_cost(self):
        result=screen.lifecycle_cost(5.,2.,100,10,index_bytes=8,transfer_seconds=.3,horizons=(1,10))
        self.assertEqual(result['resident_bytes'],118);self.assertEqual(result['horizons'],{'1':7.,'10':25.})
        self.assertEqual(result['transfer_seconds_included'],.3)
        self.assertEqual(screen.break_even(10,1,0,2),10)
        with self.assertRaises(ValueError):screen.lifecycle_cost(0,1,0,10)

    def test_lp_is_optimistic_against_enumerated_integer_oracle(self):
        a=np.array([[.2,.9],[.5,1.]]);t=np.array([[1.,3.],[1.,4.]])
        target=.8;feasible=[]
        for choices in itertools.product(range(2),repeat=2):
            if np.mean([a[q,c] for q,c in enumerate(choices)])>=target:
                feasible.append(np.mean([t[q,c] for q,c in enumerate(choices)]))
        answer=screen.oracle_budget_bound(a,t,target,solver_seconds=2)
        self.assertTrue(answer['lp']['optimal']);self.assertTrue(answer['milp']['optimal'])
        self.assertAlmostEqual(answer['milp']['objective_seconds'],min(feasible))
        self.assertLessEqual(answer['lp']['objective_seconds'],min(feasible)+1e-10)
        self.assertLessEqual(answer['lp']['objective_seconds'],answer['fixed_mixture']['seconds']+1e-10)

    def test_infeasible_oracle_is_not_success_or_interpolation(self):
        result=screen.oracle_budget_bound([[.1,.2]],[[2.,3.]],.95,solver_seconds=1)
        self.assertFalse(result['lp']['optimal']);self.assertIsNone(result['lp']['objective_seconds'])
        self.assertIsNone(result['best_same_pipeline_fixed'])
        with self.assertRaises(ValueError):screen.oracle_budget_bound([[1]],[[float('nan')]],.9)

    def test_discovery_selection_never_retunes_on_check(self):
        base=dict(metric='recall',quality=.99,resident_bytes=100,budget=100)
        points=[dict(base,id='a',pipeline='a',role='geometry_discovery',seconds=2),
                dict(base,id='b',pipeline='b',role='geometry_discovery',seconds=3),
                dict(base,id='a',pipeline='a',role='geometry_check',seconds=9),
                dict(base,id='b',pipeline='b',role='geometry_check',seconds=.1)]
        self.assertEqual(screen.select_fixed(points,.95)['id'],'a')
        frontier=screen.frontiers(points,self.cfg)[0]
        self.assertEqual(frontier['frozen_check_audit']['id'],'a')
        self.assertFalse(frontier['check_reselection'])

    def test_incomplete_is_unresolved_not_negative_science(self):
        result=screen.seal_report(['a'],['a','b'])
        self.assertEqual(result['status'],'incomplete');self.assertEqual(result['missing'],['b'])
        self.assertFalse(result['missing_is_negative_evidence']);self.assertFalse(result['tri_validated'])

    def test_bounded_products_refuse_oversize_and_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p=screen.Products(temp,100)
            with self.assertRaises(RuntimeError):p.array('large.npy',np.zeros(100))
            self.assertFalse((Path(temp)/'large.npy').exists())
            p.json('small.json',dict(a=1))
            with self.assertRaises(FileExistsError):p.json('small.json',dict(a=2))

    def monitor(self,root,code,**changes):
        limits=dict(total_seconds=30,scientific_seconds=30,rss_bytes=8*2**30,gpu_bytes=2**30,
                    artifact_bytes=2**20,disk_bytes=2**20,free_bytes=0)
        limits.update(changes)
        return runner.monitor_worker([sys.executable,'-c',code],root,limits,use_gpu=False)

    def test_actual_owned_worker_timeout_and_closed_index(self):
        with tempfile.TemporaryDirectory() as temp:
            result=self.monitor(temp,'import time;time.sleep(30)',total_seconds=5.3,scientific_seconds=5.3)
            self.assertEqual(result['status'],'resource_stop')
            pid=json.loads((Path(temp)/'owned_worker.json').read_text())['pid']
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)
            self.assertGreater(runner.verify_index(temp),1)

    def test_actual_owned_worker_rss_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            result=self.monitor(temp,'import time;x=bytearray(1000000);time.sleep(30)',rss_bytes=1)
            self.assertEqual(result['reason'],'rss_bytes');self.assertEqual(result['status'],'resource_stop')

    def test_actual_owned_worker_artifact_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            path=str(Path(temp)/'large')
            result=self.monitor(temp,'import time;open('+repr(path)+',"wb").write(b"a"*10000);time.sleep(30)',artifact_bytes=1000)
            self.assertEqual(result['reason'],'artifact_bytes')

    def test_monitor_failure_stops_child_and_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(runner,'process_tree',side_effect=RuntimeError('monitor broken')):
                result=self.monitor(temp,'import time;time.sleep(30)')
            self.assertEqual(result['status'],'monitor_failed')
            self.assertIn('monitor broken',result['monitor_error'])

    def test_closed_index_rejects_extra_file_and_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'a').write_text('x');runner.close_index(root)
            self.assertEqual(runner.verify_index(root),1)
            (root/'a').write_text('y')
            with self.assertRaises(ValueError):runner.verify_index(root)

    def test_measure_call_includes_synchronization(self):
        events=[]
        result,seconds=screen.measure_call(lambda:events.append('call') or 7,lambda:events.append('sync'))
        self.assertEqual(events,['sync','call','sync']);self.assertEqual(result,7);self.assertGreaterEqual(seconds,0)

    def test_frozen_inventory_budgets_and_approval_caps(self):
        specs=self.cfg['representations']
        self.assertEqual(len(specs),24);self.assertEqual(sum(s['diagnostic'] for s in specs),4)
        self.assertEqual(sum(s.get('byte_matched',False) for s in specs),4)
        self.assertEqual(self.cfg['budgets'],[100,200,400,800,1600,3200,6400])
        self.assertEqual(self.cfg['limits']['total_seconds'],9000)
        self.assertEqual(self.cfg['limits']['scientific_seconds'],7200)
        self.assertEqual(self.cfg['limits']['rss_bytes'],64*2**30)
        self.assertFalse(self.cfg['tri_stage'])


if __name__=='__main__':unittest.main()
