"""Offline contracts and actual native continuation tests; no network/data roles."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from mechanism.tls_rag_step5_ann_continuation_v1 import (config,sources,sha,save,Native,
    Session,Runtime,gap_policy,gap_methods,stopping_oracle,OpenedContinuationInputs,seal_selection)


class ContinuationContracts(unittest.TestCase):
    def test_only_route2_first_trial_enabled(self):
        s=config();self.assertEqual(s['active_routes'],[2]);self.assertEqual(s['return_pool'],64)
        for key in ['predictor_training','adapter_training','paid_tri','scale_1m','formal_certification']:
            self.assertFalse(s[key])
        self.assertEqual(s['checkpoints'],[128,256,512,1024,2048,4096])

    def test_gap_policy_uses_only_past_observations(self):
        s=dict(done=0,full_distances=1024,frontier_min=1.2,result_threshold=1.,result_updates=10)
        self.assertTrue(gap_policy(s,10,.1,0,512));self.assertFalse(gap_policy(s,9,.1,0,512))
        self.assertFalse(gap_policy(s,10,.1,0,2048))
        for key,value in [('done',1),('frontier_min',None),('result_threshold',0)]:
            d=dict(s);d[key]=value;self.assertFalse(gap_policy(d,10,.1,0,512))
        with self.assertRaises(TypeError):gap_policy(s,10,.1,0,512,truth=1)
        self.assertEqual(len(gap_methods(config())),45)

    def test_quality_constrained_oracle_is_not_query_independent_mixture(self):
        s=dict(config(),quality_tolerance=dict(retention_drop=0,hit_drop=0))
        v=stopping_oracle([[1,1],[0,1]],[[1,1],[0,1]],[1.,4.],[1.,1.],s)
        self.assertAlmostEqual(v['oracle_ms'],2.5);self.assertAlmostEqual(v['mixture_ms'],4)
        self.assertAlmostEqual(v['same_endpoint_ms'],2.5);self.assertFalse(v['deployed'])
        np.testing.assert_allclose(v['oracle_weights'],[[1,0],[0,1]])

    def test_oracle_prices_conservative_monotone_and_bad_cohort_rejected(self):
        v=stopping_oracle([[1,1]],[[1,1]],[4.,1.],[1.,1.],config())
        self.assertEqual(v['prices_raw'],[4.,1.]);self.assertEqual(v['prices_monotone'],[4.,4.])
        with self.assertRaises(ValueError):stopping_oracle([[np.nan,1]],[[1,1]],[1,2],[1,1],config())
        with self.assertRaises(ValueError):stopping_oracle([[0,0]],[[0,0]],[1,2],[1,1],config())

    def test_new_report_requires_selection_and_source_seal(self):
        x=object.__new__(OpenedContinuationInputs);x.binding='toy';x._seal=None
        x.groups=dict(select=['s'],report=['r']);x._rows={'s':{'query_id':'s'},'r':{'query_id':'r'}}
        x._truth={'s':['a'],'r':['b']}
        with self.assertRaises(ValueError):x.records('report')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'selection.json';binding=seal_selection(p,x,dict(best_gap='checkpoints'))
            x.open_report(p,binding);self.assertEqual(x.records('report')[0]['truth'],['b'])
            with self.assertRaises(ValueError):x.open_report(p,'wrong')
            v=json.loads(p.read_text());v['sources']['future']='x';p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):x.open_report(p,binding)


@unittest.skipUnless(os.environ.get('STEP5_CONTINUATION_NATIVE'),'requires compiled fixed-version route2 shim')
class ContinuationNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import faiss
        cls.s=config();cls.n=Native(os.environ['STEP5_CONTINUATION_NATIVE'],cls.s)
        rng=np.random.default_rng(55821);cls.x=rng.normal(size=(1200,384));cls.x/=np.linalg.norm(cls.x,axis=1)[:,None]
        cls.x[-1]=cls.x[0]
        cls.q=rng.normal(size=(5,384));cls.q/=np.linalg.norm(cls.q,axis=1)[:,None]
        cls.index=faiss.IndexHNSWFlat(384,16);cls.index.hnsw.efConstruction=200;cls.index.hnsw.efSearch=256
        cls.index.add(cls.x.astype(np.float32))

    def compare(self,q):
        with Session(self.n,self.index,q,debug=True) as session:
            session.advance();return session.snapshot()

    def test_uninterrupted_matches_faiss_bitwise(self):
        d,i=self.index.search(self.q.astype(np.float32),64)
        for j,q in enumerate(self.q):
            sd,si,s=self.compare(q);np.testing.assert_array_equal(sd,d[j]);np.testing.assert_array_equal(si,i[j])
            self.assertTrue(s['done']);self.assertGreater(s['full_distances'],64)

    def test_many_pauses_preserve_entire_call_sequence_and_final_state(self):
        for q in self.q:
            d,i,s=self.compare(q)
            with Session(self.n,self.index,q,debug=True) as session:
                for target in [1,2,65,127,128,129,256,511,512,1025,2048]:
                    v=session.advance(target);sd,si,v=session.snapshot()
                    self.assertEqual(len(set(si)),64)
                session.advance();sd,si,v=session.snapshot()
            np.testing.assert_array_equal(d,sd);np.testing.assert_array_equal(i,si)
            for key in ['full_distances','upper_distances','expanded','neighbor_slots','batch4_calls',
                'scalar_calls','distance_sequence','state_hash']:
                self.assertEqual(s[key],v[key],key)

    def test_snapshot_is_copy_and_does_not_reorder_live_heap(self):
        with Session(self.n,self.index,self.q[0],debug=True) as session:
            session.advance(512);d,i,s=session.snapshot();d[:]=-99;i[:]=-99
            rd,ri,t=session.snapshot();self.assertEqual(s['state_hash'],t['state_hash'])
            self.assertTrue((rd>=0).all());self.assertTrue((ri>=0).all())
            session.advance();fd,fi,_=session.snapshot()
        d,i,_=self.compare(self.q[0]);np.testing.assert_array_equal(fd,d);np.testing.assert_array_equal(fi,i)

    def test_absolute_budget_and_complete_batch_accounting(self):
        with Session(self.n,self.index,self.q[0],debug=True) as session:
            s=session.advance(512);self.assertTrue(s['done'] or 512<=s['full_distances']<=515)
            self.assertEqual(s['full_distances'],4*s['batch4_calls']+s['scalar_calls'])
            t=session.advance(128);self.assertEqual(t['full_distances'],s['full_distances'])
            self.assertEqual(t['state_hash'],s['state_hash'])

    def test_natural_end_cannot_be_reopened(self):
        with Session(self.n,self.index,self.q[0],debug=True) as session:
            s=session.advance();d,i,s=session.snapshot();t=session.advance(10**6)
            self.assertTrue(t['done']);self.assertEqual(s['state_hash'],t['state_hash'])
            self.assertEqual(s['full_distances'],t['full_distances'])
            nd,ni,_=session.snapshot();np.testing.assert_array_equal(d,nd);np.testing.assert_array_equal(i,ni)

    def test_query_memory_is_owned_and_sessions_are_independent(self):
        q=self.q[0].copy()
        with Session(self.n,self.index,q) as a,Session(self.n,self.index,self.q[1]) as b:
            q[:]=100;a.advance(128);b.advance(256);a.advance();b.advance()
            ad,ai,_=a.snapshot();bd,bi,_=b.snapshot()
        for j,d,i in [(0,ad,ai),(1,bd,bi)]:
            rd,ri,_=self.compare(self.q[j]);np.testing.assert_array_equal(d,rd);np.testing.assert_array_equal(i,ri)

    def test_zero_distance_duplicates_and_gap_boundary(self):
        d,i,s=self.compare(self.x[0]);self.assertEqual(d[0],0);self.assertIn(0,i);self.assertIn(len(self.x)-1,i)
        with Session(self.n,self.index,self.x[0]) as session:
            session.advance(1);d,i,s=session.snapshot();self.assertTrue(np.isfinite(d).all())

    def test_invalid_input_closed_state_and_future_information_rejected(self):
        for q in [np.zeros(383),np.full(384,np.nan)]:
            with self.assertRaises(ValueError):Session(self.n,self.index,q)
        with patch('sys.unraisablehook') as hook:
            with self.assertRaises(TypeError):Session(self.n,self.index,self.q[0],truth=[])
            hook.assert_not_called()
        session=Session(self.n,self.index,self.q[0])
        for target in [-1,True,1.5]:
            with self.assertRaises(ValueError):session.advance(target)
        session.close();session.close()
        with self.assertRaises(ValueError):session.advance()
        with self.assertRaises(ValueError):session.snapshot()

    def test_runtime_policy_and_independent_worker_match(self):
        import faiss
        path=Path(__file__).parents[1]/'scripts/run_tls_rag_step5_ann_continuation_v1.py'
        loader=importlib.util.spec_from_file_location('continuation_test_runner',path)
        runner=importlib.util.module_from_spec(loader);loader.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);np.save(root/'corpus.npy',self.x);save(root/'ids.json',['d'+str(i) for i in range(len(self.x))])
            faiss.write_index(self.index,str(root/'index.faiss'))
            shutil.copyfile(os.environ['STEP5_CONTINUATION_NATIVE'],root/'continuation.dylib')
            paths=dict(index=str(root/'index.faiss'),index_sha=sha(root/'index.faiss'),corpus=str(root/'corpus.npy'),ids=str(root/'ids.json'))
            rows=[dict(query_id=str(j),query=q.tolist(),truth=['d0'],qrels={}) for j,q in enumerate(self.q[:2])]
            inputs=SimpleNamespace(paths=paths,groups={'synthetic':[r['query_id'] for r in rows]},records=lambda group:rows)
            trial=runner.Trial(inputs,self.n,root,self.s,perf_counter());trial.prepare('synthetic')
            methods=['original:256','native','checkpoints','fixed:128','path:512','gap:0.0:4:512']
            trial.score('synthetic',methods);trial.measure('synthetic',methods)
            self.assertEqual(len(trial.timings),36);self.assertEqual(len(trial.traces),14)
            self.assertGreater(trial.summary('synthetic','native')['mean_ms'],0)
            for method in ['path:300','original:1','gap:nan:4:512','fixed:300','bad']:
                with self.assertRaises(ValueError):trial.runtime.call(self.q[0],method)
            trial.runtime.close()


if __name__=='__main__':unittest.main()
