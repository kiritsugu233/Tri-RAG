"""Section38 proof checks, native kernels, input/cost isolation, real monitor."""
import contextlib
import hashlib
import inspect
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from mechanism import tls_rag_step5_route4_scenarios_v1 as core
from mechanism import tls_rag_step5_route4_advisor_v1 as advisor
from scripts import run_tls_rag_step5_route4_scenarios_v1 as runner


class ScenarioContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s=core.config()
        cls.s.update(d=8,k=2,parent_n=256,banks=[[64,2],[256,1]],gaussian_dimensions=[2,4],
                     pca_dimensions=[2,4],budgets=[4,8,16],pq_subquantizers=2,pq_bits=[2],pq_iterations=2,
                     timing_repeats=2,role_sizes={'advice':4,'select':4})
        core.check_versions(cls.s)
        rng=np.random.default_rng(19231)
        cls.X=rng.normal(size=(256,8));cls.X/=np.linalg.norm(cls.X,axis=1)[:,None]
        cls.ids=['d%04d'%j for j in range(256)]
        q=rng.normal(size=(8,8));q/=np.linalg.norm(q,axis=1)[:,None]
        cls.recs=[dict(query_id='q'+str(i),query=x.tolist()) for i,x in enumerate(q[:4])]

    def test_approved_parameters_and_cpu_scope(self):
        s=core.config()
        self.assertEqual(s['banks'],[[2000,4],[5000,2],[20000,1]])
        self.assertEqual(s['budgets'],[128,512,1024])
        self.assertEqual(s['limits']['seconds'],2700)
        self.assertEqual(len(core.NEW_PATHS),5)
        self.assertEqual(s['role_sizes'],{'advice':32,'select':32})
        self.assertTrue(all(not s[n] for n in ['formal_certification','fresh_independence','ssh','gpu','adapter_training']))

    def test_only_authorized_input_roles_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            np.save(p/'corpus64.npy',self.X)
            core.save(p/'ids.json',self.ids)
            core.save(p/'advice_queries.json',self.recs)
            core.save(p/'select_queries.json',[dict(r,query_id='s'+r['query_id']) for r in self.recs])
            s=dict(self.s,input_root=str(p),inputs={n:core.sha(p/n) for n in ['corpus64.npy','ids.json','advice_queries.json','select_queries.json']})
            _,_,r=core.load_observed_inputs(s)
            self.assertEqual(r,self.recs)
            with self.assertRaises(ValueError):core.load_observed_inputs(s,'select')
            core.save(p/'selection.json',{'entries':{},'select_inspected_for_choice':False})
            _,_,selected=core.load_observed_inputs(s,'select',{'path':str(p/'selection.json'),'sha256':core.sha(p/'selection.json')})
            self.assertTrue(all(r['query_id'].startswith('s') for r in selected))
            with self.assertRaises(ValueError):core.load_observed_inputs(s,'report','anything')
            with self.assertRaises(ValueError):core.load_observed_inputs(s,'cert','anything')
            (p/'ids.json').write_text('[]')
            with self.assertRaises(ValueError):core.load_observed_inputs(s)

    def test_bank_disjointness_and_stable_ids(self):
        banks=core.make_banks(self.ids,self.s)
        self.assertFalse(set(banks['n64-b0'])&set(banks['n64-b1']))
        perm=self.ids[::-1]
        other=core.make_banks(perm,self.s)
        for key in banks:
            self.assertEqual([self.ids[j] for j in banks[key]],[perm[j] for j in other[key]])
        with self.assertRaises(ValueError):core.make_banks(['a','a'],self.s)

    def test_all_real_kernels_rerank_and_complete_clock(self):
        q=np.asarray(self.recs[0]['query'])
        for spec in core.base_specs(self.s):
            model=core.build_method(self.X,self.ids,spec,self.s)
            M=2 if spec['kind'].startswith('original') else 16
            out=core.query_method(model,q,M,2)
            E=out['candidate_rows']
            direct=sorted(E,key=lambda j:(np.sum((self.X[j]-q)**2),self.ids[j]))[:2]
            self.assertEqual(out['final_ids'],[self.ids[j] for j in direct])
            self.assertEqual(out['full_ns'],sum(out[k] for k in ['query_projection_ns','screen_ns','rerank_ns','return_overhead_ns']))
            self.assertGreater(out['full_ns'],0)
            self.assertGreaterEqual(out['return_overhead_ns'],0)
            with self.assertRaises(TypeError):core.query_method(model,q,M,2,truth=[])

    def test_projection_definition_and_no_projected_normalization(self):
        model=core.build_method(self.X,self.ids,dict(kind='gaussian',m=4),self.s)
        raw=np.random.Generator(np.random.PCG64(self.s['production_seed'])).standard_normal((4,8))
        np.testing.assert_array_equal(model.P,(raw/2).astype(np.float32))
        np.testing.assert_array_equal(model.Y,self.X.astype(np.float32)@model.P.T)
        self.assertGreater(np.std(np.linalg.norm(model.Y,axis=1)),.1)
        pca=core.build_method(self.X,self.ids,dict(kind='pca',m=4),self.s)
        np.testing.assert_allclose(pca.P@pca.P.T,np.eye(4),atol=2e-7)

    def test_ideal_gaussian_confirmation_matches_its_fp64_geometry(self):
        model=core.Model(self.X,self.ids,dict(kind='gaussian',m=4))
        model.P=core.gaussian_projection(8,4,self.s['production_seed'],maximum=4)
        model.Y=self.X@model.P.T
        q=np.asarray(self.recs[0]['query'])
        E,_=core.candidates(model,q,16)
        d=np.sum((model.Y-q@model.P.T)**2,axis=1)
        direct=sorted(range(len(self.X)),key=lambda j:(d[j],self.ids[j]))[:16]
        self.assertEqual(E.tolist(),direct)

    def test_pq_cutoff_ties_use_all_equal_codes_and_string_ids(self):
        X=np.repeat(self.X[:16],16,axis=0)
        ids=['z%04d'%(255-j) for j in range(256)]
        model=core.build_method(X,ids,dict(kind='pq',bits=2),self.s)
        E,stages=core.candidates(model,X[0],4)
        D,I=model.index.search(X[0:1].astype(np.float32),len(X))
        direct=sorted(zip(D[0],I[0]),key=lambda z:(float(z[0]),ids[z[1]]))[:4]
        self.assertEqual(E.tolist(),[int(j) for _,j in direct])
        self.assertTrue(stages['pq_cutoff_tie_full_scan'])

    def test_probe_reuses_one_pass_and_equals_fixed_outputs(self):
        for spec in core.base_specs(self.s):
            if spec['kind'].startswith('original'):continue
            model=core.build_method(self.X,self.ids,spec,self.s)
            real=core.candidates
            with patch.object(core,'candidates',wraps=real) as counter:
                values=core.probe_model(model,self.recs,self.s['budgets'],self.s)
                self.assertEqual(counter.call_count,len(self.recs))
            for M,records in values.items():
                for r,rec in zip(self.recs,records):
                    fixed=core.query_method(model,np.asarray(r['query']),int(M),2)
                    self.assertEqual(rec['candidate_rows'],fixed['candidate_rows'])
                    self.assertEqual(rec['final_ids'],fixed['final_ids'])

    def test_model_cache_binds_corpus_ids_seed_dimension_and_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model';spec=dict(kind='gaussian',m=4)
            model=core.build_method(self.X,self.ids,spec,self.s)
            core.store_model(model,p)
            got=core.load_model(self.X,self.ids,p,spec,self.s['production_seed'])
            np.testing.assert_array_equal(got.Y,model.Y)
            with self.assertRaises(ValueError):core.load_model(self.X,self.ids[::-1],p,spec)
            with self.assertRaises(ValueError):core.load_model(self.X,self.ids,p,dict(kind='gaussian',m=2))
            with self.assertRaises(ValueError):core.load_model(self.X,self.ids,p,spec,999)
            bad=self.X.copy();bad[0]*=-1
            with self.assertRaises(ValueError):core.load_model(bad,self.ids,p,spec)
            with (p/'P.npy').open('ab') as f:f.write(b'corrupt')
            with self.assertRaises(ValueError):core.load_model(self.X,self.ids,p,spec)

    def test_count_bounds_exhaustive_rank_distributions(self):
        orders=list(itertools.permutations(range(6)))
        weights=np.arange(1,len(orders)+1,dtype=float);weights/=weights.sum()
        mu=np.zeros(2)
        for order,w in zip(orders,weights):
            for i in [0,1]:mu[i]+=w*sum(j in [2,3,4,5] for j in order[:order.index(i)])
        for M in range(2,7):
            intervals=advisor.count_risk_interval(mu,mu,2,2,M,6)
            loss=sum(w*sum(i not in order[:M] for i in [0,1])/2 for order,w in zip(orders,weights))
            self.assertLessEqual(intervals['lower'],loss+1e-14)
            self.assertGreaterEqual(intervals['upper'],loss-1e-14)
        self.assertAlmostEqual(advisor.count_risk_interval([2.8],[2.8],1,1,3,5)['lower'],.4)
        self.assertEqual(advisor.count_risk_interval([1],[1],1,1,5,5)['upper'],0)
        for args in [([2],[1],1,1,2,5),([-1],[1],1,1,2,5),([0],[5],1,1,2,5)]:
            with self.assertRaises(ValueError):advisor.count_risk_interval(*args)

    def test_conservative_pair_unknown_domain_has_zero_lower(self):
        X=np.array([[1.,0.],[1.,0.],[0.,2.],[0.,5.]])
        s=dict(self.s,k=1,gaussian_dimensions=[2],budgets=[1,3],risk_block=2)
        records=[dict(query_id='q',query=[0.,0.])]
        v=advisor.assess_geometry(X,['a','b','c','d'],records,{'q':['a']},s)[0]
        dim=v['dimensions']['2']
        self.assertEqual(v['conservative_pairs'],1)
        self.assertAlmostEqual(dim['mu_upper'][0]-dim['mu_lower'][0],1)

    def test_large_upper_is_unknown_not_danger_and_upper_ablation(self):
        geometry=[{'dimensions':{str(m):{'budgets':{str(M):{'lower':0.,'upper':.5} for M in self.s['budgets']}}
                                for m in self.s['gaussian_dimensions']}}]
        states=advisor.expected_states(geometry,self.s)
        self.assertEqual({v['state'] for v in states.values()},{'unknown'})
        for d in geometry[0]['dimensions'].values():
            for x in d['budgets'].values():x['lower']=.2
        self.assertEqual({v['state'] for v in advisor.expected_states(geometry,self.s).values()},{'high_expected_risk'})
        self.assertEqual({v['state'] for v in advisor.expected_states(geometry,self.s,False).values()},{'unknown'})

    def test_always_unknown_is_actual_exact_fallback_without_truth_or_tri(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(core,'truth',side_effect=AssertionError('free truth forbidden')):
            v=advisor.run_probe_policy(self.X,self.ids,self.recs,self.s,'always_unknown',Path(tmp)/'policy')
            self.assertEqual(v['selected']['spec']['kind'],'original64')
            self.assertEqual(v['costs']['truth_seconds'],0)
            self.assertEqual(v['attempted'],[])
            self.assertEqual(v['input_roles'],['advice'])
        self.assertNotIn('select',inspect.signature(advisor.run_probe_policy).parameters)

    def test_tri_interval_can_skip_proved_bad_but_upper_cannot(self):
        geometry=[{'dimensions':{str(m):{'budgets':{str(M):{'lower':.8,'upper':.9} for M in self.s['budgets']}}
                                for m in self.s['gaussian_dimensions']}}]
        with tempfile.TemporaryDirectory() as tmp,patch.object(advisor,'assess_geometry',return_value=geometry):
            interval=advisor.run_probe_policy(self.X,self.ids,self.recs,self.s,'tri_interval',Path(tmp)/'interval')
            upper=advisor.run_probe_policy(self.X,self.ids,self.recs,self.s,'tri_upper',Path(tmp)/'upper')
            self.assertEqual(interval['attempted'],[])
            self.assertEqual(len(upper['attempted']),2)
            self.assertGreater(upper['costs']['probe_seconds'],0)

    def test_low_expected_risk_still_pays_concrete_model_confirmation(self):
        geometry=[{'dimensions':{str(m):{'budgets':{str(M):{'lower':0.,'upper':0.} for M in self.s['budgets']}}
                                for m in self.s['gaussian_dimensions']}}]
        with tempfile.TemporaryDirectory() as tmp,patch.object(advisor,'assess_geometry',return_value=geometry):
            v=advisor.run_probe_policy(self.X,self.ids,self.recs,self.s,'tri_interval',Path(tmp)/'policy')
            self.assertTrue(v['attempted'])
            self.assertGreater(v['costs']['probe_seconds'],0)
            self.assertTrue(v['observations'])

    def test_lifecycle_clock_hand_fixture_and_quality_gate(self):
        self.assertAlmostEqual(core.lifecycle_seconds(.5,2000000,1000000,10),.511)
        self.assertEqual(core.lifecycle_seconds(.5,2000000,1000000,1),.502)
        with self.assertRaises(ValueError):core.lifecycle_seconds(0,1,1,0)
        self.assertFalse(core.eligible({'mean_loss':.006,'worst_loss':.1},self.s))
        self.assertFalse(core.eligible({'mean_loss':0.,'worst_loss':.2},self.s))

    def test_outer_query_clock_includes_return_and_conversion(self):
        model=core.build_method(self.X,self.ids,dict(kind='original64'),self.s)
        real=core.query_method
        def delayed(*a,**kw):
            value=real(*a,**kw);time.sleep(.002);return value
        with patch.object(core,'query_method',side_effect=delayed):
            measured=core.measure_queries(model,self.recs,2,self.s)
        self.assertGreaterEqual(measured['mean_ns'],2000000)
        for values in measured['samples'].values():
            for v in values:self.assertEqual(v['full_ns'],sum(v[k] for k in ['query_projection_ns','screen_ns','rerank_ns','return_overhead_ns']))

    def test_role_and_input_seals_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            core.save(p/'inputs/advice_queries.json',self.recs)
            core.save(p/'inputs/select_queries.json',self.recs)
            core.save(p/'input_manifest.json',{'advice_sha256':core.sha(p/'inputs/advice_queries.json')})
            self.assertEqual(runner.queries(p,'advice','B'),self.recs)
            with self.assertRaises(ValueError):runner.queries(p,'select','A')
            with self.assertRaises(ValueError):runner.queries(p,'select','B')
            core.save(p/'A_builds.json',{});core.save(p/'A_families.json',{})
            h=runner.seal(p,'A_selection_seal.json',{})
            core.save(p/'select_input_seal.json',{'sha256':core.sha(p/'inputs/select_queries.json'),'A_selection_sha256':h})
            self.assertEqual(runner.queries(p,'select','A'),self.recs)
            with self.assertRaises(ValueError):runner.queries(p,'select','B')
            (p/'A_builds.json').write_text('{"tampered":true}')
            with self.assertRaises(ValueError):runner.queries(p,'select','A')

    def test_real_resource_preflight_and_owned_child_monitor(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'blocked'
            with patch.object(runner,'descendant_processes',side_effect=PermissionError('real monitor denied')):
                with self.assertRaises(PermissionError):runner.supervisor(p)
            self.assertFalse(p.exists())
        proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
        try:
            self.assertGreater(sum(r for _,r in runner.descendant_processes(os.getpid())),0)
            t=time.monotonic();runner.terminate_owned_tree(proc)
            self.assertLess(time.monotonic()-t,3)
            self.assertLess(proc.returncode,0)
        finally:
            if proc.poll() is None:proc.kill();proc.wait()

    def test_resource_limit_and_monitor_exception_stop_real_sleep(self):
        real_popen=subprocess.Popen
        def only_science(args,*a,**kw):
            if len(args)>2 and str(args[1])==runner.__file__ and args[2]=='science':
                return real_popen([sys.executable,'-c','import time;time.sleep(20)'],*a,**kw)
            return real_popen(args,*a,**kw)
        for fail_monitor in [False,True]:
            with self.subTest(fail_monitor=fail_monitor),tempfile.TemporaryDirectory() as tmp:
                s=core.config();s['limits']=dict(s['limits'],seconds=.05,free_bytes=0)
                real_monitor=runner.descendant_processes;calls=[]
                def monitor(pid):
                    calls.append(pid)
                    if fail_monitor and len(calls)>1:raise RuntimeError('monitor lost')
                    return real_monitor(pid)
                with patch.object(core,'config',return_value=s),patch.object(runner.subprocess,'Popen',side_effect=only_science),patch.object(runner,'descendant_processes',side_effect=monitor),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit):runner.supervisor(Path(tmp)/'root')
                status=json.loads((Path(tmp)/'root/status.json').read_text())
                self.assertEqual(status['reason'],'monitor_failure' if fail_monitor else 'cumulative_wall_limit')
                self.assertLess(status['exit_code'],0)
                self.assertLess(status['supervision_seconds'],3)

    def test_shared_epoch_source_and_input_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            f={'sources':runner.source_map(),'config':core.config(),'deadline_unix':time.time()+10}
            core.save(p/'source_freeze.json',f)
            runner.assert_frozen(p)
            core.save(p/'input_manifest.json',{})
            core.save(p/'input_seal.json',{'manifest_sha256':core.sha(p/'input_manifest.json'),'source_freeze_sha256':core.sha(p/'source_freeze.json')})
            runner.assert_frozen(p)
            (p/'input_manifest.json').write_text('{"tampered":true}')
            with self.assertRaises(ValueError):runner.assert_frozen(p)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f['deadline_unix']=time.time()-1;core.save(p/'source_freeze.json',f)
            with self.assertRaises(RuntimeError):runner.assert_frozen(p)

    def test_actual_frozen_worker_without_select_or_production_data(self):
        s=core.config();rng=np.random.default_rng(245)
        X=rng.normal(size=(2000,384));X/=np.linalg.norm(X,axis=1)[:,None]
        Q=rng.normal(size=(32,384));Q/=np.linalg.norm(Q,axis=1)[:,None]
        recs=[dict(query_id='synthetic-q'+str(i),query=q.tolist()) for i,q in enumerate(Q)]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);b=p/'banks/n2000-b0';b.mkdir(parents=True)
            np.save(b/'corpus64.npy',X);core.save(b/'ids.json',['synthetic-d'+str(i) for i in range(len(X))])
            core.save(p/'inputs/advice_queries.json',recs)
            bank={k:{'path':str((b/f).relative_to(p)),'sha256':core.sha(b/f)} for k,f in [('vectors','corpus64.npy'),('ids','ids.json')]}
            core.save(p/'source_freeze.json',{'sources':runner.source_map(),'config':s,'deadline_unix':time.time()+120})
            core.save(p/'input_manifest.json',{'banks':{'n2000-b0':bank},'advice_sha256':core.sha(p/'inputs/advice_queries.json')})
            core.save(p/'input_seal.json',{'manifest_sha256':core.sha(p/'input_manifest.json'),'source_freeze_sha256':core.sha(p/'source_freeze.json')})
            core.save(p/'job.json',{'mode':'build','bank':'n2000-b0','spec':{'kind':'original64'},'repeat':0})
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable,runner.__file__,'worker','--root',str(p),'--job',str(p/'job.json'),'--output',str(p/'output.json')],
                           env=env,check=True,capture_output=True,timeout=15)
            out=json.loads((p/'output.json').read_text())
            self.assertEqual(len(out['advice']['10']['records']),32)
            self.assertGreater(out['deploy_prequery_seconds'],0)
            self.assertFalse((p/'inputs/select_queries.json').exists())
            def run_job(name,job):
                core.save(p/(name+'.json'),job)
                subprocess.run([sys.executable,runner.__file__,'worker','--root',str(p),'--job',str(p/(name+'.json')),'--output',str(p/(name+'-out.json'))],
                               env=env,check=True,capture_output=True,timeout=30)
                return json.loads((p/(name+'-out.json')).read_text())
            family=run_job('family',{'mode':'family','bank':'n2000-b0','family':'gaussian'})
            self.assertEqual(len(family['records']),9)
            tri=run_job('advisor',{'mode':'advisor','bank':'n2000-b0','policy':'tri_interval'})
            self.assertEqual(len(tri['states']),9)
            self.assertFalse(tri['reads_scenario_A_results'])
            core.save(p/'A_builds.json',out);core.save(p/'A_families.json',family)
            h=runner.seal(p,'A_selection_seal.json',{'synthetic':family['selected']})
            select=[dict(r,query_id='s'+r['query_id']) for r in recs]
            core.save(p/'inputs/select_queries.json',select)
            core.save(p/'select_input_seal.json',{'sha256':core.sha(p/'inputs/select_queries.json'),'A_selection_sha256':h})
            ev=run_job('evaluate',{'mode':'evaluate','phase':'A','bank':'n2000-b0',
                'spec':family['selected']['spec'],'budgets':[family['selected']['M']],'model_directory':family['model_directory']})
            self.assertEqual(len(ev['G']),32)
            self.assertEqual(len(ev['results'][str(family['selected']['M'])]['measurements']['samples']),32)


if __name__=='__main__':unittest.main()
