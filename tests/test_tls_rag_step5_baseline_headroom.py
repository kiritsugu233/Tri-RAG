"""Synthetic offline regressions for the explicitly approved baseline screen."""
import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mechanism.tls_rag_step5_baseline_headroom import (ObservedInputs,GRID,METHODS,
    SPEC,sha,save,sources,fingerprint,extract_observed_subset,remote_source,quality,
    original_exact,projected_fixed,original_ann,execute,select_fixed,seal_selection,
    perfect_information_bound,timing_ids,make_request)
from mechanism.tls_rag_step5_audit import ExactScan
from tri_rag_harness.projection import dense_gaussian_projection,project_rows
from tri_rag_harness.tls_rag_step5_ann import HnswCandidateIndex
from tri_rag_harness.tls_rag_step5_protocol import load_protocol


def fixture():
    rng=np.random.default_rng(872)
    v=rng.normal(size=(300,8));v/=np.linalg.norm(v,axis=1)[:,None]
    q=rng.normal(size=8);q/=np.linalg.norm(q)
    ids=['d'+str(i) for i in range(len(v))]
    matrix=dense_gaussian_projection(4,8,55117)
    reader=SimpleNamespace(corpus=v,projected=project_rows(v,matrix),id_array=np.array(ids),
        row_by_id={i:n for n,i in enumerate(ids)})
    rows,_,_=ExactScan(ids,v,31).search(q,10)
    truth=np.array(ids)[rows].tolist()
    record=dict(query_id='q',query=q.tolist(),truth=truth,qrels=[truth[0],'outside-corpus'],archived_exact_hit=1.)
    index=HnswCandidateIndex(ids,v,load_protocol()['hnsw'],55113)
    return SimpleNamespace(reader=reader,matrix=matrix,ann=index),record


class BaselineHeadroomTests(unittest.TestCase):
    def test_original_and_projected_use_same_reference_geometry(self):
        inputs,record=fixture()
        result=execute(inputs,record,'original_exact')
        self.assertEqual(result['candidate_ids'],record['truth'])
        self.assertEqual(result['metrics']['retention'],1.)
        q=np.array(record['query']);qp=project_rows(q[None],inputs.matrix)[0]
        for m in GRID:
            result=projected_fixed(inputs.reader,q,inputs.matrix,m)
            rr,_,_=ExactScan(inputs.reader.id_array,inputs.reader.projected,31).search(qp,m)
            self.assertEqual(result['candidate_ids'],inputs.reader.id_array[rr].tolist())
            self.assertEqual(result['original_distances'],m)
            self.assertEqual(result['projected_distances'],300)

    def test_original_ann_actual_sets_and_no_truth_in_decision(self):
        inputs,record=fixture()
        one=execute(inputs,record,'original_ann64')
        changed=copy.deepcopy(record);changed['qrels']=[]
        two=execute(inputs,changed,'original_ann64')
        self.assertEqual(one['candidate_ids'],two['candidate_ids'])
        self.assertEqual(one['context_ids'],two['context_ids'])
        self.assertEqual(one['original_distances'],64)
        self.assertEqual(two['metrics']['hit'],0.)

    def test_truth_mismatch_stops_instead_of_refreshing(self):
        inputs,record=fixture();record['truth']=list(reversed(record['truth']))
        with self.assertRaisesRegex(ValueError,'truth/Hit'):execute(inputs,record,'original_exact')

    def test_quality_hit_may_decrease_and_outside_qrels_are_valid(self):
        r=dict(truth=['a']*10,qrels=['a','outside'],archived_exact_hit=1.)
        self.assertEqual(quality(['a'],['b'],r)['hit_delta'],-1.)

    def test_selection_only_eligible_fixed_no_eval_fallback(self):
        methods={m:{'metrics':{'retention':1.,'hit':1.,'hit_delta':0.}} for m in METHODS}
        costs={m:float(i+1) for i,m in enumerate(METHODS)}
        self.assertEqual(select_fixed([{'methods':methods}],costs),32)
        for m in GRID:methods['projected_fixed:'+str(m)]['metrics']['retention']=.8
        self.assertIsNone(select_fixed([{'methods':methods}],costs))

    def test_perfect_oracle_uses_mean_action_costs_and_keeps_infeasibility(self):
        rows=[]
        for i in range(2):
            methods={name:{'metrics':dict(retention=1.,hit=1.,hit_delta=0.)} for name in METHODS}
            methods['projected_fixed:32']['metrics']['retention']=1. if i==0 else .8
            rows.append(dict(query_id=str(i),methods=methods))
        costs={m:10. for m in METHODS};costs.update({'original_exact':20.,'original_ann64':.5,
            'projected_fixed:32':1.,'projected_fixed:64':2.})
        result=perfect_information_bound(rows,costs)
        self.assertTrue(result['oracle']['optimal'])
        self.assertEqual(result['oracle']['model_mean_ms'],1.)
        self.assertEqual(result['oracle']['headroom_by_control']['original_ann64']['gross_ms'],-.5)
        for row in rows:
            for m in GRID:row['methods']['projected_fixed:'+str(m)]['metrics']['retention']=.5
        result=perfect_information_bound(rows,costs)
        self.assertFalse(result['oracle']['optimal'])

    def test_eval_gate_and_selection_seal_are_binding(self):
        with tempfile.TemporaryDirectory() as td:
            inp=ObservedInputs.__new__(ObservedInputs)
            inp.groups={'selection':[1],'evaluation':[2]};inp._eval_seal=None;inp.input_binding='abc'
            with self.assertRaises(ValueError):inp.records('evaluation')
            p=Path(td)/'seal.json'
            binding=seal_selection(p,inp,64,[],{})
            inp.authorize_evaluation(p,binding)
            self.assertEqual(inp.records('evaluation'),[2])
            v=json.loads(p.read_text());v['fixed_budget']=128;p.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'seal'):inp.authorize_evaluation(p,binding)
            with self.assertRaises(FileExistsError):seal_selection(p,inp,128,[],{})

    def test_sampling_identity_only_order_invariant_and_parent_sha_rejected(self):
        rows=[dict(query_id=str(i)) for i in range(90)]
        self.assertEqual(timing_ids(rows,'selection'),timing_ids(rows[::-1],'selection'))
        self.assertEqual(len(timing_ids(rows,'selection')),64)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.json';p.write_text('{}')
            with self.assertRaises(ValueError):make_request(p,'b','a')

    def test_loader_rejects_unapproved_request_before_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);save(p/'observed.json',dict(request={'bad':1},schema=SPEC['schema']))
            with self.assertRaisesRegex(ValueError,'request'):ObservedInputs(p,{'good':2})

    def test_complete_small_input_loader_and_download_checksum_guard(self):
        import faiss
        inputs,record=fixture()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            np.save(root/'corpus.npy',inputs.reader.corpus)
            np.save(root/'projected.npy',inputs.reader.projected)
            save(root/'ids.json',inputs.reader.id_array.tolist())
            faiss.write_index(inputs.ann.index,str(root/'original.faiss'))
            a=copy.deepcopy(record);a['query_id']='selection-q'
            b=copy.deepcopy(record);b['query_id']='evaluation-q'
            request={'groups':{'selection':['selection-q'],'evaluation':['evaluation-q']},
                'hnsw':load_protocol()['hnsw']}
            pack={'schema':SPEC['schema'],'request':request,'groups':{'selection':[a],'evaluation':[b]},
                'files':{name:{'bytes':(root/name).stat().st_size,'sha256':sha(root/name)}
                    for name in ['corpus.npy','projected.npy','ids.json','original.faiss']},
                'forbidden_roles_opened':False,'opened_roles':['tune']}
            save(root/'observed.json',pack)
            with patch.dict(SPEC,{'corpus_size':300,'dimension':8,'m_prime':4}):
                loaded=ObservedInputs(root,request)
                self.assertEqual(execute(loaded,loaded.records('selection')[0],'original_exact')['metrics']['retention'],1.)
                with self.assertRaises(ValueError):loaded.records('evaluation')
                (root/'ids.json').write_text('[]')
                with self.assertRaisesRegex(ValueError,'checksum'):ObservedInputs(root,request)

    def test_existing_output_is_not_modified_even_on_rejected_run(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);inp=root/'input';inp.mkdir();save(inp/'request.json',{})
            out=root/'result';out.mkdir();(out/'sentinel').write_text('preserve')
            script=Path(__file__).resolve().parents[1]/'scripts/run_tls_rag_step5_baseline_headroom.py'
            result=subprocess.run([sys.executable,str(script),'--run',str(out),'--inputs',str(inp)],
                capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual([p.name for p in out.iterdir()],['sentinel'])

    def test_remote_export_only_observed_groups_and_no_closed_role_reads(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);bundle=root/'bundle';archive=root/'archive';bundle.mkdir();archive.mkdir()
            (archive/'indexes').mkdir();(archive/'tune').mkdir();(bundle/'labels').mkdir()
            ids=['a','b','c','d'];groups={'selection':['a'],'evaluation':['c']}
            metadata={'sources':{'f':'h'},'protocol':'proto','roles':{'tune':{'ids':fingerprint(ids)}}}
            save(bundle/'metadata.json',metadata);save(bundle/'tune.ids.json',ids)
            np.save(bundle/'tune.npy',np.arange(32).reshape(4,8))
            save(bundle/'labels/tune.json',{q:['d0'] for q in ids})
            np.save(bundle/'corpus.npy',np.ones((3,8)));save(bundle/'corpus_ids.json',['d0','d1','d2'])
            np.save(archive/'indexes/projected.npy',np.ones((3,4)))
            (archive/'indexes/original.faiss').write_bytes(b'fixture-not-loaded')
            save(archive/'indexes/setup.json',{'binding':'binding'})
            for n in ('reservation.json','labels_opened.json'):
                save(archive/'tune'/n,{'binding':'binding','stage':'tune'})
            (archive/'tune/truth.jsonl').write_text(''.join(json.dumps(dict(query_id=q,exact_top_k=['d0']*10,exact_hit=1.))+'\n' for q in ids))
            bm={str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()}
            save(bundle/'artifact_manifest.json',bm)
            am={}
            for role in ('indexes','tune'):
                save(archive/role/'artifact_manifest.json',{p.name:sha(p) for p in (archive/role).iterdir()})
                am[role]=sha(archive/role/'artifact_manifest.json')
            request={'schema':SPEC['schema'],'bundle':str(bundle),'archive':str(archive),
                'parent_config':{'bundle_binding':fingerprint(bm),'archive_binding':'binding','archive_manifests':am},
                'legacy_sources':metadata['sources'],'legacy_protocol':'proto','groups':groups}
            old_open=Path.open
            def guarded(path,*a,**kw):
                if any(part in ('cert','latency','test') for part in path.parts):raise AssertionError('closed role read')
                return old_open(path,*a,**kw)
            with patch.object(Path,'open',guarded):out=extract_observed_subset(request)
            self.assertEqual([r['query_id'] for r in out['groups']['selection']],['a'])
            self.assertEqual([r['query_id'] for r in out['groups']['evaluation']],['c'])
            self.assertEqual(set(out['files']),{'corpus.npy','ids.json','projected.npy','original.faiss'})
            self.assertFalse(out['forbidden_roles_opened'])
            scope={};printed=io.StringIO()
            with contextlib.redirect_stdout(printed):exec(remote_source(request),scope)
            self.assertEqual(json.loads(printed.getvalue()),out)
            # Standalone exporter generated the same derived package without writes.
            self.assertEqual(scope['extract_observed_subset'](request),out)


if __name__=='__main__':unittest.main()
