"""CPU, synthetic, offline tests of the newly authorized scientific boundary."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mechanism.tls_rag_step5_audit import (CONFIG, Reader, Ledger, ExactScan,
    ResourceStop, decompose, flat_prefix, sample_groups, save, phase_a)
from mechanism.tls_rag_step5_exact_policy import (PrefixSession, feature_state,
    values, fit_models, calibrate, adjusted, decide, metrics, interval, phase_b, pipeline)
from tri_rag_harness.projection import dense_gaussian_projection, project_rows
from tri_rag_harness.tls_rag_step5_policy import build_state as legacy_features
from tri_rag_harness.tls_rag_step5_protocol import sha256, source_binding, load_protocol
from tri_rag_harness.utils import fingerprint


def config():
    return json.loads(CONFIG.read_text())


def normalized(x):
    return x/np.linalg.norm(x, axis=1)[:,None]


def fixture(root):
    """Tiny real-format allowlisted parents; unopened-role files are tripwires."""
    cfg = config()
    cfg.update(corpus_size=300, dimension=8, m_prime=4, minimum_fit=2, bootstrap_replicates=30)
    cfg['samples'] = dict(fit=4,calibration=4,selection=4,evaluation=4)
    cfg['timing'].update(headroom_queries=2,warmup_queries=1,repeats=1)
    bundle, archive = root/'input/bundle', root/'archive'
    bundle.mkdir(parents=True);archive.mkdir()
    rng = np.random.default_rng(951)
    corpus = normalized(rng.normal(size=(300,8)))
    ids = ['doc-'+str(i) for i in range(len(corpus))]
    matrix = dense_gaussian_projection(4,8,cfg['projection_seed'])
    projected = project_rows(corpus,matrix)
    (archive/'indexes').mkdir()
    np.save(archive/'indexes/projected.npy',projected)
    save(archive/'indexes/setup.json',{'binding':cfg['archive_binding']})
    np.save(bundle/'corpus.npy',corpus);save(bundle/'corpus_ids.json',ids)
    role_info, counts = {}, {}
    for role in ['cal_model_fit','cal_bound_fit','tune']:
        qids = [role+'-'+str(i) for i in range(8)]
        queries = normalized(rng.normal(size=(8,8)))
        counts[role] = len(qids)
        role_info[role] = {'ids':fingerprint(qids),'families':fingerprint(qids),'texts':fingerprint(qids)}
        save(bundle/(role+'.ids.json'),qids);np.save(bundle/(role+'.npy'),queries)
        r = archive/role;r.mkdir()
        save(r/'reservation.json',{'binding':cfg['archive_binding'],'stage':role})
        save(r/'labels_opened.json',{'binding':cfg['archive_binding'],'stage':role})
        truths, decisions, labels = [], [], {}
        index = ExactScan(ids,corpus,31)
        pi = ExactScan(ids,projected,31)
        for qid,q in zip(qids,queries):
            rr,_,_ = index.search(q,10)
            truth = np.asarray(ids)[rr].tolist()
            truths.append({'query_id':qid,'exact_top_k':truth,'exact_hit':1.})
            labels[qid] = [truth[0]]
            stages=[]
            for m in cfg['anchors']:
                rr,_,_ = pi.search(project_rows(q[None],matrix)[0],m)
                found = np.asarray(ids)[rr].tolist()
                stages.append({'budget':m,'raw_ids':found,'candidate_ids':found})
            decisions.append({'query_id':qid,'rows':{'all':{'stages':stages}}})
        (bundle/'labels').mkdir(exist_ok=True)
        save(bundle/'labels'/(role+'.json'),labels)
        for name,items in [('truth.jsonl',truths),('retrieval.decisions.jsonl',decisions)]:
            (r/name).write_text(''.join(json.dumps(x)+'\n' for x in items))
    save(bundle/'metadata.json',{'sources':source_binding(),'protocol':fingerprint(load_protocol()),
        'roles':role_info,'role_counts':counts})
    for name in ['indexes','cal_model_fit','cal_bound_fit','tune']:
        r=archive/name
        save(r/'artifact_manifest.json',{p.name:sha256(p) for p in r.iterdir()})
        cfg['archive_manifests'][name] = sha256(r/'artifact_manifest.json')
    manifest = {str(p.relative_to(bundle)):sha256(p) for p in bundle.rglob('*') if p.is_file()}
    # Forbidden files are deliberately absent from usable inputs. Their opaque
    # hash entries are allowed, but reading any corresponding content is not.
    for role in ['cert','latency','test']:
        (bundle/(role+'.npy')).write_text('DO NOT OPEN')
        manifest[role+'.npy'] = 'opaque-unopened-hash'
    save(bundle/'artifact_manifest.json',manifest)
    cfg['bundle_binding'] = fingerprint(manifest)
    return cfg,bundle,archive


class MechanismMathTests(unittest.TestCase):
    def test_exact_scan_matches_brute_force_ties_and_ranks(self):
        ids = ['z','b','a','x','y']
        vv = np.array([[1.,0],[1.,0],[1.,0],[2.,0],[0.,0]])
        rr,dd,ranks = ExactScan(ids,vv,2).search(np.zeros(2),3,[0,2])
        self.assertEqual([ids[i] for i in rr],['y','a','b'])
        self.assertEqual([r['rank'] for r in ranks],[4,2])
        self.assertEqual([r['W'] for r in ranks],[1,0])
        for r in ranks:self.assertEqual(r['rank'],1+r['Z']+r['W'])
        small,_,_=ExactScan(ids,vv,3).search(np.zeros(2),2)
        self.assertEqual(small.tolist(),rr[:2].tolist())

    def test_decomposition_keeps_rescues_not_additive_error(self):
        d=decompose(['a','b'],['a','x'],['b','y'])
        self.assertEqual(d['omissions'],['a']);self.assertEqual(d['rescues'],['b'])
        self.assertEqual(d['before_retention'],d['after_retention'])
        rng=np.random.default_rng(42)
        for _ in range(100):
            t,e,h=[set(rng.choice(20,5,replace=False)) for _ in range(3)]
            result=decompose(t,e,h)
            self.assertAlmostEqual(1-result['after_retention'],
                (len(result['projection_or_prior_omissions'])+len(result['omissions'])-len(result['rescues']))/len(t))

    def test_flat_cutoff_tie_resolved_over_entire_group(self):
        class Flat:
            def search(self,q,k):return np.array([[0.,1.,1.]],np.float32),np.array([[3,0,1]])
            def range_search(self,q,radius):return np.array([0,4]),np.array([1.,1.,1.,0.]),np.array([0,1,2,3])
        rr,_,ties=flat_prefix(Flat(),np.zeros(2),np.array(['z','b','a','x']),[2])
        self.assertEqual(rr.tolist(),[3,2]);self.assertTrue(ties[2])

    def test_sampling_deterministic_identity_only_and_disjoint(self):
        cfg=config();cfg['samples']=dict(fit=3,calibration=3,selection=3,evaluation=4)
        parent={r:[r+str(i) for i in range(12)] for r in ['cal_model_fit','cal_bound_fit','tune']}
        a=sample_groups(parent,cfg)
        b=sample_groups({r:list(reversed(v)) for r,v in parent.items()},cfg)
        self.assertEqual(a,b)
        self.assertFalse(set(a['selection'])&set(a['evaluation']))
        parent['cal_bound_fit'][0]=parent['tune'][0]
        with self.assertRaises(ValueError):sample_groups(parent,cfg)

    def test_query_block_bootstrap_preserves_pairing_and_repeats(self):
        cfg=config();cfg['bootstrap_replicates']=60
        a=np.arange(1,25).reshape(8,3)
        ratio=interval(a*2,a,'ratio',cfg)
        self.assertEqual((ratio['point'],ratio['lower'],ratio['upper']),(2.,2.,2.))
        delta=interval(a+3,a,cfg=cfg)
        self.assertEqual((delta['point'],delta['lower'],delta['upper']),(3.,3.,3.))
        self.assertEqual(interval(a,a,'ratio',cfg),interval(a,a,'ratio',cfg))


class MechanismBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.cfg,self.bundle,self.archive=fixture(self.root)
        self.ledger=Ledger(self.root/'study',self.cfg,reserve_parent=False)

    def reader(self):return Reader(self.bundle,self.archive,self.cfg,self.ledger)

    def test_reader_never_opens_forbidden_roles_and_keeps_eval_closed(self):
        original=Path.open
        opened=[]
        def guarded(path,*args,**kwargs):
            if path.name in {'cert.npy','latency.npy','test.npy'}:raise AssertionError('forbidden access')
            opened.append(str(path));return original(path,*args,**kwargs)
        with patch.object(Path,'open',guarded):
            r=self.reader()
            r.truth('selection');r.labels('selection')
            for method in [r.truth,r.labels]:
                with self.assertRaises(ValueError):method('evaluation')
            with self.assertRaises(ValueError):r.query('evaluation',r.groups['evaluation'][0])
            with self.assertRaises(ValueError):r.verify('bundle','cert.npy')
            self.assertFalse(any(x['role'] in {'cert','latency','test'} for x in r.access))
            self.ledger.freeze({'selected':'closed'})
            self.ledger.authorize_evaluation()
            self.assertEqual(len(r.truth('evaluation')),4)

    def test_input_tampering_rejected_without_refreshing_binding(self):
        (self.bundle/'corpus_ids.json').write_text('[]')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):self.reader()

    def test_no_retry_parent_reservation_and_seal_tampering(self):
        parent=self.root/'other';parent.mkdir()
        a=Ledger(parent/'one',self.cfg)
        a.begin('phase_a')
        with self.assertRaises(FileExistsError):a.begin('phase_a')
        with self.assertRaises(FileExistsError):Ledger(parent/'replacement',self.cfg)
        with self.assertRaises(ValueError):a.authorize_evaluation()
        a.freeze({'x':1})
        (a.root/'selection_seal.json').write_text('{}')
        with self.assertRaises(ValueError):a.authorize_evaluation()

    def test_resource_stop_preserves_reserved_stage(self):
        self.ledger.begin('a')
        self.ledger.cfg=copy.deepcopy(self.cfg)
        self.ledger.cfg['limits']['seconds']=-1
        with self.assertRaises(ResourceStop):self.ledger.check()
        self.assertTrue((self.ledger.root/'a.reserved.json').exists())

    def test_new_namespace_does_not_change_legacy_source_binding(self):
        original=source_binding()
        self.assertTrue(all(not p.startswith('mechanism/') for p in original))
        r=self.reader()
        self.assertEqual(r.metadata['sources'],original)

    def test_features_match_reviewed_geometry_and_no_hidden_tri(self):
        r=self.reader();qid=r.groups['selection'][0]
        q=r.query('selection',qid);qp=project_rows(q[None],r.matrix)[0]
        rr,dd,_=ExactScan(r.ids,r.projected,31).search(qp,64)
        s=PrefixSession(r,q,qp,self.cfg);rec=s.expand(rr[:32],dd[:32])
        with patch('mechanism.tls_rag_step5_exact_policy._observed_pair',side_effect=AssertionError('hidden Tri')):
            b=feature_state(s,rec,'B',self.cfg)
            bd=feature_state(s,rec,'B+D',self.cfg)
            self.assertEqual(b['T'],[]);self.assertEqual(bd['triplet_pair_count'],0)
        full=feature_state(s,rec,'B+D+T',self.cfg)
        legacy=legacy_features(SimpleNamespace(query=q,original=r.corpus,row_by_id=r.row_by_id,
            original_cache=s.original_cache,projected_cache=s.projected_cache),dict(rec),
            {'features':self.cfg['features'],'m_prime':self.cfg['m_prime'],'schema':'tls-rag-step5-ann-v1'})
        np.testing.assert_allclose(full['base'],legacy.base,rtol=0,atol=0)
        np.testing.assert_allclose(full['T']+full['D'],legacy.risk,rtol=0,atol=0)
        self.assertEqual(len(values(full,'B')[0]),13)
        self.assertEqual(len(values(full,'B+D+T')[0]),19)
        rec2=s.expand(rr[:64],dd[:64])
        self.assertEqual(rec2['original_distances'],32)
        with self.assertRaises(ValueError):feature_state(s,rec,'B',self.cfg)
        with self.assertRaises(ValueError):feature_state(s,{**rec2,'budget':256},'B',self.cfg)

    def test_fit_calibration_are_matched_and_invalid_expands(self):
        examples=[]
        for i in range(8):
            state={'stage':0,'budget':32,'base':[float(i)]*13,'D':[float(i)]*2,'T':[float(i)]*4,
                   'base_valid':True,'D_valid':True,'T_valid':True}
            examples.append((state,.7+i*.02))
        models=fit_models(examples,self.cfg)
        margins,_=calibrate(examples,models,self.cfg)
        s=examples[0][0]
        self.assertIsNotNone(adjusted(s,'B+D',models,margins))
        invalid={**s,'T_valid':False}
        self.assertFalse(decide(invalid,{'family':'B+D+T','threshold':.8},models,margins,self.cfg)['stop'])
        with self.assertRaises(ValueError):decide(s,{'family':'B','threshold':.999},models,margins,self.cfg)

    def test_real_pipeline_uses_own_budget_and_scoring_cache_without_hidden_tri(self):
        r=self.reader();qid=r.groups['selection'][0]
        with patch('mechanism.tls_rag_step5_exact_policy._observed_pair',side_effect=AssertionError('hidden Tri')):
            fixed=pipeline(r,'selection',qid,self.cfg,self.ledger,fixed=64)
            adaptive=pipeline(r,'selection',qid,self.cfg,self.ledger,
                policy={'family':'B+D','threshold':.8},models={},margins={})
        self.assertEqual(len(fixed['record']['candidate_ids']),64)
        self.assertEqual(fixed['original_distances'],64)
        self.assertEqual(adaptive['original_distances'],256)
        self.assertEqual(adaptive['features_ms']>0,True)
        self.assertEqual(fixed['features_ms'],0.)
        self.assertGreater(fixed['pipeline_ms'],fixed['gather_score_ms'])
        self.assertGreaterEqual(fixed['scan_and_selection_ms'],fixed['selection_ms'])

    def test_complete_two_phase_fixture_with_real_faiss_when_available(self):
        try:import faiss
        except ImportError:self.skipTest('real FAISS optional locally; required on server')
        if faiss.__version__ != self.cfg['faiss_version']:self.skipTest('pinned FAISS required on server')
        r=self.reader()
        with patch('builtins.print'):
            cache=phase_a(r,self.ledger,self.cfg)
            result=phase_b(r,self.ledger,self.cfg,cache,{'fixture':True})
        self.assertTrue((self.ledger.root/'phase_a.complete.json').is_file())
        self.assertTrue((self.ledger.root/'selection_seal.json').is_file())
        self.assertTrue(self.ledger.evaluation_open)
        self.assertFalse(result['formal_certification'])
        self.assertEqual(result['prediction']['primary_adjusted_mae_ratio']['paired_complete_queries'],4)
        self.ledger.archive()
        manifest=json.loads((self.ledger.root/'artifacts.json').read_text())
        self.assertTrue(all(sha256(self.ledger.root/n)==h for n,h in manifest.items()))


if __name__=='__main__':unittest.main()
