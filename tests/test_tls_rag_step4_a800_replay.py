import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import sys
from types import SimpleNamespace as NS
import numpy as np
from unittest.mock import patch

from tri_rag_harness import tls_rag_step4_a800_replay as a
from tri_rag_harness import tls_rag_step4_probe as v2
from tri_rag_harness import tls_rag_step4_retention_v3 as v3
from test_tls_rag_step4_probe import fixture
from test_tls_rag_step4_retention_v3 import write_bundle


class A800ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.replay=a.load_protocol();self.retention=v3.load_protocol()
        self.data,self.env,self.roles=fixture()
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(v2,'load_protocol',return_value=self.data))
        self.stack.enter_context(patch.object(v3,'load_protocol',return_value=self.retention))
        self.stack.enter_context(patch.object(a,'load_protocol',return_value=self.replay))
        self.previous=self.root/'input';self.previous.mkdir()
        bundle=self.previous/'bundle'
        write_bundle(bundle,self.data,self.env,self.roles)
        (bundle/'binding.json').unlink()
        v2.save(bundle/'role_audit.json',[])
        source=self.previous/'source';source.mkdir()
        for name in a.adapter.SOURCES:
            (source/name).write_text('synthetic source fixture\n')
        model=self.previous/'model_snapshot';model.mkdir();(model/'config.json').write_text('{}')
        transport=a.transport_record(a.OFFICIAL_ENDPOINT,self.data['embedding'])
        registration={'transport':transport,'protocol':self.replay,'data_protocol':self.data,
                      'source_code_sha256':a.sources(),'source_urls':a.adapter.SOURCES}
        v2.save(self.previous/'registration.json',registration)
        binding={'transport':transport,'model_sha256':{'config.json':v2.sha256(model/'config.json')},'schema':a.BINDING_SCHEMA,'replay_protocol':self.replay,'data_protocol':self.data,
            'source_code_sha256':a.sources(),'source_urls':a.adapter.SOURCES,
            'source_sha256':{p.name:v2.sha256(p) for p in source.iterdir()},
            'embedding':self.data['embedding'],'embedding_device':'cpu',
            'historical_probe_observed':True,'independent_validation':False,
            'registration_sha256':v2.sha256(self.previous/'registration.json'),
            'files':{p.name:v2.sha256(p) for p in bundle.iterdir()}}
        v2.save(bundle/'binding.json',binding)

    def edit_binding(self,**changes):
        path=self.previous/'bundle/binding.json'
        data=json.loads(path.read_text());data.update(changes);path.write_text(json.dumps(data))

    def execute(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return a.run_replay(self.previous,self.root/'output')

    def test_valid_new_binding_and_old_loader_rejection(self):
        _,binding,env,roles,check=a.validate_rebuilt_bundle(self.previous)
        check();self.assertEqual(roles,self.roles)
        self.assertEqual(env.upstream.corpus_ids,self.env.upstream.corpus_ids)
        with self.assertRaisesRegex(ValueError,'wrong bundle schema'):
            v2.load_bundle(self.previous/'bundle',self.data)

    def test_reject_old_schema_or_false_independence(self):
        self.edit_binding(schema='tls_rag_nfcorpus_probe_binding_v2')
        with self.assertRaisesRegex(ValueError,'provenance mismatch'):a.validate_rebuilt_bundle(self.previous)
        self.edit_binding(schema=a.BINDING_SCHEMA,independent_validation=True)
        with self.assertRaisesRegex(ValueError,'provenance mismatch'):a.validate_rebuilt_bundle(self.previous)

    def test_tampering_and_source_changes(self):
        _,_,_,_,check=a.validate_rebuilt_bundle(self.previous)
        path=self.previous/'bundle/queries.npy';path.write_bytes(path.read_bytes()+b'x')
        with self.assertRaisesRegex(ValueError,'input changed'):check()
        with self.assertRaisesRegex(ValueError,'hash mismatch'):a.validate_rebuilt_bundle(self.previous)

    def test_symlink_payload_rejected(self):
        path=self.previous/'bundle/roles.json';other=self.root/'roles.json'
        path.rename(other);path.symlink_to(other)
        with self.assertRaisesRegex(ValueError,'symlink'):a.validate_rebuilt_bundle(self.previous)

    def test_root_mount_link_is_supported(self):
        link=self.root/'mount';link.symlink_to(self.previous,target_is_directory=True)
        a.validate_rebuilt_bundle(link)

    def test_complete_replay_closure_and_no_independence_claim(self):
        original=a.guard.read_json;opened=[]
        def read(path):
            if str(path).endswith('.labels.json'):
                role=Path(path).name.removesuffix('.labels.json');opened.append(role)
                a.guard.check_closed(self.root/'output/engine_replay',role)
            return original(path)
        with patch.object(a.guard,'read_json',side_effect=read):result=self.execute()
        self.assertEqual(opened,['query_tune','query_probe'])
        self.assertEqual(result['status'],'replay_targets_met')
        self.assertTrue(result['historical_probe_observed'])
        self.assertFalse(result['independent_validation']);self.assertFalse(result['latency_claim'])
        self.assertEqual(result['schema'],a.RESULT_SCHEMA)
        self.assertTrue((self.previous/'a800-replay-attempt/reservation.json').exists())
        with self.assertRaisesRegex(ValueError,'already attempted'):
            a.run_replay(self.previous,self.root/'another-output')
        manifest=json.loads((self.root/'output/artifact_manifest.json').read_text())
        self.assertTrue(all(v2.sha256(self.root/'output'/n)==h for n,h in manifest.items()))

    def test_failed_tune_does_not_open_probe(self):
        protocol=copy.deepcopy(self.retention);protocol['acceptance']['minimum_retention']=1.1
        with patch.object(v3,'load_protocol',return_value=protocol),patch.object(v3,'PROTOCOL_FP',v2.fingerprint(protocol)):
            result=self.execute()
        self.assertEqual(result['status'],'no_tune_candidate')
        self.assertFalse(any((self.root/'output/engine_replay').glob('query_probe.*')))

    def test_existing_output_and_nested_output_rejected(self):
        (self.root/'output').mkdir()
        with self.assertRaises(FileExistsError):self.execute()
        with self.assertRaisesRegex(ValueError,'separate'):
            a.run_replay(self.previous,self.previous/'result')

    def test_interruption_preserves_attempt(self):
        with patch.object(v2,'prepare_role',side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):self.execute()
        with self.assertRaisesRegex(ValueError,'already attempted'):
            a.run_replay(self.previous,self.root/'retry')

    def test_script_explicit_exit_handling(self):
        script=a.ROOT/'scripts/run_tls_rag_step4_repaired_acceptance.sh'
        self.assertNotIn('set -e',script.read_text())
        proc=subprocess.run(['sh',str(script)],cwd=a.ROOT,capture_output=True)
        self.assertEqual(proc.returncode,2)

    def test_preparation_uses_cpu_pinned_model_and_new_schema(self):
        target=self.root/'prepared'
        corpus=[{'_id':qid,'text':text} for qid,text in
                zip(self.env.upstream.corpus_ids,self.env.upstream.corpus_texts)]
        queries=[{'_id':q.query_id,'text':q.text} for q in self.env.queries]
        all_vectors={q.text:q.embedding for q in self.env.queries}
        all_vectors.update(dict(zip(self.env.upstream.corpus_texts,self.env.upstream.corpus_embeddings)))
        calls=[]; downloads=[]; snapshots=[]
        class Model:
            def __init__(inner,*args,**kwargs):calls.append(kwargs)
            def float(inner):return inner
            def eval(inner):return inner
            def encode(inner,texts,**kwargs):
                self.assertEqual(kwargs['batch_size'],32)
                return np.vstack([all_vectors[text] for text in texts])
        def table(path):
            return NS(to_pylist=lambda:corpus if path.name=='corpus.parquet' else queries)
        parquet=NS(read_table=table)
        def snapshot(**kwargs):
            snapshots.append(kwargs)
            path=Path(kwargs['local_dir']);path.mkdir();(path/'config.json').write_text('{}');(path/'1_Pooling').mkdir();(path/'1_Pooling/config.json').write_text('{}')
        modules={'huggingface_hub':NS(snapshot_download=snapshot),'torch':NS(manual_seed=lambda seed:None,use_deterministic_algorithms=lambda value:None),
                 'pyarrow':NS(parquet=parquet),'pyarrow.parquet':parquet,
                 'sentence_transformers':NS(SentenceTransformer=Model)}
        def download(url,path):
            downloads.append(url)
            path.write_text('synthetic pinned source')
            return v2.sha256(path)
        versions={'numpy':'1.26.4','scipy':'1.13.0','sentence-transformers':'3.4.1',
                  'transformers':'4.48.3','pyarrow':'19.0.1'}
        qrels=[(q.query_id,self.env.upstream.corpus_ids[0],1) for q in self.env.queries]
        with patch.dict(a.os.environ,{'TLS_RAG_HF_ENDPOINT':'https://hf-mirror.com','HF_ENDPOINT':'https://hf-mirror.com'}),patch.dict(sys.modules,modules),patch.object(a.importlib.metadata,'version',side_effect=versions.__getitem__), \
             patch.object(a.adapter,'download',side_effect=download), \
             patch.object(a.adapter,'read_train_qrels',return_value=({q.query_id for q in self.env.queries},qrels)), \
             patch.object(a,'environment_record',return_value={'synthetic':True}),contextlib.redirect_stdout(io.StringIO()):
            a.prepare_rebuilt_bundle(target)
        self.assertEqual(calls[0]['device'],'cpu')
        self.assertEqual(snapshots[0]['revision'],self.data['embedding']['revision'])
        self.assertEqual(snapshots[0]['endpoint'],'https://hf-mirror.com')
        self.assertTrue(calls[0]['local_files_only'])
        self.assertEqual(downloads,list(a.transport_record('https://hf-mirror.com',self.data['embedding'])['source_urls'].values()))
        _,binding,_,roles,_=a.validate_rebuilt_bundle(target)
        self.assertEqual(binding['schema'],a.BINDING_SCHEMA)
        self.assertEqual(roles,self.roles)
        self.assertFalse((target/'evaluation').exists())
        with self.assertRaises(FileExistsError):a.prepare_rebuilt_bundle(target)

    def test_endpoint_allowlist_and_mapping(self):
        for endpoint in a.ALLOWED_ENDPOINTS:
            record=a.transport_record(endpoint,self.data['embedding'])
            for name,url in a.adapter.SOURCES.items():
                self.assertEqual(record['source_urls'][name],endpoint+url[len(a.OFFICIAL_ENDPOINT):])
        for endpoint in ('', 'http://hf-mirror.com', 'https://hf-mirror.com/',
                         'https://hf-mirror.com.evil', 'https://user@hf-mirror.com',
                         'https://hf-mirror.com?x=1', ' https://hf-mirror.com'):
            with self.assertRaises(ValueError):a.transport_record(endpoint,self.data['embedding'])
        with patch.dict(a.os.environ,{'HF_ENDPOINT':'https://hf-mirror.com','TLS_RAG_HF_ENDPOINT':a.OFFICIAL_ENDPOINT}):
            with self.assertRaisesRegex(ValueError,'conflicts'):a.declared_transport(self.data['embedding'])

    def test_transport_tampering_rejected(self):
        record=a.transport_record(a.OFFICIAL_ENDPOINT,self.data['embedding'])
        record['source_urls']['corpus.parquet']='https://hf-mirror.com/wrong'
        self.edit_binding(transport=record)
        with self.assertRaisesRegex(ValueError,'transport provenance'):a.validate_rebuilt_bundle(self.previous)

    def test_raw_source_and_model_hashes_checked(self):
        for path in (self.previous/'source/corpus.parquet', self.previous/'model_snapshot/config.json'):
            old=path.read_bytes();path.write_bytes(old+b'x')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):a.validate_rebuilt_bundle(self.previous)
            path.write_bytes(old)
        _,_,_,_,check=a.validate_rebuilt_bundle(self.previous)
        (self.previous/'model_snapshot/config.json').write_text('changed')
        with self.assertRaisesRegex(ValueError,'input changed'):check()

    def test_config_is_frozen(self):
        path=self.root/'config.json';path.write_text('{}')
        self.stack.close()
        with patch.object(a,'CONFIG',path),self.assertRaisesRegex(ValueError,'protocol changed'):
            a.load_protocol()


if __name__=='__main__':unittest.main()
