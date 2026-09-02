import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from tri_rag_harness.pdctp_embedding_audit import (
    PDCTPEmbeddingAuditError,
    audit_pdctp_embedding_cache,
)
from tri_rag_harness.pdctp_fiqa_dataset import (
    FIVE_ROLES,
    PDCTPFiQADatasetError,
    prepare_pdctp_fiqa_text_inputs,
)
from tri_rag_harness.text_embeddings import (
    build_or_load_text_embedding_cache,
    load_text_embedding_config,
)
from tri_rag_harness.utils import fingerprint, write_json


ROOT = Path(__file__).resolve().parents[1]


def _identity(path):
    value = path.read_bytes()
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def _archive_identity(path):
    value = path.read_bytes()
    return {
        "bytes": len(value),
        "md5": hashlib.md5(value, usedforsecurity=False).hexdigest(),  # nosec B324
        "sha256": hashlib.sha256(value).hexdigest(),
    }


def _member_identity(archive, name):
    value = archive.read(name)
    return {"path": name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def _fingerprinted(path, value):
    value = dict(value)
    value["fingerprint"] = fingerprint(value)
    write_json(path, value)
    return value


class _FrozenFixture:
    def __init__(self, root):
        self.root = root
        self.archive = root / "fiqa.zip"
        corpus = [
            {"_id": "d0", "title": " T0 ", "text": " body zero "},
            {"_id": "d1", "title": "", "text": ""},
        ]
        queries = [{"_id": f"q{i}", "text": f" query {i} "} for i in range(5)]
        with zipfile.ZipFile(self.archive, "w") as output:
            for name, rows in (("fiqa/corpus.jsonl", corpus), ("fiqa/queries.jsonl", queries)):
                info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
                output.writestr(
                    info,
                    "".join(
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                        for row in rows
                    ).encode(),
                )
        with zipfile.ZipFile(self.archive) as archive:
            members = {
                "corpus": _member_identity(archive, "fiqa/corpus.jsonl"),
                "queries": _member_identity(archive, "fiqa/queries.jsonl"),
            }
        source_base = {
            "source": {"archive": _archive_identity(self.archive), "members": members},
            "counts": {"corpus": 2, "source_queries": 5},
        }
        self.source = root / "source.json"
        source = _fingerprinted(self.source, source_base)

        roles_value = {}
        for index, role in enumerate(FIVE_ROLES):
            ids = [f"pdctp-beir-fiqa:query:q{index}"]
            roles_value[role] = {
                "n": 1,
                "ordered_id_hash": fingerprint(ids),
                "ordered_ids": ids,
            }
        self.roles = root / "roles.json"
        roles = _fingerprinted(
            self.roles,
            {
                "roles": roles_value,
                "all_roles_initially_closed": True,
                "authorizes_outcome_access": False,
            },
        )

        self.continuity_config = root / "continuity.json"
        raw_config = json.loads(
            (ROOT / "configs" / "real_scifact_e5_base_v2_embeddings.json").read_text()
        )
        raw_config["dataset_manifest_fingerprint"] = "0" * 64
        raw_config["model"]["embedding_dimension"] = 4
        raw_config["model"]["snapshot_allow_patterns"] = ["config.json"]
        raw_config["encoding"]["batch_size"] = 2
        raw_config["runtime"]["required_packages"] = {"fixture-runtime": "1.0"}
        write_json(self.continuity_config, raw_config)
        protocol_embedding = {
            "continuity_source_config_sha256": _identity(self.continuity_config)["sha256"],
            "model": {
                key: raw_config["model"][key]
                for key in (
                    "name",
                    "revision",
                    "embedding_dimension",
                    "max_sequence_length",
                    "trust_remote_code",
                )
            },
            "formatting": raw_config["formatting"],
            "encoding": {
                key: raw_config["encoding"][key]
                for key in (
                    "model_dtype",
                    "output_dtype",
                    "l2_normalize",
                    "deterministic_algorithms",
                    "allow_tf32",
                    "attention_implementation",
                    "cublas_workspace_config",
                )
            },
            "runtime_packages": {"fixture-runtime": "1.0"},
        }
        self.config_fingerprint = "a" * 64
        protocol_base = {
            "config_fingerprint": self.config_fingerprint,
            "decision": "READY_FOR_DATASET_AND_EMBEDDING_AUDIT_ONLY",
            "authorizes_method_evaluation": False,
            "authorizes_protected_outcome_access": False,
            "resolved_inputs": {
                "source_audit": {**_identity(self.source), "fingerprint": source["fingerprint"]}
            },
            "resolved_roles": {
                "assignment_fingerprint": roles["fingerprint"],
                "counts": {role: 1 for role in FIVE_ROLES},
                "ordered_id_hashes": {
                    role: roles_value[role]["ordered_id_hash"] for role in FIVE_ROLES
                },
            },
            "protocol": {
                "source_gate": {"source_audit_fingerprint": source["fingerprint"]},
                "dataset": {
                    "name": "fixture FiQA",
                    "version": "fixture-v1",
                    "id_namespace": "pdctp-beir-fiqa",
                    "archive_sha256": _archive_identity(self.archive)["sha256"],
                    "corpus_size": 2,
                    "source_query_count": 5,
                    "empty_documents": {
                        "source_count": 1,
                        "policy": "replace_empty_title_and_text_with_marker_v1",
                        "replacement_text": "[EMPTY_DOCUMENT]",
                        "formatted_embedding_text": "passage: [EMPTY_DOCUMENT]",
                        "silent_deletion_allowed": False,
                    },
                },
                "embedding": protocol_embedding,
            },
        }
        self.protocol = root / "protocol.json"
        protocol = _fingerprinted(self.protocol, protocol_base)
        self.protocol_fingerprint = protocol["fingerprint"]
        self.state = root / "state.json"
        state = _fingerprinted(
            self.state,
            {
                "config_fingerprint": self.config_fingerprint,
                "calibration_opened": False,
                "certification_opened": False,
                "latency_opened": False,
                "test_opened": False,
                "selection_fingerprint": None,
                "certification_result_fingerprint": None,
                "latency_result_fingerprint": None,
            },
        )
        protocol_body = json.loads(self.protocol.read_text())
        protocol_body.pop("fingerprint")
        protocol_body["initial_guard_state_fingerprint"] = state["fingerprint"]
        _fingerprinted(self.protocol, protocol_body)
        self.protocol_fingerprint = json.loads(self.protocol.read_text())["fingerprint"]

    def prepare(self, output):
        return prepare_pdctp_fiqa_text_inputs(
            self.protocol, self.roles, self.source, self.archive, output
        )

    def embedding_config(self, dataset_manifest, path):
        raw = json.loads(self.continuity_config.read_text())
        raw["dataset_manifest_fingerprint"] = dataset_manifest["fingerprint"]
        write_json(path, raw)
        return load_text_embedding_config(path)


class _Provider:
    def __init__(self, dimension=4):
        self.dimension = dimension
        self.stats = {}

    def encode(self, texts, *, batch_size, role):
        self.stats[role] = {
            "n": len(texts),
            "minimum": 3,
            "maximum": 9,
            "mean": 5.0,
            "p95": 8.0,
            "max_sequence_length": 512,
            "truncated": 0,
            "truncated_fraction": 0.0,
        }
        return np.asarray(
            [[len(text) + 1, sum(text.encode()) % 101 + 1, index + 1, 3] for index, text in enumerate(texts)],
            dtype=np.float32,
        )

    def metadata(self):
        snapshot = {"config.json": {"bytes": 12, "sha256": "f" * 64}}
        return {
            "provider": "sentence_transformers_v1",
            "model": {
                "name": "intfloat/e5-base-v2",
                "revision": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
                "snapshot_files": snapshot,
                "snapshot_fingerprint": fingerprint(snapshot),
            },
            "runtime": {
                "packages": {"fixture-runtime": "1.0"},
                "device": {"requested": "cuda", "resolved": "cuda:0"},
                "deterministic_algorithms": True,
                "allow_tf32": False,
                "model_dtype": "float32",
                "attention_implementation": "eager",
                "cublas_workspace_config": ":4096:8",
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "input_token_lengths": self.stats,
            },
        }


class PDCTPFiQAEmbeddingGateTests(unittest.TestCase):
    def test_checked_in_real_embedding_audit_is_self_consistent(self):
        audit_path = (
            ROOT / "artifacts" / "pdctp_fiqa_e5_v1" / "embedding_audit.json"
        )
        audit = json.loads(audit_path.read_text())
        claimed = audit.pop("fingerprint")
        self.assertEqual(
            claimed,
            "54af315d5b94b43a81be71ea29ab860635f0748a97108e0cda120a510947dd71",
        )
        self.assertEqual(fingerprint(audit), claimed)
        self.assertEqual(audit["decision"], "READY_TO_OPEN_QUERY_CAL")
        self.assertEqual(
            audit["embedding_manifest_fingerprint"],
            "079545ef7c6af8ab27a5c8382dbd8174905f1bb537df59e94d572b6c2f2b04c1",
        )
        self.assertEqual(audit["input_token_lengths"]["corpus"]["truncated"], 2446)
        self.assertEqual(audit["input_token_lengths"]["queries"]["truncated"], 0)
        self.assertTrue(audit["checks"]["all_roles_remained_closed"])
        self.assertFalse(audit["scope_guards"]["contains_qrels_or_relevance"])
        self.assertEqual(
            hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "c0a98c945c2efc70ac376a684b575127a0184086524b246c59e06ead834e811f",
        )

    def test_checked_in_embedding_request_pins_real_text_manifest_and_e5(self):
        config = load_text_embedding_config(
            ROOT / "configs" / "pdctp_fiqa_e5_base_v2_embeddings.json"
        )
        self.assertEqual(
            config.config_fingerprint,
            "dce9c5f590c0348672dc3ab6f90a8e07e5b170c2174a5c2aab5b9eaeabc8bc78",
        )
        self.assertEqual(
            config.dataset_manifest_fingerprint,
            "bfc25daad8d2d382390a0a42c3aa03b96e965965ba17c2065aaf8bef00903240",
        )
        self.assertEqual(config.model.name, "intfloat/e5-base-v2")
        self.assertEqual(
            config.model.revision,
            "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        )
        self.assertEqual(config.model.embedding_dimension, 768)
        self.assertTrue(config.encoding.l2_normalize)
        self.assertTrue(config.encoding.deterministic_algorithms)
        self.assertFalse(config.encoding.allow_tf32)

    def test_text_only_preparation_is_deterministic_and_keeps_empty_marker(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            fixture = _FrozenFixture(root)
            first, second = root / "first", root / "second"
            fixture.prepare(first)
            fixture.prepare(second)
            for name in (
                "corpus.jsonl",
                "queries.jsonl",
                "splits.json",
                "empty_documents.json",
                "formatted_text_hashes.json",
                "dataset_manifest.json",
            ):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            self.assertFalse((first / "qrels.jsonl").exists())
            corpus = [json.loads(line) for line in (first / "corpus.jsonl").read_text().splitlines()]
            self.assertEqual(corpus[1]["text"], "[EMPTY_DOCUMENT]")
            self.assertTrue(corpus[1]["source_content_empty"])
            manifest = json.loads((first / "dataset_manifest.json").read_text())
            self.assertEqual(manifest["adapter"], "pdctp_fiqa_text_only_v1")
            self.assertFalse(manifest["scope_guards"]["contains_qrels"])
            self.assertEqual(manifest["counts"], {"corpus": 2, "queries": 5, "empty_corpus_items": 1})

    def test_preparation_rejects_changed_archive_before_publication(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            fixture = _FrozenFixture(root)
            with fixture.archive.open("ab") as handle:
                handle.write(b"changed")
            output = root / "output"
            with self.assertRaisesRegex(PDCTPFiQADatasetError, "archive identity mismatch"):
                fixture.prepare(output)
            self.assertFalse(output.exists())

    def test_embedding_build_accepts_text_only_profile_and_audit_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            fixture = _FrozenFixture(root)
            prepared = root / "prepared"
            fixture.prepare(prepared)
            dataset_manifest = json.loads((prepared / "dataset_manifest.json").read_text())
            embedding_path = root / "embedding.json"
            config = fixture.embedding_config(dataset_manifest, embedding_path)
            cache = root / "cache"
            build_or_load_text_embedding_cache(config, prepared, cache, provider=_Provider())
            outputs = []
            for name in ("audit-one", "audit-two"):
                output = root / name
                audit_pdctp_embedding_cache(
                    fixture.protocol,
                    fixture.state,
                    fixture.roles,
                    fixture.continuity_config,
                    embedding_path,
                    prepared,
                    cache,
                    output,
                )
                outputs.append(output)
            for name in ("embedding_audit.json", "report.md"):
                self.assertEqual((outputs[0] / name).read_bytes(), (outputs[1] / name).read_bytes())
            audit = json.loads((outputs[0] / "embedding_audit.json").read_text())
            self.assertEqual(audit["decision"], "READY_TO_OPEN_QUERY_CAL")
            self.assertTrue(audit["checks"]["all_roles_remained_closed"])

    def test_independent_audit_rejects_tampered_cache_array(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            fixture = _FrozenFixture(root)
            prepared = root / "prepared"
            fixture.prepare(prepared)
            dataset_manifest = json.loads((prepared / "dataset_manifest.json").read_text())
            embedding_path = root / "embedding.json"
            config = fixture.embedding_config(dataset_manifest, embedding_path)
            cache = root / "cache"
            build_or_load_text_embedding_cache(config, prepared, cache, provider=_Provider())
            array_path = cache / "query_embeddings.f32.npy"
            values = np.load(array_path)
            values[0] *= np.float32(-1.0)
            np.save(array_path, values, allow_pickle=False)
            with self.assertRaisesRegex(PDCTPEmbeddingAuditError, "metadata mismatch"):
                audit_pdctp_embedding_cache(
                    fixture.protocol,
                    fixture.state,
                    fixture.roles,
                    fixture.continuity_config,
                    embedding_path,
                    prepared,
                    cache,
                    root / "audit",
                )


if __name__ == "__main__":
    unittest.main()
