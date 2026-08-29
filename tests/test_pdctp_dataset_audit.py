import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tri_rag_harness.pdctp_dataset_audit import (
    PDCTPDatasetAuditError,
    audit_pdctp_beir_source,
    load_pdctp_dataset_audit_config,
)
from tri_rag_harness.utils import fingerprint


ROOT = Path(__file__).resolve().parents[1]


class PDCTPDatasetAuditTests(unittest.TestCase):
    @staticmethod
    def _write_archive(path: Path, *, missing_doc=False, cross_test_text=False):
        corpus = [
            {"_id": f"d{i}", "title": f"Title {i}", "text": f"Body {i}"}
            for i in range(8)
        ]
        corpus[0] = {"_id": "d0", "title": "", "text": ""}
        queries = []
        for i in range(12):
            text = f"Train query {i}"
            if i == 1:
                text = "Repeated TRAIN query"
            if i == 2:
                text = "  repeated   train query "
            queries.append({"_id": f"tr{i}", "text": text})
        queries.extend(
            [
                {"_id": "dv0", "text": "Dev query"},
                {"_id": "dv1", "text": "Shared development query"},
                {"_id": "tr_shared", "text": "Ｓｈａｒｅｄ development query"},
                {"_id": "te0", "text": "Test query"},
                {"_id": "te1", "text": "Other test query"},
            ]
        )
        if cross_test_text:
            queries[0]["text"] = "test QUERY"

        def jsonl(rows):
            return "".join(json.dumps(row) + "\n" for row in rows)

        def qrels(rows):
            return "query-id\tcorpus-id\tscore\n" + "".join(
                f"{query_id}\t{doc_id}\t{score}\n"
                for query_id, doc_id, score in rows
            )

        train_ids = [f"tr{i}" for i in range(12)] + ["tr_shared"]
        train_rows = [
            (query_id, "missing" if missing_doc and index == 0 else f"d{index % 8}", 1)
            for index, query_id in enumerate(train_ids)
        ]
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("fixture/corpus.jsonl", jsonl(corpus))
            archive.writestr("fixture/queries.jsonl", jsonl(queries))
            archive.writestr("fixture/qrels/train.tsv", qrels(train_rows))
            archive.writestr(
                "fixture/qrels/dev.tsv", qrels([("dv0", "d0", 1), ("dv1", "d1", 1)])
            )
            archive.writestr(
                "fixture/qrels/test.tsv", qrels([("te0", "d2", 1), ("te1", "d3", 1)])
            )

    @staticmethod
    def _write_config(path: Path, archive: Path, *, sha256=None):
        data = archive.read_bytes()
        raw = {
            "schema_version": 1,
            "audit": "pdctp_beir_source_audit_v1",
            "dataset_name": "Fixture",
            "dataset_version": "fixture-v1",
            "id_namespace": "fixture-audit",
            "minimum_relevance": 1,
            "source": {
                "url": "https://example.invalid/fixture.zip",
                "metadata_url": "https://example.invalid/metadata",
                "archive_md5": hashlib.md5(data).hexdigest(),
                "archive_sha256": sha256 or hashlib.sha256(data).hexdigest(),
                "archive_bytes": len(data),
                "archive_root": "fixture",
                "corpus_member": "corpus.jsonl",
                "queries_member": "queries.jsonl",
                "qrels_members": {
                    "train": "qrels/train.tsv",
                    "dev": "qrels/dev.tsv",
                    "test": "qrels/test.tsv",
                },
            },
            "license": {
                "upstream_component": "fixture",
                "upstream_identifier": "non-commercial-use-only",
                "upstream_url": "https://example.invalid/license",
                "commercial_use_permitted": False,
                "redistributor_disclaimer_url": "https://example.invalid/disclaimer",
            },
            "role_plan": {
                "seed": 19,
                "query_cert_required": 4,
                "cal_fraction_of_remaining_train": 0.5,
                "minimum_role_sizes": {
                    "query_cal": 1,
                    "query_tune": 1,
                    "query_cert": 4,
                    "query_latency": 1,
                    "query_test": 1,
                },
            },
        }
        path.write_text(json.dumps(raw), encoding="utf-8")

    def test_audit_is_reproducible_and_proves_five_role_capacity(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            archive = directory / "fixture.zip"
            config_path = directory / "config.json"
            self._write_archive(archive, cross_test_text=True)
            self._write_config(config_path, archive)
            config = load_pdctp_dataset_audit_config(config_path)
            first = audit_pdctp_beir_source(config, archive, directory / "first")
            second = audit_pdctp_beir_source(config, archive, directory / "second")
            for name in first:
                self.assertEqual(first[name].read_bytes(), second[name].read_bytes())

            report = json.loads(first["source_audit.json"].read_text())
            witness = json.loads(
                first["role_feasibility_witness.json"].read_text()
            )
            capacity = report["eligibility"]["five_role_capacity"]
            self.assertTrue(capacity["passed"])
            self.assertTrue(capacity["ids_disjoint"])
            self.assertTrue(capacity["normalized_texts_disjoint"])
            self.assertEqual(capacity["role_counts"]["query_cert"], 4)
            self.assertEqual(capacity["role_counts"]["query_latency"], 3)
            self.assertEqual(
                capacity["excluded_non_test_queries_matching_test_text"], 1
            )
            self.assertFalse(witness["authorizes_method_evaluation"])
            self.assertEqual(set(witness["roles"]), {
                "query_cal",
                "query_tune",
                "query_cert",
                "query_latency",
                "query_test",
            })
            role_sets = [set(ids) for ids in witness["roles"].values()]
            self.assertFalse(
                any(
                    left.intersection(right)
                    for index, left in enumerate(role_sets)
                    for right in role_sets[index + 1 :]
                )
            )
            all_ids = set().union(*role_sets)
            self.assertNotIn("fixture-audit:query:tr0", all_ids)
            self.assertEqual(report["decision"], "GO_TO_PROTOCOL_FREEZE")
            self.assertEqual(report["counts"]["empty_corpus_items"], 1)
            self.assertEqual(
                report["counts"]["qrels"]["dev"][
                    "positive_rows_to_empty_corpus"
                ],
                1,
            )
            self.assertFalse(report["scope_guards"]["contains_query_or_corpus_text"])
            self.assertFalse(
                report["scope_guards"]["contains_retrieval_or_policy_outcomes"]
            )

    def test_archive_sha256_mismatch_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            archive = directory / "fixture.zip"
            config_path = directory / "config.json"
            output = directory / "output"
            self._write_archive(archive)
            self._write_config(config_path, archive, sha256="0" * 64)
            config = load_pdctp_dataset_audit_config(config_path)
            with self.assertRaisesRegex(
                PDCTPDatasetAuditError, "archive identity mismatch"
            ):
                audit_pdctp_beir_source(config, archive, output)
            self.assertFalse(output.exists())

    def test_missing_qrel_document_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            archive = directory / "fixture.zip"
            config_path = directory / "config.json"
            self._write_archive(archive, missing_doc=True)
            self._write_config(config_path, archive)
            config = load_pdctp_dataset_audit_config(config_path)
            with self.assertRaisesRegex(
                PDCTPDatasetAuditError, "missing_documents=.*missing"
            ):
                audit_pdctp_beir_source(config, archive, directory / "output")

    def test_checked_in_fiqa_config_pins_source_license_and_power(self):
        config = load_pdctp_dataset_audit_config(
            ROOT / "configs" / "pdctp_fiqa_source_audit_v1.json"
        )
        self.assertEqual(config.dataset_name, "BEIR FiQA-2018")
        self.assertEqual(
            config.source["archive_md5"], "17918ed23cd04fb15047f73e6c3bd9d9"
        )
        self.assertEqual(
            config.source["archive_sha256"],
            "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02",
        )
        self.assertFalse(config.license["commercial_use_permitted"])
        self.assertEqual(config.role_plan["query_cert_required"], 1567)
        self.assertEqual(
            config.config_fingerprint,
            "17b561b18b4c721f9bec843afe5351a5da038a4b928a97f5c41ad8c797b92487",
        )

    def test_checked_in_fiqa_audit_is_self_consistent_and_outcome_free(self):
        artifact_root = ROOT / "artifacts" / "pdctp_fiqa_source_audit_v1"
        report_path = artifact_root / "source_audit.json"
        witness_path = artifact_root / "role_feasibility_witness.json"
        report = json.loads(report_path.read_text())
        witness = json.loads(witness_path.read_text())

        report_without_fingerprint = dict(report)
        self.assertEqual(
            report_without_fingerprint.pop("fingerprint"),
            fingerprint(report_without_fingerprint),
        )
        witness_without_fingerprint = dict(witness)
        self.assertEqual(
            witness_without_fingerprint.pop("fingerprint"),
            fingerprint(witness_without_fingerprint),
        )
        self.assertEqual(
            report["eligibility"]["five_role_capacity"]["role_counts"],
            {
                "query_cal": 1966,
                "query_tune": 1967,
                "query_cert": 1567,
                "query_latency": 500,
                "query_test": 648,
            },
        )
        self.assertEqual(report["counts"]["empty_corpus_items"], 38)
        self.assertEqual(report["decision"], "GO_TO_PROTOCOL_FREEZE")
        self.assertFalse(report["scope_guards"]["authorizes_method_evaluation"])
        self.assertFalse(witness["authorizes_method_evaluation"])
        self.assertEqual(
            report["artifacts"]["role_feasibility_witness.json"]["sha256"],
            hashlib.sha256(witness_path.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
