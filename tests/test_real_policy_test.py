import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_real_policy_certify import _fixture as _cert_fixture

from tri_rag_harness.real_policy_certify import (
    load_real_policy_certification_config,
    run_real_policy_certification,
)
from tri_rag_harness.real_policy_test import (
    RealPolicyTestError,
    load_real_policy_test_config,
    run_real_policy_test,
)
from tri_rag_harness.utils import fingerprint, stable_id_hash, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "real_scifact_policy_test.json"
CERT_CONFIG_PATH = ROOT / "configs" / "real_scifact_policy_certify.json"


def _jsonl(path, rows):
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _file_identity(path):
    import hashlib

    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _fixture(root):
    fixture = _cert_fixture(root)
    certification_config = load_real_policy_certification_config(
        fixture.config_path
    )
    validation = {
        "dataset_manifest": fixture.dataset_manifest,
        "embedding_manifest": fixture.embedding_manifest,
        "request_fingerprint": fixture.request_fingerprint,
    }
    certification_run = root / "certification"
    with patch(
        "tri_rag_harness.real_policy_certify.load_text_embedding_config",
        return_value=fixture.embedding_config,
    ), patch(
        "tri_rag_harness.real_policy_certify.validate_text_embedding_cache",
        return_value=validation,
    ):
        run_real_policy_certification(
            certification_config,
            fixture.prepared,
            Path("fixture-embedding-config.json"),
            fixture.cache,
            fixture.policy_run,
            certification_run,
        )

    query_rows = [
        json.loads(line)
        for line in (fixture.prepared / "queries.jsonl").read_text().splitlines()
    ]
    test_ids = [
        row["query_id"] for row in query_rows if row["split"] == "query_test"
    ]
    corpus_ids = json.loads((fixture.cache / "corpus_ids.json").read_text())
    _jsonl(
        fixture.prepared / "qrels.jsonl",
        [
            {
                "query_id": test_ids[0],
                "doc_id": corpus_ids[0],
                "relevance": 2,
                "split": "query_test",
            },
            {
                "query_id": test_ids[0],
                "doc_id": corpus_ids[1],
                "relevance": 1,
                "split": "query_test",
            },
            {
                "query_id": test_ids[1],
                "doc_id": corpus_ids[2],
                "relevance": 1,
                "split": "query_test",
            },
        ],
    )
    cert_manifest = json.loads(
        (certification_run / "manifest.json").read_text(encoding="utf-8")
    )
    certifications = json.loads(
        (certification_run / "certifications.json").read_text(encoding="utf-8")
    )
    decisions = {
        name: "PASS" if value["passed"] else "FAIL"
        for name, value in certifications["certificates"].items()
    }
    config_raw = {
        "schema_version": 1,
        "benchmark": "real_frozen_policy_test_v1",
        "certification_config_fingerprint": certification_config.config_fingerprint,
        "evaluation_split": "query_test",
        "query_split": {
            "n": len(test_ids),
            "id_hash": stable_id_hash(test_ids),
        },
        "certification_source": {
            "manifest_fingerprint": cert_manifest["fingerprint"],
            "result_fingerprint": cert_manifest["result_fingerprint"],
            "certificates_fingerprint": certifications["fingerprint"],
            "query_cert_n": cert_manifest["query_cert_n"],
            "query_cert_id_hash": cert_manifest["query_cert_id_hash"],
            "terminal": True,
            "failure_behavior": "terminal_no_retuning_no_budget_expansion",
            "decisions": decisions,
        },
        "evidence": {
            "source": "prepared_query_test_qrels",
            "cutoffs": [1, 2],
            "include_exact_original_reference": True,
        },
        "reporting": {
            "role": "descriptive_frozen_policy_test_no_selection_no_certificate",
            "policies": [
                "fixed_reference",
                "monotone_binned",
                "tri_predict",
            ],
            "post_test_selection": "forbidden",
            "new_certification": "forbidden",
            "retuning": "forbidden",
        },
    }
    test_config_path = root / "test.json"
    write_json(test_config_path, config_raw)
    return SimpleNamespace(
        **vars(fixture),
        certification_config=certification_config,
        certification_run=certification_run,
        test_config_path=test_config_path,
        validation=validation,
        test_ids=test_ids,
    )


class RealPolicyTestTests(unittest.TestCase):
    def test_checked_in_protocol_freezes_test_and_terminal_cert_identities(self):
        config = load_real_policy_test_config(CONFIG_PATH)
        certification_config = load_real_policy_certification_config(
            CERT_CONFIG_PATH
        )
        self.assertEqual(config.evaluation_split, "query_test")
        self.assertEqual(config.query_split_n, 300)
        self.assertEqual(
            config.query_split_id_hash,
            "d1e72b0e72d42e4753b016fb92d25ceac745db61439e470a2e79936d15dd260b",
        )
        self.assertEqual(
            config.certification_source.result_fingerprint,
            "81e1e984a735215a9faa99a50991b51dd28c73b1a11e9fa24a0d6e8785088c4d",
        )
        self.assertEqual(
            config.certification_source.decisions,
            {
                "fixed_reference": "PASS",
                "monotone_binned": "PASS",
                "tri_predict": "FAIL",
            },
        )
        self.assertEqual(
            config.certification_config_fingerprint,
            certification_config.config_fingerprint,
        )
        self.assertEqual(
            config.config_fingerprint,
            "149eac226a2e948b3a56d0eff09217f72e28e18f490efe44cfc690bf4b318bbc",
        )

    def test_config_rejects_non_test_scope_selection_certificate_and_retuning(self):
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        mutations = (
            ("scope", "query_cert"),
            ("selection", "allowed"),
            ("certificate", "allowed"),
            ("retuning", "allowed"),
            ("reference", False),
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            for index, (kind, value) in enumerate(mutations):
                changed = json.loads(json.dumps(raw))
                if kind == "scope":
                    changed["evaluation_split"] = value
                elif kind == "selection":
                    changed["reporting"]["post_test_selection"] = value
                elif kind == "certificate":
                    changed["reporting"]["new_certification"] = value
                elif kind == "retuning":
                    changed["reporting"]["retuning"] = value
                else:
                    changed["evidence"]["include_exact_original_reference"] = value
                path = directory / f"mutation-{index}.json"
                write_json(path, changed)
                with self.subTest(kind=kind):
                    with self.assertRaises(RealPolicyTestError):
                        load_real_policy_test_config(path)

    def test_certification_tamper_is_rejected_before_test_data_access(self):
        with tempfile.TemporaryDirectory() as directory_name:
            fixture = _fixture(Path(directory_name))
            path = fixture.certification_run / "certifications.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["terminal"] = False
            write_json(path, value)
            config = load_real_policy_test_config(fixture.test_config_path)
            with patch(
                "tri_rag_harness.real_policy_test.validate_text_embedding_cache",
                side_effect=AssertionError("protected test cache must not be accessed"),
            ):
                with self.assertRaisesRegex(
                    RealPolicyTestError, "certification result artifact identity"
                ):
                    run_real_policy_test(
                        config,
                        fixture.certification_config,
                        fixture.prepared,
                        Path("unused-embedding-config.json"),
                        fixture.cache,
                        fixture.policy_run,
                        fixture.certification_run,
                        Path(directory_name) / "out",
                    )

    def test_tiny_test_is_descriptive_auditable_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            fixture = _fixture(directory)
            config = load_real_policy_test_config(fixture.test_config_path)
            output_a = directory / "test-a"
            output_b = directory / "test-b"
            with patch(
                "tri_rag_harness.real_policy_test.load_text_embedding_config",
                return_value=fixture.embedding_config,
            ), patch(
                "tri_rag_harness.real_policy_test.validate_text_embedding_cache",
                return_value=fixture.validation,
            ):
                run_real_policy_test(
                    config,
                    fixture.certification_config,
                    fixture.prepared,
                    Path("fixture-embedding-config.json"),
                    fixture.cache,
                    fixture.policy_run,
                    fixture.certification_run,
                    output_a,
                )
                run_real_policy_test(
                    config,
                    fixture.certification_config,
                    fixture.prepared,
                    Path("fixture-embedding-config.json"),
                    fixture.cache,
                    fixture.policy_run,
                    fixture.certification_run,
                    output_b,
                )
            for name in (
                "manifest.json",
                "per_query.jsonl",
                "summary.json",
                "report.md",
            ):
                self.assertEqual(
                    (output_a / name).read_bytes(), (output_b / name).read_bytes()
                )
            self.assertFalse((output_a / "certifications.json").exists())
            records = [
                json.loads(line)
                for line in (output_a / "per_query.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual({row["split"] for row in records}, {"query_test"})
            self.assertEqual(
                stable_id_hash([row["query_id"] for row in records]),
                stable_id_hash(fixture.test_ids),
            )
            for row in records:
                relevance = row["relevance_by_doc_id"]
                self.assertTrue(relevance)
                self.assertIn("exact_original_evidence_metrics", row)
                for policy in row["policies"].values():
                    self.assertEqual(
                        policy["candidate_overlap"], policy["reranked_overlap"]
                    )
                    self.assertEqual(
                        policy["embedding_retention"],
                        policy["candidate_overlap"] / 2,
                    )
                    self.assertIn("evidence_metrics", policy)
                self.assertTrue(
                    row["policies"]["tri_predict"]["compiled_decision_match"]
                )
            summary = json.loads((output_a / "summary.json").read_text())
            self.assertEqual(summary["data_scope"], "query_test_only")
            self.assertEqual(
                summary["evaluation_role"],
                "descriptive_frozen_policy_test_no_selection_no_certificate",
            )
            self.assertEqual(summary["post_test_selection"], "forbidden")
            self.assertEqual(summary["new_certification"], "forbidden")
            self.assertEqual(summary["retuning"], "forbidden")
            self.assertEqual(
                summary["certification_source"]["decisions"],
                config.certification_source.decisions,
            )
            for value in summary["policies"].values():
                self.assertEqual(value["work_per_query"]["projected_scan_count"], 1)
                self.assertIn("evidence_metrics", value)
                self.assertNotIn("certificate", value)
            timings = json.loads((output_a / "timings.json").read_text())
            self.assertEqual(timings["projected_scan_count_per_query"], 1)
            manifest = json.loads((output_a / "manifest.json").read_text())
            for name, identity in manifest["result_artifacts"].items():
                self.assertEqual(identity, _file_identity(output_a / name))
            result_identity = {
                "config_fingerprint": manifest["config_fingerprint"],
                "certification_config_fingerprint": manifest[
                    "certification_config_fingerprint"
                ],
                "dataset_manifest_fingerprint": manifest[
                    "dataset_manifest_fingerprint"
                ],
                "embedding_manifest_fingerprint": manifest[
                    "embedding_manifest_fingerprint"
                ],
                "policy_source_result_fingerprint": manifest["policy_source"][
                    "result_fingerprint"
                ],
                "compiled_deployment_fingerprint": manifest["deployment"][
                    "policy_fingerprint"
                ],
                "certification_result_fingerprint": manifest[
                    "certification_source"
                ]["result_fingerprint"],
                "query_test_id_hash": manifest["query_test_id_hash"],
                "artifacts": manifest["result_artifacts"],
            }
            self.assertEqual(
                manifest["result_fingerprint"], fingerprint(result_identity)
            )


if __name__ == "__main__":
    unittest.main()
