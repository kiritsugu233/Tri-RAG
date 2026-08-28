import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_real_policy_certify import (
    _file_identity,
    _fixture as _policy_fixture,
    _jsonl,
)

from tri_rag_harness.real_policy_certify import (
    load_real_policy_certification_config,
)
from tri_rag_harness.real_tune_diagnostics import (
    RealTuneDiagnosticsError,
    load_real_tune_diagnostics_config,
    run_real_tune_diagnostics,
)
from tri_rag_harness.utils import fingerprint, stable_id_hash, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "real_scifact_tune_diagnostics.json"
BINDING_CONFIG_PATH = ROOT / "configs" / "real_scifact_policy_certify.json"


def _fixture(root):
    fixture = _policy_fixture(root)
    binding = load_real_policy_certification_config(fixture.config_path)
    query_rows = [
        json.loads(line)
        for line in (fixture.prepared / "queries.jsonl").read_text().splitlines()
    ]
    tune_ids = [row["query_id"] for row in query_rows if row["split"] == "query_tune"]
    corpus_ids = json.loads((fixture.cache / "corpus_ids.json").read_text())
    _jsonl(
        fixture.prepared / "qrels.jsonl",
        [
            {
                "query_id": tune_ids[0],
                "doc_id": corpus_ids[0],
                "relevance": 2,
                "split": "query_tune",
            },
            {
                "query_id": tune_ids[0],
                "doc_id": corpus_ids[1],
                "relevance": 1,
                "split": "query_tune",
            },
        ],
    )
    config_raw = {
        "schema_version": 1,
        "benchmark": "real_tune_only_evidence_diagnostics_v1",
        "policy_binding_config_fingerprint": binding.config_fingerprint,
        "policy_source_result_fingerprint": binding.policy_source.result_fingerprint,
        "evaluation_split": "query_tune",
        "query_split": {
            "n": len(tune_ids),
            "id_hash": stable_id_hash(tune_ids),
        },
        "evidence": {
            "source": "prepared_query_tune_qrels",
            "cutoffs": [1, 2],
            "candidate_metrics": ["evidence_hit", "evidence_recall"],
            "final_context_metrics": ["evidence_hit", "evidence_recall", "ndcg"],
            "empty_qrel_behavior": "record_null_metrics_and_do_not_drop_query",
            "include_exact_original_reference": True,
        },
        "matched_comparisons": {
            "fixed_cost_rule": "bracket_adaptive_mean_budget_on_frozen_grid",
            "fixed_quality_metrics": [
                "embedding_retention",
                "candidate_evidence_recall",
                "final_context_evidence_recall_at_k_ctx",
            ],
            "fixed_quality_rule": "smallest_frozen_fixed_budget_with_mean_metric_at_least_adaptive_mean",
            "role": "posthoc_query_tune_diagnostic_not_policy_selection",
        },
        "shuffled_lid": {
            "seed": 31,
            "repetitions": 7,
            "unit": "permute_pilot_lid_and_validity_pairs_across_query_tune_ids",
            "policies": ["monotone_binned", "tri_predict"],
            "metrics": [
                "embedding_retention",
                "candidate_evidence_recall",
                "final_context_evidence_recall_at_k_ctx",
            ],
            "p_value": "one_plus_controls_at_least_observed_over_repetitions_plus_one",
        },
        "reporting": {
            "role": "posthoc_tune_only_evidence_and_allocation_diagnostics",
            "protected_split_access": "forbidden",
            "policy_selection": "forbidden",
            "new_certification": "forbidden",
            "retuning": "forbidden",
        },
    }
    config_path = root / "diagnostics.json"
    write_json(config_path, config_raw)
    return fixture, binding, config_path, tune_ids


class RealTuneDiagnosticsTests(unittest.TestCase):
    def test_checked_in_config_is_tune_only_and_frozen(self):
        config = load_real_tune_diagnostics_config(CONFIG_PATH)
        binding = load_real_policy_certification_config(BINDING_CONFIG_PATH)
        self.assertEqual(config.evaluation_split, "query_tune")
        self.assertEqual(config.query_split_n, 403)
        self.assertEqual(config.shuffle_seed, 31013)
        self.assertEqual(config.shuffle_repetitions, 1000)
        self.assertEqual(config.policy_binding_config_fingerprint, binding.config_fingerprint)
        self.assertEqual(
            config.config_fingerprint,
            "1b5d8a47ebf64a42c0757cae1d460198a0b1a585044ed4c4d0e900f3972337c6",
        )

    def test_config_rejects_protected_scope_selection_and_recertification(self):
        original = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        mutations = (
            ("scope", "query_test"),
            ("protected", "allowed"),
            ("selection", "allowed"),
            ("certificate", "allowed"),
            ("retuning", "allowed"),
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            for index, (kind, value) in enumerate(mutations):
                changed = json.loads(json.dumps(original))
                if kind == "scope":
                    changed["evaluation_split"] = value
                elif kind == "protected":
                    changed["reporting"]["protected_split_access"] = value
                elif kind == "selection":
                    changed["reporting"]["policy_selection"] = value
                elif kind == "certificate":
                    changed["reporting"]["new_certification"] = value
                else:
                    changed["reporting"]["retuning"] = value
                path = directory / f"mutation-{index}.json"
                write_json(path, changed)
                with self.subTest(kind=kind):
                    with self.assertRaises(RealTuneDiagnosticsError):
                        load_real_tune_diagnostics_config(path)

    def test_tiny_diagnostics_are_auditable_tune_only_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            fixture, binding, config_path, tune_ids = _fixture(directory)
            config = load_real_tune_diagnostics_config(config_path)
            validation = {
                "dataset_manifest": fixture.dataset_manifest,
                "embedding_manifest": fixture.embedding_manifest,
                "request_fingerprint": fixture.request_fingerprint,
            }
            output_a = directory / "diagnostics-a"
            output_b = directory / "diagnostics-b"
            with patch(
                "tri_rag_harness.real_tune_diagnostics.load_text_embedding_config",
                return_value=fixture.embedding_config,
            ), patch(
                "tri_rag_harness.real_tune_diagnostics.validate_text_embedding_cache",
                return_value=validation,
            ):
                for output in (output_a, output_b):
                    run_real_tune_diagnostics(
                        config,
                        binding,
                        fixture.prepared,
                        Path("fixture-embedding-config.json"),
                        fixture.cache,
                        fixture.policy_run,
                        output,
                    )
            for name in (
                "per_query.jsonl",
                "fixed_grid.json",
                "shuffled_controls.jsonl",
                "summary.json",
                "report.md",
            ):
                self.assertEqual(
                    (output_a / name).read_bytes(), (output_b / name).read_bytes()
                )
            self.assertFalse((output_a / "certifications.json").exists())
            self.assertFalse((output_a / "selection.json").exists())
            records = [
                json.loads(line)
                for line in (output_a / "per_query.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual({row["split"] for row in records}, {"query_tune"})
            self.assertEqual(stable_id_hash([records[0]["query_id"]]), stable_id_hash(tune_ids))
            self.assertEqual(set(records[0]["fixed_grid"]), {"4", "8", "12", "20"})
            for entry in records[0]["fixed_grid"].values():
                self.assertEqual(entry["candidate_overlap"], entry["reranked_overlap"])
                self.assertIn("candidate_set_evidence", entry)
                self.assertIn("final_context_evidence", entry)
            controls = [
                json.loads(line)
                for line in (output_a / "shuffled_controls.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(controls), 7)
            summary = json.loads((output_a / "summary.json").read_text())
            self.assertEqual(summary["data_scope"], "query_tune_only")
            self.assertEqual(summary["protected_split_access"], "forbidden")
            self.assertEqual(summary["policy_selection"], "forbidden")
            self.assertEqual(summary["new_certification"], "forbidden")
            self.assertEqual(summary["retuning"], "forbidden")
            self.assertIn("matched_fixed", summary["policies"]["monotone_binned"])
            self.assertIn("matched_fixed", summary["policies"]["tri_predict"])
            manifest = json.loads((output_a / "manifest.json").read_text())
            for name, identity in manifest["result_artifacts"].items():
                self.assertEqual(identity, _file_identity(output_a / name))
            result_identity = {
                "config_fingerprint": manifest["config_fingerprint"],
                "policy_binding_config_fingerprint": manifest[
                    "policy_binding_config_fingerprint"
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
                "query_tune_id_hash": manifest["query_tune_id_hash"],
                "artifacts": manifest["result_artifacts"],
            }
            self.assertEqual(manifest["result_fingerprint"], fingerprint(result_identity))


if __name__ == "__main__":
    unittest.main()
