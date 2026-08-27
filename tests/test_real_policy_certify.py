import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tri_rag_harness.certification import make_certificate
from tri_rag_harness.policies import (
    CompiledTriPredictPolicy,
    FixedBudgetPolicy,
    MonotoneBinnedPolicy,
    TriPredictPolicy,
)
from tri_rag_harness.projection import projection_metadata
from tri_rag_harness.real_policy_certify import (
    RealPolicyCertificationError,
    load_real_policy_certification_config,
    run_real_policy_certification,
)
from tri_rag_harness.utils import array_fingerprint, fingerprint, stable_id_hash, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "real_scifact_policy_certify.json"
FEATURE_VERSION = "pilot_rerank_lid_rounded_9_v2"
COMPILED_ROLE = "platform_deployment_artifact_excluded_from_scientific_identity"


def _file_identity(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _jsonl(path, rows):
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _embedded(value, field="fingerprint"):
    result = dict(value)
    result[field] = fingerprint(result)
    return result


def _fixture(root):
    prepared = root / "prepared"
    cache = root / "cache"
    policy_run = root / "policy"
    prepared.mkdir()
    cache.mkdir()
    policy_run.mkdir()

    rng = np.random.default_rng(19)
    corpus = rng.normal(size=(20, 4))
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.normal(size=(35, 4))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    corpus = corpus.astype(np.float32)
    queries = queries.astype(np.float32)
    corpus_ids = [f"fixture:doc:{index:02d}" for index in range(len(corpus))]
    tune_ids = ["fixture:query:tune"]
    cert_ids = [f"fixture:query:cert:{index:02d}" for index in range(32)]
    test_ids = ["fixture:query:test:00", "fixture:query:test:01"]
    query_ids = tune_ids + cert_ids + test_ids
    np.save(cache / "corpus_embeddings.f32.npy", corpus, allow_pickle=False)
    np.save(cache / "query_embeddings.f32.npy", queries, allow_pickle=False)
    (cache / "corpus_ids.json").write_text(json.dumps(corpus_ids), encoding="utf-8")
    (cache / "query_ids.json").write_text(json.dumps(query_ids), encoding="utf-8")
    query_rows = []
    for query_id in query_ids:
        split = (
            "query_tune"
            if query_id in tune_ids
            else "query_cert"
            if query_id in cert_ids
            else "query_test"
        )
        query_rows.append({"query_id": query_id, "split": split, "text": query_id})
    _jsonl(prepared / "queries.jsonl", query_rows)

    dataset_fingerprint = "a" * 64
    embedding_config_fingerprint = "b" * 64
    embedding_request_fingerprint = "c" * 64
    embedding_manifest_fingerprint = "d" * 64
    frozen_projection_fingerprint = "e" * 64
    corpus_array_fingerprint = array_fingerprint(corpus)
    projection_fingerprint = projection_metadata(
        dimension=4,
        m_prime=2,
        seed=23,
        normalization=True,
        embedding_model="fixture-model@fixture-revision",
        corpus_hash=corpus_array_fingerprint,
    )["fingerprint"]

    grid = [4, 8, 12, 20]
    fixed = FixedBudgetPolicy(8, grid, 4)
    fixed_policy = _embedded(fixed.serialize())
    monotone = MonotoneBinnedPolicy(
        edges=[3.0],
        budgets=[4, 8],
        grid=grid,
        fallback_budget=20,
        target=0.5,
        feature_version=FEATURE_VERSION,
    )
    analytic = TriPredictPolicy(
        corpus_size=20,
        m_prime=2,
        k_gt=2,
        grid=grid,
        target=0.8,
        max_rank_samples=16,
        feature_version=FEATURE_VERSION,
    )
    compiled = CompiledTriPredictPolicy.compile(
        analytic, lid_min=1.0, lid_max=100.0, validation_samples=5
    )
    fixed_artifact = _embedded(
        {
            "schema_version": 1,
            "kind": "real_fixed_policy_grid_v1",
            "data_scope": "query_tune_only",
            "candidates": [{"policy": fixed_policy, "evaluation": {}}],
            "selected_reference_budget": 8,
        }
    )
    selection = _embedded(
        {
            "schema_version": 2,
            "kind": "real_tune_policy_selection_v2",
            "data_scope": "query_tune_only",
            "selected": {
                "fixed_reference_budget": 8,
                "monotone_binned_policy_fingerprint": monotone.serialize()[
                    "fingerprint"
                ],
                "tri_predict_policy_fingerprint": analytic.serialize()["fingerprint"],
            },
        },
        "selection_fingerprint",
    )
    summary = _embedded(
        {
            "schema_version": 2,
            "kind": "real_tune_policy_summary_v2",
            "data_scope": "query_tune_only",
        }
    )
    _jsonl(
        policy_run / "per_query.jsonl",
        [{"query_id": tune_ids[0], "split": "query_tune"}],
    )
    write_json(policy_run / "selection.json", selection)
    write_json(policy_run / "fixed_policies.json", fixed_artifact)
    write_json(policy_run / "monotone_binned_policy.json", monotone.serialize())
    write_json(policy_run / "tri_predict_policy.json", analytic.serialize())
    write_json(policy_run / "summary.json", summary)
    (policy_run / "report.md").write_text("fixture tune report\n", encoding="utf-8")
    write_json(policy_run / "compiled_tri_predict_policy.json", compiled.serialize())
    result_names = (
        "per_query.jsonl",
        "selection.json",
        "fixed_policies.json",
        "monotone_binned_policy.json",
        "tri_predict_policy.json",
        "summary.json",
        "report.md",
    )
    result_artifacts = {
        name: _file_identity(policy_run / name) for name in result_names
    }
    result_identity = {
        "config_fingerprint": "1" * 64,
        "dataset_manifest_fingerprint": dataset_fingerprint,
        "embedding_manifest_fingerprint": embedding_manifest_fingerprint,
        "dimension_selection_result_fingerprint": "2" * 64,
        "query_tune_id_hash": stable_id_hash(tune_ids),
        "selection_fingerprint": selection["selection_fingerprint"],
        "artifacts": result_artifacts,
    }
    result_fingerprint = fingerprint(result_identity)
    compiled_identity = _file_identity(policy_run / "compiled_tri_predict_policy.json")
    manifest = {
        "schema_version": 2,
        "kind": "real_tune_policy_manifest_v2",
        "data_scope": "query_tune_only",
        "config_fingerprint": "1" * 64,
        "dataset_manifest_fingerprint": dataset_fingerprint,
        "embedding_manifest_fingerprint": embedding_manifest_fingerprint,
        "dimension_selection_result_fingerprint": "2" * 64,
        "query_tune_id_hash": stable_id_hash(tune_ids),
        "selection_fingerprint": selection["selection_fingerprint"],
        "frozen_projection_fingerprint": frozen_projection_fingerprint,
        "projection_fingerprint": projection_fingerprint,
        "policy_fingerprints": {
            "fixed_grid": fixed_artifact["fingerprint"],
            "monotone_binned": monotone.serialize()["fingerprint"],
            "tri_predict": analytic.serialize()["fingerprint"],
        },
        "result_artifacts": result_artifacts,
        "result_fingerprint": result_fingerprint,
        "deployment": {
            "role": COMPILED_ROLE,
            "policy_fingerprint": compiled.serialize()["fingerprint"],
            "reference_policy_fingerprint": analytic.serialize()["fingerprint"],
            "artifacts": {
                "compiled_tri_predict_policy.json": compiled_identity,
            },
        },
        "software": {"python": "fixture", "numpy": "fixture"},
    }
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(policy_run / "manifest.json", manifest)

    dataset_manifest = {
        "fingerprint": dataset_fingerprint,
        "splits": {
            "query_tune": {"n": len(tune_ids), "id_hash": stable_id_hash(tune_ids)},
            "query_cert": {"n": len(cert_ids), "id_hash": stable_id_hash(cert_ids)},
            "query_test": {"n": len(test_ids), "id_hash": stable_id_hash(test_ids)},
        },
    }
    embedding_manifest = {
        "fingerprint": embedding_manifest_fingerprint,
        "arrays": {
            "corpus": {"array_fingerprint": corpus_array_fingerprint},
        },
    }
    config_raw = {
        "schema_version": 1,
        "benchmark": "real_frozen_policy_certification_v1",
        "dataset_manifest_fingerprint": dataset_fingerprint,
        "embedding_config_fingerprint": embedding_config_fingerprint,
        "embedding_request_fingerprint": embedding_request_fingerprint,
        "embedding_manifest_fingerprint": embedding_manifest_fingerprint,
        "evaluation_split": "query_cert",
        "query_split": {"n": len(cert_ids), "id_hash": stable_id_hash(cert_ids)},
        "policy_source": {
            "manifest_fingerprint": manifest["fingerprint"],
            "result_fingerprint": result_fingerprint,
            "selection_fingerprint": selection["selection_fingerprint"],
            "fixed_grid_fingerprint": fixed_artifact["fingerprint"],
            "fixed_reference_budget": 8,
            "fixed_reference_policy_fingerprint": fixed_policy["fingerprint"],
            "monotone_binned_fingerprint": monotone.serialize()["fingerprint"],
            "analytic_tri_predict_fingerprint": analytic.serialize()["fingerprint"],
            "compiled_tri_predict_fingerprint": compiled.serialize()["fingerprint"],
            "compiled_artifact_sha256": compiled_identity["sha256"],
            "compiled_policy_role": COMPILED_ROLE,
        },
        "projection": {
            "family": "dense_gaussian_n0_variance_1_over_m_prime",
            "seed": 23,
            "m_prime": 2,
            "post_projection_normalize": False,
            "frozen_projection_fingerprint": frozen_projection_fingerprint,
            "projection_fingerprint": projection_fingerprint,
        },
        "search": {
            "normalized_inputs": True,
            "distance": "squared_l2",
            "arithmetic": "numpy_float64",
            "stable_tie_break": "lexicographic_doc_id",
            "query_batch_size": 8,
            "k_ctx": 1,
            "k_gt": 2,
            "m_pilot": 4,
            "s_lid": 3,
            "min_lid_neighbors": 2,
            "m_grid": grid,
            "fixed_reference_budget": 8,
            "pilot_expansion_reuse": "one_projected_scan_and_reuse_pilot_original_distances",
        },
        "lid": {
            "decimal_places": 9,
            "feature_version": FEATURE_VERSION,
            "clip_min": 1.0,
            "clip_max": 100.0,
            "duplicate_tolerance": 1e-12,
            "fallback": 100.0,
        },
        "certification": {
            "metric": "embedding_neighbor_retention_at_k_gt",
            "alpha": 0.05,
            "target": 0.5,
            "planned_n": len(cert_ids),
            "statistic_role": "independent_empirical_bernstein_certificate",
            "policies": ["fixed_reference", "monotone_binned", "tri_predict"],
            "familywise_adjustment": "none_three_predeclared_standalone_certificates_no_selection",
            "failure_behavior": "terminal_no_retuning_no_budget_expansion",
            "evidence_labels_used": False,
        },
    }
    config_path = root / "cert.json"
    write_json(config_path, config_raw)
    return SimpleNamespace(
        prepared=prepared,
        cache=cache,
        policy_run=policy_run,
        config_path=config_path,
        dataset_manifest=dataset_manifest,
        embedding_manifest=embedding_manifest,
        embedding_config=SimpleNamespace(
            config_fingerprint=embedding_config_fingerprint,
            model=SimpleNamespace(
                embedding_dimension=4,
                name="fixture-model",
                revision="fixture-revision",
            ),
        ),
        request_fingerprint=embedding_request_fingerprint,
    )


class RealPolicyCertificationTests(unittest.TestCase):
    def test_checked_in_protocol_freezes_cert_scope_and_genoa_bundle(self):
        config = load_real_policy_certification_config(CONFIG_PATH)
        self.assertEqual(config.evaluation_split, "query_cert")
        self.assertEqual(config.query_split_n, 404)
        self.assertEqual(config.fixed_reference_budget, 768)
        self.assertEqual(config.m_prime, 192)
        self.assertEqual(config.certification_alpha, 0.05)
        self.assertEqual(config.certification_target, 0.95)
        self.assertEqual(
            config.policy_source.compiled_tri_predict_fingerprint,
            "687d47f7fa1f93babaec6049ceaf825929445a92be52df48f26945ab38b42c30",
        )
        self.assertEqual(
            config.config_fingerprint,
            "e5545a4aa4c07a1bc188870538c7d346ff26faf38f135c03d4b32f4a18c7ce74",
        )

    def test_config_rejects_noncert_scope_and_posthoc_selection(self):
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        mutations = (
            ("scope", "query_test", "query_cert only"),
            ("familywise", "select_best_after_cert", "protocol is not frozen"),
            ("failure", "increase_budget_on_fail", "protocol is not frozen"),
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            for index, (kind, value, message) in enumerate(mutations):
                changed = json.loads(json.dumps(raw))
                if kind == "scope":
                    changed["evaluation_split"] = value
                elif kind == "familywise":
                    changed["certification"]["familywise_adjustment"] = value
                else:
                    changed["certification"]["failure_behavior"] = value
                path = directory / f"mutation-{index}.json"
                write_json(path, changed)
                with self.subTest(kind=kind):
                    with self.assertRaisesRegex(RealPolicyCertificationError, message):
                        load_real_policy_certification_config(path)

    def test_frozen_policy_tamper_is_rejected_before_protected_data_access(self):
        with tempfile.TemporaryDirectory() as directory_name:
            fixture = _fixture(Path(directory_name))
            compiled_path = fixture.policy_run / "compiled_tri_predict_policy.json"
            compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
            compiled["states"][0]["budget"] = 20
            write_json(compiled_path, compiled)
            config = load_real_policy_certification_config(fixture.config_path)
            with patch(
                "tri_rag_harness.real_policy_certify.validate_text_embedding_cache",
                side_effect=AssertionError("protected cache must not be accessed"),
            ):
                with self.assertRaisesRegex(
                    RealPolicyCertificationError, "compiled.*fingerprint"
                ):
                    run_real_policy_certification(
                        config,
                        fixture.prepared,
                        Path("unused-embedding-config.json"),
                        fixture.cache,
                        fixture.policy_run,
                        Path(directory_name) / "out",
                    )

    def test_tiny_certification_is_auditable_terminal_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            fixture = _fixture(directory)
            config = load_real_policy_certification_config(fixture.config_path)
            validation = {
                "dataset_manifest": fixture.dataset_manifest,
                "embedding_manifest": fixture.embedding_manifest,
                "request_fingerprint": fixture.request_fingerprint,
            }
            output_a = directory / "cert-a"
            output_b = directory / "cert-b"
            with patch(
                "tri_rag_harness.real_policy_certify.load_text_embedding_config",
                return_value=fixture.embedding_config,
            ), patch(
                "tri_rag_harness.real_policy_certify.validate_text_embedding_cache",
                return_value=validation,
            ):
                run_real_policy_certification(
                    config,
                    fixture.prepared,
                    Path("fixture-embedding-config.json"),
                    fixture.cache,
                    fixture.policy_run,
                    output_a,
                )
                run_real_policy_certification(
                    config,
                    fixture.prepared,
                    Path("fixture-embedding-config.json"),
                    fixture.cache,
                    fixture.policy_run,
                    output_b,
                )
            for name in (
                "manifest.json",
                "per_query.jsonl",
                "certifications.json",
                "summary.json",
                "report.md",
            ):
                self.assertEqual((output_a / name).read_bytes(), (output_b / name).read_bytes())
            rows = [
                json.loads(line)
                for line in (output_a / "per_query.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 32)
            self.assertEqual({row["split"] for row in rows}, {"query_cert"})
            self.assertTrue(
                all(row["policies"]["tri_predict"]["compiled_decision_match"] for row in rows)
            )
            self.assertTrue(
                all(
                    value["candidate_overlap"] == value["reranked_overlap"]
                    for row in rows
                    for value in row["policies"].values()
                )
            )
            certificates = json.loads((output_a / "certifications.json").read_text())
            self.assertTrue(certificates["terminal"])
            self.assertEqual(
                certificates["failure_behavior"],
                "terminal_no_retuning_no_budget_expansion",
            )
            self.assertEqual(set(certificates["certificates"]), {
                "fixed_reference",
                "monotone_binned",
                "tri_predict",
            })
            for name, certificate in certificates["certificates"].items():
                expected = make_certificate(
                    [row["policies"][name]["embedding_retention"] for row in rows],
                    alpha=0.05,
                    target=0.5,
                    policy_fingerprint=certificate["policy_fingerprint"],
                    split_hash=stable_id_hash([row["query_id"] for row in rows]),
                    metric="embedding_neighbor_retention_at_k_gt",
                    planned_n=32,
                )
                self.assertEqual(certificate, expected)
            summary = json.loads((output_a / "summary.json").read_text())
            self.assertEqual(summary["compiled_reference_match_n"], 32)
            for value in summary["policies"].values():
                self.assertEqual(value["work_per_query"]["projected_scan_count"], 1)
            timings = json.loads((output_a / "timings.json").read_text())
            self.assertEqual(timings["projected_scan_count_per_query"], 1)
            manifest = json.loads((output_a / "manifest.json").read_text())
            for name, identity in manifest["result_artifacts"].items():
                self.assertEqual(identity, _file_identity(output_a / name))
            result_identity = {
                "config_fingerprint": manifest["config_fingerprint"],
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
                "query_cert_id_hash": manifest["query_cert_id_hash"],
                "artifacts": manifest["result_artifacts"],
            }
            self.assertEqual(manifest["result_fingerprint"], fingerprint(result_identity))
            self.assertFalse((fixture.prepared / "qrels.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
