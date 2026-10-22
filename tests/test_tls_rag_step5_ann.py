from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tri_rag_harness.tls_rag_step5_ann import (CandidateSession, ExactCandidateIndex,
    HnswCandidateIndex, expand_candidate_pool, stable_top)
from tri_rag_harness.tls_rag_step5_benchmark import (Runtime, run_stage, self_test, synthetic_protocol)
from tri_rag_harness.tls_rag_step5_data import (family_key, freeze_roles, load_bundle,
    normalize_text, normalized, sample_corpus, text_hash, write_bundle)
from tri_rag_harness.tls_rag_step5_policy import (DeployableState, build_state, choose,
    fit_models, policy_grid, predict, calibrate)
from tri_rag_harness.tls_rag_step5_protocol import (RunLedger, load_protocol,
    paired_bootstrap, quality_bounds, seal, sha256, verify_seal)
from tri_rag_harness.utils import write_json


class Step5AnnTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.p = synthetic_protocol()

    def tearDown(self):
        self.temp.cleanup()

    def state(self, stage=0, valid=True):
        return DeployableState(self.p["schema"], stage, self.p["budget_grid"][stage],
                               tuple(float(i) for i in range(13)), tuple(float(i) for i in range(6)), valid, valid)

    def test_stable_boundary_uses_string_ids(self):
        self.assertEqual(stable_top(["z", "10", "2", "a"], [0, 1, 1, 1], 3).tolist(), [0, 1, 2])

    def test_exact_reference_across_blocks_and_ties(self):
        ids = ("z", "b", "a", "c")
        values = np.array([[1, 0], [0, 1], [-1, 0], [0, -1.]])
        self.assertEqual(ExactCandidateIndex(ids, values, 2).search(np.zeros(2), 3), ("a", "b", "c"))

    def test_exact_matches_direct_bruteforce(self):
        rng = np.random.default_rng(7)
        corpus, query = rng.normal(size=(77, 8)), rng.normal(size=8)
        ids = tuple(map(str, range(77)))
        d = np.sum((corpus - query) ** 2, axis=1)
        expected = tuple(np.asarray(ids)[np.lexsort((np.asarray(ids), d))[:13]])
        self.assertEqual(ExactCandidateIndex(ids, corpus, 9).search(query, 13), expected)

    def test_exact_rejects_invalid_ids_query_and_k(self):
        with self.assertRaises(ValueError):
            ExactCandidateIndex((1, 2), np.ones((2, 3)))
        index = ExactCandidateIndex(("a", "b"), np.ones((2, 3)))
        for k in (True, 0, 3):
            with self.assertRaises(ValueError):
                index.search(np.ones(3), k)
        with self.assertRaises(ValueError):
            index.search(np.full(3, np.nan), 1)

    def test_non_nested_expansion_keeps_previous_members(self):
        result = expand_candidate_pool(("a", "b"), ("b", "c", "d", "e"),
                                       {"b": 0., "c": 3., "d": 2., "e": 1.}, 4)
        self.assertEqual(result, ("a", "b", "e", "d"))

    def test_missing_or_duplicate_ann_result_is_failure(self):
        for raw in (("a", "a"), ("a",)):
            with self.assertRaises(ValueError):
                expand_candidate_pool((), raw, {"a": 0.}, 2)

    def test_original_scores_reused_across_restarted_search(self):
        class Fake:
            calls = 0
            def search(self, query, k):
                self.calls += 1
                return ("a", "b") if k == 2 else ("b", "c", "d", "e")
        ids = tuple("abcde")
        vectors = np.arange(10.).reshape(5, 2)
        fake = Fake()
        session = CandidateSession(ids, vectors, vectors, np.zeros(2), np.zeros(2), fake, 1)
        first, second = session.expand(2), session.expand(4)
        self.assertEqual(fake.calls, 2)
        self.assertEqual(first["original_distances"] + second["original_distances"], 4)
        self.assertEqual(len(session.original_cache), 4)
        self.assertTrue(set(first["candidate_ids"]) <= set(second["candidate_ids"]))

    def test_feature_reads_only_current_pool_and_is_bounded(self):
        rng = np.random.default_rng(17)
        vectors = normalized(rng.normal(size=(96, 24)))
        ids = tuple(f"d-{i}" for i in range(96))
        query = normalized(rng.normal(size=(1, 24)))[0]
        index = ExactCandidateIndex(ids, vectors)
        session = CandidateSession(ids, vectors, vectors[:, :12], query, query[:12],
            ExactCandidateIndex(ids, vectors[:, :12]), 3)
        record = session.expand(8)
        before = deepcopy(session.original_cache)
        state = build_state(session, record, self.p)
        self.assertEqual(before, session.original_cache)
        self.assertLessEqual(len(record["feature_sample_ids"]), self.p["features"]["sample_size"] + 3 + self.p["features"]["shell_size"])
        self.assertEqual(state.budget, 8)
        self.assertTrue(set(record["feature_sample_ids"]) <= set(record["candidate_ids"]))

    def test_controller_rejects_label_dictionary(self):
        with self.assertRaises(TypeError):
            choose({"retention": 1.}, policy_grid(self.p)[0], {}, {}, self.p)

    def test_uncalibrated_expands_then_terminal_stop(self):
        policy = policy_grid(self.p)[0]
        self.assertEqual(choose(self.state(), policy, {}, {}, self.p)["action"], "EXPAND")
        final = choose(self.state(stage=2), policy, {}, {}, self.p)
        self.assertEqual(final["action"], "STOP")
        self.assertEqual(final["reason"], "terminal_without_quality_assertion")

    def test_controller_rejects_grid_and_policy_drift(self):
        policy = policy_grid(self.p)[0]
        with self.assertRaises(ValueError):
            choose(replace(self.state(), budget=9), policy, {}, {}, self.p)
        with self.assertRaises(ValueError):
            choose(self.state(), {**policy, "threshold": .1}, {}, {}, self.p)

    def test_new_fit_and_independent_residual_margin(self):
        examples = [(replace(self.state(), base=tuple(float(i + j) for i in range(13))), .6 + .01 * j) for j in range(16)]
        models = fit_models(examples, self.p)
        self.assertIsNotNone(models["tri:0"])
        low_examples = [(state, target - .2) for state, target in examples]
        margins, records = calibrate(low_examples, models, self.p)
        self.assertGreater(margins["tri:0:0.5"], .19)
        self.assertIsNone(margins["tri:1:0.5"])
        self.assertEqual(len(records), 6)

    def test_text_and_lexical_family_canonicalization(self):
        self.assertEqual(normalize_text("Ａ  B\n"), "a b")
        self.assertEqual(family_key("How much: 12 costs?"), family_key("costs how much 15"))

    def test_hash_sample_reproducible_and_order_independent(self):
        path = self.root / "corpus.tsv"
        lines = [f"{i}\tpassage {i}\n" for i in range(40)]
        path.write_text("".join(lines))
        a, count = sample_corpus(path, 12, 17)
        path.write_text("".join(reversed(lines)))
        b, _ = sample_corpus(path, 12, 17)
        self.assertEqual(a, b)
        self.assertEqual(count, 40)

    def test_role_family_history_corpus_exclusion(self):
        p = deepcopy(self.p)
        p["roles"] = {r: 1 for r in p["roles"]}
        queries = [("q" + str(i), "unique " + chr(97 + i)) for i in range(15)]
        queries += [("duplicate", "a unique"), ("copy", "corpus copied"), ("history", "seen before")]
        positives = {qid: {"doc"} for qid, _ in queries}
        history = {"text_hashes": [text_hash("seen before")], "family_hashes": [], "source_ids": [], "complete": True}
        roles, audit = freeze_roles(queries, positives, ["corpus copied"], p, history)
        chosen = [row for rows in roles.values() for row in rows]
        self.assertEqual(len({family_key(text) for _, text in chosen}), 6)
        self.assertEqual(audit["excluded"]["corpus_copy"], 1)
        self.assertEqual(audit["excluded"]["history"], 1)
        self.assertFalse(audit["semantic_family_independence_established"])

    def test_insufficient_families_stops_before_encoding(self):
        with self.assertRaises(ValueError):
            freeze_roles([("q", "text")], {"q": {"doc"}}, [], self.p)

    def test_artifact_tampering_rejected(self):
        path = self.root / "seal"
        path.mkdir()
        (path / "a").write_text("first")
        seal(path)
        (path / "a").write_text("changed")
        with self.assertRaises(ValueError):
            verify_seal(path)

    def test_future_role_and_early_labels_rejected(self):
        ledger = RunLedger(self.root / "run", "fixture", self.p)
        with self.assertRaises(FileNotFoundError):
            ledger.reserve("cert")
        ledger.reserve("cal_model_fit")
        with self.assertRaises(FileNotFoundError):
            ledger.authorize_labels("cal_model_fit")
        with self.assertRaises(ValueError):
            ledger.authorize_labels("latency")

    def test_crash_reservation_cannot_be_reused(self):
        ledger = RunLedger(self.root / "run", "fixture", self.p)
        ledger.reserve("cal_model_fit")
        with self.assertRaises(FileExistsError):
            ledger.reserve("cal_model_fit")

    def test_pass_cannot_bypass_decision_and_label_closure(self):
        ledger = RunLedger(self.root / "run", "fixture", self.p)
        ledger.reserve("cal_model_fit")
        with self.assertRaises(ValueError):
            ledger.finish("cal_model_fit", "passed")

    def test_closed_decisions_are_immutable(self):
        ledger = RunLedger(self.root / "run", "fixture", self.p)
        path = ledger.reserve("cal_model_fit") / "x.decisions.jsonl"
        path.write_text('{}\n')
        ledger.close_decisions("cal_model_fit")
        path.write_text('{"changed":true}\n')
        with self.assertRaises(ValueError):
            ledger.authorize_labels("cal_model_fit")

    def test_failed_or_inconclusive_stage_locks_successors(self):
        for status in ("failed", "inconclusive", "no_tune_candidate"):
            ledger = RunLedger(self.root / status, "fixture", self.p)
            ledger.reserve("cal_model_fit")
            ledger.finish("cal_model_fit", status)
            with self.assertRaises(ValueError):
                ledger.reserve("cal_bound_fit")

    def test_protocol_binding_rejects_changed_budget(self):
        path = self.root / "run"
        RunLedger(path, "fixture", self.p)
        altered = deepcopy(self.p)
        altered["budget_grid"][0] = 9
        with self.assertRaises(ValueError):
            RunLedger(path, "fixture", altered)

    def test_quality_bounds_range_scaling_and_power(self):
        _, passed = quality_bounds(np.ones(8192), np.zeros(8192), np.zeros(8192), self.p)
        self.assertTrue(passed)
        bounds, passed = quality_bounds(np.ones(8), np.zeros(8), np.zeros(8), self.p)
        self.assertFalse(passed)
        self.assertLess(bounds["hit_delta"]["lower"], -.02)
        with self.assertRaises(ValueError):
            quality_bounds([1, 1], [2, 2], [0, 0], self.p)

    def test_paired_bootstrap_preserves_query_repeat_blocks(self):
        fixed = np.arange(1., 41.).reshape(20, 2)
        result = paired_bootstrap(fixed * .7, fixed, self.p)
        again = paired_bootstrap(fixed * .7, fixed, self.p)
        self.assertEqual(result, again)
        self.assertAlmostEqual(result["mean_ratio_upper"], .7)
        with self.assertRaises(ValueError):
            paired_bootstrap(fixed * 0, fixed, self.p)

    def test_self_test_real_pipeline_and_labels_order(self):
        result = self_test(self.root / "selftest", False)
        self.assertFalse(result["ann_integration_exercised"])
        self.assertIn(result["last_stage"], ("tune",))
        run = self.root / "selftest/run"
        for role in ("cal_model_fit", "cal_bound_fit"):
            path = run / role
            self.assertTrue((path / "decisions_closed.json").exists())
            self.assertTrue((path / "labels_opened.json").exists())
            verify_seal(path)
        self.assertFalse((run / "cert").exists())
        self.assertFalse((run / "latency").exists())
        self.assertFalse((run / "test").exists())
        p = synthetic_protocol()
        load_bundle(self.root / "selftest/bundle", p)
        altered = deepcopy(p)
        altered["seeds"]["projection"] += 1
        with self.assertRaises(ValueError):
            load_bundle(self.root / "selftest/bundle", altered)

    def test_seeded_pipeline_reproduces_scientific_records(self):
        for name in ("a", "b"):
            self_test(self.root / name, False)
        def scientific(value):
            if isinstance(value, dict):
                return {k: scientific(v) for k, v in value.items() if k not in {"timing_ms", "projection_ms"}}
            if isinstance(value, list):
                return [scientific(v) for v in value]
            return value
        for role in ("cal_model_fit", "cal_bound_fit", "tune"):
            paths = [self.root / name / "run" / role / "retrieval.decisions.jsonl" for name in ("a", "b")]
            records = [[scientific(json.loads(line)) for line in path.read_text().splitlines()] for path in paths]
            self.assertEqual(records[0], records[1])
        for role, name in (("cal_model_fit", "models.json"), ("cal_bound_fit", "margins.json")):
            self.assertEqual((self.root / "a/run" / role / name).read_bytes(), (self.root / "b/run" / role / name).read_bytes())

    def test_six_stage_pipeline_with_explicit_synthetic_gate_fixture(self):
        # These permissive thresholds exercise success transitions, not a real
        # quality/latency claim. The registered real protocol is never altered.
        p = deepcopy(self.p)
        p["acceptance"].update(retention=0., hit_drop=1., retention_drop=1.,
            distance_ratio=1e6, latency_ratio=1e6, p95_ratio=1e6)
        p["timing"].update(warmup=1, repeats=1, bootstrap_replicates=8)
        p["roles"] = {role: 36 if role in ("cal_model_fit", "cal_bound_fit") else 8 for role in p["roles"]}
        rng = np.random.default_rng(33)
        corpus = normalized(rng.normal(size=(96, 24)))
        ids = tuple(f"doc-{i:04}" for i in range(96))
        vectors, roles, positives = {}, {}, {}
        exact = ExactCandidateIndex(ids, corpus)
        for role, count in p["roles"].items():
            vectors[role] = normalized(rng.normal(size=(count, 24)))
            roles[role] = [(f"{role}-{i}", f"fixture {role} {i}") for i in range(count)]
            for (qid, _), query in zip(roles[role], vectors[role]):
                positives[qid] = set(exact.search(query, 1))
        bundle = self.root / "bundle"
        write_bundle(bundle, ids, corpus, roles, vectors, positives, p, {"kind": "synthetic"})
        rt = Runtime(bundle, self.root / "run", p, "exact_fixture")
        for role in p["roles"]:
            run_stage(rt, role)
            self.assertEqual(rt.ledger.result(role)["status"], "passed", role)
        self.assertFalse((rt.ledger.root / "latency/labels_opened.json").exists())
        self.assertFalse(rt.ledger.result("test")["formal_certification"])
        for role in p["roles"]:
            verify_seal(rt.ledger.root / role)
        with self.assertRaises(FileExistsError):
            rt.ledger.reserve("test")
        path = rt.ledger.root / "cal_model_fit/models.json"
        path.write_text('{}')
        with self.assertRaises(ValueError):
            rt.ledger.reserve("test")

    @unittest.skipUnless(importlib.util.find_spec("faiss"), "optional real FAISS integration")
    def test_real_hnsw_end_to_end_self_test(self):
        result = self_test(self.root / "hnsw", True)
        self.assertTrue(result["ann_integration_exercised"])
        self.assertEqual(result["last_stage"], "tune")

    @unittest.skipUnless(importlib.util.find_spec("faiss"), "optional real FAISS integration")
    def test_real_hnsw_deterministic_build_and_ef_rule(self):
        rng = np.random.default_rng(7)
        values = normalized(rng.normal(size=(96, 24)))
        ids = tuple(f"doc-{i:04}" for i in range(96))
        a = HnswCandidateIndex(ids, values, self.p["hnsw"], 19)
        b = HnswCandidateIndex(ids, values, self.p["hnsw"], 19)
        self.assertEqual(a.search(values[0], 24), b.search(values[0], 24))
        self.assertEqual(a.index.hnsw.efSearch, 256)
        self.assertEqual(len(a.search(values[0], 64)), 64)
        a.save(self.root / "a.faiss")
        b.save(self.root / "b.faiss")
        self.assertEqual(sha256(self.root / "a.faiss"), sha256(self.root / "b.faiss"))


if __name__ == "__main__":
    unittest.main()
