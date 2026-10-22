"""Staged ANN harness. Tests/self-test are synthetic; real roles are explicit.

All scientific gates inherit L0 even though timing orchestration is maintenance.
This module never installs packages, downloads data, or accesses old run roots.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np
import scipy

from .projection import dense_gaussian_projection, project_rows
from .tls_rag_step5_ann import CandidateSession, ExactCandidateIndex, HnswCandidateIndex, TorchExactGpuIndex
from .tls_rag_step5_data import load_bundle, normalized, write_bundle
from .tls_rag_step5_policy import (DeployableState, build_state, calibrate, choose,
                                  fit_models, policy_grid, policy_id)
from .tls_rag_step5_protocol import (ROLES, RunLedger, load_protocol, paired_bootstrap,
    quality_bounds, read_rows, save_rows, seal, sha256, verify_seal)
from .utils import fingerprint, write_json


class Runtime:
    def __init__(self, bundle, run, protocol, backend="hnsw", planned_role=None):
        self.started = perf_counter()
        self.protocol = protocol
        (binding, self.ids, self.corpus, self.roles, self.queries,
         self.metadata) = load_bundle(bundle, protocol)
        if backend != "hnsw" and self.metadata["metadata"]["kind"] != "synthetic":
            raise ValueError("real experiment must use HNSW")
        self.bundle, self.backend = Path(bundle), backend
        environment = {"platform": platform.platform(), "python": sys.version,
                       "numpy": np.__version__, "scipy": scipy.__version__}
        if backend == "hnsw":
            import faiss
            environment.update(faiss=faiss.__version__, faiss_compile_options=faiss.get_compile_options())
        self.ledger = RunLedger(run, {"bundle": binding, "backend": backend, "environment": environment}, protocol)
        if planned_role is not None:
            self.ledger.check_ready(planned_role)  # reject before a real index build
        self.row_by_id = {item: row for row, item in enumerate(self.ids)}
        self.matrix = dense_gaussian_projection(protocol["m_prime"], protocol["dimension"], protocol["seeds"]["projection"])
        index_root = self.ledger.root / "indexes"
        if not index_root.exists():
            index_root.mkdir()
            projected = np.lib.format.open_memmap(index_root / "projected.npy", mode="w+", dtype=np.float64,
                                                 shape=(len(self.ids), protocol["m_prime"]))
            for start in range(0, len(self.ids), 8192):
                self.check_resources()
                projected[start:start + 8192] = project_rows(self.corpus[start:start + 8192], self.matrix)
            projected.flush()
            del projected
            if backend == "hnsw":
                self.projected = np.load(index_root / "projected.npy", mmap_mode="r")
                self.ann = HnswCandidateIndex(self.ids, self.projected, protocol["hnsw"], protocol["seeds"]["index"])
                self.ann.save(index_root / "projected.faiss")
                self.original_ann = HnswCandidateIndex(self.ids, self.corpus, protocol["hnsw"], protocol["seeds"]["index"])
                self.original_ann.save(index_root / "original.faiss")
                del self.ann, self.original_ann
            write_json(index_root / "setup.json", {"binding": self.ledger.identity, "backend": backend,
                "elapsed_seconds": perf_counter() - self.started,
                "environment": environment})
            seal(index_root)
        verify_seal(index_root)
        setup = json.loads((index_root / "setup.json").read_text())
        if setup["binding"] != self.ledger.identity:
            raise ValueError("index/run binding mismatch")
        self.projected = np.load(index_root / "projected.npy", mmap_mode="r")
        self.exact = ExactCandidateIndex(self.ids, self.corpus)
        self.exact_projected = ExactCandidateIndex(self.ids, self.projected)
        if backend == "hnsw":
            import faiss
            if faiss.__version__ != protocol["hnsw"]["faiss_version"]:
                raise ValueError("FAISS version changed")
            faiss.omp_set_num_threads(protocol["hnsw"]["threads"])
            def restored(name):
                result = HnswCandidateIndex.__new__(HnswCandidateIndex)
                result.ids, result.config = self.ids, protocol["hnsw"]
                result.index = faiss.read_index(str(index_root / (name + ".faiss")))
                return result
            self.ann, self.original_ann = restored("projected"), restored("original")
        else:
            self.ann, self.original_ann = self.exact_projected, self.exact
        self.setup_seconds = perf_counter() - self.started

    def check_resources(self):
        elapsed = perf_counter() - self.started
        previous = 0.
        if hasattr(self, "ledger"):
            setup_file = self.ledger.root / "indexes/setup.json"
            if setup_file.exists():
                previous += json.loads(setup_file.read_text()).get("elapsed_seconds", 0.)
            for role in ROLES:
                path = self.ledger.root / role / "result.json"
                if path.exists():
                    previous += json.loads(path.read_text()).get("elapsed_seconds", 0.)
        # macOS reports bytes; Linux reports KiB. A breach preserves reservation.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_gib = rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)
        limits = self.protocol["resource_limits"]
        if elapsed + previous > limits["max_hours"] * 3600 or rss_gib > limits["max_rss_gib"]:
            raise RuntimeError("resource limit exceeded; preserve run; no auto-resize or retry")

    def retrieve(self, query, method, budget=None, policy=None, models=None, margins=None):
        started = perf_counter()
        if method in {"original_exact", "original_exact_gpu"}:
            index = self.exact if method == "original_exact" else self.gpu_exact
            top = index.search(query, self.protocol["k_gt"])
            return {"method": method, "budget": len(self.ids), "stages": [],
                    "context_ids": list(top[:self.protocol["k_ctx"]]), "candidate_ids": list(top),
                    "original_distances": len(self.ids), "timing_ms": (perf_counter() - started) * 1000,
                    "projection_ms": 0.}
        projected_query = project_rows(query[None, :], self.matrix)[0] if method != "original" else query
        projection_ms = (perf_counter() - started) * 1000
        use_original = method == "original"
        index = self.original_ann if use_original else self.exact_projected if method.startswith("exact_projected") else self.ann
        session = CandidateSession(self.ids, self.corpus, self.corpus if use_original else self.projected,
                                   query, projected_query, index, self.protocol["k_ctx"], self.row_by_id)
        grid = self.protocol["budget_grid"]
        adaptive = method in {"adaptive", "exact_projected_adaptive"}
        budgets = grid if method == "all" or adaptive else [b for b in grid if b <= budget] if method.endswith("staged") else [budget]
        records = []
        for value in budgets:
            record = session.expand(value)
            if method == "all" or adaptive:
                feature_started = perf_counter()
                state = build_state(session, record, self.protocol)
                record["state"] = asdict(state)
                record["timing_ms"]["features"] = (perf_counter() - feature_started) * 1000
                controller_started = perf_counter()
                if adaptive:
                    record["decision"] = choose(state, policy, models, margins, self.protocol)
                record["timing_ms"]["controller"] = (perf_counter() - controller_started) * 1000
            records.append(record)
            if adaptive and record["decision"]["action"] == "STOP":
                break
        elapsed_ms = (perf_counter() - started) * 1000
        return {"method": method, "budget": records[-1]["budget"], "stages": records,
                "context_ids": records[-1]["context_ids"], "candidate_ids": records[-1]["candidate_ids"],
                "original_distances": sum(r["original_distances"] for r in records),
                "timing_ms": elapsed_ms, "projection_ms": projection_ms}

    def close_and_label(self, role):
        self.ledger.close_decisions(role)
        self.ledger.authorize_labels(role)
        labels = json.loads((self.bundle / "labels" / (role + ".json")).read_text())
        if set(labels) != set(self.roles[role]):
            raise ValueError("label role mismatch")
        return labels

    def ground_truth(self, role):
        result = {}
        for qid, query in zip(self.roles[role], self.queries[role]):
            self.check_resources()
            result[qid] = self.exact.search(query, self.protocol["k_gt"])
        return result


def state_from(record):
    state = dict(record)
    state["base"], state["risk"] = tuple(state["base"]), tuple(state["risk"])
    return DeployableState(**state)


def metric(record, truth, positives):
    return {"retention": len(set(record["candidate_ids"]) & set(truth)) / len(truth),
            "hit": float(bool(set(record["context_ids"]) & set(positives))),
            "original_distances": record["original_distances"]}


def cached_decision(record, policy, models, margins, protocol):
    stages = []
    for stage in record["stages"]:
        stages.append(stage)
        decision = choose(state_from(stage["state"]), policy, models, margins, protocol)
        if decision["action"] == "STOP":
            break
    final = stages[-1]
    return {"candidate_ids": final["candidate_ids"], "context_ids": final["context_ids"],
            "original_distances": sum(s["original_distances"] for s in stages)}


def run_stage(runtime, role):
    rt, p = runtime, runtime.protocol
    directory = rt.ledger.reserve(role)
    started = perf_counter()
    previous = lambda name: rt.ledger.root / name
    models = json.loads((previous("cal_model_fit") / "models.json").read_text()) if role != "cal_model_fit" else None
    margins = json.loads((previous("cal_bound_fit") / "margins.json").read_text()) if role in ROLES[2:] else None
    selection = json.loads((previous("tune") / "selection.json").read_text()) if role in ROLES[3:] else None
    if selection and (selection["models"] != fingerprint(models) or selection["margins"] != fingerprint(margins)):
        raise ValueError("selected policy calibration changed")
    def finish(status, **values):
        return rt.ledger.finish(role, status, elapsed_seconds=perf_counter() - started + rt.setup_seconds,
                                formal_certification=False, **values)
    if role == "latency":
        return run_latency(rt, directory, selection, models, margins, finish)
    def decisions():
        for qid, query in zip(rt.roles[role], rt.queries[role]):
            rt.check_resources()
            if role in ROLES[:3]:
                rows = {"all": rt.retrieve(query, "all")}
                if role == "tune":
                    for budget in p["budget_grid"]:
                        rows[f"fixed:{budget}"] = rt.retrieve(query, "fixed", budget)
                        rows[f"original:{budget}"] = rt.retrieve(query, "original", budget)
            else:
                rows = {"adaptive": rt.retrieve(query, "adaptive", policy=selection["policy"], models=models, margins=margins),
                        "fixed": rt.retrieve(query, "fixed", selection["fixed_budget"]),
                        "original": rt.retrieve(query, "original", selection["original_budget"]),
                        "fixed_staged": rt.retrieve(query, "fixed_staged", selection["fixed_budget"]),
                        "exact_projected": rt.retrieve(query, "exact_projected", selection["fixed_budget"]),
                        "exact_projected_adaptive": rt.retrieve(query, "exact_projected_adaptive", policy=selection["policy"], models=models, margins=margins)}
            yield {"query_id": qid, "rows": rows}
    save_rows(directory / "retrieval.decisions.jsonl", decisions())
    labels = rt.close_and_label(role)
    truth = rt.ground_truth(role)
    truth_records = [{"query_id": qid, "exact_top_k": list(top),
                      "exact_hit": float(bool(set(top[:p["k_ctx"]]) & set(labels[qid])))} for qid, top in truth.items()]
    save_rows(directory / "truth.jsonl", truth_records)
    exact_hit = {r["query_id"]: r["exact_hit"] for r in truth_records}
    if role in ROLES[:2]:
        examples = []
        for row in read_rows(directory / "retrieval.decisions.jsonl"):
            qid = row["query_id"]
            for stage in row["rows"]["all"]["stages"]:
                examples.append((state_from(stage["state"]), len(set(stage["candidate_ids"]) & set(truth[qid])) / p["k_gt"]))
        if role == "cal_model_fit":
            models = fit_models(examples, p)
            write_json(directory / "models.json", models)
            valid = all(models.get(f"tri:{stage}") is not None for stage in range(len(p["budget_grid"])))
        else:
            margins, residuals = calibrate(examples, models, p)
            write_json(directory / "margins.json", margins)
            save_rows(directory / "residuals.jsonl", residuals)
            valid = all(margins.get(f"tri:{stage}:{q}") is not None for stage in range(len(p["budget_grid"])) for q in p["policy"]["quantiles"])
        return finish("passed" if valid else "failed", valid_fit=valid)
    if role == "tune":
        return select_tune(rt, directory, models, margins, labels, truth, exact_hit, finish)
    retention, hit_delta, retention_delta, metrics = [], [], [], []
    for row in read_rows(directory / "retrieval.decisions.jsonl"):
        qid = row["query_id"]
        measured = {name: metric(record, truth[qid], labels[qid]) for name, record in row["rows"].items()}
        a, f = measured["adaptive"], measured["fixed"]
        retention.append(a["retention"])
        hit_delta.append(a["hit"] - exact_hit[qid])
        retention_delta.append(a["retention"] - f["retention"])
        metrics.append({"query_id": qid, "methods": measured})
    save_rows(directory / "metrics.jsonl", metrics)
    bounds, passed = quality_bounds(retention, hit_delta, retention_delta, p)
    point_pass = (np.mean(retention) >= p["acceptance"]["retention"] and np.mean(hit_delta) >= -p["acceptance"]["hit_drop"]
                  and np.mean(retention_delta) >= -p["acceptance"]["retention_drop"])
    return finish("passed" if passed else "inconclusive" if point_pass else "failed",
                  bounds=bounds, conditional_quality_bounds_passed=passed,
                  population=rt.metadata["metadata"], n=len(metrics))


def select_tune(rt, directory, models, margins, labels, truth, exact_hit, finish):
    p, acceptance = rt.protocol, rt.protocol["acceptance"]
    policies = policy_grid(p)
    summaries = {"fixed:" + str(b): [] for b in p["budget_grid"]}
    summaries.update({"original:" + str(b): [] for b in p["budget_grid"]})
    summaries.update({policy_id(policy): [] for policy in policies})
    detailed = []
    for row in read_rows(directory / "retrieval.decisions.jsonl"):
        qid = row["query_id"]
        query_metrics = {}
        for name, record in row["rows"].items():
            if name == "all":
                continue
            query_metrics[name] = metric(record, truth[qid], labels[qid])
            query_metrics[name]["timing_ms"] = record["timing_ms"]
        for policy in policies:
            query_metrics[policy_id(policy)] = metric(cached_decision(row["rows"]["all"], policy, models, margins, p), truth[qid], labels[qid])
        for name, value in query_metrics.items():
            value["hit_delta"] = value["hit"] - exact_hit[qid]
            summaries[name].append(value)
        detailed.append({"query_id": qid, "methods": query_metrics})
    save_rows(directory / "metrics.jsonl", detailed)
    means = {name: {key: float(np.mean([r[key] for r in rows])) for key in rows[0]} for name, rows in summaries.items()}
    write_json(directory / "summaries.json", means)
    def eligible(name):
        return means[name]["retention"] >= acceptance["retention"] and means[name]["hit_delta"] >= -acceptance["hit_drop"]
    fixed = [b for b in p["budget_grid"] if eligible(f"fixed:{b}")]
    original = [b for b in p["budget_grid"] if eligible(f"original:{b}")]
    if not fixed or not original:
        return finish("no_tune_candidate", reason="no quality-matched ANN comparator; protected roles remain closed")
    fixed_budget = min(fixed, key=lambda b: (means[f"fixed:{b}"]["timing_ms"], b))
    original_budget = min(original, key=lambda b: (means[f"original:{b}"]["timing_ms"], b))
    eligible_policies = [policy for policy in policies if policy["family"] == "tri" and eligible(policy_id(policy))
        and means[policy_id(policy)]["retention"] - means[f"fixed:{fixed_budget}"]["retention"] >= -acceptance["retention_drop"]
        and means[policy_id(policy)]["original_distances"] <= acceptance["distance_ratio"] * fixed_budget]
    if not eligible_policies:
        return finish("no_tune_candidate", reason="no predeclared Tri-feature candidate; cert/test remain closed")
    selected = min(eligible_policies, key=lambda policy: (means[policy_id(policy)]["original_distances"], policy_id(policy)))
    selection = {"policy": selected, "policy_id": policy_id(selected), "models": fingerprint(models), "margins": fingerprint(margins),
                 "fixed_budget": fixed_budget, "original_budget": original_budget, "fixed_selection": "tune_quality_then_measured_mean_latency"}
    write_json(directory / "selection.json", selection)
    return finish("passed", selection=selection)


def run_latency(rt, directory, selection, models, margins, finish):
    p, t = rt.protocol, rt.protocol["timing"]
    methods = ("adaptive", "fixed", "fixed_staged", "original", "exact_projected", "original_exact")
    if rt.metadata["metadata"]["kind"] == "real":
        gpu_started = perf_counter()
        rt.gpu_exact = TorchExactGpuIndex(rt.ids, rt.corpus)
        # Audit numerical identities against the canonical CPU reference on
        # latency vectors only; these are not policy-selection inputs.
        for query in rt.queries["latency"][:8]:
            if rt.gpu_exact.search(query, p["k_gt"]) != rt.exact.search(query, p["k_gt"]):
                raise ValueError("GPU reference disagrees with canonical CPU top-k")
        write_json(directory / "gpu_setup.json", {"elapsed_seconds": perf_counter() - gpu_started,
            "backend": "torch_float64_direct_L2_host_stable_merge", "torch": rt.gpu_exact.torch.__version__,
            "device": rt.gpu_exact.torch.cuda.get_device_name(0), "audited_query_count": 8})
        methods += ("original_exact_gpu",)
    rng = np.random.default_rng(p["seeds"]["timing"])
    def execute(query, name):
        return rt.retrieve(query, name, selection["original_budget"] if name == "original" else selection["fixed_budget"],
                           selection["policy"], models, margins)
    queries = rt.queries["latency"]
    for i in range(t["warmup"]):
        for name in methods:
            execute(queries[i % len(queries)], name)
    times = {name: np.zeros((len(queries), t["repeats"])) for name in methods}
    work = {name: np.zeros_like(times[name]) for name in methods}
    def records():
        for repeat in range(t["repeats"]):
            for i in rng.permutation(len(queries)):
                for name in rng.permutation(methods):
                    rt.check_resources()
                    record = execute(queries[i], str(name))
                    times[name][i, repeat] = record["timing_ms"]
                    work[name][i, repeat] = record["original_distances"]
                    yield {"query_id": rt.roles["latency"][i], "repeat": repeat, "method": str(name), **record}
    save_rows(directory / "timing.decisions.jsonl", records())
    rt.ledger.close_decisions("latency")
    latency = paired_bootstrap(times["adaptive"], times["fixed"], p)
    distance = paired_bootstrap(work["adaptive"], work["fixed"], p)
    summary = {name: {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
                      "mean_ms": float(values.mean()), "serial_qps": float(1000 / values.mean())}
               for name, values in times.items()}
    # Batch workloads include the full per-query pipeline in fixed-size blocks;
    # this is explicitly serial execution, not native batched HNSW acceleration.
    batch_records = []
    for start in range(0, len(queries), t["batch_size"]):
        block = queries[start:start + t["batch_size"]]
        for name in rng.permutation(methods):
            rt.check_resources()
            started = perf_counter()
            for query in block:
                execute(query, str(name))
            elapsed = perf_counter() - started
            batch_records.append({"start": start, "n": len(block), "method": str(name),
                                  "elapsed_ms": elapsed * 1000, "qps": len(block) / elapsed,
                                  "execution": "serial_full_pipeline_in_batch64_blocks"})
    save_rows(directory / "batch64.jsonl", batch_records)
    passed = (latency["mean_ratio_upper"] <= p["acceptance"]["latency_ratio"]
              and latency["p95_ratio_upper"] <= p["acceptance"]["p95_ratio"]
              and distance["mean_ratio_upper"] <= p["acceptance"]["distance_ratio"])
    write_json(directory / "summary.json", summary)
    return finish("passed" if passed else "failed", latency=latency, original_distance_cost=distance,
                  cpu_single_query_gate_passed=passed, failure_count=0,
                  peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  peak_rss_unit="bytes" if sys.platform == "darwin" else "KiB",
                  gpu_comparison="measured_conservative_float64_reference" if "original_exact_gpu" in methods else "synthetic_CPU_only",
                  cold_cache_comparison="index build/load separately recorded; no OS cache eviction performed",
                  general_serving_superiority_claim=False)


def synthetic_protocol():
    p = load_protocol()
    p.update(corpus_size=96, dimension=24, m_prime=12, budget_grid=[8, 24, 64], k_gt=5, k_ctx=3)
    p["roles"] = {role: 36 for role in ROLES}
    p["policy"]["minimum_fit"] = 8
    p["features"]["sample_size"] = 12
    p["features"]["lid_neighbors"] = 6
    p["timing"].update(warmup=2, repeats=2, bootstrap_replicates=64)
    return p


def self_test(output, require_faiss):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    p = synthetic_protocol()
    rng = np.random.default_rng(55139)
    corpus = normalized(rng.normal(size=(p["corpus_size"], p["dimension"])))
    ids = tuple(f"doc-{i:04}" for i in range(len(corpus)))
    roles, vectors, positives = {}, {}, {}
    for role in ROLES:
        vectors[role] = normalized(rng.normal(size=(p["roles"][role], p["dimension"])))
        roles[role] = [(f"{role}-{i}", f"synthetic {role} {i}") for i in range(p["roles"][role])]
        for (qid, _), query in zip(roles[role], vectors[role]):
            positives[qid] = set(ExactCandidateIndex(ids, corpus).search(query, 1))
    write_bundle(output / "bundle", ids, corpus, roles, vectors, positives, p,
                 {"kind": "synthetic", "audit": {"semantic_family_independence_established": False}})
    backend = "hnsw" if require_faiss else "exact_fixture"
    rt = Runtime(output / "bundle", output / "run", p, backend)
    for role in ROLES[:3]:
        run_stage(rt, role)
        if rt.ledger.result(role)["status"] != "passed":
            break
    if role != "tune":
        raise RuntimeError("synthetic calibration path failed before tune")
    summary = {"status": "synthetic_paths_checked", "backend": backend,
               "last_stage": role, "last_stage_status": rt.ledger.result(role)["status"],
               "formal_certification": False, "real_data_opened": False,
               "ann_integration_exercised": require_faiss}
    write_json(output / "self_test.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("self-test")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--require-faiss", action="store_true")
    run = commands.add_parser("run-stage")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--run", type=Path, required=True)
    run.add_argument("--role", choices=ROLES, required=True)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test(args.output, args.require_faiss)
    else:
        rt = Runtime(args.bundle, args.run, load_protocol(), planned_role=args.role)
        run_stage(rt, args.role)
        result = rt.ledger.result(args.role)
        print(json.dumps(result, indent=2))
        if result["status"] != "passed":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
