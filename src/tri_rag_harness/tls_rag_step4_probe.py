"""First real retrieval probe. Offline evaluation; no certification or latency claim.

The frozen Step 3 numerical/controller functions are reused through new versioned
parameter containers. Qrels define judged-relevance proxies, never sufficiency.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np
import scipy

from . import tls_rag_step2 as s2
from . import tls_rag_step3 as s3
from .utils import fingerprint

ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = (
    "src/tri_rag_harness/tls_rag_step4_probe.py",
    "src/tri_rag_harness/tls_rag_step4_nfcorpus.py",
    "src/tri_rag_harness/tls_rag_step3.py",
    "src/tri_rag_harness/tls_rag_step2.py",
    "src/tri_rag_harness/tri_law.py",
    "src/tri_rag_harness/embeddings.py",
    "src/tri_rag_harness/indexes.py",
    "src/tri_rag_harness/projection.py",
    "src/tri_rag_harness/utils.py",
    "configs/tls_rag_step3_synthetic_v1.json",
    "configs/tls_rag_step2_synthetic_v1.json",
)
ROLES = ("query_cal_model_fit", "query_cal_bound_fit", "query_tune", "query_probe")
PROTOCOL_PATH = ROOT / "configs/tls_rag_step4_nfcorpus_probe_v2.json"
# Updated only before the first data run; the CLI rejects edited protocols.
PROTOCOL_FINGERPRINT = "9f31fc87ec1c0d38716474c951eefb456d3bda8e3637557390e3e577e5833952"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value) -> str:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return sha256(path)


def save_rows(path: Path, rows) -> str:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return sha256(path)


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if fingerprint(protocol) != PROTOCOL_FINGERPRINT:
        raise ValueError("probe protocol changed; use a new preregistered version")
    return protocol


def canonical_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def assign_roles(queries: dict[str, str], eligible_ids, counts: dict, seed: int):
    """One deterministic representative per exact normalized-text family.

    This prevents mechanical duplicate leakage; it does NOT establish iid units
    or identify every semantic paraphrase. The probe makes descriptive claims.
    """
    if set(counts) != set(ROLES) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("invalid role counts")
    groups = {}
    for query_id in sorted(set(eligible_ids)):
        if not isinstance(query_id, str) or not query_id or query_id not in queries:
            raise ValueError("unknown or invalid eligible query ID")
        text = canonical_text(queries[query_id])
        if not text:
            raise ValueError("empty query text")
        groups.setdefault(text, []).append(query_id)
    units = sorted(groups, key=lambda text: (fingerprint([seed, text]), text))
    if len(units) < sum(counts.values()):
        raise ValueError("insufficient distinct query-text families; no automatic top-up")
    roles, audit, offset = {}, [], 0
    for role in ROLES:
        selected = units[offset:offset + counts[role]]
        roles[role] = [groups[text][0] for text in selected]
        audit.extend({"role": role, "query_id": groups[text][0],
                      "text_family_sha256": fingerprint(text),
                      "excluded_duplicate_ids": groups[text][1:]} for text in selected)
        offset += counts[role]
    return roles, audit


def component_config(protocol: dict) -> s3.Step3Config:
    upstream = s3.load_step3_config(ROOT / "configs/tls_rag_step3_synthetic_v1.json")
    if upstream.config_fingerprint != protocol["component_config_fingerprint"]:
        raise ValueError("frozen upstream component config changed")
    raw = {key: copy.deepcopy(upstream.raw[key]) for key in
           ("risk_profile", "score_model", "calibration", "controller")}
    raw["upstream_component_fingerprint"] = upstream.config_fingerprint
    raw.update(schema_version="tls_rag_step4_probe_components_v2", run_name=protocol["run_name"])
    candidates = tuple(s3.CandidateSpec(f"tls-rag-probe-row{row}-v2", row) for row in (2, 3, 4))
    raw["score_model"].update(version="tls_rag_probe_ridge_v2", fit_partition=ROLES[0])
    raw["calibration"].update(version="tls_rag_probe_reachable_cp_v2", bound_partition=ROLES[1])
    raw["controller"].update(candidates=[asdict(c) for c in candidates],
                              fixed_reference_budgets=protocol["retrieval"]["budget_grid"])
    raw["scope"] = {"descriptive_retrieval_probe_only": True}
    return s3.Step3Config(PROTOCOL_PATH, raw, candidates)


def make_environment(corpus_ids, corpus_texts, corpus_vectors, queries, query_vectors,
                     roles, protocol) -> s3.Step3Environment:
    ids = tuple(corpus_ids)
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("corpus IDs must be unique strings")
    if set(ids) & set(queries):
        raise ValueError("queries must be external to corpus IDs")
    expected_queries = [qid for role in ROLES for qid in roles[role]]
    if len(set(expected_queries)) != len(expected_queries) or set(expected_queries) != set(queries):
        raise ValueError("role/query identities must be disjoint and complete")
    r = protocol["retrieval"]
    if len(ids) < r["budget_grid"][-1]:
        raise ValueError("corpus smaller than frozen maximum budget")
    if len(corpus_texts) != len(ids) or not all(isinstance(t, str) for t in corpus_texts):
        raise ValueError("invalid corpus text rows")
    c = np.asarray(corpus_vectors, dtype=np.float64)
    q = np.asarray(query_vectors, dtype=np.float64)
    if c.shape != (len(ids), protocol["embedding"]["dimension"]) or q.shape != (len(queries), c.shape[1]):
        raise ValueError("embedding shape/model dimension mismatch")
    for values in (c, q):
        if not np.all(np.isfinite(values)) or not np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-6, rtol=0):
            raise ValueError("embeddings must be finite and normalized before projection")
    # A label-free identity audit rejects an actual corpus text reused as a query.
    document_texts = set(canonical_text(t) for t in corpus_texts)
    if any(canonical_text(t) in document_texts for t in queries.values()):
        raise ValueError("query text is a corpus item")
    matrix = s2.dense_gaussian_projection(r["m_prime"], c.shape[1], r["projection_seed"])
    config = replace(s2.load_step2_config(ROOT / "configs/tls_rag_step2_synthetic_v1.json"),
        run_name=protocol["run_name"], dimension=c.shape[1], corpus_size=len(ids),
        projection_seed=r["projection_seed"], m_prime=r["m_prime"], k_gt=r["k_gt"],
        k_ctx=r["k_ctx"], m_pilot=r["budget_grid"][0], budget_grid=tuple(r["budget_grid"]),
        maximum_expansions=len(r["budget_grid"]) - 1,
        evidence_plan_generator="query_longest_token_relevance_proxy_v2")
    query_objects = []
    for index, (query_id, text) in enumerate(queries.items()):
        tokens = re.findall(r"[a-z0-9]+", text.casefold())
        term = min(tokens, key=lambda token: (-len(token), token)) if tokens else ""
        # This plan is an explicit single relevance-proxy slot. No qrels or
        # corpus statistics enter its generator; no rich evidence claim follows.
        plan = s2.EvidencePlan("query_longest_token_relevance_proxy_v2",
                             (s2.FacetSlot("judged_relevance_proxy", 1, False, term),), False)
        query_objects.append(s2.SyntheticQuery(query_id, text, q[index], plan, True))
    objects = tuple(query_objects)
    identity = fingerprint({"corpus_ids": ids, "query_ids": list(queries), "roles": roles,
                            "protocol": protocol})
    upstream = s2.Step2Environment(config, ids, tuple(corpus_texts), c, objects,
                                   matrix, s2.project_rows(c, matrix), identity)
    return s3.Step3Environment(upstream, objects, tuple((k, tuple(v)) for k, v in roles.items()), identity)


def prepare_role(environment, config, query_ids, output, role):
    """Evaluate all prefixes offline, with only exposed inputs at each stage.

    This materialization is for outcome reconstruction. Work counters below
    describe the selected path; its wall time is NOT a serving latency measure.
    """
    prepared = {}
    path = output / f"{role}.features.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for index, query_id in enumerate(query_ids):
            query = s3.prepare_queries(environment, config, [query_id])[0]
            prepared[query_id] = query
            for stage in query.stages:
                record = {"query_id": query_id, "stage": stage.stage,
                          "retrieval": s3._retrieval_state_record(stage),
                          "features": {str(c.row): asdict(s3.make_deployable_state(stage, c))
                                       for c in config.candidates}}
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            if (index + 1) % 25 == 0:
                print(f"{role}: prepared {index + 1}/{len(query_ids)} queries", flush=True)
    return prepared


def phase_a(prepared, config, models, tables, protocol, corpus_size):
    """No annotation handle or outcome argument exists on this decision path."""
    r = protocol["retrieval"]
    records = []
    for query_id, query in prepared.items():
        for stage in query.stages:
            state = stage.decision_input
            records.append({"query_id": query_id, "method": f"fixed-{state.current_budget}",
                "stage": stage.stage, "budget": state.current_budget,
                "candidate_ids": list(state.exposed_candidate_ids),
                "reranked_ids": list(state.exact_reranked_ids), "context_ids": list(state.context_ids),
                "actions": [], "work": {"pilot_original_distances": r["budget_grid"][0],
                "expansion_original_distances": state.current_budget - r["budget_grid"][0],
                "total_original_distances": state.current_budget,
                "projected_distances": corpus_size,
                "rerank_candidates": state.current_budget, "risk_pairs": 0,
                "candidate_diversity_pairs": 0, "plan_feature_evaluations": 0,
                "controller_evaluations": 0}})
        for candidate, model_pair, table in zip(config.candidates, models, tables):
            controller = s3.CalibratedStep3Controller(candidate=candidate, gain_model=model_pair[0],
                sufficiency_model=model_pair[1], calibration_table=table,
                budget_grid=r["budget_grid"], maximum_expansions=len(r["budget_grid"]) - 1,
                delta_gain=config.raw["controller"]["delta_gain"],
                tau_sufficient=config.raw["controller"]["tau_sufficient"])
            actions, risk_pairs, rerank_work = [], 0, 0
            for stage in query.stages:
                decision = controller.choose(s3.make_deployable_state(stage, candidate))
                actions.append({"stage": stage.stage, **asdict(decision)})
                rerank_work += stage.decision_input.current_budget
                if candidate.uses_tri_law:
                    risk_pairs += (stage.risk_profile.context_boundary.attempted_pair_count +
                                   stage.risk_profile.core_frontier.attempted_pair_count)
                if decision.action is s2.Action.STOP:
                    break
            else:
                raise ValueError("controller failed to terminate on frozen grid")
            state = stage.decision_input
            records.append({"query_id": query_id, "method": candidate.candidate_id,
                "stage": stage.stage, "budget": state.current_budget,
                "candidate_ids": list(state.exposed_candidate_ids),
                "reranked_ids": list(state.exact_reranked_ids), "context_ids": list(state.context_ids),
                "actions": actions, "work": {"pilot_original_distances": r["budget_grid"][0],
                "expansion_original_distances": state.current_budget - r["budget_grid"][0],
                "total_original_distances": state.current_budget,
                "projected_distances": corpus_size,
                "rerank_candidates": rerank_work, "risk_pairs": risk_pairs,
                "candidate_diversity_pairs": sum(b * (b - 1) // 2 for b in r["budget_grid"][:stage.stage + 1]),
                "plan_feature_evaluations": len(actions) if candidate.uses_plan_facet else 0,
                "controller_evaluations": len(actions)}})
    return records


def supervision(prepared, labels):
    # Only these two proxy labels are bridged into the frozen component API.
    # Their upstream key names do not grant evidence-sufficiency semantics.
    result = {}
    for query_id, query in prepared.items():
        if query_id not in labels:
            raise ValueError("missing query annotation record")
        relevant = set(labels[query_id])
        hits = [bool(set(stage.decision_input.context_ids) & relevant) for stage in query.stages]
        result[query_id] = [{"stage": index, "current_context_is_sufficient": hit,
            "later_context_has_useful_evidence": bool(not hit and any(hits[index + 1:]))}
            for index, hit in enumerate(hits)]
    return result


def metric_rows(records, environment, labels, prepared):
    r = environment.upstream.config
    corpus_ids = np.asarray(environment.upstream.corpus_ids)
    by_query = {query.query_id: query for query in environment.queries}
    exact = {}
    for query_id in prepared:
        difference = environment.upstream.corpus_embeddings - by_query[query_id].embedding
        distances = np.einsum("ij,ij->i", difference, difference)
        order = np.lexsort((corpus_ids, distances))
        exact[query_id] = corpus_ids[order[:max(r.k_gt, r.k_ctx)]].tolist()
    rows = []
    for record in records:
        query_id = record["query_id"]
        relevant = set(labels[query_id])
        top = exact[query_id][:r.k_gt]
        candidate_set = set(record["candidate_ids"])
        overlap = len(set(top) & candidate_set)
        if overlap != len(set(top) & set(record["reranked_ids"][:r.k_gt])):
            raise AssertionError("candidate/reranked retention disagree")
        context = record["context_ids"]
        later_hit = any(set(s.decision_input.context_ids) & relevant
                        for s in prepared[query_id].stages[record["stage"] + 1:])
        hits = len(set(context) & relevant)
        rows.append({"query_id": query_id, "method": record["method"],
            "budget": record["budget"], "retention": overlap / r.k_gt,
            "hit_at_context": int(hits > 0),
            "judged_recall_at_context": hits / len(relevant) if relevant else 0.,
            "judged_precision_at_context": hits / r.k_ctx,
            "exact_hit_at_context": int(bool(set(exact[query_id][:r.k_ctx]) & relevant)),
            "exact_top_k_ids": top, "context_ids": context,
            "relevant_ids": sorted(relevant),
            "early_missed_later_hit": int(hits == 0 and later_hit),
            "early_stop": int(record["budget"] < r.budget_grid[-1]),
            "work": record["work"]})
    for query_id in prepared:
        relevant = set(labels[query_id])
        top = exact[query_id][:r.k_gt]
        context = exact[query_id][:r.k_ctx]
        hits = len(set(context) & relevant)
        rows.append({"query_id": query_id, "method": "exact-original", "budget": len(corpus_ids),
            "retention": 1., "hit_at_context": int(hits > 0),
            "judged_recall_at_context": hits / len(relevant) if relevant else 0.,
            "judged_precision_at_context": hits / r.k_ctx,
            "exact_hit_at_context": int(hits > 0), "exact_top_k_ids": top,
            "context_ids": context, "relevant_ids": sorted(relevant),
            "early_missed_later_hit": 0, "early_stop": 0,
            "work": {"pilot_original_distances": 0, "expansion_original_distances": 0,
                "reference_original_distances": len(corpus_ids), "total_original_distances": len(corpus_ids),
                "projected_distances": 0, "rerank_candidates": len(corpus_ids), "risk_pairs": 0,
                "candidate_diversity_pairs": 0, "plan_feature_evaluations": 0, "controller_evaluations": 0}})
    return rows


def aggregate(rows):
    result = {}
    for method in sorted({row["method"] for row in rows}):
        values = [row for row in rows if row["method"] == method]
        means = {key: float(np.mean([row[key] for row in values])) for key in
                 ("retention", "hit_at_context", "judged_recall_at_context",
                  "judged_precision_at_context", "exact_hit_at_context",
                  "early_missed_later_hit", "early_stop")}
        means.update(query_count=len(values),
                     mean_original_distances=float(np.mean([row["work"]["total_original_distances"] for row in values])),
                     mean_rerank_candidates=float(np.mean([row["work"]["rerank_candidates"] for row in values])),
                     mean_risk_pairs=float(np.mean([row["work"]["risk_pairs"] for row in values])))
        means["mean_candidate_diversity_pairs"] = float(np.mean([row["work"]["candidate_diversity_pairs"] for row in values]))
        means["mean_plan_feature_evaluations"] = float(np.mean([row["work"]["plan_feature_evaluations"] for row in values]))
        result[method] = means
    return result


def select_candidate(summary, config, protocol):
    target = protocol["selection"]
    eligible = []
    for candidate in config.candidates:
        row = summary[candidate.candidate_id]
        if (row["retention"] >= target["minimum_retention"] and
            row["exact_hit_at_context"] - row["hit_at_context"] <= target["maximum_hit_drop"] and
            row["mean_original_distances"] < protocol["retrieval"]["budget_grid"][-1]):
            eligible.append(candidate.candidate_id)
    return min(eligible, key=lambda key: (summary[key]["mean_original_distances"], key)) if eligible else None


def run_probe(environment, roles, protocol, open_labels: Callable, output: Path, binding=None):
    """Sequential role pipeline. open_labels is invoked only after closure receipts."""
    output.mkdir(parents=True, exist_ok=False)
    config = component_config(protocol)
    if binding is not None:
        save(output / "input_binding.json", binding)
    save(output / "evaluation_environment.json", {"python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__})
    save(output / "protocol.json", protocol)
    save(output / "roles.json", roles)
    save(output / "component_parameters.json", config.raw)
    np.save(output / "projection.npy", environment.upstream.projection_matrix, allow_pickle=False)
    receipts = {}

    def close(role, files):
        receipt = {"role": role, "files": {path.name: sha256(path) for path in files},
                   "previous_receipts": dict(receipts)}
        receipts[role] = save(output / f"{role}.phase_a_closed.json", receipt)
        return receipt

    def labels_after_close(role, receipt):
        for name, digest in receipt["files"].items():
            if sha256(output / name) != digest:
                raise ValueError("Phase A changed before label opening")
        labels = open_labels(role, tuple(roles[role]))
        if not isinstance(labels, dict) or any(
            not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values) for values in labels.values()):
            raise ValueError("invalid annotation shape")
        if set(labels) != set(roles[role]):
            raise ValueError("annotation role IDs do not match frozen assignment")
        if any(not set(values) <= set(environment.upstream.corpus_ids) for values in labels.values()):
            raise ValueError("annotation references unknown document")
        save(output / f"{role}.opened_labels.json", labels)
        return labels

    role = ROLES[0]
    prepared = prepare_role(environment, config, roles[role], output, role)
    receipt = close(role, [output / f"{role}.features.jsonl"])
    fit_labels = labels_after_close(role, receipt)
    fit_supervision = supervision(prepared, fit_labels)
    save(output / "fit_supervision.json", fit_supervision)
    models = s3.fit_score_models(config.candidates, prepared, fit_supervision, roles[role], config)
    model_fp = save(output / "score_models.json", [[m.to_dict() for m in pair] for pair in models])
    del prepared, fit_supervision, fit_labels

    role = ROLES[1]
    prepared = prepare_role(environment, config, roles[role], output, role)
    receipt = close(role, [output / f"{role}.features.jsonl", output / "score_models.json"])
    bound_labels = labels_after_close(role, receipt)
    bound_supervision = supervision(prepared, bound_labels)
    save(output / "bound_supervision.json", bound_supervision)
    tables, bound_records = s3.build_calibration_tables(candidates=config.candidates, models=models, prepared=prepared,
        supervision=bound_supervision, bound_query_ids=roles[role], config=config)
    table_fp = save(output / "calibration_tables.json", [t.to_dict() for t in tables])
    # Upstream record schema is clearly wrapped, not presented as synthetic data.
    save(output / "bound_records.json", {"schema": "tls_rag_probe_bound_records_v2",
        "label_semantics": "judged_relevance_proxy", "component_records": bound_records})
    del prepared, bound_supervision, bound_labels

    summaries, selected = {}, None
    for role in ROLES[2:]:
        prepared = prepare_role(environment, config, roles[role], output, role)
        decisions = phase_a(prepared, config, models, tables, protocol, len(environment.upstream.corpus_ids))
        decision_path = output / f"{role}.decisions.jsonl"
        save_rows(decision_path, decisions)
        dependencies = [decision_path, output / f"{role}.features.jsonl",
                        output / "score_models.json", output / "calibration_tables.json"]
        if role == "query_probe":
            dependencies.append(output / "selection.json")
        receipt = close(role, dependencies)
        labels = labels_after_close(role, receipt)
        metrics = metric_rows(decisions, environment, labels, prepared)
        save_rows(output / f"{role}.metrics.jsonl", metrics)
        summary = aggregate(metrics)
        summaries[role] = summary
        save(output / f"{role}.summary.json", summary)
        if role == "query_tune":
            selected = select_candidate(summary, config, protocol)
            save(output / "selection.json", {"selected_candidate": selected,
                 "score_models_sha256": model_fp, "calibration_tables_sha256": table_fp,
                 "tune_metrics_sha256": sha256(output / f"{role}.metrics.jsonl"),
                 "protocol_fingerprint": fingerprint(protocol), "role_fingerprint": fingerprint(roles)})
            if selected is None:
                break
        del prepared, decisions, labels, metrics
    selected_probe = summaries.get("query_probe", {}).get(selected)
    probe_met = bool(selected_probe is not None and
        selected_probe["retention"] >= protocol["selection"]["minimum_retention"] and
        selected_probe["exact_hit_at_context"] - selected_probe["hit_at_context"] <= protocol["selection"]["maximum_hit_drop"] and
        selected_probe["mean_original_distances"] < protocol["retrieval"]["budget_grid"][-1])
    deltas = {}
    for role, summary in summaries.items():
        deltas[role] = {}
        for lower, upper in zip(config.candidates, config.candidates[1:]):
            before, after = summary[lower.candidate_id], summary[upper.candidate_id]
            deltas[role][f"row{upper.row}_minus_row{lower.row}"] = {
                metric: after[metric] - before[metric] for metric in
                ("retention", "hit_at_context", "mean_original_distances",
                 "mean_rerank_candidates", "mean_risk_pairs")}
    result = {"schema": "tls_rag_nfcorpus_probe_result_v2", "selected_candidate": selected,
        "status": "no_tune_candidate" if selected is None else
                  ("descriptive_probe_targets_met" if probe_met else "descriptive_probe_targets_failed"),
        "certified": False, "latency_claim": False, "evidence_sufficiency_claim": False,
        "official_dev_test_opened": False, "summaries": summaries, "ablation_deltas": deltas}
    save(output / "result.json", result)
    lines = ["# TLS-RAG first real retrieval probe", "", f"Status: {result['status']}",
             f"Selected on tune: {selected}", "", "Descriptive results only. No scientific certification, serving latency or evidence-sufficiency claim.",
             "All-prefix offline computation time is not selected-policy latency.", "",
             "| Role | Method | Retention | Hit@context | Original distances | Rerank candidates | Risk pairs |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for role, summary in summaries.items():
        for method, row in summary.items():
            lines.append(f"| {role} | {method} | {row['retention']:.4f} | {row['hit_at_context']:.4f} | {row['mean_original_distances']:.1f} | {row['mean_rerank_candidates']:.1f} | {row['mean_risk_pairs']:.1f} |")
    lines += ["", "## Feature-family differences", "",
              "These are paired mean differences on the same queries. Negative work means less counted work.",
              "Selecting Row 2 does not establish a benefit from Tri-Law features.", "",
              "| Role | Difference | Retention | Hit@context | Original distances | Risk pairs |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for role, pairs in deltas.items():
        for pair, row in pairs.items():
            lines.append(f"| {role} | {pair} | {row['retention']:+.4f} | {row['hit_at_context']:+.4f} | {row['mean_original_distances']:+.1f} | {row['mean_risk_pairs']:+.1f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    save(output / "artifact_manifest.json", {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    return result


def load_bundle(bundle: Path, protocol: dict):
    manifest = json.loads((bundle / "binding.json").read_text())
    if manifest.get("schema") != "tls_rag_nfcorpus_probe_binding_v2":
        raise ValueError("wrong bundle schema")
    if manifest.get("source_revisions") != {
        "dataset": protocol["dataset_revision"], "qrels": protocol["qrels_revision"]}:
        raise ValueError("wrong dataset revision binding")
    if manifest["protocol_fingerprint"] != fingerprint(protocol):
        raise ValueError("bundle belongs to a different protocol")
    names = {"corpus.jsonl", "queries.jsonl", "roles.json", "corpus.npy", "queries.npy"}
    names |= {f"{role}.labels.json" for role in ROLES}
    if set(manifest["files"]) != names:
        raise ValueError("unexpected bundle file manifest")
    for name, digest in manifest["files"].items():
        path = bundle / name
        if path.is_symlink() or sha256(path) != digest:
            raise ValueError(f"bundle hash/path mismatch: {name}")
    for name in CODE_FILES:
        if manifest["source_code_sha256"].get(name) != sha256(ROOT / name):
            raise ValueError("bundle/code identity mismatch: " + name)
    if manifest["embedding"] != protocol["embedding"]:
        raise ValueError("embedding metadata mismatch")
    corpus = [json.loads(line) for line in (bundle / "corpus.jsonl").read_text().splitlines()]
    query_rows = [json.loads(line) for line in (bundle / "queries.jsonl").read_text().splitlines()]
    queries = {row["query_id"]: row["text"] for row in query_rows}
    if len(queries) != len(query_rows):
        raise ValueError("duplicate query rows")
    roles = json.loads((bundle / "roles.json").read_text())
    if set(roles) != set(ROLES) or any(len(roles[k]) != protocol["role_counts"][k] for k in ROLES):
        raise ValueError("wrong role counts")
    env = make_environment([row["passage_id"] for row in corpus], [row["text"] for row in corpus],
        np.load(bundle / "corpus.npy", mmap_mode="r", allow_pickle=False), queries,
        np.load(bundle / "queries.npy", mmap_mode="r", allow_pickle=False), roles, protocol)
    def loader(role, query_ids):
        if role not in ROLES:
            raise ValueError("unknown label role")
        path = bundle / f"{role}.labels.json"
        if path.is_symlink() or sha256(path) != manifest["files"][path.name]:
            raise ValueError("label file changed after bundle validation")
        labels = json.loads(path.read_text())
        if set(labels) != set(query_ids):
            raise ValueError("wrong label role")
        return labels
    return env, roles, loader, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = load_protocol()
    env, roles, loader, manifest = load_bundle(args.bundle.resolve(), protocol)
    result = run_probe(env, roles, protocol, loader, args.output.resolve(), binding=manifest)
    print(json.dumps({key: value for key, value in result.items() if key != "summaries"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
