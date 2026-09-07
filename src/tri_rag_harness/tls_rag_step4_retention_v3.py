"""Retention-aligned v3 successor. Reuses unopened v2 probe identities explicitly.

Calibration residual quantiles are empirical safety margins, not confidence
bounds or per-query guarantees. All ground truth is outside Controller.choose.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import tls_rag_step4_probe as v2
from .tls_rag_step4_probe import s2, s3, fingerprint, save, save_rows, sha256

ROOT = v2.ROOT
PROTOCOL_PATH = ROOT / 'configs/tls_rag_step4_retention_v3.json'
PROTOCOL_FP = '1d730efc412851c45b5ae2a600387ed4aca6c749aceaff87c30f40eebf55e026'
ROWS = (2, 3, 4)


def load_protocol():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if fingerprint(protocol) != PROTOCOL_FP:
        raise ValueError('v3 protocol changed; preregister a new version')
    if protocol['parent_protocol_fingerprint'] != fingerprint(v2.load_protocol()):
        raise ValueError('parent protocol changed')
    return protocol


def candidate(row):
    return s3.CandidateSpec(f'tls-rag-retention-row{row}-v3', row)


@dataclass(frozen=True)
class Policy:
    row: int
    threshold: float
    quantile: float

    @property
    def policy_id(self):
        return f'row{self.row}-retention-v3-t{self.threshold:.3f}-q{self.quantile:.2f}'


def policy_grid(protocol):
    return tuple(Policy(row, threshold, quantile) for row in ROWS
                 for threshold in protocol['stop_thresholds']
                 for quantile in protocol['residual_quantiles'])


@dataclass(frozen=True)
class Stage:
    retrieval: dict
    states: dict

    @property
    def decision_input(self):
        # Metric reconstruction needs only contexts. No label joins here.
        return SimpleNamespace(context_ids=self.retrieval['context_ids'])


@dataclass(frozen=True)
class Query:
    query_id: str
    stages: tuple


def parse_features(path, ids, grid):
    """Read the fixed v2 feature format after checking its content hash."""
    requested = set(ids)
    grouped = {qid: {} for qid in ids}
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if set(record) != {'query_id', 'stage', 'retrieval', 'features'}:
                raise ValueError('unexpected cached feature fields')
            qid, index = record['query_id'], record['stage']
            if qid not in requested or type(index) is not int or index not in range(len(grid)):
                raise ValueError('unknown feature query/stage')
            if index in grouped[qid] or set(record['features']) != {'2', '3', '4'}:
                raise ValueError('duplicate stage or missing feature family')
            retrieval = record['retrieval']
            if (retrieval['query_id'] != qid or retrieval['stage'] != index or
                retrieval['current_budget'] != grid[index] or
                len(retrieval['exposed_candidate_ids']) != grid[index] or
                len(set(retrieval['exposed_candidate_ids'])) != grid[index]):
                raise ValueError('cached retrieval identity mismatch')
            if set(retrieval['exact_reranked_ids']) != set(retrieval['exposed_candidate_ids']):
                raise ValueError('cached reranking is not a permutation')
            states = {}
            for row in ROWS:
                values = dict(record['features'][str(row)])
                values['context_ids'] = tuple(values['context_ids'])
                values['feature_values'] = tuple(tuple(pair) for pair in values['feature_values'])
                state = s3.Step3DeployableState(**values)
                s2.assert_deployable_only(state)
                if (state.schema_version != s3.STATE_SCHEMA or state.query_id != qid or
                    state.stage != index or state.row != row or state.current_budget != grid[index] or
                    state.remaining_grid_steps != len(grid) - index - 1 or
                    list(state.context_ids) != retrieval['context_ids'] or
                    tuple(name for name, _ in state.feature_values) != s3.feature_names_for_row(row)):
                    raise ValueError('cached state identity/schema mismatch')
                states[row] = replace(state, candidate_id=candidate(row).candidate_id)
            grouped[qid][index] = Stage(retrieval, states)
    if any(set(stages) != set(range(len(grid))) for stages in grouped.values()):
        raise ValueError('incomplete cached query stages')
    return {qid: Query(qid, tuple(grouped[qid][i] for i in range(len(grid)))) for qid in ids}


def retention_targets(prepared, environment):
    """Offline cal supervision only; never called by the controller."""
    corpus = environment.upstream.corpus_embeddings
    ids = np.asarray(environment.upstream.corpus_ids)
    queries = {q.query_id: q for q in environment.queries}
    k = environment.upstream.config.k_gt
    targets, records = {}, []
    for qid, query in prepared.items():
        difference = corpus - queries[qid].embedding
        distances = np.einsum('ij,ij->i', difference, difference)
        top = ids[np.lexsort((ids, distances))[:k]].tolist()
        targets[qid] = []
        for index, stage in enumerate(query.stages):
            value = len(set(top) & set(stage.retrieval['exposed_candidate_ids'])) / k
            targets[qid].append(value)
            records.append({'query_id': qid, 'stage': index, 'retention': value, 'exact_top_k_ids': top})
    return targets, records


def valid(state):
    return state.base_state_valid and state.required_risk_profile_valid and state.evidence_plan_valid


def fit_models(prepared, targets, protocol):
    # Reuse the numerical ridge fitter directly with continuous targets. The
    # v2 boolean supervision bridge is intentionally not used.
    config = v2.component_config(v2.load_protocol())
    config.raw['score_model'].update(version='tls_rag_stage_retention_ridge_v3',
                                    regularization=protocol['ridge_regularization'])
    models = {}
    for row in ROWS:
        for index in range(len(next(iter(prepared.values())).stages)):
            rows = [(qid, query.stages[index].states[row], targets[qid][index])
                    for qid, query in prepared.items() if valid(query.stages[index].states[row])]
            if len(rows) < protocol['minimum_calibration_queries']:
                models[row, index] = None
            else:
                models[row, index] = s3._fit_score_model(outcome='exact_top_k_retention_fraction',
                    candidate=candidate(row), rows=rows, config=config)
    return models


def calibrate(prepared, targets, models, protocol):
    calibration, records = {}, []
    for (row, index), model in models.items():
        residuals = []
        if model is not None:
            for qid, query in prepared.items():
                state = query.stages[index].states[row]
                prediction = model.predict(state)
                if not valid(state) or not prediction.valid:
                    continue
                residual = prediction.score - targets[qid][index]
                residuals.append(residual)
                records.append({'query_id': qid, 'row': row, 'stage': index,
                    'prediction': prediction.score, 'retention': targets[qid][index], 'overprediction': residual})
        for quantile in protocol['residual_quantiles']:
            powered = len(residuals) >= protocol['minimum_calibration_queries']
            calibration[row, index, quantile] = {
                'valid': powered, 'query_count': len(residuals),
                'margin': max(0., float(np.quantile(residuals, quantile, method='higher'))) if powered else 1.}
    return calibration, records


class Controller:
    def __init__(self, policy, models, calibration, grid):
        self.policy, self.models, self.calibration, self.grid = policy, models, calibration, tuple(grid)

    def choose(self, state):
        if not isinstance(state, s3.Step3DeployableState):
            raise TypeError('only a deployable state is accepted')
        s2.assert_deployable_only(state)
        if (state.row != self.policy.row or state.candidate_id != candidate(self.policy.row).candidate_id or
            state.schema_version != s3.STATE_SCHEMA or type(state.stage) is not int or
            state.stage not in range(len(self.grid)) or self.grid[state.stage] != state.current_budget or
            state.remaining_grid_steps != len(self.grid) - state.stage - 1 or
            tuple(name for name, _ in state.feature_values) != s3.feature_names_for_row(state.row)):
            raise ValueError('invalid deployable identity/schema/grid')
        model = self.models.get((state.row, state.stage))
        cell = self.calibration.get((state.row, state.stage, self.policy.quantile))
        prediction = model.predict(state) if model is not None else None
        powered = bool(valid(state) and prediction is not None and prediction.valid and cell and cell['valid'])
        score = float(np.clip(prediction.score - cell['margin'], 0., 1.)) if powered else 0.
        meets = powered and score >= self.policy.threshold
        terminal = state.stage == len(self.grid) - 1
        reason = ('retention_score_stop' if meets else 'terminal_budget_without_retention_assertion'
                  if terminal else 'retention_score_requires_expand' if powered else 'invalid_or_uncalibrated_expand')
        return {'stage': state.stage, 'budget': state.current_budget,
                'action': 'STOP' if meets or terminal else 'EXPAND_TO_NEXT_GRID_VALUE', 'reason': reason,
                'prediction': prediction.score if prediction is not None and prediction.valid else None,
                'margin': cell['margin'] if cell else None, 'calibrated_retention_score': score,
                'threshold': self.policy.threshold, 'calibration_valid': powered}


def phase_a(prepared, policies, models, calibration, grid, corpus_size, references=True):
    records = []
    for qid, query in prepared.items():
        def record(stage, method, actions, row=None):
            r = stage.retrieval
            budget = r['current_budget']
            visited = query.stages[:r['stage'] + 1]
            risk_pairs = sum(dict(s.states[3].feature_values)['context_boundary_attempted_pair_count'] +
                             dict(s.states[3].feature_values)['core_frontier_attempted_pair_count'] for s in visited)
            return {'query_id': qid, 'method': method, 'stage': r['stage'], 'budget': budget,
                'candidate_ids': r['exposed_candidate_ids'], 'reranked_ids': r['exact_reranked_ids'],
                'context_ids': r['context_ids'], 'actions': actions,
                'work': {'pilot_original_distances': grid[0], 'expansion_original_distances': budget - grid[0],
                    'total_original_distances': budget, 'projected_distances': corpus_size,
                    'rerank_candidates': sum(grid[:r['stage'] + 1]) if row else budget,
                    'risk_pairs': int(risk_pairs) if row and row >= 3 else 0,
                    'candidate_diversity_pairs': sum(b * (b - 1) // 2 for b in grid[:r['stage'] + 1]) if row else 0,
                    'plan_feature_evaluations': len(actions) if row == 4 else 0,
                    'controller_evaluations': len(actions)}}
        if references:
            records.extend(record(stage, f'fixed-{stage.retrieval["current_budget"]}', []) for stage in query.stages)
        for policy in policies:
            controller = Controller(policy, models, calibration, grid)
            actions = []
            for stage in query.stages:
                decision = controller.choose(stage.states[policy.row])
                actions.append(decision)
                if decision['action'] == 'STOP':
                    break
            records.append(record(stage, policy.policy_id, actions, policy.row))
    return records


def gates(summary, protocol):
    target = protocol['acceptance']
    return {'retention': summary['retention'] >= target['minimum_retention'],
            'context_hit_drop': summary['exact_hit_at_context'] - summary['hit_at_context'] <= target['maximum_hit_drop'],
            'original_distance_saving': summary['mean_original_distances'] < target['maximum_mean_original_distances_exclusive']}


def select(summary, policies, protocol):
    eligible = [p for p in policies if all(gates(summary[p.policy_id], protocol).values())]
    return min(eligible, key=lambda p: (summary[p.policy_id]['mean_original_distances'], p.policy_id)) if eligible else None


def audit_decisions(records):
    result = {}
    for method in sorted({r['method'] for r in records if r['actions']}):
        rows = [r for r in records if r['method'] == method]
        result[method] = {'query_count': len(rows), 'stop_budgets': {}, 'action_reasons': {}}
        for record in rows:
            key = str(record['budget'])
            result[method]['stop_budgets'][key] = result[method]['stop_budgets'].get(key, 0) + 1
            for action in record['actions']:
                key = action['reason']
                result[method]['action_reasons'][key] = result[method]['action_reasons'].get(key, 0) + 1
    return result


def run(environment, roles, protocol, feature_provider, label_loader, output, lineage=None):
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'evaluation_environment.json', {'python': platform.python_version(),
        'numpy': np.__version__, 'scipy': v2.scipy.__version__, 'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'thread_environment': {name: os.environ.get(name) for name in
                               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}})
    save(output / 'protocol.json', protocol)
    save(output / 'roles.json', roles)
    save(output / 'lineage.json', lineage or {'synthetic_fixture': True})
    grid = tuple(environment.upstream.config.budget_grid)
    receipts = {}

    def close(role, files):
        receipt = {'role': role, 'files': {p.name: sha256(p) for p in files},
                   'previous_receipts': dict(receipts), 'protocol_fingerprint': fingerprint(protocol)}
        receipts[role] = save(output / f'{role}.phase_a_closed.json', receipt)
        return receipt

    def check(receipt):
        if any(sha256(output / name) != digest for name, digest in receipt['files'].items()):
            raise ValueError('closed Phase A changed')

    models, calibration = {}, {}
    for role in v2.ROLES[:2]:
        path = feature_provider(role, output)
        prepared = parse_features(path, roles[role], grid)
        dependencies = [path]
        if models:
            dependencies.append(output / 'retention_models.json')
        receipt = close(role, dependencies)
        check(receipt)
        targets, target_rows = retention_targets(prepared, environment)
        save_rows(output / f'{role}.retention_targets.jsonl', target_rows)
        if role == v2.ROLES[0]:
            models = fit_models(prepared, targets, protocol)
            save(output / 'retention_models.json', {f'row{row}-stage{index}': model.to_dict() if model else None
                                                   for (row, index), model in models.items()})
        else:
            calibration, residual_rows = calibrate(prepared, targets, models, protocol)
            save(output / 'retention_calibration.json', {f'row{r}-stage{i}-q{q}': cell for (r, i, q), cell in calibration.items()})
            save_rows(output / 'calibration_residuals.jsonl', residual_rows)
        del prepared, targets
    policies = policy_grid(protocol)
    summaries, diagnostics, selected = {}, {}, None
    for role in v2.ROLES[2:]:
        path = feature_provider(role, output)
        prepared = parse_features(path, roles[role], grid)
        evaluated = policies if role == 'query_tune' else tuple(Policy(row, selected.threshold, selected.quantile) for row in ROWS)
        decisions = phase_a(prepared, evaluated, models, calibration, grid, len(environment.upstream.corpus_ids))
        decision_path = output / f'{role}.decisions.jsonl'
        save_rows(decision_path, decisions)
        dependencies = [path, decision_path, output / 'retention_models.json', output / 'retention_calibration.json']
        if role == 'query_probe':
            dependencies.append(output / 'selection.json')
        receipt = close(role, dependencies)
        check(receipt)
        labels = label_loader(role, tuple(roles[role]))
        if (not isinstance(labels, dict) or set(labels) != set(roles[role]) or
            any(not isinstance(v, list) or any(not isinstance(i, str) for i in v) or len(v) != len(set(v)) or
                not set(v) <= set(environment.upstream.corpus_ids) for v in labels.values())):
            raise ValueError('invalid role annotations')
        save(output / f'{role}.opened_labels.json', labels)
        metrics = v2.metric_rows(decisions, environment, labels, prepared)
        save_rows(output / f'{role}.metrics.jsonl', metrics)
        summary = v2.aggregate(metrics)
        summaries[role] = summary
        save(output / f'{role}.summary.json', summary)
        diagnostics[role] = audit_decisions(decisions)
        save(output / f'{role}.stopping_audit.json', diagnostics[role])
        if role == 'query_tune':
            selected = select(summary, policies, protocol)
            save(output / 'selection.json', {'selected_policy': asdict(selected) if selected else None,
                'selected_candidate': selected.policy_id if selected else None,
                'candidate_gates': {p.policy_id: gates(summary[p.policy_id], protocol) for p in policies},
                'models_sha256': sha256(output / 'retention_models.json'),
                'calibration_sha256': sha256(output / 'retention_calibration.json'),
                'tune_metrics_sha256': sha256(output / f'{role}.metrics.jsonl'),
                'role_fingerprint': fingerprint(roles), 'protocol_fingerprint': fingerprint(protocol)})
            if selected is None:
                break
    acceptance = gates(summaries['query_probe'][selected.policy_id], protocol) if selected else None
    deltas = {}
    if selected:
        probe = summaries['query_probe']
        for lower, upper in ((2, 3), (3, 4)):
            before = probe[Policy(lower, selected.threshold, selected.quantile).policy_id]
            after = probe[Policy(upper, selected.threshold, selected.quantile).policy_id]
            deltas[f'row{upper}_minus_row{lower}'] = {key: after[key] - before[key] for key in
                ('retention', 'hit_at_context', 'mean_original_distances', 'mean_rerank_candidates', 'mean_risk_pairs')}
    result = {'schema': 'tls_rag_retention_acceptance_v3', 'selected_candidate': selected.policy_id if selected else None,
        'status': 'no_tune_candidate' if selected is None else
                  ('heldout_targets_met' if all(acceptance.values()) else 'heldout_targets_failed'),
        'acceptance_gates': acceptance, 'probe_opened': selected is not None,
        'certified': False, 'latency_claim': False, 'evidence_sufficiency_claim': False,
        'official_dev_test_opened': False, 'matched_feature_deltas': deltas, 'summaries': summaries}
    save(output / 'result.json', result)
    lines = ['# TLS-RAG retention v3 acceptance', '', f'Status: {result["status"]}',
        f'Selected on tune: {result["selected_candidate"]}', f'Held-out gates: {acceptance}', '',
        'Descriptive held-out acceptance; no statistical certification or latency claim.',
        'Residual-adjusted scores are not confidence bounds. Audit runtime includes all prefixes.', '',
        '| Role | Method | Retention | Hit@context | Original distances | Early-stop fraction |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for role, summary in summaries.items():
        for method, row in summary.items():
            lines.append(f'| {role} | {method} | {row["retention"]:.4f} | {row["hit_at_context"]:.4f} | {row["mean_original_distances"]:.2f} | {row["early_stop"]:.4f} |')
    lines += ['', 'Matched feature comparisons use the same tune-selected threshold and margin quantile.',
              'A passing selected controller does not by itself establish a Tri-Law benefit.', '',
              '```json', json.dumps(deltas, indent=2), '```', '',
              'See query_tune.stopping_audit.json and query_probe.stopping_audit.json (if opened) for stop budgets/reasons.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    save(output / 'artifact_manifest.json', {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    return result


def verify_parent(previous, protocol):
    evaluation = previous / 'evaluation'
    for suffix in ('features.jsonl', 'decisions.jsonl', 'metrics.jsonl', 'summary.json',
                   'phase_a_closed.json', 'opened_labels.json'):
        if (evaluation / f'query_probe.{suffix}').exists():
            raise ValueError('v2 probe has already been opened/prepared; independent reuse refused')
    manifest = json.loads((evaluation / 'artifact_manifest.json').read_text())
    def verified(name):
        path = evaluation / name
        if path.is_symlink() or sha256(path) != manifest.get(name):
            raise ValueError('v2 artifact fingerprint mismatch: ' + name)
        return path
    result = json.loads(verified('result.json').read_text())
    selection = json.loads(verified('selection.json').read_text())
    if (result.get('schema') != 'tls_rag_nfcorpus_probe_result_v2' or
        result.get('status') != 'no_tune_candidate' or result.get('selected_candidate') is not None or
        selection.get('selected_candidate') is not None or 'query_probe' in result.get('summaries', {})):
        raise ValueError('requires the completed unsuccessful v2 tune run')
    environment, roles, loader, binding = v2.load_bundle(previous / 'bundle', v2.load_protocol())
    if (json.loads(verified('input_binding.json').read_text()) != binding or
        json.loads(verified('roles.json').read_text()) != roles or
        json.loads(verified('protocol.json').read_text()) != v2.load_protocol()):
        raise ValueError('v2 evaluation/bundle identity mismatch')
    cached = {role: verified(f'{role}.features.jsonl') for role in v2.ROLES[:3]}
    lineage = {'v2_binding_sha256': sha256(previous / 'bundle/binding.json'),
        'v2_result_sha256': sha256(verified('result.json')), 'v2_manifest_sha256': sha256(evaluation / 'artifact_manifest.json'),
        'reused_development_roles': list(v2.ROLES[:3]), 'preserved_unopened_probe_ids': roles['query_probe'],
        'source_code_sha256': {name: sha256(ROOT / name) for name in (*v2.CODE_FILES,
            'src/tri_rag_harness/tls_rag_step4_retention_v3.py', 'configs/tls_rag_step4_retention_v3.json')}}
    def feature_provider(role, output):
        if role in cached:
            path = verified(f'{role}.features.jsonl')
            target = output / path.name
            if target.exists():
                raise FileExistsError(target)
            shutil.copyfile(path, target)
            if sha256(target) != manifest[path.name]:
                raise ValueError('feature copy changed')
            print(f'{role}: reusing verified label-free features', flush=True)
            return target
        if role != 'query_probe':
            raise ValueError('unknown role')
        v2.prepare_role(environment, v2.component_config(v2.load_protocol()), roles[role], output, role)
        return output / f'{role}.features.jsonl'
    return environment, roles, loader, feature_provider, lineage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    protocol = load_protocol()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('use an absent v3 output; previous results are preserved')
    environment, roles, loader, provider, lineage = verify_parent(args.previous_run.resolve(), protocol)
    result = run(environment, roles, protocol, provider, loader, output, lineage)
    print(json.dumps({k: v for k, v in result.items() if k != 'summaries'}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
