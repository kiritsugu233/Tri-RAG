"""Bounded deployable features and newly fitted ANN retention controllers.

The sampled observed-pair features are diagnostics, never a recall certificate.
No evidence-plan proxy, future prefix, churn, oracle LID or label is an input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .lid import estimate_lid_from_squared_distances
from .tls_rag_step3 import build_observed_pair_risk_profile
from .utils import fingerprint

BASE_NAMES = ("budget", "stage", "original_min", "original_max", "original_mean",
              "context_boundary", "frontier_gap", "distortion_mean", "distortion_max",
              "sample_pair_mean", "sample_duplicate_pairs", "observed_rerank_lid", "lid_valid")
RISK_NAMES = ("boundary_risk_mean", "boundary_risk_max", "frontier_risk_mean",
              "frontier_risk_max", "core_log_distortion_q90", "shell_log_distortion_q90")


@dataclass(frozen=True)
class DeployableState:
    schema: str
    stage: int
    budget: int
    base: tuple
    risk: tuple
    base_valid: bool
    risk_valid: bool


def build_state(session, record, protocol):
    """Read only the current session's accepted candidates and original scores."""
    feature = protocol["features"]
    pool, context = record["candidate_ids"], record["context_ids"]
    ordered = sorted(pool, key=lambda i: (session.projected_cache[i], i))
    # Bounded evenly spaced projected ranks, plus context and current new shell.
    count = min(feature["sample_size"], len(ordered))
    sample = [ordered[i] for i in np.linspace(0, len(ordered) - 1, count, dtype=int)]
    shell = record["new_ids"][-feature["shell_size"]:]
    sample = list(dict.fromkeys(sample + context + shell))
    original = np.asarray([session.original_cache[i] for i in pool])
    projected = np.asarray([session.projected_cache[i] for i in pool])
    positive = original > 0
    distortion = projected[positive] / original[positive]
    vectors = np.asarray([session.original[session.row_by_id[i]] for i in sample], dtype=np.float64)
    pair = vectors[:, None, :] - vectors[None, :, :]
    pair_distances = np.einsum("ijd,ijd->ij", pair, pair)[np.triu_indices(len(sample), 1)]
    sorted_original = np.sort(original)
    lid = estimate_lid_from_squared_distances(sorted_original,
        s_lid=min(feature["lid_neighbors"], len(original)), min_neighbors=3,
        clip_min=1., clip_max=200., duplicate_tolerance=0., fallback=200.)
    base = (float(len(pool)), float(record["stage"]), float(original.min()),
            float(original.max()), float(original.mean()),
            session.original_cache[context[-1]],
            float(sorted_original[-1] - sorted_original[-2]) if len(pool) > 1 else 0.,
            float(distortion.mean()) if len(distortion) else 0.,
            float(distortion.max()) if len(distortion) else 0.,
            float(pair_distances.mean()) if len(pair_distances) else 0.,
            float(np.count_nonzero(pair_distances == 0)), lid.clipped, float(lid.valid))
    profile = build_observed_pair_risk_profile(
        query_embedding=session.query, exposed_candidate_ids=sample, context_ids=context,
        shell_candidate_ids=shell,
        candidate_embeddings_by_id={i: session.original[session.row_by_id[i]] for i in sample},
        projected_squared_distances_by_id={i: session.projected_cache[i] for i in sample},
        original_squared_distances_by_id={i: session.original_cache[i] for i in sample},
        m_prime=protocol["m_prime"])
    risk = (profile.context_boundary.probability_mean, profile.context_boundary.probability_maximum,
            profile.core_frontier.probability_mean, profile.core_frontier.probability_maximum,
            profile.core_log_distortion.q90, profile.shell_log_distortion.q90)
    record["feature_sample_ids"] = sample
    record["feature_pair_count"] = len(pair_distances)
    record["risk_pair_count"] = profile.context_boundary.attempted_pair_count + profile.core_frontier.attempted_pair_count
    return DeployableState(protocol["schema"], record["stage"], len(pool), base, risk,
                           bool(np.all(positive) and np.all(np.isfinite(base))),
                           bool(profile.required_profile_valid and np.all(np.isfinite(risk))))


def values(state, family):
    if family not in {"base", "tri"}:
        raise ValueError("unknown feature family")
    if type(state) is not DeployableState or state.schema != "tls-rag-step5-ann-v1":
        raise TypeError("only the new deployable state is accepted")
    return np.asarray(state.base + (state.risk if family == "tri" else ()), dtype=np.float64)


def fit_models(examples, protocol):
    models = {}
    policy = protocol["policy"]
    for family in ("base", "tri"):
        for stage in range(len(protocol["budget_grid"])):
            rows = [(state, target) for state, target in examples if state.stage == stage
                    and state.base_valid and (family == "base" or state.risk_valid)]
            key = f"{family}:{stage}"
            if len(rows) < policy["minimum_fit"]:
                models[key] = None
                continue
            x = np.vstack([values(state, family) for state, _ in rows])
            y = np.asarray([target for _, target in rows], dtype=np.float64)
            if np.any(~np.isfinite(y)) or np.any((y < 0) | (y > 1)):
                raise ValueError("invalid calibration target")
            mean, scale = x.mean(axis=0), x.std(axis=0)
            scale = np.where(scale > policy["scale_floor"], scale, 1.)
            design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
            penalty = np.eye(design.shape[1]) * policy["ridge"]
            penalty[0, 0] = 0.
            coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
            models[key] = {"mean": mean.tolist(), "scale": scale.tolist(),
                           "coefficients": coefficients.tolist(), "n": len(rows),
                           "names": list(BASE_NAMES + (RISK_NAMES if family == "tri" else ()))}
    return models


def predict(state, family, models):
    x = values(state, family)
    model = models.get(f"{family}:{state.stage}")
    if model is None or not state.base_valid or (family == "tri" and not state.risk_valid):
        return None
    if len(x) != len(model["mean"]) or len(x) != len(model["scale"]) or not np.all(np.isfinite(x)):
        raise ValueError("feature schema changed")
    coefficients = np.asarray(model["coefficients"])
    return float(coefficients[0] + ((x - model["mean"]) / model["scale"]) @ coefficients[1:])


def calibrate(examples, models, protocol):
    margins, records = {}, []
    for family in ("base", "tri"):
        for stage in range(len(protocol["budget_grid"])):
            residuals = []
            for state, target in examples:
                if state.stage == stage:
                    score = predict(state, family, models)
                    if score is not None:
                        residuals.append(score - target)
            records.append({"family": family, "stage": stage, "overprediction": residuals})
            for quantile in protocol["policy"]["quantiles"]:
                margins[f"{family}:{stage}:{quantile}"] = (
                    max(0., float(np.quantile(residuals, quantile, method="higher")))
                    if len(residuals) >= protocol["policy"]["minimum_fit"] else None)
    return margins, records


def policy_grid(protocol):
    return [{"family": family, "threshold": threshold, "quantile": quantile}
            for family in ("base", "tri") for threshold in protocol["policy"]["thresholds"]
            for quantile in protocol["policy"]["quantiles"]]


def choose(state, policy, models, margins, protocol):
    values(state, policy["family"])
    grid = protocol["budget_grid"]
    if type(state.stage) is not int or not 0 <= state.stage < len(grid) or grid[state.stage] != state.budget:
        raise ValueError("state/grid mismatch")
    if policy not in policy_grid(protocol):
        raise ValueError("unregistered policy")
    score = predict(state, policy["family"], models)
    margin = margins.get(f"{policy['family']}:{state.stage}:{policy['quantile']}")
    adjusted = float(np.clip(score - margin, 0, 1)) if score is not None and margin is not None else None
    meets = adjusted is not None and adjusted >= policy["threshold"]
    terminal = state.stage == len(grid) - 1
    return {"action": "STOP" if meets or terminal else "EXPAND", "score": adjusted,
            "reason": "score_stop" if meets else "terminal_without_quality_assertion" if terminal else "expand"}


def policy_id(policy):
    return fingerprint(policy)[:16]
