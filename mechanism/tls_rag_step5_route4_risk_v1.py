"""Approved P1 offline diagnostics; never an online budget decision API."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from tri_rag_harness.tri_law import tri_law_probability


def retention_loss_upper_bound(mu, k, g, M, pool_size):
    """Fixed-geometry expectation bound; no competitor-independence assumption."""
    mu = np.asarray(mu, dtype=np.float64)
    if not (k >= g >= 0 and M >= k and pool_size >= g) or mu.shape != (g,):
        raise ValueError('invalid fixed-pool rank domain')
    if not np.isfinite(mu).all() or np.any(mu < 0):
        raise ValueError('nonfinite/negative expected count')
    filtering = 0.0 if M >= pool_size else float(np.minimum(1, mu / (M - g + 1)).sum() / k)
    return dict(upstream=(k - g) / k, filtering=filtering, total=(k - g) / k + filtering)


def offline_pair_risk(q, X, G, C, dimensions, budgets, block=4096):
    """All C\\G competitors; geometry reused across dimensions; truth offline only.

    Undefined/equal-distance/rounding-uncertain terms receive upper bound one.
    This intentionally does not extend the protected Tri-Law domain.
    """
    G, C = np.asarray(G, dtype=np.int64), np.asarray(C, dtype=np.int64)
    H = np.intersect1d(G, C)
    competitors = np.setdiff1d(C, G)
    sums = {m: np.zeros(len(H)) for m in dimensions}
    outside = 0
    for i, row in enumerate(H):
        a = np.asarray(X[row], dtype=np.float64) - q
        aa = float(a @ a)
        for start in range(0, len(competitors), block):
            b = np.asarray(X[competitors[start:start + block]], dtype=np.float64) - q
            bb = np.einsum('ij,ij->i', b, b)
            with np.errstate(divide='ignore', invalid='ignore'):
                beta = bb / aa
                rho = (b @ a) / np.sqrt(aa * bb)
            valid = (aa > 0) & np.isfinite(beta) & (beta > 1) & np.isfinite(rho) & (np.abs(rho) <= 1)
            outside += int((~valid).sum())
            for m in dimensions:
                p = np.ones(len(bb))
                if valid.any():
                    p[valid] = tri_law_probability(beta[valid], rho[valid], int(m))
                sums[m][i] += p.sum()
    return dict(near_rows=H.tolist(), competitor_count=len(competitors),
                full_pairs=len(H) * len(competitors), conservative_pairs=outside,
                dimensions={str(m): dict(mu=sums[m].tolist(), budgets={
                    str(M): retention_loss_upper_bound(sums[m], len(G), len(H), M, len(C))
                    for M in budgets}) for m in dimensions})


def geometric_counterexamples():
    # m=2, beta=2: Y~Exp(1), p(Y)=1-exp(-Y/2).
    p = Fraction(1, 3)
    return dict(beta=2, m=2, marginal=str(p), opposite_joint=str(p),
                orthogonal_joint=str(1 - 2 * Fraction(2, 3) + Fraction(1, 2)),
                incorrect_unconditional_product=str(p * p))


def free_budget_oracle(costs, missed, k, mean_tolerance, query_tolerance):
    """Multiple-choice knapsack, exactly the fixed-policy quality constraints.

    costs[q,action] are measured mean full costs; no controller cost is charged.
    Labels and measured costs are forbidden in deployment; this is diagnostic.
    """
    costs, missed = np.asarray(costs), np.asarray(missed)
    if costs.ndim != 2 or costs.shape != missed.shape or not len(costs):
        raise ValueError('oracle shape')
    if not np.isfinite(costs).all() or np.any(costs < 0) or np.any(missed < 0) or np.any(missed != np.floor(missed)):
        raise ValueError('oracle nonfinite costs or nonintegral losses')
    maximum = int(np.floor(mean_tolerance * len(costs) * k + 1e-12))
    local = int(np.floor(query_tolerance * k + 1e-12))
    states = {0: (0.0, [])}
    for c, loss in zip(costs, missed):
        nxt = {}
        for used, (value, actions) in states.items():
            for a in range(len(c)):
                budget = used + int(loss[a])
                if loss[a] > local or budget > maximum:
                    continue
                v = value + float(c[a])
                if budget not in nxt or (v, actions + [a]) < nxt[budget]:
                    nxt[budget] = (v, actions + [a])
        states = nxt
    if not states:
        return dict(status='infeasible', deployable=False)
    used, (total, actions) = min(states.items(), key=lambda x: (x[1][0], x[0]))
    return dict(status='feasible', deployable=False, mean_cost=total / len(costs),
                total_missed=used, actions=actions, quality_budget=maximum)
