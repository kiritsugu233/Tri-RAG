"""Scheme6 A: finite-geometry bounds. Existing Tri-Law contracts are unchanged."""
from __future__ import annotations

import hashlib
import json
from math import comb

import numpy as np
from scipy.special import betainc
from scipy.stats import beta as beta_distribution, chi2, ncx2

from tri_rag_harness.tri_law import tri_law_probability


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return int(value)


def probabilities(values):
    p = np.asarray(values, np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('finite one-dimensional probabilities required')
    return p


def scalar_law(beta, rho, m):
    """Beta equivalent, with stable x=(1-gamma)/2 and endpoint branches.

    gamma may round to one. It is never used to compute the small Beta argument.
    Floating-point outputs are numerical evaluations, not outward-rounded intervals.
    """
    m = integer(m, 'm')
    beta, rho = np.broadcast_arrays(np.asarray(beta, float), np.asarray(rho, float))
    if not np.isfinite(beta).all() or not np.isfinite(rho).all() or np.any(beta <= 1) or np.any(np.abs(rho) > 1):
        raise ValueError('beta>1 and |rho|<=1 required')
    transverse = (1-np.abs(rho))*(1+np.abs(rho))
    with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
        t = ((beta-1)/np.sqrt(beta))/2/np.sqrt(transverse)
        norm = np.hypot(1, t)
        root = norm+t
        inverse = 1/root
        x = inverse**2/(1+inverse**2)
        gamma = np.where(transverse == 0, 1.0, t/norm)
        probability = (2/np.pi)*np.arctan(inverse) if m == 1 else betainc(m/2, m/2, x)
        power_upper = np.exp(-m*np.log(norm))
        exponential_upper = np.exp(-m*gamma**2/2)
    probability = np.where(transverse == 0, 0.0, probability)
    power_upper = np.where(transverse == 0, 0.0, np.minimum(.5, power_upper))
    exponential_upper = np.where(transverse == 0, 0.0, np.minimum(.5, exponential_upper))
    return {'gamma': gamma, 'beta_argument': x, 'probability': probability,
            'power_upper': power_upper, 'exponential_upper': exponential_upper}


def marginal_bounds(p, h):
    """Classical sharp marginal-only interval over unrestricted Bernoulli couplings."""
    p = probabilities(p)
    h = integer(h, 'h', 0)
    n = len(p)
    if h == 0:
        return (1.0, 1.0)
    if h > n:
        return (0.0, 0.0)
    def upper(values, threshold):
        ordered = np.sort(values)[::-1]
        tails = np.cumsum(ordered[::-1])[::-1]
        return min(1.0, min(float(tails[s]/(threshold-s)) for s in range(threshold)))
    return max(0.0, 1-upper(1-p, n-h+1)), upper(p, h)


def cross_inversion_bound(expectation, n, k, M, ell):
    n, k, M, ell = (integer(v, name) for v, name in zip((n, k, M, ell), ('N', 'k', 'M', 'ell')))
    if not k <= M <= n or ell > k or not np.isfinite(expectation) or not 0 <= expectation <= k*(n-k):
        raise ValueError('invalid cross-inversion bound inputs')
    if ell > min(k, n-M):
        return 0.0
    return min(1.0, expectation/(ell*(M-k+ell)))


def cantelli_bound(mean, variance_upper, h):
    if not np.isfinite([mean, variance_upper]).all() or mean < 0 or variance_upper < 0:
        raise ValueError('invalid moments')
    h = integer(h, 'h', 0)
    return 1.0 if h <= mean else float(variance_upper/(variance_upper+(h-mean)**2))


def factorial_bound(joint_sum_upper, n, h, order=2):
    n, h, order = (integer(v, name, 0 if name == 'n' else 1) for v, name in zip((n, h, order), ('n', 'h', 'order')))
    if order > h or not np.isfinite(joint_sum_upper) or not 0 <= joint_sum_upper <= comb(n, order):
        raise ValueError('invalid factorial moment bound')
    return 0.0 if h > n else min(1.0, joint_sum_upper/comb(h, order))


def boundary_remainder_bound(local_upper, eta, h, t=0):
    h, t = integer(h, 'h'), integer(t, 't', 0)
    if t >= h or not np.isfinite([local_upper, eta]).all() or not 0 <= local_upper <= 1 or eta < 0:
        raise ValueError('invalid boundary decomposition')
    # local_upper must bound P(Z_local >= h-t), not P(Z_local >= h).
    return min(1.0, local_upper+eta/(t+1))


def second_order_bounds(p, joint_upper, h):
    """Known upper pair probabilities; returns an explicit inconsistent status.

    Production R6-A uses coordinate-stream simultaneous CP upper endpoints here.
    These are extra offline statistical inputs, not quantities implied by Tri alone.
    """
    p = probabilities(p)
    n = len(p)
    h = integer(h, 'h')
    u = np.asarray(joint_upper, float)
    if u.shape != (n, n) or not np.isfinite(u).all() or np.any((u < 0) | (u > 1)) or not np.array_equal(u, u.T):
        raise ValueError('invalid pair matrix')
    if h > n:
        return {'cantelli': 0.0, 'factorial2': 0.0, 'status': 'impossible_count'}
    joint_sum = float(np.triu(u, 1).sum())
    mean = float(p.sum())
    variance = mean+2*joint_sum-mean**2
    lower_joint = np.maximum(0, p[:, None]+p[None, :]-1)
    inconsistent_pair = np.any(np.triu(u-lower_joint, 1) < 0)
    if variance < 0 or inconsistent_pair:
        return {'cantelli': 1.0, 'factorial2': 1.0, 'status': 'inconsistent_joint_constraints', 'variance_upper': variance}
    return {'cantelli': cantelli_bound(mean, variance, h),
            'factorial2': factorial_bound(joint_sum, n, h) if h >= 2 and n >= 2 else 1.0,
            'status': 'upper_moments', 'variance_upper': variance}


def conditional_probabilities(beta, alpha, m, t):
    beta, alpha = np.broadcast_arrays(np.asarray(beta, float), np.asarray(alpha, float))
    if beta.ndim != 1 or not np.isfinite(beta).all() or not np.isfinite(alpha).all() or np.any(beta <= 1):
        raise ValueError('invalid conditional geometry')
    residual = beta-alpha**2
    if np.any(residual < 0):
        raise ValueError('non-PSD conditional geometry')
    t = np.asarray(t, float)
    if t.ndim != 1 or np.any(t < 0) or not np.isfinite(t).all():
        raise ValueError('finite nonnegative t required')
    result = np.zeros((len(t), len(beta)))
    for j, variance in enumerate(residual):
        if variance:
            result[:, j] = (chi2.cdf(t/variance, integer(m, 'm')) if alpha[j] == 0 else ncx2.cdf(t/variance, integer(m, 'm'), alpha[j]**2*t/variance))
    return result


def conditional_marginal_bounds(beta, alpha, m, h, bins=8192):
    """Monotone Riemann enclosure; finite FP64 CDF error is NOT certified."""
    beta, alpha = np.asarray(beta, float), np.asarray(alpha, float)
    bins = integer(bins, 'bins')
    if beta.shape != alpha.shape or beta.ndim != 1 or np.any(np.abs(alpha) > 1):
        raise ValueError('monotone enclosure requires |alpha|<=1')
    h = integer(h, 'h', 0)
    n = len(beta)
    if h == 0 or h > n:
        lo, hi = (1.0, 1.0) if h == 0 else (0.0, 0.0)
        return {'lower': lo, 'upper': hi, 'riemann_gap_upper': 0.0, 'roundoff_certified': False}
    grid = chi2.ppf(np.arange(1, bins)/bins, integer(m, 'm'))
    p = conditional_probabilities(beta, alpha, m, grid)
    def upper_rows(values, threshold):
        ordered = np.sort(values, axis=1)[:, ::-1]
        tails = np.cumsum(ordered[:, ::-1], axis=1)[:, ::-1]
        return np.minimum(1.0, np.min(tails[:, :threshold]/(threshold-np.arange(threshold)), axis=1))
    # p(0)=0. At infinity use the safe endpoint one, also when |alpha|=1.
    lower = float((1-upper_rows(1-p, n-h+1)).sum()/bins)
    upper = float((upper_rows(p, h).sum()+1)/bins)
    return {'lower': lower, 'upper': upper, 'riemann_gap_upper': 1/bins, 'roundoff_certified': False}


def clopper_pearson_envelope(definite, possible, trials, family_size, alpha=.05):
    """Two tails per event, Bonferroni over a PRESEALED complete catalogue."""
    trials, family_size = integer(trials, 'trials'), integer(family_size, 'family_size')
    low, high = np.broadcast_arrays(np.asarray(definite), np.asarray(possible))
    if not np.issubdtype(low.dtype, np.integer) or not np.issubdtype(high.dtype, np.integer) or np.any(low < 0) or np.any(high > trials) or np.any(low > high) or not 0 < alpha < 1:
        raise ValueError('invalid event counts/confidence family')
    tail = alpha/(2*family_size)
    with np.errstate(invalid='ignore'):
        lo = np.where(low == 0, 0.0, beta_distribution.ppf(tail, low, trials-low+1))
        hi = np.where(high == trials, 1.0, beta_distribution.isf(tail, high+1, trials-high))
    return lo, hi


def decision(interval, delta):
    lo, hi = interval
    if not 0 <= lo <= hi <= 1 or not 0 < delta < 1:
        raise ValueError('invalid risk interval/threshold')
    return 'pass' if hi <= delta else ('fail' if lo > delta else 'undecided')


def geometry_probabilities(gram, k, m):
    gram = np.asarray(gram, float)
    d = np.diag(gram)
    if gram.shape != (len(d), len(d)) or not np.isfinite(gram).all() or not np.all(d[k:] > d[:k].max()):
        raise ValueError('strict original GT boundary required')
    beta = d[None, k:]/d[:k, None]
    rho = gram[:k, k:]/np.sqrt(d[:k, None]*d[None, k:])
    # Only a roundoff-sized Cauchy-Schwarz violation may be clipped.
    if np.any(np.abs(rho) > 1+8*np.finfo(float).eps):
        raise ValueError('invalid Gram correlations')
    rho = np.clip(rho, -1, 1)
    return np.asarray(tri_law_probability(beta, rho, m)), beta, rho
