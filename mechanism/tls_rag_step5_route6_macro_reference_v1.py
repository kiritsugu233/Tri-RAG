"""R6-M0: order-statistic omission integrals, with discretization enclosures.

FP64 special functions are audited, never advertised as directed rounding.
No Gaussian samples, empirical labels, or arbitrary-Gram approximations enter here.
"""
import heapq
import math
import time

import numpy as np
from scipy.stats import chi2, ncx2

AUDIT_TOL = 2.0**-40


class NumericFailure(RuntimeError):
    pass


class BudgetStop(RuntimeError):
    pass


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name)
    return int(value)


def _model(d2, k, m, c):
    d2 = np.asarray(d2, dtype=float)
    k, m = _integer(k, 'k'), _integer(m, 'm')
    if d2.ndim != 1 or len(d2) <= k or not np.all(np.isfinite(d2)) or np.any(d2 <= 0):
        raise ValueError('positive finite d2, with nonempty GT and outsiders required')
    if isinstance(c, bool) or c not in (0, .5):
        raise ValueError('only independent and c=1/2 models')
    return d2, k, m, float(c)


def omission_from_cross_counts(counts, k, M, ell):
    """Exact deterministic event: counts[j] is the number of GTs beaten by j."""
    k, ell = _integer(k, 'k'), _integer(ell, 'ell')
    counts = np.asarray(counts)
    if counts.ndim != 1 or counts.dtype.kind not in 'iu' or np.any(counts < 0) or np.any(counts > k):
        raise ValueError('cross counts')
    M = _integer(M, 'M')
    if ell > k or not k <= M <= k+len(counts):
        raise ValueError('event domain')
    return bool(np.count_nonzero(counts >= ell) >= M-k+ell)


class Audit:
    def __init__(self):
        self.max_pmf_mass_error = 0.
        self.min_cdf_mass = 0.
        self.max_nesting_drift = 0.
        self.max_partition_reset_widening = 0.

    def mass(self, value):
        error = abs(float(np.sum(value))-1.)
        self.max_pmf_mass_error = max(self.max_pmf_mass_error, error)
        if error > AUDIT_TOL or not np.all(np.isfinite(value)) or np.any(value < 0):
            raise NumericFailure('nonconserving/nonpositive PMF')

    def difference(self, value):
        self.min_cdf_mass = min(self.min_cdf_mass, float(np.min(value)))
        if not np.all(np.isfinite(value)) or np.any(value < -AUDIT_TOL):
            raise NumericFailure('CDF nesting or nonfinite mass')

    def record(self):
        return dict(vars(self), tolerance=AUDIT_TOL, roundoff_certified=False)


def poisson_binomial_pmf(probabilities, audit=None):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Bernoulli probabilities')
    pmf = np.array([1.])
    for probability in p:
        pmf = np.convolve(pmf, [1-probability, probability])
    (audit or Audit()).mass(pmf)
    return pmf


def conditional_distance_cdf(s, d2, m, c, r=0.):
    d2, _, m, c = _model(d2, 1, m, c)
    if not np.isfinite(r) or r < 0 or math.isnan(s) or s < 0:
        raise ValueError('s/r')
    if s == 0:
        return np.zeros_like(d2)
    if s == math.inf:
        return np.ones_like(d2)
    x = m*s/(d2*(1-c))
    value = chi2.cdf(x, m) if c == 0 or r == 0 else ncx2.cdf(x, m, c*r/(1-c))
    if not np.all(np.isfinite(value)) or np.any(value < 0) or np.any(value > 1):
        raise NumericFailure('CDF outside [0,1]')
    return value


def target_order_cdf(probabilities, t, audit=None):
    t = _integer(t, 't')
    if t > len(probabilities):
        raise ValueError('order')
    return float(np.sum(poisson_binomial_pmf(probabilities, audit)[t:]))


class _Nodes:
    """One node computes all N CDFs and both PMFs; every event shares it."""
    def __init__(self, d2, k, m, c, events, max_nodes, check):
        self.d2, self.k, self.m, self.c = _model(d2, k, m, c)
        self.events = []
        for M, ell in events:
            M, ell = _integer(M, 'M'), _integer(ell, 'ell')
            if not k <= M <= len(d2) or ell > k:
                raise ValueError('event domain')
            self.events.append((M, ell))
        if not self.events or len(set(self.events)) != len(self.events):
            raise ValueError('nonempty distinct events')
        self.max_nodes = _integer(max_nodes, 'max_nodes')
        self.check = check or (lambda: None)
        self.audit, self.cache = Audit(), {}
        self.impossible = np.array([M-k+ell > len(d2)-k for M, ell in self.events])

    def get(self, r, s):
        if s in (0., math.inf):
            value = float(s == math.inf)
            return np.full(len(self.events), value), np.where(self.impossible, 0., value)
        key = (0. if self.c == 0 else r, s)
        if key not in self.cache:
            self.check()
            if len(self.cache) >= self.max_nodes:
                raise BudgetStop('cdf_node_limit')
            f = conditional_distance_cdf(s, self.d2, self.m, self.c, key[0])
            gt = poisson_binomial_pmf(f[:self.k], self.audit)
            out = poisson_binomial_pmf(f[self.k:], self.audit)
            a = np.array([np.sum(gt[self.k-ell+1:]) for _, ell in self.events])
            b = np.array([np.sum(out[M-self.k+ell:]) for M, ell in self.events])
            self.cache[key] = a, b
        return self.cache[key]


class _Inner:
    def __init__(self, nodes, r_gt, r_out):
        self.nodes, self.r_gt, self.r_out = nodes, r_gt, r_out
        self.heap, self.serial = [], 0
        first = self.cell(0., math.inf)
        self.add(first)
        self.lower, self.upper = first[2].astype(np.longdouble), first[3].astype(np.longdouble)

    def cell(self, left, right):
        fl, _ = self.nodes.get(self.r_gt, left)
        fr, _ = self.nodes.get(self.r_gt, right)
        _, gl = self.nodes.get(self.r_out, left)
        _, gr = self.nodes.get(self.r_out, right)
        w = fr-fl
        self.nodes.audit.difference(w)
        self.nodes.audit.difference(gr-gl)
        return left, right, w*gl, w*gr

    def add(self, cell):
        heapq.heappush(self.heap, (-float(np.max(cell[3]-cell[2])), self.serial, cell))
        self.serial += 1

    def refine(self):
        _, _, cell = self.heap[0]
        left, right = cell[:2]
        if math.isinf(right):
            scale = float(np.mean(self.nodes.d2))*(1-self.nodes.c)*(1+self.nodes.c*max(self.r_gt, self.r_out)/(self.nodes.m*(1-self.nodes.c)))
            middle = max(scale, 2*left)
        else:
            middle = left+(right-left)/2
        if not left < middle < right:
            raise BudgetStop('distance_partition_precision')
        # Complete both new cells before replacing the old valid enclosure.
        a, b = self.cell(left, middle), self.cell(middle, right)
        heapq.heappop(self.heap)
        self.add(a)
        self.add(b)
        self.lower += a[2].astype(np.longdouble)+b[2]-cell[2]
        self.upper += a[3].astype(np.longdouble)+b[3]-cell[3]

    def bounds(self):
        return np.asarray(self.lower, dtype=float), np.asarray(self.upper, dtype=float)

    def partition(self):
        return [[cell[0], 'infinity' if math.isinf(cell[1]) else cell[1]] for _, _, cell in sorted(self.heap, key=lambda row: row[2][0])]


def conditional_omission_bracket(d2, k, m, c, events, r_gt=0., r_out=0.,
                                 width=2.**-14, max_nodes=32768):
    """Independent conditional GT/O groups; unequal r values are intentional."""
    if not all(np.isfinite(x) and x >= 0 for x in (r_gt, r_out)):
        raise ValueError('conditional factors')
    return macro_reference_interval(d2, k, m, c, events, width, max_nodes,
                                    conditional_factors=(r_gt, r_out))


def macro_reference_interval(d2, k, m, c, events, width=2.**-14,
                             max_nodes=32768, check=None, checkpoint=None,
                             conditional_factors=None):
    """Error-driven nested rectangles. Budget exhaustion retains the last bracket.

    Outer refinement uses theta(a,b) <= theta(r,r) <= theta(b,a), never
    monotonicity of theta(r). Inner refinement splits the largest gap cell.
    """
    if not np.isfinite(width) or not 0 < width < 1:
        raise ValueError('width')
    nodes = _Nodes(d2, k, m, c, events, max_nodes, check)
    started, steps, reason = time.monotonic(), 0, None
    inner = _Inner(nodes, *(conditional_factors or (0., 0.))) if c == 0 or conditional_factors is not None else None
    outer = [] if inner else [(0., math.inf, 1., None, None)]
    lower, upper = np.zeros(len(events)), np.where(nodes.impossible, 0., 1.)

    def outer_values(cell):
        _, b, w, low, high = cell
        if math.isinf(b):
            return np.zeros(len(events)), w*np.where(nodes.impossible, 0., 1.)
        return w*low.bounds()[0], w*high.bounds()[1]

    def report(partitions=False):
        rows = [dict(M=M, ell=ell, discretization_bracket=[float(lo), float(hi)],
                     gap=float(hi-lo), status='numerically_resolved' if hi-lo <= width else 'unresolved')
                for (M, ell), lo, hi in zip(events, lower, upper)]
        result = dict(events=rows, nodes=len(nodes.cache), steps=steps, stop_reason=reason,
                      elapsed_seconds=time.monotonic()-started, audit=nodes.audit.record(),
                      roundoff_certified=False, width_goal=width)
        if partitions:
            result['partition'] = (dict(distance=inner.partition()) if inner else [
                dict(r=[a, 'infinity' if math.isinf(b) else b], mass=w,
                     lower_distance=None if low is None else low.partition(),
                     upper_distance=None if high is None else high.partition())
                for a, b, w, low, high in outer])
        return result

    try:
        while float(np.max(upper-lower)) > width:
            nodes.check()
            if inner:
                inner.refine()
                lo, hi = inner.bounds()
            else:
                values = [outer_values(cell) for cell in outer]
                index = max(range(len(outer)), key=lambda i: float(np.max(values[i][1]-values[i][0])))
                a, b, w, low, high = outer[index]
                gap = float(np.max(values[index][1]-values[index][0]))
                if low is not None:
                    inner_error = w*(low.bounds()[1]-low.bounds()[0]+high.bounds()[1]-high.bounds()[0])
                else:
                    inner_error = np.zeros(len(events))
                if low is not None and float(np.max(inner_error)) > gap/2:
                    chosen = max((low, high), key=lambda obj: float(np.max(obj.bounds()[1]-obj.bounds()[0])))
                    chosen.refine()
                else:
                    # Actual chi-square masses, not assumed exact quantile weights.
                    fa, fb = chi2.cdf(a, m), chi2.cdf(b, m)
                    middle = float(chi2.ppf(fa+(fb-fa)/2, m))
                    if not a < middle < b:
                        raise BudgetStop('factor_partition_precision')
                    new = []
                    for x, y in ((a, middle), (middle, b)):
                        weight = float(chi2.sf(x, m) if math.isinf(y) else chi2.cdf(y, m)-chi2.cdf(x, m))
                        nodes.audit.difference(np.array([weight]))
                        new.append((x, y, weight, None if math.isinf(y) else _Inner(nodes, x, y),
                                    None if math.isinf(y) else _Inner(nodes, y, x)))
                    outer[index:index+1] = new
                lo = np.sum([outer_values(cell)[0] for cell in outer], axis=0, dtype=np.longdouble).astype(float)
                hi = np.sum([outer_values(cell)[1] for cell in outer], axis=0, dtype=np.longdouble).astype(float)
            drift = max(0., float(np.max(lower-lo)), float(np.max(hi-upper)))
            if inner:
                nodes.audit.max_nesting_drift = max(nodes.audit.max_nesting_drift, drift)
            else:
                # A new outer cell starts with coarse inner partitions. Its raw
                # bracket may widen; intersect with all earlier valid partitions.
                nodes.audit.max_partition_reset_widening = max(nodes.audit.max_partition_reset_widening, drift)
            if (inner and drift > AUDIT_TOL) or np.any(np.maximum(lower, lo) > np.minimum(upper, hi)+AUDIT_TOL):
                raise NumericFailure('reversed or non-nested enclosure')
            # Intersection retains all previously valid partitions; record FP drift.
            lower, upper = np.maximum(lower, lo), np.minimum(upper, hi)
            steps += 1
            if checkpoint and steps % 256 == 0:
                checkpoint(report())
    except BudgetStop as exc:
        reason = str(exc)
    except NumericFailure as exc:
        reason = 'numeric_failure: '+str(exc)
    return report(partitions=True)
