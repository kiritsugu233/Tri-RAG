# Exact Dense-Gaussian Tri-Law Specification

This document is the implementation contract for the exact single-triplet law in *Predict Before You Project*. It separates the paper's exact theorem from Tri-Predict and from this project's query-adaptive extension.

## 1. Scope

Tri-Law answers one question:

> Given a query, one strictly closer point, one strictly farther point, and a dense Gaussian projection, what is the exact probability that their projected distance order is inverted?

It does not aggregate all competitors, predict full recall, select `M`, or certify a deployed policy.

## 2. Geometry and inputs

Let:

```text
a = x_plus  - q
b = x_minus - q
d_plus  = norm(a)
d_minus = norm(b)
e_plus  = a / d_plus
e_minus = b / d_minus
```

Require distinct positive distances and strict ambient ordering `d_minus > d_plus`.

Define:

```text
beta = d_minus^2 / d_plus^2 > 1
rho  = dot(e_plus, e_minus) in [-1, 1]
```

The projection matrix has shape `(m_prime, d)` with independent entries:

```text
Pi_ij ~ Normal(mean=0, variance=1/m_prime)
```

For NumPy:

```python
Pi = rng.normal(
    loc=0.0,
    scale=1.0 / np.sqrt(m_prime),
    size=(m_prime, d),
)
```

## 3. Exact marginal law

The inversion event is:

```text
I = { norm(Pi @ b) < norm(Pi @ a) }
```

For `abs(rho) < 1`, define:

```text
D = (1 + beta)^2 - 4 * beta * rho^2
s = sqrt(D)
r = (s + beta - 1) / (s - beta + 1)
```

If `W ~ F(m_prime, m_prime)`, then:

```text
P(I) = P(W >= r)
```

Use the survival function rather than `1 - cdf`:

```python
probability = scipy.stats.f.sf(r, dfn=m_prime, dfd=m_prime)
```

### Stable threshold evaluation

The direct denominator suffers cancellation as `abs(rho)` approaches one. For non-collinear inputs, the equivalent expression is:

```text
r = (s + beta - 1)^2 / (4 * beta * (1 - rho^2))
```

Use this stable expression near collinearity. Clip a floating-point `rho` only to `[-1,1]`; do not silently replace a genuinely non-collinear input by the collinear branch. Record and test the chosen numerical tolerance.

### Boundary cases

- `abs(rho) == 1`: return exactly zero; collinearity makes inversion impossible for `beta > 1`.
- `rho == 0`: `r = beta`, so the probability is `F.sf(beta, m_prime, m_prime)`.
- `beta <= 1`: reject the input because it does not describe a strictly farther competitor.
- `m_prime < 1` or nonintegral: reject the input.
- nonfinite inputs: reject the input.

## 4. Orthogonal conditional law

Under `rho = 0`, let:

```text
Y_plus = norm(Pi @ e_plus)^2
```

and condition on `Y_plus = y`. The exact conditional inversion probability is:

```text
P(I | Y_plus = y)
  = ChiSquareCDF(df=m_prime, x=m_prime * y / beta)
```

API:

```python
def tri_law_conditional_orthogonal(
    y: ArrayLike,
    beta: ArrayLike,
    m_prime: int,
) -> np.ndarray:
    ...
```

Require `y >= 0` and `beta > 1`. Use `scipy.stats.chi2.cdf`.

This conditional formula is the branch used by Tri-Predict after it replaces actual `beta` with an LID rank-distance model.

## 5. Required public API

```python
def tri_law_threshold(beta: ArrayLike, rho: ArrayLike) -> np.ndarray:
    """Return r(beta, rho); return +inf at the collinear boundary."""


def tri_law_probability(
    beta: ArrayLike,
    rho: ArrayLike,
    m_prime: int,
) -> np.ndarray:
    """Exact dense-Gaussian probability of one triplet inversion."""


def tri_law_conditional_orthogonal(
    y: ArrayLike,
    beta: ArrayLike,
    m_prime: int,
) -> np.ndarray:
    """Exact rho=0 conditional inversion probability."""
```

Support scalar and broadcastable array inputs. Document whether scalar inputs return Python floats or zero-dimensional arrays and test that behavior.

## 6. Required deterministic tests

### Algebraic identities

For several `beta > 1`:

- `tri_law_threshold(beta, 0) == beta` within floating-point tolerance;
- probability at `abs(rho) == 1` is exactly zero;
- probability is symmetric in `rho`;
- probability is maximized at `rho = 0` for fixed `beta`;
- probability decreases as `beta` increases for fixed `rho` and `m_prime`;
- all returned probabilities lie in `[0,1]`.

### Conditional-to-marginal identity

Under `rho = 0`, numerically integrate:

```text
E_Y[ChiSquareCDF(m_prime * Y / beta)]
```

where `m_prime * Y ~ ChiSquare(m_prime)`, and compare it with:

```text
F.sf(beta, m_prime, m_prime)
```

Use deterministic quadrature or a high-accuracy numerical expectation, not a loose Monte Carlo test, for this identity.

### Input validation

Test `beta <= 1`, invalid `rho`, invalid `m_prime`, negative `y`, NaN, infinity, and incompatible broadcast shapes.

## 7. Required Monte Carlo conformance test

For each chosen `(beta, rho, m_prime)`:

1. Construct two unit directions in at least two ambient dimensions:

   ```text
   e_plus  = [1, 0, ...]
   e_minus = [rho, sqrt(1-rho^2), 0, ...]
   ```

2. Set `a = e_plus` and `b = sqrt(beta) * e_minus`.
3. Draw independent dense Gaussian projection matrices with variance `1/m_prime`.
4. Estimate the fraction satisfying `norm(Pi @ b) < norm(Pi @ a)`.
5. Compare with `tri_law_probability(beta, rho, m_prime)`.

Predeclare the tolerance using binomial sampling uncertainty, for example:

```text
abs(empirical - exact)
  <= 5 * sqrt(exact * (1 - exact) / n_trials) + absolute_floor
```

Use fixed seeds and include cases covering:

- `rho = 0`;
- positive and negative nonzero `rho` with equal absolute value;
- small and large distance gaps;
- at least two projected dimensions;
- a near-collinear but non-collinear case handled separately with an appropriate trial budget.

Keep the default unit-test trial count small enough for CPU CI. A slower high-trial validation may be marked as an optional test.

## 8. Paper-to-code boundary

The following is exact and paper-conformant:

```text
actual beta + actual rho + m_prime -> exact single-triplet probability
```

The following introduces Tri-Predict approximations:

```text
beta_jl = (l/j)^(2/Lambda_med)
rho = 0
conditional independence
mean-field thresholding
aggregation over j
```

This project then introduces additional, unproven extensions:

```text
Lambda_med -> query-local lambda_q estimated from pilot candidates
one fixed M -> query-adaptive M(q)
corpus-as-query population -> external RAG query distribution
```

Reports, module names, and tests must preserve these boundaries.
