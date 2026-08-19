# Statistical Certification

## 1. What is certified

The primary certificate covers the realized end-to-end retrieval policy after all of the following are frozen:

- corpus and embeddings;
- fixed projection matrix;
- fixed projected dimension;
- pilot retrieval and LID estimator;
- binning or analytic budget policy;
- fallback and clipping rules;
- exact reranker;
- query distribution represented by the independently sampled certification queries.

For each certification query `q_i`, define the bounded variable:

```text
r_i = embedding_retention(q_i) in [0, 1]
```

The policy may choose a different `M(q_i)` for every query. This does not invalidate the bound: after the policy is frozen, `r_i` is simply the bounded outcome of that fixed adaptive procedure on an i.i.d. query.

The certificate does not prove the analytic Tri-Predict approximation and does not guarantee answer correctness.

## 2. Overall empirical-Bernstein lower bound

For `n >= 2` independent certification queries, sample mean `r_bar`, unbiased sample variance `v_hat`, and failure probability `alpha`:

```text
radius = sqrt(2 * v_hat * log(2 / alpha) / n)
         + 7 * log(2 / alpha) / (3 * (n - 1))

lower_bound = max(0, r_bar - radius)
```

The certificate passes when:

```text
lower_bound >= tau_cert
```

Save `n`, `r_bar`, `v_hat`, `alpha`, both radius terms, unclipped lower bound, clipped lower bound, and pass/fail.

## 3. Evidence certificate

Evidence metrics are also bounded and may be certified separately:

- binary evidence hit is in `{0,1}`;
- fractional evidence recall is in `[0,1]`.

Do not reuse the embedding certificate as evidence certification. Produce a distinct bound and state its target.

## 4. Per-bin certification

The primary claim should be the overall adaptive-policy bound because it directly reflects deployment behavior and has better sample efficiency.

If making simultaneous claims for `B` LID bins, use a family-wise allocation such as:

```text
alpha_bin = alpha_total / B
```

and compute a separate empirical-Bernstein lower bound within each bin. Bin edges must have been frozen using `query_tune`. Each certified bin needs at least two queries; in practice, require a substantially larger configured minimum or merge bins before certification.

Never delete a difficult bin after observing its certification outcome.

## 5. Policy selection versus certification

Valid sequence:

```text
query_tune -> choose bins, budgets, thresholds, safety correction
freeze policy and run manifest
query_cert -> compute one-time certificate
query_test -> final descriptive performance and relationship analysis
```

Invalid examples:

- increasing `M` after seeing a failed certificate and recomputing on the same certification set;
- choosing between binned and analytic policies based on certification performance;
- changing LID clipping after inspecting failed certification queries;
- reporting the better of multiple projection seeds tested on the same certification set without correction.

If iterative development after certification is necessary, create a new untouched certification split or explicitly relabel the old set as tuning data.

## 6. Sample-size planning

Before running certification, choose a tolerated statistical radius `w` and solve for the smallest `n` such that the worst-case bounded-variable variance schedule satisfies:

```text
empirical_bernstein_radius(v_worst(n), n, alpha) <= w
```

For the unbiased sample variance of values in `[0,1]`, use the same conservative finite-sample variance ceiling as the source paper or verify the chosen ceiling carefully in tests. Save the planning method and resulting `n`.

Do not invent a small `n` merely because the available dataset is small. If the available certification set cannot support the desired width, report the wider bound or weaken the claim.

## 7. Finite benchmark versus query population

Two claims must be worded differently:

- Evaluating every query in a fixed benchmark gives exact performance on that finite benchmark split.
- An empirical-Bernstein statement about future queries requires treating certification queries as i.i.d. draws from the target query distribution.

Document the assumed deployment query distribution and any known shift between benchmark and production queries.

## 8. LLM nondeterminism

Retrieval certification is unaffected by the generator if retrieval is deterministic. For answer-quality certification, decoding randomness adds another source of variation.

For the MVP:

- use deterministic decoding where supported;
- store model/version, prompt, temperature, and seed;
- present answer metrics descriptively unless a separate sampling design is implemented.

## 9. Certification artifact schema

Suggested structure:

```json
{
  "policy_fingerprint": "...",
  "split_hash": "...",
  "metric": "embedding_retention",
  "n": 0,
  "mean": 0.0,
  "unbiased_variance": 0.0,
  "alpha": 0.05,
  "radius_variance_term": 0.0,
  "radius_range_term": 0.0,
  "radius_total": 0.0,
  "lower_bound_unclipped": 0.0,
  "lower_bound": 0.0,
  "target": 0.95,
  "passed": false,
  "created_at": "..."
}
```

The report generator must derive its displayed certificate from this artifact, not recompute it from rounded aggregate values.
