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

## 10. Frozen real SciFact protocol

The real certification runner uses three policies selected entirely on
`query_tune`: fixed `M=768`, the frozen monotone-binned policy, and the frozen
analytic Tri-Predict policy. The exact Genoa compiled Tri-Predict table is a
separately bound deployment input and must match the analytic decision on every
certification query. These are three predeclared standalone certificates at
`alpha=0.05`; certification results are not used to select among them.

Config fingerprint
`e5545a4aa4c07a1bc188870538c7d346ff26faf38f135c03d4b32f4a18c7ce74`
freezes target `0.95`, all 404 certification IDs, the projection and LID
contract, all policy identities, and terminal failure behavior. The runner
validates the complete frozen policy bundle before selecting `query_cert` and
writes all outcomes even when a lower bound fails. Development and regression
testing used synthetic fixtures only before the single frozen Genoa run.

## 11. Terminal SciFact certification result

Slurm job `373780` on `genoa02` evaluated all 404 frozen certification IDs once
at commit `1625f3b`. The three standalone empirical-Bernstein results are:

| policy | decision | mean M | mean retention | radius | lower bound | candidate saving | coordinate saving |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed `M=768` | PASS | 768.000 | 0.985396 | 0.027085 | 0.958311 | 0.00% | 0.00% |
| monotone binned | PASS | 673.901 | 0.980446 | 0.028142 | 0.952304 | 12.25% | 4.17% |
| Tri-Predict | FAIL | 1119.515 | 0.972525 | 0.030044 | 0.942480 | -45.77% | -15.58% |

Tri-Predict's point estimate is above `0.95`, but certification concerns the
lower bound. Its tune lower bound exceeded the target by only `0.000276`; on
cert the mean fell by `0.007128` and the radius rose by `0.000668`, lowering the
bound by `0.007796`. The mean shift is therefore the dominant immediate cause
of failure, while increased variance also makes the bound slightly wider.

The archive `scifact-policy-cert-373780-audit.tar.gz` has SHA-256
`4fd19b3b205c92d42596700e845da99b732261531d6c222d73375d57fc7ef12b`.
An independent reduction rehashed every input/output artifact, reconstructed
the result identity, verified all 404 cert IDs, reproduced all empirical-
Bernstein terms directly from query records, reproduced every monotone and
compiled Tri-Predict decision, and checked 1,212 candidate/rerank overlap and
retention identities. Result, manifest, and certificate fingerprints are
`81e1e984a735215a9faa99a50991b51dd28c73b1a11e9fa24a0d6e8785088c4d`,
`ddc1208ea17eed9b616a68141e9d03cb85c2c56c3b8ce4564c17681fafc99f61`,
and `f738545f7871568925201182311a4f14b9036f8f2eb80a943b5ba76ea5e5a22f`.

These decisions are terminal for this split. In particular, Tri-Predict may
not be retuned, corrected, or given a larger budget using these 404 outcomes.
The test split remains available only for descriptive evaluation of the
already frozen policies; it cannot create a replacement certificate.

The frozen descriptive test runner enforces this distinction structurally. It
binds this terminal result before test access, evaluates all three policies,
and reports test retention, cost proxies, and evidence metrics without a test
confidence bound or policy selection. Its output must not be called a new
certificate, even if a descriptive test mean is above a certification target.
