# Architecture and Contracts

## 1. System boundary

The harness isolates the effect of Gaussian dimensionality reduction and adaptive candidate budgeting before introducing approximate indexes.

```text
corpus text -> embedding -> normalize -> fixed Gaussian projection -> exact projected index
                                |                                  |
                                +-------- original exact index ----+

external query -> embed/normalize/project
               -> pilot projected retrieval
               -> original distances to pilot candidates
               -> local LID estimate
               -> adaptive M(q)
               -> projected candidate expansion
               -> exact original rerank
               -> top-k_ctx evidence/context
               -> optional generator
```

The primary causal comparison changes only the budget policy. Embeddings, projection, indexes, queries, and reranker remain fixed.

## 2. Symbols

| Symbol | Meaning |
|---|---|
| `N` | number of corpus passages |
| `d` | original embedding dimension |
| `m_prime` | one fixed projected dimension |
| `k_gt` | number of original-space nearest neighbors whose retention is measured |
| `k_ctx` | number of passages supplied to the generator |
| `M_pilot` | small projected shortlist used to estimate query difficulty |
| `M(q)` | final projected shortlist for query `q` |
| `M_grid` | allowed discrete candidate budgets |
| `lambda_q` | estimated query-local intrinsic dimensionality |
| `tau_predict` | predicted retention threshold used by the policy |
| `tau_cert` | certified realized-retention target |
| `alpha` | certificate failure probability |

Require `M(q) >= max(M_pilot, k_gt, k_ctx)`.

## 3. Geometry conventions

Let normalized embeddings be `x` and `q`, and let the projection matrix be:

```text
Pi[i, j] ~ Normal(0, 1 / m_prime)
```

Here the second parameter is the variance. In NumPy/SciPy code, construct the matrix with standard deviation `1 / sqrt(m_prime)`:

```python
Pi = rng.normal(0.0, 1.0 / np.sqrt(m_prime), size=(m_prime, d))
```

Projected vectors are `z = Pi @ x`. Do not normalize `z`.

Search uses squared Euclidean distance. On unit-normalized original vectors, original squared-L2 ranking is equivalent to cosine ranking. Projected squared-L2 ranking preserves the setting assumed by the dense-Gaussian triplet law.

## 4. Exact Tri-Law primitive

Implement the paper's exact single-triplet inversion law before Tri-Predict. The authoritative engineering specification is `TRI_LAW_SPEC.md`.

For an actual query/neighbor/non-neighbor triplet, let:

```text
beta = squared_distance_to_non_neighbor / squared_distance_to_neighbor > 1
rho  = cosine between the two unit displacement vectors
```

For `abs(rho) < 1`, define:

```text
D = (1 + beta)^2 - 4 * beta * rho^2
r(beta, rho) = (sqrt(D) + beta - 1) / (sqrt(D) - beta + 1)
P(inversion) = survival_function_F(r(beta, rho); m_prime, m_prime)
```

For `abs(rho) == 1`, the inversion probability is zero. For `rho == 0`, `r(beta, 0) = beta`, giving the worst-case orthogonal probability `P(F >= beta)`.

This exact law uses actual triplet geometry. It is not the same as replacing `beta` with an LID rank model and aggregating many inversions. Keep it in `tri_law.py`; place the approximation in `tri_predict.py`.

## 5. Query-local LID

For sorted positive distances `0 < r_1 <= ... <= r_s`, use the standard maximum-likelihood/Hill form:

```text
lambda_hat = -1 / mean_i(log(r_i / r_s)),  i = 1, ..., s-1
```

Implementation requirements:

- define whether distances are Euclidean or squared Euclidean; convert squared distances to Euclidean before this formula;
- exclude the boundary distance `r_s` from the mean so the denominator is not diluted by `log(1)=0`;
- reject nonpositive distances and handle duplicate passages deterministically;
- require a configurable minimum usable neighbor count;
- clip only for numerical stability, with clip thresholds recorded in the manifest.

Provide two modes:

### `pilot_rerank` (primary, deployable)

1. Retrieve `M_pilot` candidates in projected space.
2. Compute original-space distances from the query to those candidates.
3. Sort these distances and estimate LID from the nearest `s_lid` usable values.

This mode is deployable but biased if the pilot set misses important local neighbors. Quantifying that bias against `oracle_exact` is a required diagnostic.

### `oracle_exact` (diagnostic only)

Estimate LID from exact original-space nearest neighbors. Never use this result in the headline efficiency or deployment comparison.

## 6. Budget policies

All policies implement a common interface conceptually equivalent to:

```python
class BudgetPolicy:
    def fit(self, tune_records, config) -> "BudgetPolicy": ...
    def choose(self, query_features) -> int: ...
    def serialize(self) -> dict: ...
```

`choose` must use only features available at inference time.

### 6.1 Fixed policy

Returns one configured budget. Run it for every value in `M_grid`. These curves define the strongest simple baseline and the fixed budget used for matched comparisons.

### 6.2 Monotone binned empirical policy

1. Compute LID quantile cut points using `query_tune` only.
2. Assign tune queries to bins.
3. For each bin and candidate budget, compute query-level embedding retention.
4. Select the smallest budget satisfying the tune target plus safety margin.
5. Enforce nondecreasing budgets across increasing-LID bins.

The serialized policy contains only cut points, output budgets, feature version, and metadata. It contains no query IDs or labels.

### 6.3 Query-adaptive Tri-Predict policy

Reproduce the paper's analytic predictor with query-local `lambda_q`. The paper itself uses one global robust summary `Lambda_med`; replacing it with `lambda_q` is the proposed Query-Adaptive Tri-RAG extension.

For true-neighbor rank `j <= k_gt` and competing corpus rank `l > k_gt`, model the distance-ratio term as:

```text
beta_jl = (l / j) ** (2 / lambda_q)
```

Under the worst-case orthogonal specialization `rho = 0`, and conditioned on projected neighbor scale `Y_j = y`, model the inversion probability:

```text
p_jl(y) = ChiSquareCDF(df=m_prime, x=m_prime * y / beta_jl)
```

This is Proposition 3 / equation (6), not the full arbitrary-`rho` Tri-Law.

Define expected non-neighbor outrank count:

```text
h_j(y) = sum over l=k_gt+1,...,N-1 of p_jl(y)
```

Find the unique threshold `y_j_star` satisfying:

```text
h_j(y_j_star) = M - j
```

Handle the paper's boundary explicitly:

```text
if M - j >= N - k_gt - 1:
    y_j_star = infinity
    retention_probability_j = 1
```

and predict:

```text
R_hat(lambda_q, m_prime, k_gt, M, N)
  = mean over j=1,...,k_gt of ChiSquareCDF(
      df=m_prime,
      x=m_prime * y_j_star
    )
```

The finite sum may first be implemented exactly for small corpora, then replaced by deterministic quadrature/geometric rank sampling. Unit tests must compare the approximation against the exact sum on small `N`.

The aggregation relies on all of the following paper assumptions/relaxations, which must be named in reports:

1. LID power-law rank-distance model replaces actual pairwise gaps.
2. A single `Lambda_med` summarizes heterogeneous query geometry in the paper; this project instead estimates `lambda_q` from a pilot shortlist.
3. The worst-case orthogonal specialization sets `rho = 0`.
4. The structural surrogate assumes exactly `j - 1` true neighbors remain ahead of the ambient rank-`j` neighbor after projection.
5. Competing non-neighbor inversion indicators are treated as conditionally independent given `Y_j`.
6. Mean-field thresholding replaces the stochastic Poisson-Binomial retention event with `h_j(y) <= M - j`.

Choose:

```text
M(q) = min { M in M_grid : R_hat(lambda_q, ...) >= tau_predict }
```

If no value passes, return `max(M_grid)` and emit `policy_saturated=true` in the per-query record.

Important: substituting an estimated query-local LID into the paper's global mean-field predictor is a proposed extension, not a proven theorem. Phrase results accordingly.

## 7. Retrieval interfaces

Use small adapters so exact NumPy/FAISS implementations can be interchanged:

```python
class VectorIndex:
    def search(self, queries, k) -> SearchResult: ...

class ExpandingRetriever:
    def pilot(self, query, k) -> PilotResult: ...
    def expand(self, query, pilot_result, final_k) -> SearchResult: ...
```

`SearchResult` contains stable document IDs, distances, and backend timing. Tie-breaking must be deterministic by document ID or stable corpus row.

The first backend may redo projected search during expansion. Record:

- `pilot_search_ms`;
- `expansion_search_ms`;
- `pilot_original_distance_count`;
- `additional_original_distance_count`;
- `rerank_ms`;
- `total_retrieval_ms`.

## 8. Data and artifact contracts

Recommended input tables:

### `corpus.parquet`

- `doc_id: string`
- `text: string`
- optional `title: string`

### `queries.parquet`

- `query_id: string`
- `split: enum(query_tune, query_cert, query_test)`
- `query_text: string`
- optional `answer: string | list[string]`

### `qrels.parquet`

- `query_id: string`
- `doc_id: string`
- `relevance: int`

Recommended cached arrays:

- `corpus_embeddings.f32.npy`
- `query_embeddings.f32.npy`
- `projected_corpus.f32.npy`
- `projected_queries.f32.npy`
- explicit row-to-ID JSON or Parquet maps.

Recommended run directory:

```text
runs/<run_id>/
  manifest.json
  policy.json
  per_query.parquet
  certification.json
  aggregates.json
  timings.json
  report.md
  logs/
```

`manifest.json` must include dataset/version, hashes, embedding model/revision, normalization, projection distribution/seed/dimension, all query split hashes, `k_gt`, `k_ctx`, `M_pilot`, `M_grid`, LID configuration, policy configuration, and software environment.

## 9. Per-query record

At minimum:

- query ID and split;
- policy name/version;
- LID mode, raw estimate, clipped estimate, and valid-distance count;
- LID bin if applicable;
- chosen `M(q)` and saturation flag;
- original exact top-`k_gt` IDs or a stable hash/reference;
- projected candidate IDs or a stable reference;
- embedding retention fraction in `[0,1]`;
- candidate evidence recall and post-rerank evidence recall;
- top-`k_ctx` evidence hit;
- pilot/expansion/rerank timings and distance counts;
- optional answer score and generation metadata.

## 10. Failure modes to expose

- too few distinct pilot distances for LID;
- policy saturation at maximum budget;
- empty evidence labels;
- LID bin with fewer than two certification queries;
- query or corpus embedding mismatch;
- post-projection normalization accidentally enabled;
- cache fingerprint mismatch;
- answer generator failure or nondeterminism.

Do not silently drop affected queries. Preserve their IDs and status in the run output.
