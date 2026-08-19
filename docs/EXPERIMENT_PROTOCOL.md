# Experiment Protocol

## 1. Experimental claims

The harness should test four separate claims. Do not collapse them into one number.

1. **Predictive claim:** query LID carries information about the candidate budget needed to retain original embedding neighbors.
2. **Efficiency claim:** adaptive `M(q)` reduces mean reranking work or latency versus a fixed budget at matched certified retention.
3. **Evidence claim:** preserving embedding neighbors tends to preserve labeled evidence, but the relationship is empirical rather than guaranteed.
4. **Generation claim:** evidence retention is associated with answer quality, but embedding retention alone does not certify answer correctness.

## 2. Dataset choice

Use external query-to-corpus datasets; never manufacture the main evaluation by treating corpus rows as queries.

Recommended staged approach:

- Stage 1: a small retrieval dataset with explicit qrels, such as SciFact, for fast harness debugging and evidence metrics.
- Stage 2: a question-answering dataset with a fixed document corpus, evidence annotations, and reference answers for downstream generation.

Keep dataset loading behind an adapter. Pin the exact dataset revision or archive hash. Do not make the research harness depend on a particular hosted dataset API.

## 3. Query splits

Create three disjoint, frozen sets:

- `query_tune`: LID bin edges, empirical budget mapping, analytic safety correction, and all policy selection.
- `query_cert`: one-time certification of the fully frozen adaptive policy.
- `query_test`: final unbiased performance, cost, evidence, and answer-quality reporting.

If data is limited, prefer fewer LID bins over reusing certification queries for tuning. Stratify only using label-free metadata known before evaluation. Save split IDs and their hash.

## 4. Fixed experimental constants

Freeze before certification:

- corpus and text preprocessing;
- embedding model and revision;
- normalization;
- projection type, seed, and `m_prime`;
- exact-search implementation and tie-breaking;
- `k_gt`, `k_ctx`, `M_pilot`, `s_lid`, and `M_grid`;
- LID clipping/fallback rules;
- policy parameters and safety margin;
- target retention and `alpha`;
- generator prompt, model, decoding settings, and judge if answer evaluation is enabled.

## 5. Suggested MVP configuration

Use this only as a starting point and record any changes before looking at certification results:

```text
k_gt       = 10
k_ctx      = 5
M_pilot    = 32
s_lid      = 20
M_grid     = [32, 64, 128, 256, 512]
LID bins   = 4 quantile bins
tau_predict = 0.95
tau_cert    = 0.95
alpha       = 0.05
projection  = dense Gaussian
search      = exact squared L2
```

Choose `m_prime` in a small preliminary study on `query_tune` only, then freeze it. A reasonable initial sweep is a few powers or common embedding fractions, but the final MVP evaluates only the selected fixed dimension on certification/test queries.

## 6. Baselines

Required:

1. Full-dimensional exact retrieval: quality upper/reference point, not a cost-matched baseline.
2. Fixed `M` for every value in `M_grid`.
3. Monotone binned empirical adaptive policy.
4. Query-adaptive analytic Tri-Predict policy.
5. Oracle-LID variants of policies 3/4, clearly labeled diagnostic-only.

Useful falsification controls:

- shuffled-LID policy preserving the same marginal budget distribution;
- random budget assignment with the same mean budget;
- fixed budget nearest to the adaptive policy's mean cost.

## 7. Metrics

### 7.1 Embedding-neighbor retention

Let `GT_k(q)` be the exact original-space top-`k_gt`, and `C_M(q)` the projected candidates:

```text
embedding_retention(q) = |GT_k(q) intersect C_M(q)| / k_gt
```

With exact original-space reranking, every retained member of `GT_k(q)` outranks every nonmember, so this also equals overlap of the reranked approximate top-`k_gt` with exact `GT_k(q)`.

Report mean, median, lower quantiles, failure rate below target, and the empirical-Bernstein lower bound.

### 7.2 Evidence metrics

At candidate `M(q)` and final context `k_ctx`, report:

- binary evidence hit: at least one relevant document retrieved;
- evidence recall: fraction of known relevant documents retrieved;
- graded nDCG if relevance grades exist;
- evidence loss relative to full-dimensional exact retrieval.

Queries with no qrels must be explicitly excluded from evidence denominators and counted separately.

### 7.3 Answer metrics

Depending on the dataset:

- exact match and token F1;
- citation/evidence correctness;
- a bounded `[0,1]` judge score with prompt/version saved;
- abstention/failure rate.

Use deterministic decoding where possible. Treat LLM-as-judge results as a separate analysis, not ground truth.

### 7.4 Cost and latency

Report:

- mean, median, P95, and P99 `M(q)`;
- fraction of queries at each budget;
- mean original-space distance evaluations;
- estimated reranking FLOPs or bytes read;
- pilot, expansion, reranking, and end-to-end latency separately;
- policy computation overhead;
- index and embedding memory.

Primary efficiency metric:

```text
candidate_saving = 1 - mean(M_adaptive(q)) / M_fixed_certified
```

where `M_fixed_certified` is the smallest fixed budget passing the same certificate on the same certification queries.

## 8. Matched comparisons

Compare policies in two ways:

### Quality-matched

Compare adaptive policy against the smallest fixed budget whose certified lower bound reaches the same `tau_cert`.

### Cost-matched

Compare against a fixed or randomized policy with approximately equal mean candidate count. This tests whether allocating larger budgets to high-LID queries is better than spending the same budget uniformly.

Do not compare adaptive mean cost against an arbitrarily selected fixed budget.

## 9. Relationship analysis

Save all three levels per query and analyze:

```text
LID -> chosen M -> embedding retention -> evidence retention -> answer score
```

Minimum analyses:

- LID versus minimum successful budget on `query_test`;
- embedding retention versus evidence hit/recall;
- evidence hit/recall versus answer score;
- answer score stratified by embedding-retention bands;
- same plots split by LID bin and saturation status.

Use rank correlations and bootstrap confidence intervals as descriptive analyses. Avoid causal language: shared query difficulty confounds these relationships.

## 10. Ablations

Required:

- `pilot_rerank` versus `oracle_exact` LID;
- LID-adaptive versus shuffled-LID cost-matched policy;
- empirical binned versus analytic Tri policy;
- different `M_pilot` values on tune/test, without retuning on certification;
- safety correction on/off.

Optional after MVP:

- several fixed values of `m_prime`;
- multiple projection seeds;
- alternative embedding models;
- approximate projected indexes;
- sparse projection families.

## 11. Headline result table

Produce one table with rows for each policy and columns:

- deployable?;
- mean/P95 `M`;
- embedding retention mean;
- certified lower bound;
- evidence hit@`k_ctx`;
- evidence recall@`k_ctx`;
- answer metric, if run;
- mean/P95 retrieval latency;
- candidate saving versus certified fixed baseline.

Also report projection seed, `m_prime`, dataset split sizes, and confidence level adjacent to the table.
