# Step 3: Diagnose and Repair Tri-Predict Failure

## Status and authority

This file defines the next independent milestone after Calibrated Tri-Predict
v2. Read it completely before taking any action. It supplements, but does not
replace, `AGENTS.md` and `AGENT_CALIBRATED_TRI_PREDICT.md`; the stricter rule
always wins.

Calibrated Tri-Predict v2 is frozen. Its opened `query_cal` and `query_tune`
results may be used only as diagnostic evidence. They must not be used to
retune, replace, or relabel any v2 candidate. Any algorithmic repair must use
new v3 names, versions, fingerprints, run namespaces, and a separate branch.

## Primary objective

Find the actual reason that Tri-Predict fails to produce a cost-effective
query-adaptive policy, prove the cause with layer-by-layer evidence, and
implement the smallest defensible repair as Calibrated Tri-Predict v3.

Do not begin by adding another large framework. Begin by testing the scientific
assumptions between the exact dense-Gaussian Tri-Law and the final budget
decision. A successful Step 3 must distinguish a mathematical approximation
failure from an input-estimation failure, calibration-target mismatch,
fallback behavior, and a selection-contract problem.

## Read before editing

Read every file completely, in this order:

1. `AGENTS.md`
2. `AGENT_CALIBRATED_TRI_PREDICT.md`
3. `AGENT_CALIBRATED_TRI_PREDICT_STEP3.md`
4. `docs/TRI_LAW_SPEC.md`
5. `docs/CALIBRATED_TRI_PREDICT_PROTOCOL.md`
6. `docs/FIQA_QUERY_CAL_GATE.md`
7. `docs/FIQA_QUERY_TUNE_GATE.md`
8. `docs/IMPLEMENTATION_PLAN.md`
9. `STATUS.md`
10. `src/tri_rag_harness/tri_law.py`
11. `src/tri_rag_harness/tri_predict.py`
12. `src/tri_rag_harness/lid.py`
13. `src/tri_rag_harness/pdctp_features.py`
14. `src/tri_rag_harness/pdctp_calibration.py`
15. `src/tri_rag_harness/pdctp_policies.py`
16. `src/tri_rag_harness/pdctp_fiqa_query_cal.py`
17. `src/tri_rag_harness/pdctp_fiqa_query_tune.py`
18. the corresponding tests for every implementation file above
19. the accepted source, protocol, embedding, query-cal, and query-tune audit
    JSON files under `artifacts/`

Do not read, list, extract, summarize, hash, or otherwise inspect the returned
`pdctp-fiqa-query-cert-376924.tar.gz` beyond the checksum already supplied by
the user. Do not access any `query_cert`, `query_latency`, or `query_test`
record or outcome.

## Frozen evidence that must be reproduced first

Treat the following values as claims to verify, not assumptions to trust:

- query-tune archive SHA-256:
  `3cfbeb6abd65b4e01991bc79065c2c244c8bc356f86deddae88a9ae4b7084969`
- query-tune manifest fingerprint:
  `7cf01ee872a59ee28e6dd0a0c5ffa10ab556b9c0746e3b0a96b8454bdb31836e`
- selection fingerprint:
  `8db86a98eab28deaf6ba173ab78e8336a5bc23b2fc2916653cfbe6b2696cb9ee`
- post-tune protocol-state fingerprint:
  `55ecdf8e3cc53d554d6476569d34cd309e4e0f182a5a86e634741ef1a9dd97b5`
- selected fixed budget: `768`
- selected full-PDCTP mean budget: approximately `1892.7636`
- selected full-PDCTP retention mean: approximately `0.9807321`
- full-PDCTP feature-invalid terminal fallbacks: `31/1967`
- selected full-PDCTP common work exceeds fixed by approximately `7.318%`
- among feature-valid tune queries, full PDCTP has mean budget approximately
  `1000.1488`, slightly better matched-budget retention, but worse
  matched-budget candidate evidence
- raw pilot LID has large negative bias relative to query-cal oracle LID
- the oracle-LID and LID-calibrated Tri-Predict curves are conservative, while
  Raw Tri-Predict driven by raw pilot LID is optimistic

Before interpreting these results, independently validate the archive SHA, all
run-file hashes, every record and candidate fingerprint, selection replay,
protocol state, and every upstream binding. Extraction must go to a fresh
temporary directory and must not overwrite repository artifacts.

## Scope and prohibitions

Allowed in Step 3:

- CPU/network-free synthetic experiments;
- existing local query-cal and query-tune artifacts;
- existing local FiQA source and embedding caches, restricted to the already
  opened `query_cal` and `query_tune` identities;
- deterministic exploratory cross-validation inside `query_cal`;
- implementation of new v3-only modules, tests, configs, and documentation
  after the root cause has been demonstrated;
- exact float64 computation and deterministic parallel execution.

Forbidden in Step 3:

- any access to query-cert, query-latency, or query-test identities or outcomes;
- reading the returned query-cert archive;
- downloading FiQA or any other dataset;
- running an LLM or answer generation;
- fitting or selecting a new v2 policy;
- changing any v2 candidate, threshold, feature, fallback, split, hypothesis,
  artifact, fingerprint, or protocol state;
- modifying Raw Tri-Predict v1 behavior or schemas;
- changing formulas, tolerances, or numerical precision for speed;
- treating tune diagnostics as certification or a positive result.

## Required diagnosis: isolate every approximation layer

Construct a diagnostic matrix in which each row changes exactly one layer.
At minimum include:

1. **Exact Tri-Law layer**
   - Reconfirm the exact dense-Gaussian single-triplet law against Monte Carlo.
   - Preserve variance `1/m_prime`, NumPy scale `1/sqrt(m_prime)`, normalized
     pre-projection vectors, no projected renormalization, and squared L2.
   - This layer must remain separate from all Tri-Predict approximations.

2. **Finite-rank aggregation layer**
   - Compare exact finite-rank summation with the deterministic geometric-rank
     quadrature used by Tri-Predict.
   - Quantify error by neighbor rank, LID, budget, and projection dimension.

3. **Rank-distance power-law layer**
   - On query-cal only, compare the LID-implied rank-distance curve with actual
     sorted original-space distances.
   - Use the existing `actual_distance_retention_grid` diagnostic where its
     strict-gap preconditions hold.
   - Report failures separately when ties make that diagnostic invalid.

4. **Mean-field layer**
   - Compare expected inversion counts and predicted retention with direct
     projection Monte Carlo on fixed query/corpus distance profiles.
   - Determine whether dependence among competitors or top-k neighbors creates
     systematic bias that a scalar correction cannot remove.

5. **LID-input layer**
   - Compare oracle exact LID, raw pilot-rerank LID, and calibrated pilot LID.
   - Report signed log bias, RMSE, MAE, clipping, failure rates, and deterministic
     query-cal cross-validation.
   - Determine whether the pilot candidate construction itself biases the
     neighbor-distance sample used by the LID estimator.

6. **Calibration-target layer**
   - Test whether oracle exact LID is the correct target for a Tri-Predict
     decision model.
   - Compare it with an exploratory query-cal-only effective Tri-LID: the scalar
     LID that best aligns the complete predicted retention curve with the
     realized curve. This is diagnostic only until a v3 protocol is frozen.
   - Determine whether one scalar LID can match both low- and high-budget parts
     of the curve.

7. **Budget-residual layer**
   - Reconstruct residual targets and predictions on query-cal.
   - Report continuous quantile coverage, grid-level operational coverage,
     signed bias, MAE, saturation, and grouped residuals.
   - Explain the consequence of `k_gt=10`: retention levels `0.95`, `0.98`, and
     `1.0` all map to the same per-query required-budget event.

8. **Fallback and selection layers**
   - Separate invalid-feature fallback cost from valid-query policy cost.
   - Count terminal decisions caused by invalid LID, invalid features, residual
     saturation, and genuine target nonattainment.
   - Compare every selected method with fixed retrieval at matched mean budget
     and matched common coordinate work.
   - Show which selection constraints bind and whether the selection contract
     can admit a candidate that is necessarily more expensive than fixed.

For every comparison, keep query-level records or a deterministic derivation
from existing query-level records. Aggregate-only evidence is insufficient.

## Required causal conclusion

The diagnosis must explicitly answer:

1. Is the exact Tri-Law implementation wrong?
2. Is finite-rank quadrature materially inaccurate?
3. Does the LID rank-distance model fail even with oracle LID?
4. How much error is caused by pilot LID bias?
5. Does the LID calibrator overfit, or is oracle LID the wrong target?
6. Does one scalar effective dimension fail to describe the retention curve?
7. Does the residual calibrator repair prediction error or merely compensate
   for it with a high-variance budget correction?
8. How much of the cost failure is attributable to terminal fallback?
9. Would the frozen selection rule accept an inefficient method by design?

Do not implement a repair until the evidence identifies the smallest failing
layer. If more than one layer fails, preserve an ablation that changes one
layer at a time.

## V3 implementation rules

Before the first algorithm edit, create or switch to a separate
`codex/calibrated-tri-predict-v3` branch. Never rewrite or amend the frozen v2
history.

Any repair must:

- use new class/method names, schemas, versions, fingerprints, config names,
  artifact directories, and run namespaces;
- retain Raw Tri-Predict v1 and Calibrated Tri-Predict v2 as immutable baselines;
- expose exactly the same narrow deployable decision input: pilot distances,
  pilot-derived features, and explicit validity fields only;
- exclude evidence labels, exact top-k identities, realized recall, oracle LID,
  and protected outcomes at inference;
- be testable on CPU with a tiny synthetic dataset and no network;
- keep the exact Tri-Law implementation untouched unless Step 3 first proves a
  conformance failure;
- preserve float64 mathematics and deterministic seeds;
- store cache-free scientific artifacts so a performance cache cannot affect
  fingerprints.

Candidate v3 repairs may be investigated, but none is pre-approved as correct:

- calibrating an effective Tri-LID against the full retention curve rather than
  oracle exact LID;
- using more than one deployable curve-shape parameter if one scalar LID is
  conclusively insufficient;
- a cal-only cross-fitted or conformal budget correction with explicit coverage;
- a hierarchical nonterminal fallback frozen before tune;
- a selection gate requiring matched-work quality and actual tune-side cost
  superiority before certification is allowed;
- removal of scientifically duplicate candidate axes.

Choose the smallest repair supported by the causal evidence. Do not combine all
ideas into a new large framework.

## Runtime requirement

Tri-Predict prediction curves depend on LID and the frozen numerical problem,
not on the policy threshold. Compute one immutable float64 prediction grid per
exact query/LID input and reuse it across thresholds.

The reusable cache identity must include at least:

```text
(float64 LID bit pattern, m_prime, k_gt, corpus_size, ordered budget grid,
 max_rank_samples, numerical implementation version)
```

The cache must not be serialized into scientific artifacts. Prove on a complete
tiny candidate suite that cached and uncached execution produces identical
budget vectors, selection objects, artifact values, and fingerprints. Do not
switch precision or modify mathematical tolerances.

## Step 3 acceptance criteria

Step 3 is complete only when:

- all allowed input artifacts and fingerprints have been independently audited;
- the failure has been localized with the layer-by-layer diagnostic matrix;
- a written causal conclusion distinguishes implementation bugs from scientific
  model failures;
- the smallest v3 repair has network-free CPU tests and one-factor ablations;
- cached and uncached prediction decisions are bit/value identical;
- Raw Tri-Predict v1 and all v2 behavior remain unchanged;
- no protected role was accessed;
- the full CPU test suite passes;
- `STATUS.md` and `docs/IMPLEMENTATION_PLAN.md` are updated;
- the new work is committed on the v3 branch with explicit local push, cluster
  pull, and manual `salloc`/`srun` instructions.

If a fresh real-data five-role protocol is required to evaluate v3, stop after
the network-free implementation and query-cal-only diagnosis. Do not download a
dataset or reuse v2 tune/cert/test outcomes for v3 fitting, selection, or claims.

## First deliverable

The first Step 3 deliverable is not code. It is a concise evidence table showing
which of the eight approximation layers above pass or fail, the size and
direction of each error, and the single smallest proposed v3 change. Only then
begin implementation.
