# Query-Adaptive Tri-RAG Harness

This directory is the handoff package for implementing a research harness for:

> Fixed projected dimension `m_prime` with a query-adaptive candidate budget `M(q)`, driven by query-local intrinsic dimensionality (LID), calibrated on held-out external queries, and evaluated from embedding-neighbor retention through evidence recall to answer quality.

The immediate goal is a small, reproducible MVP. It is not a production RAG service and it does not attempt to reproduce every experiment from *Predict Before You Project*.

## Read in this order

1. `AGENTS.md` - non-negotiable implementation rules and scope.
2. `docs/ARCHITECTURE.md` - pipeline, modules, interfaces, and artifact contracts.
3. `docs/TRI_LAW_SPEC.md` - exact paper-conformant Tri-Law formulas, API, numerical rules, and tests.
4. `docs/EXPERIMENT_PROTOCOL.md` - datasets, splits, baselines, metrics, and experiment matrix.
5. `docs/CERTIFICATION.md` - valid statistical certification procedure.
6. `docs/IMPLEMENTATION_PLAN.md` - milestones, tests, and acceptance criteria.

## Research question

For a fixed corpus, fixed embedding model, fixed Gaussian projection matrix, and fixed projected dimension `m_prime`, can a deployable estimate of query-local LID choose a smaller candidate budget `M(q)` for easy queries and a larger one for hard queries while:

- meeting a target embedding-neighbor retention rate;
- preserving evidence recall and downstream answer quality;
- reducing mean original-space reranking work relative to a fixed-`M` policy;
- retaining an empirical-Bernstein lower-confidence certificate on an independent query set?

## MVP decision

Use one fixed `m_prime` and one fixed projection/index. Do not maintain multiple indexes or choose `m_prime` per query.

For each external query:

1. Embed and L2-normalize it.
2. Project it with the fixed dense Gaussian matrix.
3. Retrieve a small pilot shortlist `M_pilot` in projected squared-L2 space.
4. Compute original-space distances only for the pilot candidates.
5. Estimate query LID from those original-space pilot distances.
6. Choose `M(q)` from a discrete budget grid.
7. Expand projected retrieval to `M(q)`, cache/reuse pilot work, and exactly rerank in the original embedding space.
8. Return the top `k_ctx` passages to the RAG generator.

The exact backend caches top-`M_max` from one projected scan, so step 7 slices the cached ranking rather than scanning the corpus again. The retrieval-only latency benchmark retains an explicit legacy double-scan control and reports both paths separately.

## Required theoretical primitive

Before implementing analytic Tri-Predict, implement the paper's exact single-triplet Tri-Law as an independent module described in `docs/TRI_LAW_SPEC.md`. Tri-Law itself does not choose `M(q)`; it supplies the exact inversion probability that motivates the orthogonal conditional branch aggregated by Tri-Predict.

The harness must preserve the distinction:

```text
exact Tri-Law for one triplet
  -> orthogonal conditional specialization
  -> LID rank-distance model
  -> structural and mean-field approximations
  -> Tri-Predict expected-retention estimate
  -> query-adaptive M(q) extension
```

## Two policies are required

Implement both so the harness remains useful even before the paper's analytic predictor is fully reproduced.

### Policy A: monotone binned empirical policy

- Fit LID quantile bins on `query_tune` only.
- Within each bin, choose the smallest budget that reaches the target mean embedding retention plus a configurable safety margin.
- Enforce nondecreasing `M` with increasing LID, using a cumulative maximum or isotonic procedure.
- Treat this as the walking-skeleton adaptive baseline, not the main theoretical contribution.

### Policy B: query-adaptive Tri-Predict policy

- Implement and test `docs/TRI_LAW_SPEC.md` first.
- Implement the Tri-Predict equations described in `docs/ARCHITECTURE.md` using query-local `lambda_q` in place of the paper's global median LID.
- Choose the smallest budget in the grid whose predicted recall reaches `tau_predict`.
- Optionally learn one scalar safety correction on `query_tune`; never tune it on `query_cert` or `query_test`.
- Record whether LID came from `pilot_rerank` or `oracle_exact`. Only `pilot_rerank` is deployable.

## Initial success criterion

On at least one external-query retrieval dataset:

- the adaptive policy's empirical-Bernstein lower bound on query-level embedding retention is at least the configured target;
- evidence recall is no worse than the matched fixed-`M` baseline by more than the declared tolerance;
- mean `M(q)` is at least 20% lower than the smallest fixed budget that passes the same certificate;
- all policies are evaluated on the same frozen projection, corpus, embeddings, and query splits;
- the complete run can be reproduced from a single manifest and seed set.

If the 20% efficiency target fails, report the negative result rather than altering the evaluation split or target after seeing certification/test outcomes.

## Expected top-level command

The future implementation should converge on a command similar to:

```bash
python -m tri_rag_harness.run --config configs/mvp_scifact.yaml
```

The exact CLI framework is an implementation choice. One command must be able to run or resume the complete MVP and write a self-contained run directory.

## Deliverables from the implementation agent

- Python package and CLI.
- Unit and integration tests.
- One small default configuration.
- Machine-readable run artifacts.
- A Markdown summary generated from those artifacts.
- Explicit documentation of any deviation from this design.
