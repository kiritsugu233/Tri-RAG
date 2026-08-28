# Agent Instructions

These instructions apply to the entire `query-adaptive-tri-rag-harness` directory.

## Mission

Build a rigorous experimental harness for fixed-`m_prime`, query-adaptive `M(q)` retrieval. Optimize for a trustworthy, runnable MVP rather than breadth.

## Scope boundaries

In scope:

- external queries that are disjoint from corpus items;
- normalized dense text embeddings;
- one fixed dense Gaussian projection and one fixed projected dimension;
- a standalone implementation and Monte Carlo validation of the paper's exact dense-Gaussian Tri-Law;
- exact projected-space search for the first correctness-oriented backend;
- exact original-space reranking of retrieved candidates;
- pilot-based query LID estimation;
- binned and analytic adaptive-budget policies;
- independent statistical certification;
- embedding retention, evidence recall, cost, latency, and optional answer evaluation.

Out of scope for the MVP:

- choosing a projection dimension per query;
- training a new embedding model;
- HNSW/PQ/IVF approximation effects;
- sparse projection families;
- distributed serving;
- claiming a formal Tri-Law theorem for the full adaptive RAG system;
- claiming that embedding retention guarantees answer correctness.

## Non-negotiable correctness rules

1. Normalize corpus and query embeddings before projection when using cosine-equivalent retrieval.
2. Use a dense Gaussian projection with entries `N(0, 1/m_prime)`, where `1/m_prime` is the variance. NumPy's `normal(..., scale=...)` takes a standard deviation, so code must use `scale=1/sqrt(m_prime)`, never `1/m_prime`.
3. Do not renormalize vectors after projection. The paper's dense-Gaussian distance law is for projected Euclidean norms, not cosine similarity after projected renormalization.
4. Search with squared L2 distance in both the normalized original space and projected space.
5. Freeze the embedding model, corpus, projection seed, `m_prime`, budget grid, and data splits before certification.
6. Never use evidence labels, answer labels, exact top-k identities, or realized recall to choose `M(q)` at inference time.
7. `oracle_exact` LID is diagnostic only. Main deployment claims must use `pilot_rerank` LID.
8. Do not select a policy and certify it on the same queries. Policy selection uses `query_tune`; certification uses `query_cert`; final reporting uses `query_test`.
9. If any policy hyperparameter is changed after inspecting `query_cert`, certification is invalid and must be rerun on a fresh independent split.
10. Every stochastic component must have an explicit seed in the run manifest.
11. Keep query-level records. Aggregate-only CSV files are insufficient for auditing or recomputing bounds.
12. Report the pilot pass, expansion pass, and original-space reranking costs separately.
13. Keep `tri_law_probability(beta, rho, m_prime)` separate from Tri-Predict. The former is the exact single-triplet law; the latter aggregates the orthogonal conditional specialization through additional LID, structural, independence, and mean-field approximations.

## Calibrated Tri-Predict v2 addendum

These additional rules apply only on the successor branch and do not alter the
tagged Raw Tri-Predict v1 baseline:

1. Add a separate `query_cal` role for fitting calibration parameters. Policy
   candidate selection still uses `query_tune`; scientific certification uses
   `query_cert`; label-free systems measurement uses `query_latency`; final
   reporting uses `query_test`.
2. `oracle_exact` LID may supervise the pilot-LID calibrator on `query_cal`
   only. It remains forbidden at inference and in tune/cert/latency/test policy
   decisions. Main deployment claims must use the frozen pilot-distance
   calibrator and deployable pilot inputs.
3. Realized retention and exact top-k identities may label `query_cal` budget-
   residual fitting records, but may never enter a policy decision at inference.
4. Preserve Raw Tri-Predict behavior and schemas. Calibrated methods require
   new names, versions, fingerprints, and run namespaces.
5. Do not use the observed SciFact cert/test records to fit or select v2. Use a
   new dataset and fresh cal/tune/cert/latency/test identities.
6. A positive v2 claim requires independent retention/evidence certification
   and measured paired latency superiority. Candidate-count reduction alone is
   not a latency result.

## Engineering rules

- Start with NumPy/SciPy and an exact vector-search backend. Add FAISS only behind a small adapter.
- Prefer memory-mapped arrays and batched matrix operations; do not materialize a full query-by-corpus distance matrix for large datasets.
- IDs must be stable strings at data boundaries. Array row numbers belong only in explicit ID-map artifacts.
- Cache artifacts using content-derived fingerprints that include the embedding model, normalization flag, corpus hash, projection seed, and dimension.
- Refuse to reuse a cache when its metadata does not match the current run.
- All metrics and confidence bounds must be reproducible from saved per-query records.
- A failed target is a valid output. Do not silently enlarge budgets after certification.
- Tests must run on CPU with a tiny synthetic dataset and no network access.

## Required implementation order

1. Build a tiny synthetic end-to-end walking skeleton.
2. Add exact original and projected retrieval.
3. Implement and validate the exact Tri-Law and its orthogonal conditional specialization according to `docs/TRI_LAW_SPEC.md`.
4. Add pilot-based and oracle LID estimators.
5. Add fixed-budget and monotone binned policies.
6. Add query-level logging and empirical-Bernstein certification.
7. Add analytic Tri-Predict policy only after Tri-Law conformance tests pass.
8. Add one real external-query dataset adapter.
9. Add evidence metrics.
10. Add answer generation only after the retrieval harness passes acceptance tests.

Do not begin with LLM answer generation. It is the most expensive and least diagnostic component.

## Required tests

- projection entries have the expected scale within statistical tolerance;
- projected vectors are not renormalized;
- exact Tri-Law matches Monte Carlo inversion rates within a predeclared binomial-error tolerance;
- exact Tri-Law returns zero for the collinear boundary and its orthogonal specialization reduces to the `F(m_prime, m_prime)` tail at threshold `beta`;
- marginalizing the orthogonal conditional chi-square law numerically agrees with the orthogonal marginal Tri-Law;
- exact original top-k matches a brute-force reference on a toy dataset;
- candidate retention equals the overlap after exact original reranking;
- LID estimator rejects zero/duplicate/insufficient distances cleanly;
- budget policy only emits values from the configured grid and never below `max(k_gt, M_pilot)`;
- monotone policy never reduces budget for a higher-LID bin;
- fixed-budget recall is nondecreasing with budget on exact search;
- empirical-Bernstein radius matches a hand-computed fixture;
- tune/cert/test IDs are disjoint;
- changing projection metadata invalidates cached projected embeddings;
- the same manifest and seeds reproduce identical per-query results.

## Handoff behavior

At the end of each milestone, update `docs/IMPLEMENTATION_PLAN.md` with checked boxes and add a short `STATUS.md` containing:

- what runs;
- exact command used;
- tests passed/failed;
- current artifacts;
- next task;
- known deviations or risks.
