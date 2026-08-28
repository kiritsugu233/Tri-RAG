# Implementation Plan

This checklist is the working plan for the implementation agent. Check items only after tests and artifacts exist.

## Milestone 0: repository skeleton

- [x] Add `pyproject.toml` with a minimal supported Python version and pinned/locked dependencies.
- [x] Create `src/tri_rag_harness/`, `tests/`, `configs/`, `scripts/`, and `runs/` conventions.
- [x] Implement structured configuration loading and validation.
- [x] Implement deterministic seed setup and run-manifest creation.
- [x] Add CPU-only CI/local test command.
- [x] Add `STATUS.md`.

Acceptance:

- one command validates a config and writes a manifest;
- tests run without network access.

## Milestone 1: synthetic walking skeleton

- [x] Generate a tiny clustered corpus and disjoint external queries with known relevant clusters.
- [x] Implement embedding-array ingestion without any text model dependency.
- [x] L2-normalize original vectors.
- [x] Implement exact batched squared-L2 search with deterministic ties.
- [x] Compute exact original top-`k_gt` ground truth.
- [x] Generate and apply dense Gaussian projection using variance `1/m_prime` (`scale=1/sqrt(m_prime)` in NumPy).
- [x] Implement exact projected retrieval and original exact reranking.
- [x] Implement `tri_law_probability(beta, rho, m_prime)` in `tri_law.py`.
- [x] Implement the orthogonal conditional chi-square law.
- [x] Add deterministic Monte Carlo conformance experiments over several `(beta, rho, m_prime)` cases.
- [x] Test collinear, orthogonal, monotonicity, and marginalization identities from `docs/TRI_LAW_SPEC.md`.
- [x] Save query-level records and aggregate metrics.

Acceptance:

- fixed-`M` retention is nondecreasing over the configured budget grid;
- a run is byte-for-byte or value-for-value reproducible under the same seeds;
- projected vectors are demonstrably not renormalized.
- Tri-Law conformance tests pass under predeclared statistical/numerical tolerances.

## Milestone 2: LID and adaptive baseline

- [x] Implement robust Hill/MLE LID estimator.
- [x] Implement `oracle_exact` LID.
- [x] Implement two-stage `pilot_rerank` LID with cached pilot distances.
- [x] Quantify pilot-LID error versus oracle-LID.
- [x] Implement fixed-budget policies.
- [x] Implement monotone binned empirical policy fit on tune queries.
- [x] Implement fallback behavior for invalid LID estimates.

Acceptance:

- policy emits only allowed budgets;
- no labels or exact-neighbor identities are accessed by `choose`;
- tune/cert/test leakage tests pass;
- per-query logs expose pilot overhead and LID failures.

## Milestone 3: certification

- [x] Implement empirical-Bernstein radius and lower bound.
- [x] Add hand-calculated numeric fixtures.
- [x] Implement overall adaptive-policy certificate.
- [x] Implement optional Bonferroni-corrected per-bin certificates.
- [x] Implement sample-size planning or an explicit insufficient-sample warning.
- [x] Serialize `certification.json` and generate a report section from it.

Acceptance:

- a frozen synthetic policy can pass/fail certification deterministically;
- policy or split fingerprint changes invalidate old certification artifacts;
- the report cannot claim a pass when the stored artifact says failure.

## Milestone 4: analytic query-adaptive Tri-Predict

- [x] Refuse to start this milestone unless the exact Tri-Law conformance tests pass.
- [x] Implement exact finite-rank summation for `h_j(y)` on small `N`.
- [x] Implement stable chi-square CDF evaluation.
- [x] Implement bounded root finding for `y_j_star`.
- [x] Implement `y_j_star = infinity` and unit retention when `M - j >= N - k_gt - 1`.
- [x] Implement predicted retention across `M_grid`.
- [x] Implement deterministic rank integration/geometric sampling for larger `N`.
- [x] Test approximation against exact summation on small problems.
- [x] Implement `M(q)` selection and saturation logging.
- [x] Add optional scalar safety correction fitted only on tune queries.
- [x] Add a tune-only global `m_prime`/threshold sweep with a serialized selection rule.
- [x] Freeze the selected dimension before an independent fresh certification run.
- [x] Reject cert/test records at the dimension-selection boundary.

Acceptance:

- predicted recall is nondecreasing in `M` within numerical tolerance;
- increasing LID does not accidentally yield a lower configured budget after monotonic enforcement;
- analytic and empirical adaptive policies can be compared through the same interface.

## Milestone 4.5: retrieval-only systems benchmark

- [x] Add a memmap-compatible streaming exact squared-L2 backend.
- [x] Eliminate repeated pilot/expansion projected scans in the main harness.
- [x] Retain a legacy double-scan control in the latency benchmark.
- [x] Add realistic `100k x 768` and `1M x 1024` configurations.
- [x] Record query projection, search, LID, Tri-Predict, expansion, rerank, and total latency.
- [x] Record p50/p95/p99, distance counts, bytes scanned, cache size, and process RSS.
- [x] Verify that reuse and double scan make identical decisions under exact search.
- [x] Reproduce the 100k baseline on Genoa with one BLAS thread.
- [x] Run the 1M scale-up after the 100k gate passes.
- [x] Compile a frozen analytic policy into adjacent-float64 LID decision intervals.
- [x] Add fingerprinted artifact loading and dense/reference boundary equivalence tests.
- [x] Reproduce compiled-policy lookup latency and exact decision equivalence on Genoa.
- [x] Add an optional exact FAISS `IndexFlatL2` CPU/GPU adapter after archiving the exact baseline.
- [x] Add offline adapter, boundary-tie refusal, and benchmark integration tests.
- [x] Share one FAISS GPU resource pool across original/projected indexes.
- [x] Resolve bounded FAISS boundary ties with one-scan overfetch and recorded deterministic refinement.
- [x] Pass conformance against a real FAISS CPU build on the cluster.
- [x] Pass the 100k FAISS GPU smoke/latency gate before running the 1M comparison.
- [x] Reject FAISS GPU `k + boundary guard > 2048` before fixture generation.
- [x] Complete the 1M FAISS CPU/GPU comparison with a separately frozen `M_max=1984` grid.

Acceptance:

- no query-by-corpus distance matrix is materialized;
- reuse performs one projected scan and the legacy control performs two;
- both Tri-Predict paths choose identical budgets and retention;
- a non-NumPy backend must match NumPy candidate sets at `k_gt`, `M_pilot`, and
  every budget cutoff, row-aligned squared distances, compiled-policy
  decisions, reranked top-k rows, and retention before artifacts are accepted;
- FAISS boundary refinement must report its requested neighbors, host distance
  evaluations, and latency without counting an unreported second scan;
- run artifacts contain stage latency, work, memory, environment, and query-level records.

## Milestone 5: one real external-query retrieval dataset

- [x] Implement a dataset adapter producing corpus, external queries, splits, qrels, and optional answers.
- [x] Pin dataset revision/hash and store license/source metadata.
- [x] Implement pluggable text embedding with caching and explicit model revision.
- [x] Run a fingerprint-bound exact original-space baseline on `query_tune` only.
- [x] Run exact original/projected retrieval in memory-safe batches.
- [x] Freeze one `m_prime` using tune queries only.
- [x] Run required policies on tune, then freeze them.
- [x] Freeze cross-platform LID precision and separate compiled deployment identity.
- [x] Reproduce and independently audit the frozen scientific policy result on Genoa.
- [x] Implement and freeze a certification-only runner using synthetic fixtures only.
- [x] Certify on untouched certification queries.
- [x] Implement and freeze a descriptive test-only runner using synthetic fixtures only.
- [x] Evaluate the three frozen policies once on test queries without selection or recertification.

Acceptance:

- all required manifest fields and per-query records exist;
- at least one fixed policy and both adaptive policies complete end to end;
- failures and policy saturation are visible in the report.

## Milestone 6: evidence evaluation

- [x] Compute candidate and final-context evidence hit/recall.
- [ ] Compute separate evidence empirical-Bernstein bounds when requested.
- [x] Add matched fixed-cost and fixed-quality comparisons.
- [x] Add LID/retention/evidence relationship tables or plots.
- [x] Add shuffled-LID control.

The frozen test run reports auditable final-context evidence metrics at cutoffs
1/5/10. The posthoc tune-only run additionally records candidate-set evidence
separately from the exact original reranked context for all 403 tune queries and
all 16 frozen budgets. Slurm job `374032` reproduced the diagnostic twice, and
an independent reduction replayed the full fixed grid, relationship strata,
matched comparisons, and all 1,000 shuffled-LID controls. Separate evidence
confidence bounds remain optional and were not requested; no new certificate
was created from these posthoc outcomes.

Acceptance:

- qrel-free queries are handled explicitly;
- evidence results can be regenerated from saved query-level artifacts;
- the report distinguishes embedding retention from evidence recall.

## Milestone 7: optional answer generation

- [ ] Add a generator adapter only after retrieval milestones pass.
- [ ] Freeze prompt, model/version, decoding parameters, and context formatting.
- [ ] Cache prompts, contexts, raw outputs, and scores.
- [ ] Compute deterministic answer metrics where available.
- [ ] Analyze answer quality by evidence and embedding-retention strata.

Acceptance:

- retrieval can still run independently of the generator;
- generator failures do not remove query records;
- no text claims that embedding certification guarantees answer correctness.

## Final report requirements

- [ ] State corpus/query sizes and prove query IDs are external/disjoint.
- [ ] State `d`, `m_prime`, projection seed, `M_pilot`, `M_grid`, `k_gt`, and `k_ctx`.
- [ ] Report policy budget distributions.
- [ ] Identify the smallest certified fixed-`M` baseline.
- [ ] Report adaptive saving against that baseline.
- [ ] Report overall and optional per-bin certificates.
- [ ] Report pilot-versus-oracle LID gap.
- [ ] Report embedding, evidence, and answer metrics separately.
- [ ] Document all negative results, deviations, and remaining threats to validity.

## Suggested future code tree

```text
query-adaptive-tri-rag-harness/
  AGENTS.md
  START_HERE.md
  pyproject.toml
  configs/
    mvp_scifact.yaml
  src/tri_rag_harness/
    config.py
    manifest.py
    data/
    embeddings/
    projection.py
    indexes/
    lid.py
    tri_law.py
    tri_predict.py
    policies/
    retrieval.py
    metrics.py
    certification.py
    reporting.py
    run.py
  tests/
  docs/
  runs/
```

## Stop/go gates

1. Do not implement Tri-Predict until exact Tri-Law conformance tests pass.
2. Do not add a real embedding model until the synthetic walking skeleton passes.
3. Do not implement answer generation until retrieval certification works.
4. Do not optimize approximate indexes until the exact-search results establish a benefit.
5. Do not claim a positive result unless the adaptive policy beats the properly matched certified fixed baseline.
