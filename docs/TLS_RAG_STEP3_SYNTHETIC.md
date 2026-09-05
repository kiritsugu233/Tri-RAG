# TLS-RAG Step 3 synthetic observed-pair harness

Status: implemented and locally verified on 2026-09-05. This document covers
only the CPU-only, network-free, synthetic Step 3 harness. It does not authorize
Step 4.

## What runs

The isolated `tri_rag_harness.tls_rag_step3` runner reuses the frozen Step 2
corpus, normalization, dense Gaussian projection, exact projected search,
stable string-ID ties, prefix reuse, newly exposed original-distance cache,
exact original reranking, two-action enum, context builder, evidence definition,
and Phase A/Phase B separation.

Step 3 adds:

- current-prefix-only `context_boundary_risk` and `core_frontier_risk` profiles
  using the unchanged `tri_law_probability(beta, rho, m_prime)` API;
- explicit tied, zero, nonfinite, collinear, invalid, and duplicate-projected-
  distance counts; probability mean/max/q50/q90/q99; core/shell log-distortion
  quantiles; and explicit previous-state delta values/validity;
- transparent standardized ridge score models with clipped `[0,1]` scores and
  deterministic invalid/no-imputation behavior;
- salted-hash-disjoint `synthetic_model_fit`, `synthetic_bound_fit`, and
  `synthetic_evaluation` partitions, 24 external queries each;
- candidate/stage/bin/reachability-specific exact one-sided Clopper--Pearson
  intervals with Bonferroni allocation over 3 candidates, 3 stages, 2 bins,
  and 2 outcomes; cells below 8 independent query IDs use `[0,1]`;
- conservative `STOP` only when the remaining-event upper limit is at most
  `0.65`, the current-sufficiency lower limit is at least `0.15`, and every
  state, model, profile, and cell validity check passes; and
- Row 1 fixed-grid references plus Rows 2--4 with exact one-factor feature
  additions. No winner is selected.

The probabilities in the observed-pair profiles are dependent ex-ante
geometric features. They are not posterior missing-evidence probabilities,
relevance probabilities, sufficiency probabilities, or guarantees.

## Exact runner command

From the repository root:

```bash
PYTHONPATH=src python -m tri_rag_harness.tls_rag_step3 \
  --config configs/tls_rag_step3_synthetic_v1.json \
  --output /tmp/tls-rag-step3-run
```

The output directory must be absent or empty. The runner refuses to overwrite a
nonempty directory.

## Artifacts and reproducibility

Two final fresh runs were written to:

- `/tmp/tls-rag-step3-final-a.53pJPm/run`
- `/tmp/tls-rag-step3-final-b.W6tpKt/run`

All 19 portable artifacts compared byte-for-byte equal. `timings.json` is
nonportable and excluded from portable fingerprints.

- manifest: `f445fb1179cf42674c5a6c46873a558ea3916f4fce53691562d8702f2b3cae52`
- Phase A: `5272b52bc008e52d8d6cc132fcb19b91166258a3dd27b064f219297052e94e52`
- Phase B: `1754cf5e325fa8c0c950ab42ec3db26b560d96c6e52f2eb229d4cd3567a6d0a2`
- Row 2 calibration table: `98253edff39d75ef2509a199d700716f5c6456b004dc473acd0063c74f43fd1f`
- Row 3 calibration table: `d0684bc2d89da8cd8729a05359c867259bd9629af44c7b20357ee10c10365a6e`
- Row 4 calibration table: `f107d49d6c327671a5f66e9b7446315ba8de2d76308c0815e2ecc633436763b2`

The portable set contains manifest/config/fixture/projection/ID-map identities,
risk and feature registries, the partition proof, candidate registry, score
models, calibration tables with reachable query IDs, fit and bound query-stage
records, closed Phase A decisions, separate Phase B supervision, ablation
records, work counters, reconstructable aggregates, and the denial report.

## Verification

Focused command:

```bash
PYTHONPATH=src python -m unittest discover -s tests \
  -p 'test_tls_rag_step3.py' -v
```

Result: 20 passed, 0 failed, 0 skipped in 2.804 seconds.

Full CPU command:

```bash
./scripts/run_tests.sh
```

Result: 200 tests run, 199 passed, 0 failed, and 1 expected optional real-FAISS
skip in 27.644 seconds. The frozen Step 2 Phase A and Phase B fingerprints remain
`78c4e4869ffca61a7a82ab014b9a3bd9c1513824c6d7d83ad6c2180f4428c2f3`
and `a3d3620538c76bcc8a64b17c8dac619ac4b279be13abd43308758b43efda56e4`.

## Scope and residual risks

This is a small synthetic code-path exercise. The thresholds and intervals are
not selected operating points, the fixture is not an exchangeability audit for
real queries, and the timing records are not a systems benchmark. Underpowered
cells intentionally force expansion. A terminal context may remain
insufficient. No real/protected data, real role, archive, approximate index,
GPU, network, dependency download, LLM, answer generation, answer evaluation,
selection, certification, latency claim, evidence guarantee, answer-quality
claim, or Step 4 runner was used.

Step 4 remains blocked pending separate user review and authorization.
