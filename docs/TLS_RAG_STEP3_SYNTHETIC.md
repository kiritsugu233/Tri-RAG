# TLS-RAG Step 3 synthetic observed-pair harness

Status: closed on 2026-09-06 after local verification and a user-confirmed
cluster replay with all gates passing. This document covers only the CPU-only,
network-free, synthetic Step 3 harness. Step 4 is authorized only through the
new Step 3-to-Step 4 handoff and remains a separate milestone.

## What runs

The isolated `tri_rag_harness.tls_rag_step3` runner reuses the frozen Step 2
corpus, normalization, dense Gaussian projection, exact projected search,
stable string-ID ties, prefix reuse, newly exposed original-distance cache,
exact original reranking, two-action enum, context builder, evidence definition,
and Phase A/Phase B separation. Step 2 Phase A compatibility first checks the
original exact fingerprint and also verifies a frozen 12-decimal semantic
trajectory fingerprint so harmless cross-platform BLAS roundoff does not mask
unchanged actions, IDs, budgets, contexts, validity flags, or work records.
Phase B is independently rebound to that semantic Phase A identity and hashes
every supervision field, so a platform-specific raw Phase A hash cannot cause a
false Phase B rejection while any label or material numeric change still fails.

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

Two final fresh compatibility-audit runs were written to:

- `/tmp/tls-rag-step3-audit-a.FbkMwf/run`
- `/tmp/tls-rag-step3-audit-b.Jmkdnf/run`

All 19 portable artifacts compared byte-for-byte equal. `timings.json` is
nonportable and excluded from portable fingerprints. This byte-equality result
is a same-host reproducibility check. The manifest deliberately retains each
host's observed raw Step 2 identities as audit evidence, so its bytes and
fingerprint may differ across hosts whose BLAS roundoff changes those raw
identities; cross-host acceptance uses the frozen semantic identities instead.

- manifest: `864031f5a3a160db09b1b637b68529677d56f12b7ebbb04bf0f3e42d2c1c671e`
- config: `3901dd0c742944ebd8bb23893488a8ab4950e28e4c48ac7c4fa143873f8d0101`
- Phase A: `5272b52bc008e52d8d6cc132fcb19b91166258a3dd27b064f219297052e94e52`
- Phase B: `1754cf5e325fa8c0c950ab42ec3db26b560d96c6e52f2eb229d4cd3567a6d0a2`
- Step 2 Phase A semantic trajectory: `119e46b2670c8830a4de7da57a94cc990e6316d990f6c8a8bd8417385be8b5ec`
- Step 2 Phase B semantic supervision: `ef498b9ed0a10f76c6af753471b9935f3586eabd0cc1992dadb4c492472bf97b`
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

Result: 23 passed, 0 failed, 0 skipped in 2.749 seconds. This includes a
full compatibility-path regression that substitutes different raw Phase A and
Phase B hashes while preserving every semantic record, then proves that an
action change and a supervision-label change are still rejected.

Full CPU command:

```bash
./scripts/run_tests.sh
```

Result: 203 tests run, 202 passed, 0 failed, and 1 expected optional real-FAISS
skip in 27.578 seconds. The frozen Step 2 Phase A and Phase B reference
fingerprints remain
`78c4e4869ffca61a7a82ab014b9a3bd9c1513824c6d7d83ad6c2180f4428c2f3`
and `a3d3620538c76bcc8a64b17c8dac619ac4b279be13abd43308758b43efda56e4`.
Tests do not require a platform's observed raw hashes to equal those local
references. Runner and tests call the same compatibility validator: each phase
may use its exact identity or must match its complete frozen semantic identity.
Larger numerical, structural, action, or supervision-label changes are rejected
with observed and expected hashes. A cross-host replay must not compare its raw
manifest fingerprint to the local value above as an acceptance gate.

The user reported that the final cluster replay also passed completely on
2026-09-06. No Slurm job ID, node, environment versions, or cluster timings were
provided, so none are inferred here. That replay closes Step 3; its logs should
be retained with the cluster checkout's exact commit and environment record.

## Scope and residual risks

This is a small synthetic code-path exercise. The thresholds and intervals are
not selected operating points, the fixture is not an exchangeability audit for
real queries, and the timing records are not a systems benchmark. Underpowered
cells intentionally force expansion. A terminal context may remain
insufficient. No real/protected data, real role, archive, approximate index,
GPU, network, dependency download, LLM, answer generation, answer evaluation,
selection, certification, latency claim, evidence guarantee, answer-quality
claim, or Step 4 runner was used.

Step 3 is closed. Step 4 may begin only from the dedicated Step 3-to-Step 4
handoff and its restricted reading list.
