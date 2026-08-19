# Status

Updated: 2026-08-19

## What runs

Milestones 0 through 4 are implemented as a CPU-only, network-free synthetic harness. The single run command generates external tune/cert/test queries, normalizes embeddings, builds one fixed dense-Gaussian projection, runs exact original/projected squared-L2 retrieval, fits and freezes both monotone-binned and query-adaptive Tri-Predict policies on tune queries, evaluates each policy independently, and writes auditable artifacts.

The exact single-triplet Tri-Law and orthogonal conditional specialization remain independent of Tri-Predict. Tri-Predict adds the documented LID rank-distance, orthogonality, structural, conditional-independence, and mean-field approximations. It uses exact finite-rank summation for small competitor populations and deterministic geometric rank strata for larger populations.

Empirical policy float boundaries are canonicalized to 12 decimal places before both decisions and fingerprinting, eliminating the approximately `1e-15` local-versus-Genoa fingerprint drift observed in the first Slurm baseline.

## Exact commands

Full synthetic run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m tri_rag_harness.run \
  --config configs/synthetic_mvp.json \
  --output runs/synthetic_mvp
```

Configuration-only validation and manifest creation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m tri_rag_harness.run \
  --config configs/synthetic_mvp.json \
  --output /tmp/tri-rag-validate-only \
  --validate-only
```

CPU/offline tests:

```bash
scripts/run_tests.sh
```

## Tests passed/failed

- Passed: 34
- Failed: 0
- Runtime in the current environment: approximately 1.8 seconds
- Added coverage includes cross-platform policy-float canonicalization, exact `h_j(y)` term-by-term agreement with the orthogonal conditional law, geometric rank-strata population conservation and approximation error, root residuals, the infinite-root/unit-retention boundary, budget monotonicity, LID-to-budget monotonicity, saturation, analytic/empirical interface compatibility, and tune-only scalar safety correction.
- The pre-Milestone-4 baseline also passed all then-current 22 tests on Slurm job `371035`, node `genoa05`, using Python `3.9.23`, NumPy `1.26.4`, and SciPy `1.13.0` from micromamba environment `tri-rag`.

## Current artifacts

`runs/synthetic_mvp/` contains:

- `manifest.json`
- `policy.json`
- `tri_predict_policy.json`
- `per_query.jsonl` with 512 query-level records
- `tri_predict_per_query.jsonl` with 512 query-level records
- `certification.json`
- `tri_predict_certification.json`
- `aggregates.json`
- `timings.json`
- `tri_predict_timings.json`
- `report.md`

The default frozen run has 160 corpus items and 512 disjoint external queries. Its overall adaptive certificate passes: mean retention `0.9227`, empirical-Bernstein lower bound `0.8640`, target `0.80`, and `n=256`. The planned sample size for radius `0.15` is 180, so the overall certification sample is sufficient.

This is not a positive efficiency result. The four fitted LID bins choose `[32, 32, 32, 48]`; the smallest fixed budget passing the same certificate is `M=32`, so certification-split candidate saving is `-0.1074`. Bonferroni-corrected per-bin lower bounds also fail the `0.80` target. These outcomes are preserved in the artifacts and report.

The uncorrected Tri-Predict policy also passes the synthetic development certificate, with mean retention `0.9008`, lower bound `0.8391`, mean certification budget `39.2812`, and 9 saturated certification queries. Its candidate saving against fixed `M=32` is `-0.2275`, another negative efficiency result. The optional 90th-percentile additive safety correction is implemented and tested but disabled in the default config because its tune-only fitted value (`0.2287`) saturates every synthetic query at `M=80`.

## Next task

First, push the Milestone 4 changes and repeat the 34-test/default-run baseline on Genoa to confirm the canonical policy fingerprint across platforms. Then begin Milestone 5 with one pinned real external-query dataset adapter and a frozen text-embedding model. Generation remains out of scope until real retrieval and evidence evaluation pass.

## Known deviations and risks

- Configuration is JSON rather than YAML, and query-level output is JSONL rather than Parquet, to keep the first pass runnable with only the already available NumPy/SciPy stack. The artifacts remain machine-readable and auditable.
- The checked-in default is synthetic only; no external dataset, text embedding model, evidence evaluation, or answer generation has been added.
- The current synthetic pilot LID differentiates the hardest fitted bin, but the allocation is not efficient relative to the certified fixed baseline. The negative result is a dataset/policy outcome, not hidden by retuning certification data.
- The current synthetic certification split has been inspected repeatedly during implementation. Its artifacts validate code paths but must not be presented as a fresh research claim or reused to choose new hyperparameters. Real-data policy selection and certification require newly frozen independent splits.
- Tri-Predict's exact rank summation is intentionally correctness-oriented and currently costs several milliseconds per synthetic query. Large real corpora should use and validate the deterministic rank approximation before performance claims.
- Runtime timestamps and timing measurements are intentionally nondeterministic. Policy, metric, certificate, candidate, and reranked-ID values reproduce under the same manifest and seeds.
- The repository is now connected to GitHub; Slurm runs remain user-executed and their logs should be retained alongside commit IDs and environment versions.
