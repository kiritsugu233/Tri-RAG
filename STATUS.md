# Status

Updated: 2026-08-19

## What runs

Milestones 0 through 3 are implemented as a CPU-only, network-free synthetic harness. The single run command generates external tune/cert/test queries, normalizes embeddings, builds one fixed dense-Gaussian projection, runs exact original/projected squared-L2 retrieval, fits and freezes a monotone binned pilot-LID policy on tune queries, certifies it on untouched certification queries, evaluates test queries, and writes auditable artifacts.

The exact single-triplet Tri-Law and orthogonal conditional specialization are implemented independently of the adaptive policy. Tri-Predict is not implemented yet.

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

- Passed: 22
- Failed: 0
- Runtime in the current environment: approximately 0.3 seconds
- Coverage includes projection scale and non-renormalization, cache metadata invalidation, exact-search reference agreement, candidate/rerank overlap, Tri-Law algebra and input validation, deterministic quadrature, six Monte Carlo cases including near-collinearity, LID failure modes, policy grid/monotonicity, empirical-Bernstein fixtures, split isolation, report/certificate consistency, and deterministic end-to-end replay excluding declared timestamps/timings.

## Current artifacts

`runs/synthetic_mvp/` contains:

- `manifest.json`
- `policy.json`
- `per_query.jsonl` with 512 query-level records
- `certification.json`
- `aggregates.json`
- `timings.json`
- `report.md`

The default frozen run has 160 corpus items and 512 disjoint external queries. Its overall adaptive certificate passes: mean retention `0.9227`, empirical-Bernstein lower bound `0.8640`, target `0.80`, and `n=256`. The planned sample size for radius `0.15` is 180, so the overall certification sample is sufficient.

This is not a positive efficiency result. The four fitted LID bins choose `[32, 32, 32, 48]`; the smallest fixed budget passing the same certificate is `M=32`, so certification-split candidate saving is `-0.1074`. Bonferroni-corrected per-bin lower bounds also fail the `0.80` target. These outcomes are preserved in the artifacts and report.

## Next task

Milestone 4: implement the analytic query-adaptive Tri-Predict policy. The required Tri-Law stop/go gate is satisfied by the conformance suite. Start with exact finite-rank summation and bounded root finding on small corpora, and do not alter the existing certification split based on the current result.

## Known deviations and risks

- Configuration is JSON rather than YAML, and query-level output is JSONL rather than Parquet, to keep the first pass runnable with only the already available NumPy/SciPy stack. The artifacts remain machine-readable and auditable.
- The checked-in default is synthetic only; no external dataset, text embedding model, evidence evaluation, or answer generation has been added.
- The current synthetic pilot LID differentiates the hardest fitted bin, but the allocation is not efficient relative to the certified fixed baseline. The negative result is a dataset/policy outcome, not hidden by retuning certification data.
- Runtime timestamps and timing measurements are intentionally nondeterministic. Policy, metric, certificate, candidate, and reranked-ID values reproduce under the same manifest and seeds.
- This directory is not currently a Git repository, so no commit or CI workflow was created; the local CPU/offline test command is the acceptance path.
