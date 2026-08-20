# Status

Updated: 2026-08-19

## What runs

Milestones 0 through 4 are implemented as a CPU-only, network-free synthetic harness. The single run command generates external tune/cert/test queries, normalizes embeddings, builds one fixed dense-Gaussian projection, runs exact original/projected squared-L2 retrieval, fits and freezes both monotone-binned and query-adaptive Tri-Predict policies on tune queries, evaluates each policy independently, and writes auditable artifacts. A separate two-stage command now performs a predeclared global `m_prime`/Tri-Predict-threshold sweep on tune only, writes frozen selection artifacts, and then evaluates one fresh certification split.

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

Tune-only global dimension selection followed by fresh certification:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m tri_rag_harness.mprime_sweep \
  --config configs/synthetic_mprime_sweep_fresh.json \
  --output runs/synthetic_mprime_sweep_fresh
```

Extended 12-dimension sweep with independent seeds:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m tri_rag_harness.mprime_sweep \
  --config configs/synthetic_mprime_sweep_extended_fresh.json \
  --output runs/synthetic_mprime_sweep_extended_fresh
```

## Tests passed/failed

- Passed: 39
- Failed: 0
- Runtime in the current environment: approximately 3.1 seconds
- Added coverage includes cross-platform policy-float canonicalization, exact `h_j(y)` term-by-term agreement with the orthogonal conditional law, geometric rank-strata population conservation and approximation error, root residuals, the infinite-root/unit-retention boundary, budget monotonicity, LID-to-budget monotonicity, saturation, analytic/empirical interface compatibility, and tune-only scalar safety correction.
- Attribution coverage verifies that actual squared-distance ratios reproduce the rank-model prediction when the assumed power law is exact, that attribution modes produce complete query-level artifacts, and that different synthetic seeds create disjoint stable ID namespaces.
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

Additional diagnostic run directories:

- `runs/attribution_m16/`: pilot/oracle/actual-beta attribution summary and 512 query-level records;
- `runs/synthetic_attribution_fresh/`: separately seeded 928-query repair run with empirical and analytic artifacts.
- `runs/synthetic_mprime_sweep_fresh/`: tune-only five-dimension sweep, frozen selection artifacts, and a 768-query fresh certificate.
- `runs/synthetic_mprime_sweep_extended_fresh/`: independently seeded 12-dimension sweep and a 1024-query fresh certificate.

The default frozen run has 160 corpus items and 512 disjoint external queries. Its overall adaptive certificate passes: mean retention `0.9227`, empirical-Bernstein lower bound `0.8640`, target `0.80`, and `n=256`. The planned sample size for radius `0.15` is 180, so the overall certification sample is sufficient.

This is not a positive efficiency result. The four fitted LID bins choose `[32, 32, 32, 48]`; the smallest fixed budget passing the same certificate is `M=32`, so certification-split candidate saving is `-0.1074`. Bonferroni-corrected per-bin lower bounds also fail the `0.80` target. These outcomes are preserved in the artifacts and report.

The uncorrected Tri-Predict policy also passes the synthetic development certificate, with mean retention `0.9008`, lower bound `0.8391`, mean certification budget `39.2812`, and 9 saturated certification queries. Its candidate saving against fixed `M=32` is `-0.2275`, another negative efficiency result. The optional 90th-percentile additive safety correction is implemented and tested but disabled in the default config because its tune-only fitted value (`0.2287`) saturates every synthetic query at `M=80`.

The completed attribution experiment is documented in `docs/ATTRIBUTION.md`. On the development certification split, pilot-rank MAE is `0.1369`, oracle-rank MAE is `0.2448`, and actual-distance-beta MAE is `0.1132`. Oracle LID therefore does not repair prediction; the LID rank model contributes some error, while most residual error remains in the downstream approximation stack.

A separately frozen fresh synthetic run uses data seed `7301`, projection seed `8111`, `m_prime=4`, analytic threshold `0.89`, 256 tune queries, 512 certification queries, and 160 test queries. It passes with mean retention `0.8535`, lower bound `0.8130`, mean `M=63.6953`, and `20.38%` saving versus the smallest certified fixed budget `M=80`. It has 125 saturated certification queries and remains a synthetic result.

The rigorous tune-only global sweep uses previously unused data/projection seeds `12011`/`13007`, 512 tune queries, candidate dimensions `[4, 8, 12, 16, 24]`, and a predeclared threshold grid. It froze `m_prime=24` and threshold `0.91` before certification. On 768 fresh certification queries, Tri-Predict passes with mean retention `0.847917`, empirical-Bernstein lower bound `0.817234`, mean `M=28.40625`, 7 saturated queries, and `11.2305%` candidate saving versus the smallest certified fixed budget `M=32`. The selection and policy fingerprints are recorded in `docs/MPRIME_SWEEP.md`.

The independent extended sweep covers twelve dimensions from `2` through `32` with seeds `16001`/`17011`. Its predeclared rule froze `m_prime=8`, threshold `0.95`; the 1024-query fresh certificate passes with lower bound `0.817544`, mean `M=54.863281`, and `31.4209%` saving versus certified fixed `M=80`. This result exposes a metric problem rather than establishing that dimension 8 is globally optimal: the dimension-specific fixed baseline jumps from 80 to 48 to 32 to 20, and fixed `M=48` missed the selected run's certificate by only `0.000207`. The full analysis is in `docs/MPRIME_SWEEP.md`.

## Next task

Repeat the 39 tests and extended sweep command on Genoa, preserving the Slurm log and checking the selection/policy fingerprints. Before any further dimension selection, replace the discontinuous relative-saving objective with a predeclared cross-dimension compute objective and densify the fixed-budget grid, using new tune/cert seeds. Real-data work follows after that synthetic design issue is resolved.

## Known deviations and risks

- Configuration is JSON rather than YAML, and query-level output is JSONL rather than Parquet, to keep the first pass runnable with only the already available NumPy/SciPy stack. The artifacts remain machine-readable and auditable.
- The checked-in default is synthetic only; no external dataset, text embedding model, evidence evaluation, or answer generation has been added.
- The current synthetic pilot LID differentiates the hardest fitted bin, but the allocation is not efficient relative to the certified fixed baseline. The negative result is a dataset/policy outcome, not hidden by retuning certification data.
- The current synthetic certification split has been inspected repeatedly during implementation. Its artifacts validate code paths but must not be presented as a fresh research claim or reused to choose new hyperparameters. Real-data policy selection and certification require newly frozen independent splits.
- The new global sweep avoids that old split and enforces tune-only selection in code. Its positive `11.23%` result is candidate-count efficiency, not a latency claim; `m_prime=24` increases projected-search arithmetic relative to `m_prime=16`.
- The extended sweep shows that maximizing relative saving against a dimension-specific certified fixed baseline is unstable across dimensions and seeds. Its selected `m_prime=8` policy passes independently, but the `31.42%` saving is amplified by a coarse-grid certification cliff and is not a global cost optimum.
- Tri-Predict's exact rank summation is intentionally correctness-oriented and currently costs several milliseconds per synthetic query. Large real corpora should use and validate the deterministic rank approximation before performance claims.
- Runtime timestamps and timing measurements are intentionally nondeterministic. Policy, metric, certificate, candidate, and reranked-ID values reproduce under the same manifest and seeds.
- The repository is now connected to GitHub; Slurm runs remain user-executed and their logs should be retained alongside commit IDs and environment versions.
