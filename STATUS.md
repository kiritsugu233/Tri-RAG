# Status

Updated: 2026-08-21

## What runs

Milestones 0 through 4 and the retrieval-only systems benchmark are implemented as CPU-only, network-free harnesses. The certification run generates external tune/cert/test queries, normalizes embeddings, builds one fixed dense-Gaussian projection, runs exact original/projected squared-L2 retrieval, fits and freezes both monotone-binned and query-adaptive Tri-Predict policies on tune queries, evaluates each policy independently, and writes auditable artifacts. A separate two-stage command performs a predeclared global `m_prime`/Tri-Predict-threshold sweep on tune only, writes frozen selection artifacts, and then evaluates one fresh certification split.

Pilot and expansion now reuse one exact projected scan in the main harness: the backend retains top `M_max`, exposes the pilot prefix, and slices the cached ranking after `M(q)` is chosen. A separate retrieval-only benchmark provides a memmap-compatible streaming exact backend plus an explicit legacy double-scan control at `100k x 768` and `1M x 1024` scale.

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

Retrieval-only 100k/d768 latency baseline:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.retrieval_benchmark \
  --config configs/retrieval_latency_100k_d768.json \
  --output runs/retrieval_latency-100k
```

## Tests passed/failed

- Passed: 41
- Failed: 0
- Runtime in the current environment: approximately 3.4 seconds
- Added coverage includes cross-platform policy-float canonicalization, exact `h_j(y)` term-by-term agreement with the orthogonal conditional law, geometric rank-strata population conservation and approximation error, root residuals, the infinite-root/unit-retention boundary, budget monotonicity, LID-to-budget monotonicity, saturation, analytic/empirical interface compatibility, and tune-only scalar safety correction.
- Attribution coverage verifies that actual squared-distance ratios reproduce the rank-model prediction when the assumed power law is exact, that attribution modes produce complete query-level artifacts, and that different synthetic seeds create disjoint stable ID namespaces.
- The pre-Milestone-4 baseline also passed all then-current 22 tests on Slurm job `371035`, node `genoa05`, using Python `3.9.23`, NumPy `1.26.4`, and SciPy `1.13.0` from micromamba environment `tri-rag`.
- The extended sweep passed all 39 tests on Slurm job `371643`, node `genoa04`, commit `d5ec795abf0ca604c90ac2b5300708232874ef32`, using Python `3.9.23`, NumPy `1.26.4`, and SciPy `1.13.0`.

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
- `runs/retrieval_latency_smoke/`: local structural validation of all four latency paths.
- `runs/retrieval_latency_100k_d768_local/`: local real-scale structural run; Mac timings are diagnostic only.

The default frozen run has 160 corpus items and 512 disjoint external queries. Its overall adaptive certificate passes: mean retention `0.9227`, empirical-Bernstein lower bound `0.8640`, target `0.80`, and `n=256`. The planned sample size for radius `0.15` is 180, so the overall certification sample is sufficient.

This is not a positive efficiency result. The four fitted LID bins choose `[32, 32, 32, 48]`; the smallest fixed budget passing the same certificate is `M=32`, so certification-split candidate saving is `-0.1074`. Bonferroni-corrected per-bin lower bounds also fail the `0.80` target. These outcomes are preserved in the artifacts and report.

The uncorrected Tri-Predict policy also passes the synthetic development certificate, with mean retention `0.9008`, lower bound `0.8391`, mean certification budget `39.2812`, and 9 saturated certification queries. Its candidate saving against fixed `M=32` is `-0.2275`, another negative efficiency result. The optional 90th-percentile additive safety correction is implemented and tested but disabled in the default config because its tune-only fitted value (`0.2287`) saturates every synthetic query at `M=80`.

The completed attribution experiment is documented in `docs/ATTRIBUTION.md`. On the development certification split, pilot-rank MAE is `0.1369`, oracle-rank MAE is `0.2448`, and actual-distance-beta MAE is `0.1132`. Oracle LID therefore does not repair prediction; the LID rank model contributes some error, while most residual error remains in the downstream approximation stack.

A separately frozen fresh synthetic run uses data seed `7301`, projection seed `8111`, `m_prime=4`, analytic threshold `0.89`, 256 tune queries, 512 certification queries, and 160 test queries. It passes with mean retention `0.8535`, lower bound `0.8130`, mean `M=63.6953`, and `20.38%` saving versus the smallest certified fixed budget `M=80`. It has 125 saturated certification queries and remains a synthetic result.

The rigorous tune-only global sweep uses previously unused data/projection seeds `12011`/`13007`, 512 tune queries, candidate dimensions `[4, 8, 12, 16, 24]`, and a predeclared threshold grid. It froze `m_prime=24` and threshold `0.91` before certification. On 768 fresh certification queries, Tri-Predict passes with mean retention `0.847917`, empirical-Bernstein lower bound `0.817234`, mean `M=28.40625`, 7 saturated queries, and `11.2305%` candidate saving versus the smallest certified fixed budget `M=32`. The selection and policy fingerprints are recorded in `docs/MPRIME_SWEEP.md`.

The independent extended sweep covers twelve dimensions from `2` through `32` with seeds `16001`/`17011`. Its predeclared rule froze `m_prime=8`, threshold `0.95`; the 1024-query fresh certificate passes with lower bound `0.817544`, mean `M=54.863281`, and `31.4209%` saving versus certified fixed `M=80`. This result exposes a metric problem rather than establishing that dimension 8 is globally optimal: the dimension-specific fixed baseline jumps from 80 to 48 to 32 to 20, and fixed `M=48` missed the selected run's certificate by only `0.000207`. The full analysis is in `docs/MPRIME_SWEEP.md`.

Slurm job `371643` exactly reproduced the selection, frozen config, sweep result, Tri-Predict policy, and Tri-Predict certificate byte for byte. The aggregate files differ only in the test-split mean pilot/oracle LID gap at approximately `4e-15`; the manifest differs only in timestamp and software platform fields. On Genoa, Tri-Predict averaged `6.0325 ms/query`, of which `5.9988 ms` was analytic policy computation. The empirical policy path averaged `0.0404 ms/query`. Thus the current analytic implementation has no demonstrated wall-clock benefit on this tiny corpus despite reducing candidate count.

The retrieval-only benchmark is documented in `docs/RETRIEVAL_LATENCY.md`. Slurm job `371643` completed the controlled 100k/d768 run on `genoa04` at commit `fc2ce25d7fb29c1f005d61ddc5c847981ebe7e3b`; all 41 tests passed. Reuse performs one projected scan (`N` distances, 38.4 MB per query) while the control performs two (`2N`, 76.8 MB), and all 64 paired queries have identical budgets and retention. Mean latency fell from `63.9405` to `61.0718 ms/query` (4.49%), but scalar Tri-Predict alone costs `55.7710 ms/query`. All 64 decisions saturate at `M=1024`; mean retention is only `0.096875`, with zero top-10 retention on 23 queries. Peak RSS was `444,706,816` bytes. The systems/reuse gate passes, while the semantic and adaptive-efficiency gate fails as expected for the normalized-Gaussian fixture.

## Next task

Run and archive the 1M/d1024 exact benchmark on Genoa as a systems-scaling experiment. Preserve the one-versus-two scan assertions and report generation/setup time separately from measured query latency. Do not interpret it as a positive adaptive-policy result: the 100k Gaussian fixture saturated every Tri-Predict decision and failed to preserve useful top-10 retention. After the exact 1M run, add the real external-query embedding adapter before making policy-quality claims; FAISS CPU/GPU should be compared against the archived exact baselines. Cross-dimension policy selection remains blocked on a predeclared compute objective and denser fixed-budget grid.

## Known deviations and risks

- Configuration is JSON rather than YAML, and query-level output is JSONL rather than Parquet, to keep the first pass runnable with only the already available NumPy/SciPy stack. The artifacts remain machine-readable and auditable.
- The checked-in default is synthetic only; no external dataset, text embedding model, evidence evaluation, or answer generation has been added.
- The current synthetic pilot LID differentiates the hardest fitted bin, but the allocation is not efficient relative to the certified fixed baseline. The negative result is a dataset/policy outcome, not hidden by retuning certification data.
- The current synthetic certification split has been inspected repeatedly during implementation. Its artifacts validate code paths but must not be presented as a fresh research claim or reused to choose new hyperparameters. Real-data policy selection and certification require newly frozen independent splits.
- The new global sweep avoids that old split and enforces tune-only selection in code. Its positive `11.23%` result is candidate-count efficiency, not a latency claim; `m_prime=24` increases projected-search arithmetic relative to `m_prime=16`.
- The extended sweep shows that maximizing relative saving against a dimension-specific certified fixed baseline is unstable across dimensions and seeds. Its selected `m_prime=8` policy passes independently, but the `31.42%` saving is amplified by a coarse-grid certification cliff and is not a global cost optimum.
- On Genoa, scalar Tri-Predict root solving dominates measured retrieval latency (`5.9988` of `6.0325 ms/query`) on the 160-item synthetic corpus. Candidate-count saving must not be presented as latency saving; vectorization, lookup-table caching, or a validated approximation is required before serving claims.
- The retrieval latency fixture uses normalized Gaussian vectors with realistic shapes and memory traffic, not embeddings from a text model. It is a systems benchmark only; semantic retrieval conclusions require the real external-query adapter.
- The Genoa 100k run passes the systems gate but not the policy gate: all queries saturate at `M=1024`, mean top-10 retention is `0.096875`, and Tri-Predict accounts for 91.32% of reuse-path latency. The 1M run can establish scaling behavior only.
- Tri-Predict's exact rank summation is intentionally correctness-oriented and currently costs several milliseconds per synthetic query. Large real corpora should use and validate the deterministic rank approximation before performance claims.
- Runtime timestamps and timing measurements are intentionally nondeterministic. Policy, metric, certificate, candidate, and reranked-ID values reproduce under the same manifest and seeds.
- The repository is now connected to GitHub; Slurm runs remain user-executed and their logs should be retained alongside commit IDs and environment versions.
