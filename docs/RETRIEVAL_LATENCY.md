# Retrieval-only latency benchmark

Updated: 2026-08-25

## Purpose

`tri_rag_harness.retrieval_benchmark` measures retrieval latency at realistic
embedding dimensions and corpus sizes without invoking an LLM. It is separate
from statistical certification: latency observations must not select or repair
a certified policy.

The benchmark uses deterministic, external, normalized Gaussian query and
corpus vectors stored as `float32` memmaps. Their dimensions and memory traffic
are realistic, but they are not embeddings from a text model and support no
semantic-quality claim.

## Compared paths

Every measured query runs four paths:

1. `original_space_fixed_m`: one streaming exact scan in original space;
2. `projected_space_fixed_m`: query projection, one projected scan to fixed `M`,
   then exact original-space candidate reranking;
3. `projected_tri_predict_reuse`: one projected scan retaining top `M_max`, a
   pilot prefix, original pilot distances, LID, Tri-Predict, cached expansion,
   and exact reranking;
4. `projected_tri_predict_double_scan`: the legacy control with an independent
   pilot scan and expansion scan.

The reuse path computes exactly `N` projected distances per query. The legacy
control computes `2N`. Both must choose the same `M(q)`, candidates, and final
retention under exact search. Original pilot distances are cached and reused by
reranking in both paths.

The main synthetic certification harness now uses the same one-scan invariant:
it scans to `M_max` once, obtains the pilot from the prefix, and slices the
cached ranking after the policy decision.

## Compiled Tri-Predict serving policy

The retrieval benchmark no longer evaluates scalar Tri-Predict roots in the
measured query path. Before queries start, it compiles the frozen analytic
policy over the configured LID clipping interval into monotone float64 decision
intervals. Compilation locates each transition down to adjacent representable
positive float64 values. Runtime inference performs one interval lookup.

The compiled artifact records the analytic reference-policy fingerprint,
numeric and hexadecimal transition boundaries, budget/saturation states, LID
domain, validation counts, and its own fingerprint. Loading refuses a modified
artifact, a mismatched analytic policy, inconsistent hexadecimal boundaries,
or an incompatible input domain. Invalid or out-of-domain LID inputs use the
maximum-budget fallback.

Compilation validates linear, geometric, exact-boundary, and immediately
adjacent float64 inputs. Every benchmark also evaluates the analytic reference
on every observed reuse-path LID after latency measurement and refuses to emit
results if budget, fallback, or saturation differs. Predicted-retention values
remain reference-only diagnostics and are intentionally not synthesized by the
decision-only serving artifact.

The local 10k smoke run produced five decision states and zero equivalence
mismatches. The analytic reference averaged 5.8602 ms/decision; frozen lookup
averaged 0.0021 ms/decision, a measured 2,845x speedup. These Mac timings are a
structural result only.

The local 100k/d768 run produced seven decision states, loaded its frozen
artifact in 0.1093 ms, and reproduced all 64 previous LID values, budgets, and
retention values exactly. Reference validation averaged 38.4606 ms/decision;
lookup averaged 0.0021 ms/decision. The measured reuse path fell from 44.7511
to 4.2712 ms/query across separate local runs. Compilation took 17.9965 seconds
and remains explicitly outside per-query latency. This is also diagnostic until
the controlled Genoa result below.

### Genoa compiled-policy result

Slurm job `373123` ran both configurations on `genoa00` at commit
`0407c831c263e7505543b3701b84e7cd4a4b4bd0`. The environment used Python
`3.9.23`, NumPy `1.26.4`, SciPy `1.13.0`, and one BLAS thread; all 43 tests
passed. The combined audit archive has SHA-256
`12f2480532a221260ff2e2fd9089570b3ea3459c3578e1e67d01b5c90384bb22`.

| Corpus | old analytic reuse | compiled reuse | mean reduction | old p95 | compiled p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100k/d768 | 61.0718 ms | 5.2163 ms | 91.46% | 62.3908 ms | 5.4572 ms |
| 1M/d1024 | 146.7481 ms | 87.5176 ms | 40.36% | 151.3177 ms | 90.2264 ms |

At 100k, analytic validation averaged 55.8130 ms/decision and compiled lookup
averaged 0.002544 ms, a 21,939x speedup. Compilation took 26.106 seconds and
artifact loading took 0.149 ms. The reuse scan remained stable at 4.593 ms and
now accounts for 88.05% of end-to-end reuse latency. Removing the shared scalar
policy overhead increases the observable one-scan benefit over the double-scan
control from 4.49% to 32.89% mean latency.

At 1M, analytic validation averaged 58.3822 ms/decision and lookup averaged
0.002587 ms, a 22,564x speedup. Compilation took 27.898 seconds and artifact
loading took 0.142 ms. The reuse scan remained stable at 85.827 ms and now
accounts for 98.07% of end-to-end reuse latency. The one-scan benefit over the
double-scan control increases from 24.71% to 35.22% mean latency.

The old and new Genoa runs have identical corpus, projected-corpus, and query
hashes. Across all four methods and every query, LID values, budgets,
saturation, retention, scan counts, distance counts, and byte counts are
identical. Both analytic reference artifacts are unchanged. The 100k compiled
artifact, including every hexadecimal transition boundary, is also byte-value
identical between Mac and Genoa.

Compilation is an offline policy-build cost. A serving process loads the frozen
artifact in approximately 0.15 ms and does not repay the 26--28 second build
cost. Even if a process compiled at startup, the measured per-query saving would
amortize that cost after roughly 470 queries. The benchmark process wall time
must not be confused with serving latency because it includes policy
compilation, corpus generation, projection generation, and reference audits.

This completes the exact CPU systems gate for a FAISS adapter. It does not
repair the Gaussian quality result: every query still saturates at the maximum
budget, and retention remains exactly 9.6875% at 100k and 6.875% at 1M.

## Memory-bounded exact backend

`StreamingExactSquaredL2Index` scans an ndarray or memmap in fixed row blocks
and keeps only global top-k candidates. It never materializes a
query-by-corpus distance matrix. Each result records:

- scan count;
- vector distance evaluations;
- distance-coordinate work;
- vector bytes scanned;
- measured search latency.

Top-k ties use stable corpus row number, corresponding to the benchmark's
implicit stable `doc-{row}` IDs.

## Optional exact FAISS backend

The same benchmark now accepts `--backend faiss-cpu` and
`--backend faiss-gpu`. Both use float32 `IndexFlatL2`; IVF, PQ, HNSW, and other
approximate indexes remain out of scope. The adapter remains behind the small
single-query index interface used by the NumPy reference, so projection, pilot
LID, compiled policy lookup, candidate reranking, and query-level accounting
are shared across backends.

Before query measurement, a FAISS run performs a mandatory NumPy conformance
probe in both original and projected space. It checks candidate-set identity
at every prefix consumed by the harness (`k_gt`, `M_pilot`, and every budget
grid value), compares squared-L2 distances after aligning by corpus row, then
checks the compiled-policy decision, reranked top-k rows, and embedding
retention. A mismatch aborts before final artifacts are written. Because FAISS
does not promise the benchmark's row-stable result at an exact top-k distance
tie, the adapter overfetches a fixed guard in the same FAISS scan, recomputes
that small returned pool with the NumPy squared-L2 formula, and selects by
`(distance,row)`. It still aborts if the guard does not close the raw tie band;
it never hides the event with an unaccounted second scan.
The guard defaults to 64, is configurable with
`--faiss-boundary-tie-overfetch`, and is included in the manifest and
reproducibility fingerprint.

The first 100k FAISS CPU gate found two nearly equal projected rows exchanged
at positions 969/970. All semantic cutoff sets, including top-1024, were
identical and the maximum row-aligned distance difference was
`3.5762786865234375e-7`; the exchange came from float32 accumulation order and
could not change pilot input or any candidate budget. Conformance therefore
records and accepts order-only permutations strictly inside retained prefixes.
A row crossing any semantic cutoff remains a hard failure even if distances
are numerically close. Dedicated tests cover both cases.

The first measured 100k GPU attempt then exposed an exact float32-quantized tie
at the original-space top-512 boundary. Deterministic refinement now reports
its latency, number of host distance evaluations, and requested neighbor count
separately. Projected scan counters continue to count full-corpus FAISS scans;
refinement work is not folded into or hidden behind those counters. CPU and GPU
use the same overfetch/refinement rule for a controlled comparison.

The run records host index construction and host-to-device transfer for both
indexes. Per-query records add backend query-upload, device/CPU search, and
result-download fields. When FAISS GPU and its PyTorch tensor bridge are both
available, the three GPU stages are synchronized and timed separately. With
the synchronous NumPy API, transfers are included in `backend_search_ms` and
the explicit transfer fields remain zero. `search_ms` and end-to-end latency
also include adapter validation and stable ordering around that call, so stage
components must not be assumed to sum exactly to total latency.

Original-space and projected-space GPU indexes share one
`StandardGpuResources` instance. This avoids reserving a separate FAISS scratch
memory pool for each index; the manifest records which index created the pool
and which reused it. The 10k A100 smoke audit initially exposed approximately
3.5 GiB of device use for only 6.4 MB of vectors when two pools were created,
so shared-resource behavior is a required gate before the 100k comparison.

`gpu_memory.json` contains `nvidia-smi` snapshots before index construction,
after both original/projected indexes are resident, and after measured queries,
along with `CUDA_VISIBLE_DEVICES` and `SLURM_JOB_GPUS`. These are device-level
snapshots, not allocator-specific ownership accounting. CPU process peak RSS
continues to be reported separately.

FAISS is intentionally optional rather than a base dependency: all required
offline CPU tests still run without it. The local suite uses an exact test
double for adapter/control-flow coverage and conditionally runs an additional
real-FAISS CPU conformance test when the module is installed. Slurm job
`373268` passed that real CPU test and the initial 10k A100 correctness smoke;
the shared-resource memory rerun and 100k latency comparison remain required.

Example commands, after the cluster environment passes the FAISS probe:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.retrieval_benchmark \
  --config configs/retrieval_latency_100k_d768.json \
  --output runs/faiss-cpu-100k \
  --backend faiss-cpu \
  --faiss-threads 1

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.retrieval_benchmark \
  --config configs/retrieval_latency_100k_d768.json \
  --output runs/faiss-gpu-100k \
  --backend faiss-gpu \
  --gpu-device 0 \
  --faiss-threads 1
```

## Configurations

### Smoke

`configs/retrieval_latency_smoke.json` uses `N=10,000`, `d=128`, and
`m_prime=32`. It is for local validation, not reporting.

### Genoa baseline

`configs/retrieval_latency_100k_d768.json` uses:

- `N=100,000`;
- 64 measured and 4 warmup queries;
- `d=768`, `m_prime=96`;
- fixed `M=512`;
- `M_grid=[32,64,128,256,512,1024]`;
- approximately 307.2 MB original and 38.4 MB projected corpus storage.

### Genoa scale-up

`configs/retrieval_latency_1m_d1024.json` uses:

- `N=1,000,000`;
- 32 measured and 2 warmup queries;
- `d=1024`, `m_prime=128`;
- fixed `M=1024`;
- `M_grid=[64,128,256,512,1024,2048]`;
- approximately 4.096 GB original and 512 MB projected corpus storage.

The 100k run is the first gate. Run the 1M configuration only after the 100k
artifacts pass scan-count and memory checks.

## Recorded artifacts

- `manifest.json`: config, hashes, environment, generation durations, projection
  metadata, thread environment, and reproducibility identity;
- `summary.json`: per-method latency percentiles, quality, budget, work, memory,
  and reuse comparison;
- `per_query.jsonl`: query-level stage timings and work counters;
- `memory.json`: memmap sizes, query/projection sizes, cache size, and peak RSS;
- `gpu_memory.json`: GPU environment and device-memory snapshots for FAISS GPU;
- `tri_predict_policy.json`;
- `tri_predict_compiled_policy.json`: loadable decision intervals tied to the
  analytic reference fingerprint;
- `report.md`.

Stages include query projection, original scan, pilot scan, original pilot
distance computation, LID estimation, Tri-Predict, expansion, deterministic
FAISS refinement, original rerank, and end-to-end latency. Summary latency
includes mean, p50, p95, and p99.

## Reproducible CPU command

Set BLAS thread counts before Python starts. The first baseline is deliberately
single-threaded so per-query latency and memory traffic are interpretable:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
python3 -m tri_rag_harness.retrieval_benchmark \
  --config configs/retrieval_latency_100k_d768.json \
  --output runs/retrieval_latency-100k
```

The generated memmaps are intentionally inside the run directory and ignored
by Git. Audit bundles should include JSON/JSONL/Markdown/log artifacts but omit
the `data/` directory.

## Local structural result

The 100k/d768 configuration completed locally and confirmed the expected work
invariant: reuse performed 100,000 projected distances and scanned 38.4 MB per
query, versus 200,000 distances and 76.8 MB for the legacy control. It selected
the same budgets and retention. Mac latency is not a Genoa result and is not
used for serving claims.

## Genoa 100k result

Slurm job `371643` ran on `genoa04` at commit
`fc2ce25d7fb29c1f005d61ddc5c847981ebe7e3b` with Python `3.9.23`, NumPy
`1.26.4`, SciPy `1.13.0`, and one thread for each configured BLAS runtime. All
41 offline tests passed before the benchmark. The audit archive has SHA-256
`3b4d5a76d2a0005a237f1e86df6b8ba992c9b2b51a3a4a9f649700672d999c10`.

| Path | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | mean M | mean retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original fixed | 17.0579 | 17.0539 | 17.1631 | 17.2223 | 512 | 1.0000 |
| projected fixed | 3.5866 | 3.3435 | 4.2215 | 4.2990 | 512 | 0.0578 |
| Tri-Predict, reuse | 61.0718 | 60.9780 | 62.3908 | 63.8409 | 1024 | 0.0969 |
| Tri-Predict, double scan | 63.9405 | 64.1006 | 65.1323 | 65.5186 | 1024 | 0.0969 |

The reuse path evaluated `100,000` projected distances and scanned 38.4 MB per
query. The control evaluated `200,000` and scanned 76.8 MB. Reuse therefore
removed exactly 50% of projected distance work and bytes. It reduced mean total
latency by 4.49% and p95 by 4.21%. Query-level audit found identical budgets and
retention for all 64 paired queries; reuse was faster for 63 of 64 individual
pairs. The one slower pair is timing noise rather than a decision mismatch.

All 64 pilot LID estimates were valid, with raw LID from `34.43` to `84.96`
(mean `54.30`), but all 64 policy decisions saturated at the maximum budget
`M=1024`. Tri-Predict itself averaged 55.7710 ms, or 91.32% of the reuse path's
total time. Its mean top-10 embedding retention was only 9.69%; 23 of 64
queries retained none of the original top-10. Projected fixed `M=512` retained
5.78% on average and had zero retention on 37 of 64 queries.

The operational 100k gate passes: the run completed within the memory bound,
the exact backend remained streaming, work counters prove one-scan reuse, and
reuse preserved decisions and quality. Peak process RSS was approximately
424.1 MiB (`444,706,816` bytes), while the node reported approximately 738 GiB
available before the run. The semantic/policy-efficiency gate does not pass:
the normalized-Gaussian fixture has no text semantics, the adaptive policy
saturates every query, and its analytic computation makes it slower than both
fixed baselines.

The 1M run is therefore authorized only as a systems-scaling measurement. It
may test scan bandwidth, tail latency, memory, and the crossover between scan
cost and the current scalar analytic overhead; it must not be reported as
evidence that Tri-Predict is adaptive, quality-preserving, or latency-efficient.
Because the checked-in 1M configuration also changes `d` from 768 to 1024 and
`m_prime` from 96 to 128, it is a second deployment-shaped operating point,
not a controlled estimate of corpus-size scaling alone.
Meaningful policy evaluation still requires the real external-query embedding
adapter and an independently tuned/frozen policy.

## Genoa 1M result

Slurm job `371643` also ran the 1M/d1024 configuration on `genoa04` at commit
`37e60d5da57817efe3af8a7874b16206586f672c`. The environment again used Python
`3.9.23`, NumPy `1.26.4`, SciPy
`1.13.0`, and one BLAS thread. All 41 offline tests passed. The audit archive
has SHA-256
`fa6b8763d0ebff0dd7421cc6d81cd5b3dfad7fb4b0cb7e4ab00cdcabe83b8c77`.

| Path | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | mean M | mean retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original fixed | 193.2208 | 193.2184 | 193.6270 | 193.8720 | 1024 | 1.0000 |
| projected fixed | 64.4080 | 63.8469 | 66.7349 | 66.8708 | 1024 | 0.0312 |
| Tri-Predict, reuse | 146.7481 | 146.0945 | 151.3177 | 152.4968 | 2048 | 0.0688 |
| Tri-Predict, double scan | 194.9228 | 195.4949 | 198.2416 | 198.7729 | 2048 | 0.0688 |

The reuse path scanned 512 MB and evaluated one million projected distances per
query. The control scanned 1.024 GB and evaluated two million. Reuse reduced
mean latency by 24.71% and p95 by 23.67%, and was faster on all 32 paired
queries. The paired saving averaged 48.175 ms and ranged from 37.264 to
52.673 ms. All paired budgets and retention values were identical.

At this operating point the single top-`M_max` projected scan averaged
85.902 ms, or 58.54% of reuse-path latency. Tri-Predict averaged 59.298 ms, or
40.41%. This is the intended systems crossover: scan work has become the
largest stage, so eliminating the second pass has a material end-to-end effect.
The scalar policy cost nevertheless remains too large for a serving path. Its
latency grew only 1.06x from 100k while the reuse scan grew 18.78x, consistent
with policy computation depending primarily on the frozen rank approximation
rather than scanning every corpus item.

The quality result remains negative. All 32 LID estimates were valid, with raw
LID from `42.17` to `71.14` (mean `58.36`), but all 32 decisions saturated at
`M=2048`. Mean top-10 retention was 6.875%, and 17 of 32 queries had zero
retention. Projected fixed `M=1024` retained 3.125% on average and had zero
retention on 24 queries. Although the maximum budget doubled relative to the
100k run, its corpus fraction fell from 1.024% to 0.2048%, which helps explain
the lower retention on the Gaussian fixture.

Original and projected corpus storage were 4.096 GB and 512 MB; measured peak
RSS was 4.364 GiB. Corpus generation took 72.19 seconds and projection
generation took 4.74 seconds. The complete benchmark process took 99.88 seconds
of wall time and exited successfully.

The exact CPU baseline is now sufficient to begin a FAISS adapter experiment,
but raw backend latency and end-to-end latency must remain separate. Before an
end-to-end GPU claim, the frozen analytic policy should be compiled into tested
LID-to-budget decision boundaries or an equivalently validated lookup, because
a faster projected backend would otherwise expose the roughly 59 ms scalar
policy computation as the dominant cost. Real embedding data is still required
for any retention, evidence, or adaptive-policy claim.

## Interpretation limits

- Exact NumPy streaming search is the correctness baseline, not the final
  production backend.
- Gaussian vectors reproduce dimensions and memory traffic but not real text
  embedding anisotropy, LID, or neighbor structure.
- Different methods may use different candidate budgets; latency and retention
  must be read together.
- Page cache, BLAS implementation, CPU frequency, NUMA placement, and thread
  count are part of the benchmark environment.
- Candidate-count saving is not latency saving. Tri-Predict root solving,
  projected scans, and memory traffic are all reported independently.
- Compilation time is setup cost and is reported separately. The per-query
  policy stage measures only the loaded frozen lookup.
- FAISS CPU/GPU comparisons come only after the exact Genoa baseline passes.
- Real FAISS CPU conformance and an initial 10k A100 correctness smoke pass.
  The shared-resource memory rerun and 100k CPU/GPU comparison are still
  pending, so no acceleration claim exists yet.
