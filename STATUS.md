# Status

Updated: 2026-08-27

## What runs

Milestones 0 through 4 are implemented as a network-free harness with a CPU default, and the retrieval-only systems benchmark additionally supports optional exact FAISS CPU/GPU backends. The certification run generates external tune/cert/test queries, normalizes embeddings, builds one fixed dense-Gaussian projection, runs exact original/projected squared-L2 retrieval, fits and freezes both monotone-binned and query-adaptive Tri-Predict policies on tune queries, evaluates each policy independently, and writes auditable artifacts. A separate two-stage command performs a predeclared global `m_prime`/Tri-Predict-threshold sweep on tune only, writes frozen selection artifacts, and then evaluates one fresh certification split.

Pilot and expansion now reuse one exact projected scan in the main harness: the backend retains top `M_max`, exposes the pilot prefix, and slices the cached ranking after `M(q)` is chosen. A separate retrieval-only benchmark provides a memmap-compatible streaming exact backend plus an explicit legacy double-scan control at `100k x 768` and `1M x 1024` scale.

An optional exact FAISS adapter is now implemented behind that benchmark interface for CPU and one NVIDIA GPU. It uses float32 `IndexFlatL2`, records index-build/host-to-device timing, separates GPU upload/search/download when PyTorch tensor interop is available, captures device-memory snapshots, and retains the one-scan reuse/control accounting. Before measurement it must match NumPy candidate sets at every consumed cutoff and row-aligned squared distances, compiled-policy decisions, reranked top-k rows, and retention. A fixed one-scan overfetch guard plus canonical NumPy refinement restores the stable-row boundary contract; a tie band that is not closed by the guard remains a terminal error. Real FAISS 1.10.0 CPU conformance, the 10k A100 correctness/shared-resource gates, and the 100k and separately frozen 1M CPU/GPU latency gates passed on Slurm job `373268`.

The first A100 10k smoke at commit `40b2312` passed all correctness gates but exposed approximately 3.5 GiB of GPU use for only 6.4 MB of index vectors. Each index had independently allocated a FAISS scratch pool. The adapter now shares one `StandardGpuResources` instance across original/projected indexes and records that ownership in the manifest; its A100 rerun passed the explicit memory-reduction assertion before the 100k attempt.

The same allocation used Python `3.9.23`, NumPy `1.26.4`, SciPy `1.13.0`, FAISS GPU `1.10.0` built for CUDA `12.1.1`, and PyTorch `2.5.1` with CUDA `12.1` on one A100-SXM4-80GB. All then-current 47 tests passed without a skip. CPU/GPU smoke records matched exactly on every measured query's budget, retention, LID, fallback/saturation state, scan count, and distance count. At 10k, FAISS CPU was faster than GPU for single-query requests because GPU launch and transfer overhead dominated; this is expected and is not extrapolated to 100k or 1M.

The first 100k FAISS CPU attempt stopped at the pre-measurement NumPy gate. The projected top-1024 results differed only by exchanging positions 969/970: every pilot/budget cutoff candidate set was identical and maximum row-aligned distance error was `3.5763e-7`. This is float32 accumulation-order variation, not a candidate, policy, or retention change. The gate now compares exact candidate sets at `k_gt`, `M_pilot`, and every configured budget, plus row-aligned distances and downstream decisions. It accepts and records permutations strictly inside a retained prefix but still rejects any row crossing a semantic cutoff.

The subsequent 100k CPU run completed, but the GPU run stopped on an exact float32-quantized tie at the original top-512 boundary. The adapter now overfetches a fixed guard within one FAISS search, recomputes that small pool with the canonical NumPy squared-L2 formula, and deterministically selects by distance/row. Refinement latency, requested neighbors, and extra host distance evaluations are recorded separately. A tie band larger than the guard remains a terminal error rather than triggering an unreported second full scan.

The fresh 100k CPU/GPU run at commit `05ccf91` completed on the same A100 allocation after all 51 tests passed. The CPU/GPU runs have identical corpus, projected-corpus, query, projection, analytic-policy, and compiled-policy identities. Their 64-query audit records agree on every selected budget, retention value, LID result, fallback/saturation flag, scan count, and distance count. Both backend conformance probes report zero mismatches. CPU/GPU mean latencies in milliseconds were `12.4316/1.9796` for original fixed, `2.7146/1.8744` for projected fixed, `3.7243/3.4357` for Tri-Predict reuse, and `5.4599/4.3035` for the double-scan control. The GPU therefore accelerates original exact search by 6.28x, but Tri-Predict reuse by only 1.08x because transfer, deterministic refinement, LID, lookup, and original reranking remain exposed at this scale.

The A100 reuse path still performs exactly one projected full-corpus search and reduces projected distance work by 50% relative to the control. This lowers mean/p95 GPU latency by `20.17%/20.22%`; the corresponding CPU reductions are `31.79%/31.83%`. GPU query upload, search, result download, and deterministic refinement are all separately nonzero and recorded. Device memory rose from 0 to 2307 MiB when the shared original/projected indexes were installed and remained at 2307 MiB after all queries, so no query-loop growth was observed. The raw index vectors occupy approximately 329.6 MiB; the remaining approximately 1.93 GiB includes the CUDA context, shared FAISS resources/scratch, and other device allocations and is not attributed to a specific allocator by `nvidia-smi`.

The first 1M FAISS attempt on job `373268` completed the CPU run but stopped the GPU run during pre-measurement conformance. The original 1M grid ends at `M=2048`; adding the required 64-neighbor stable-boundary guard requested top-2112, while this FAISS GPU build supports k-selection only through 2048. This is a backend capability mismatch, not an out-of-memory or retrieval mismatch. The harness now rejects such a configuration before creating the output directory or generating embeddings. A separate frozen FAISS operating-point config retains the same seeds, `N=1M`, `d=1024`, `m_prime=128`, and fixed `M=1024`, but uses `M_max=1984`, so the full `1984+64=2048` guard remains valid. Results from that policy grid must not be presented as the old `M_max=2048` policy result.

The corrected 1M CPU/GPU pair completed on job `373268` at commit `0d387a1` after all 54 tests passed with real FAISS. Both runs have identical data, projection, analytic-policy, compiled-policy, and configuration identities; their serialized analytic and compiled policies are byte-identical. An independent local audit of all 128 records per backend found exact agreement in query/method identity, budget, retention, LID, validity/failure, saturation/fallback, scan counts, distance counts, refinement counts, and requested-neighbor counts. Each backend's mandatory NumPy probe matched rows and distances exactly at every consumed cutoff, with zero downstream mismatches.

At 1M, FAISS GPU improves original fixed from `157.9387` to `16.6473 ms/query` (9.49x), projected fixed from `22.8045` to `14.7312 ms` (1.55x), Tri-Predict reuse from `24.9773` to `18.3728 ms` (1.36x), and the double-scan control from `45.4543` to `23.6353 ms` (1.92x). Reuse removes exactly 50% of projected distance work and lowers GPU mean/p95 latency by `22.27%/19.12%`; CPU mean/p95 reductions are `45.05%/45.04%`. Device memory is `0/6383/6383 MiB` before indexes/after indexes/after queries. Raw index vectors account for approximately 4394.5 MiB, leaving approximately 1988.5 MiB of context/shared-resource/other overhead and zero observed query-loop growth.

This closes the exact 1M FAISS systems gate but not the adaptive-efficiency or quality gate. All 32 adaptive queries saturate at `M=1984`; mean top-10 retention is `0.065625`, 18 queries have zero retention, and GPU reuse remains 10.37% slower than GPU original fixed and 24.72% slower than GPU projected fixed. The audit archive SHA-256 is `9f589694b7fddbf6ef0d134468a753b54806fae04f2d816b6c409f1e07edc2aa`.

The retrieval benchmark now compiles the frozen analytic Tri-Predict policy into adjacent-float64 LID decision intervals before query measurement. Its serving path loads a fingerprint-checked artifact and uses one interval lookup per query. Compilation and analytic-reference validation are outside measured retrieval latency. Every observed query LID is checked against the analytic reference after measurement, and a mismatch aborts artifact generation. The main certification harness retains the analytic policy so its predicted-retention diagnostics and existing certificate identity do not change.

Milestone 5 now has a network-free BEIR ZIP adapter and a pinned SciFact configuration. Adapter v2 verifies the publisher-listed archive MD5, records archive/member SHA-256 identities, rejects corrupt or structurally inconsistent data, and gives corpus and external queries separate stable-ID namespaces. It also normalizes query text with NFKC/casefold/whitespace collapse, excludes development text duplicated in untouched official test, and assigns remaining development duplicate-text groups wholly to tune or cert using a seeded label-free ordering. Canonical artifacts contain no timestamps or machine paths and reproduce byte for byte.

The original v1 preparation completed on 2026-08-27 at commit `aff63e4` and passed all ID-level and artifact checks, but an independent post-embedding audit found four duplicate query-text groups, three crossing tune/cert/test. Dataset fingerprint `6f54d75d95c40569f7382270e833c8602afd317042e2a791118e4a15992038df` is now quarantined from downstream claims. Adapter v2 was regenerated on Slurm job `373564` and then independently reproduced byte for byte from the returned source ZIP. Its accepted fingerprint is `4a73586d3a29a0567287e501ac3c06c998af661cdc74dbc589e7525a7924f903`: 5,183 documents, 1,107 retained queries, 1,256 qrels, and 403/404/300 tune/cert/test queries. Development IDs `1291` and `871` were excluded in favor of their official-test text duplicates, and all normalized-text intersections are empty.

A pluggable text-embedding cache is implemented and bound to the repaired dataset fingerprint. The frozen provider is `intfloat/e5-base-v2` at Hugging Face commit `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, with explicit E5 query/passage formatting, 768 dimensions, 512-token maximum, float32 computation/output, canonical L2 normalization, deterministic algorithms, eager attention, TF32 disabled, cuDNN deterministic mode, and a required deterministic cuBLAS workspace configuration. Slurm job `373564` on `a100-1` passed all 66 then-current tests and created then reused the repaired cache. Independent audit rehashed every input/output, validated IDs/qrels/norms/splits, and accepted embedding fingerprint `2ec53ce38e226129ba0feffcd28ba1da1081e0627ad8e54f4a60e430c341e914`. The returned archive SHA-256 is `dddd51c97d04171f253820131ca37feae450e1ba2b620ed83bf2e9de29e0dd63`. The earlier v1-bound embedding fingerprint remains quarantined.

The first real retrieval runner now provides a strict exact original-space baseline on `query_tune` only and rejects cert/test scope at config load. It binds the complete dataset/config/request/cache identities, uses normalized squared L2 with NumPy float64 and stable document-ID ties, writes 403 query-level evidence records, separates timing from deterministic result identity, and revalidates every input before search. Two local runs over the accepted arrays matched byte for byte with result fingerprint `2921f39dc051bc3331da8bf9b0ddc6c584dcd1f043099d8dda353653a1926b1c`. Tune evidence hit/recall/nDCG are `0.803970/0.786849/0.713828` at context cutoff 5 and `0.866005/0.850124/0.735501` at neighbor-reference cutoff 10. No cert/test retrieval outcome was computed.

Slurm job `373780` reproduced that exact original-space result twice on Genoa at commit `37f68fc`; both deterministic runs are byte-identical and all 71 then-current tests passed with one expected optional real-FAISS skip. Search timings were `0.5372` and `0.5336 ms/query` and are correctly excluded from result identity. Independent local reduction of the returned query-level records reproduced every evidence aggregate and artifact hash. The accepted baseline audit archive SHA-256 is `c91a402eea55de127381f82f21f3a988ced679959b88eed868604292fda1af6d`.

A real tune-only fixed-dimension sweep is now implemented with a strict schema that rejects protected splits and mutations of the common cost objective. Config fingerprint `3265e303c5249a6b90868f5234d333eca3f1fc4bc28c12cdb710382e2b71eabd` freezes dense Gaussian seed `27011`, 12 dimensions from 16 through 768, no projected renormalization, `M_pilot=32`, one shared 16-value budget grid ending at the full 5,183-document corpus, and a tune retention-score target of `0.95`. Selection minimizes absolute coordinate work `(N+d)*m_prime+d*M`; it does not use dimension-specific candidate-saving denominators or evidence labels.

Two local runs over the accepted real arrays matched byte for byte for every deterministic artifact, and an independent audit recomputed all 192 dimension/budget empirical-Bernstein statistics from 4,836 query-level records. The frozen tune choice is `m_prime=192`, fixed reference `M=768`, mean retention `0.985360`, lower bound `0.958051`, and theoretical coordinate-work reduction `56.48%` versus an original full scan. The selection/result/frozen-projection fingerprints are `093588a27e0d588b9407d02fe5c5ed7e46f6a5fdc02a1881738abbde4eda01fb`, `5dcb0a5f17cc1f2f1684a38c71ede5c8dfc1de709f98496156959e14fbea7558`, and `8a9a1148527db16c43bc3fedf6da1ac79ae00c0d76f2a31321cd6d9fe049809e`. This is a tune selection result, not a certificate, latency claim, evidence/test result, or answer-quality claim.

Slurm job `373780` reproduced the dimension sweep twice on Genoa at commit
`07c28e1`; all six deterministic artifacts are byte-identical across the two
runs and match the local identities above. The returned archive SHA-256 is
`99cf703cd384555305d8d526224ccc01208c76539cd79acb45b8ea600b737b21`.
An independent local audit rehashed every artifact and recomputed all 192
empirical-Bernstein bounds before accepting the archive.

The required real policies have now been fit and frozen on `query_tune` only at
`m_prime=192`. The repaired protocol-v2 config fingerprint
`47d37917974869641951a0155e71ffbb76f676d8229ff606fef56fabc83ba812`
freezes the pilot/LID contract, nine-decimal deployable LID canonicalization,
complete budget grid, common coordinate-work objective, monotone-bin target
grid, dense near-one Tri-Predict threshold grid, and tune-only residual-
correction grid. It rejects cert/test scope, determinism-contract mutation, and
evidence labels. Two independent local v2 runs produced byte-identical
scientific artifacts for all 403 tune queries, and an independent reduction
reproduced the selected-policy bounds and result identity.

The fixed reference remains `M=768` with mean retention `0.985360` and tune
lower bound `0.958051`. The selected monotone-binned policy uses budgets
`[384, 512, 768, 1024]`, averages `M=672.397`, and has mean/lower-bound
retention `0.983623/0.956282`; relative to the fixed tune reference this is
`12.45%` fewer candidates and `4.24%` less common coordinate work. The selected
Tri-Predict policy requires analytic target `0.99995`, averages `M=1092.548`,
and has mean/lower-bound retention `0.979653/0.950276`; it uses `42.26%` more
candidates and `14.39%` more coordinate work than fixed. This is a negative
Tri-Predict tune-efficiency result, not an independent certificate. The pilot
versus oracle clipped-LID mean absolute gap is `14.8902`, with both estimators
valid on all 403 tune queries; oracle LID remains diagnostic only.

The v2 policy result/selection fingerprints are
`2c31279a8f8038eebb049b0630548b2edd533ee5e1e01adb6cbd0a41e7e9bcb8`
and `2eaed81134b1621e9f2fd2f072a3c800cb6eaa8610bcaeb02f0a8465c34509f1`.
The fixed-grid/monotone/analytic policy fingerprints are
`50c332ea015bda36f30803b31f480777d0129720e2bc7988b13d63fe8c6cea0f`,
`7734ac4efb84a66af837028a289422ff21ce77d1e9cfae68f484d52d10286f38`,
and `7838e1be673932f38c5b4db9d1cea06e168565b0b7feba3014bef618f89d4423`.
The local compiled deployment fingerprint is
`4530a8a5bc9ef8d3c9858da8774490f98ffdf722fae943356205abd185dadd7b`;
it is deliberately excluded from selection and scientific result identity.
The exact target-1 boundary now requires retrieving the complete corpus instead
of accepting a finite-budget special-function value rounded to one; a full
corpus also retains exact unit retention regardless of a tune-fit correction.
This decision-semantic change is explicitly serialized as Tri-Predict policy
version 2, so old and new behavior cannot share a policy fingerprint.

The first Genoa policy run at commit `516c1e4` was internally reproducible: two
runs on job `373780` were byte-identical, and all selected policies, budgets,
retention values, and empirical-Bernstein bounds matched the Mac exactly. Its
cross-platform fingerprint assertion nevertheless failed. Independent audit of
archive SHA-256
`a74ce1d5ad1a13f4c7851deccac9314bfbec378c5c085194d134eeac1fd3bb13`
found only two causes: 447 LID fields across 222 query records differed in the
last decimal places (maximum `8.01e-12`), and 13 of 15 adjacent-float compiled
boundaries differed by at most `7.82e-14`. Canonicalizing LID to nine decimals
makes all 403 Mac/Genoa features identical, and replaying both selected policies
at that precision changes zero budgets. Protocol v2 therefore separates the
platform-bound compiled lookup from scientific identity while still binding it
to the analytic policy and requiring zero validation mismatches.

The local compiled-policy 100k/d768 structural run created seven states, loaded the artifact in `0.1093 ms`, and exactly matched all 64 prior local LID values, budgets, and retention values. Analytic validation averaged `38.4606 ms/decision`; lookup averaged `0.0021 ms/decision`. Reuse-path latency fell from `44.7511` to `4.2712 ms/query` across the old and new local runs. Compilation cost `17.9965 s` once at setup. These are not Genoa serving claims.

Slurm job `373123` reproduced the compiled policy on `genoa00` at commit `0407c831c263e7505543b3701b84e7cd4a4b4bd0`; all 43 tests passed. At 100k, lookup averaged `0.002544 ms` versus `55.8130 ms` for the analytic reference (21,939x), and reuse fell from `61.0718` to `5.2163 ms/query` with p95 `5.4572 ms`. At 1M, lookup averaged `0.002587 ms` versus `58.3822 ms` (22,564x), and reuse fell from `146.7481` to `87.5176 ms/query` with p95 `90.2264 ms`. Artifact loading took `0.149/0.142 ms`; offline compilation took `26.106/27.898 s`. Every old/new query-level LID, budget, saturation flag, retention value, and work counter is identical, and both reference policies are unchanged. The exact CPU and compiled-policy systems gates pass.

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

Optional exact FAISS CPU/GPU selection adds, respectively:

```bash
--backend faiss-cpu --faiss-threads 1
--backend faiss-gpu --gpu-device 0 --faiss-threads 1
```

Pinned SciFact preparation (after downloading the configured archive):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.beir_dataset \
  --config configs/real_scifact_dataset.json \
  --archive data/source/scifact.zip \
  --output data/prepared/scifact-dedup-v2
```

Pinned SciFact E5 embedding cache on an NVIDIA node:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.text_embeddings \
  --config configs/real_scifact_e5_base_v2_embeddings.json \
  --dataset data/prepared/scifact-dedup-v2 \
  --output data/embeddings/scifact-e5-base-v2-dedup-v2 \
  --device cuda:0 \
  --model-cache data/model_cache
```

Tune-only exact original-space SciFact baseline:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.real_original_baseline \
  --config configs/real_scifact_original_exact_tune.json \
  --dataset data/prepared/scifact-dedup-v2 \
  --embedding-config configs/real_scifact_e5_base_v2_embeddings.json \
  --embedding-cache data/embeddings/scifact-e5-base-v2-dedup-v2 \
  --output runs/scifact-original-exact-tune
```

Tune-only fixed SciFact projection-dimension selection:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.real_dimension_sweep \
  --config configs/real_scifact_fixed_dimension_tune.json \
  --dataset data/prepared/scifact-dedup-v2 \
  --embedding-config configs/real_scifact_e5_base_v2_embeddings.json \
  --embedding-cache data/embeddings/scifact-e5-base-v2-dedup-v2 \
  --original-baseline runs/scifact-original-exact-tune \
  --output runs/scifact-fixed-dimension-tune
```

Tune-only SciFact policy fitting at the frozen projection:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.real_policy_tune \
  --config configs/real_scifact_policy_tune.json \
  --dataset data/prepared/scifact-dedup-v2 \
  --embedding-config configs/real_scifact_e5_base_v2_embeddings.json \
  --embedding-cache data/embeddings/scifact-e5-base-v2-dedup-v2 \
  --dimension-selection runs/scifact-fixed-dimension-tune \
  --output runs/scifact-policy-tune
```

## Tests passed/failed

- Passed locally: 85
- Skipped locally: 1 conditional real-FAISS CPU conformance test because FAISS is not installed on the Mac environment
- Failed: 0
- Runtime in the current environment: approximately 7.5 seconds
- FAISS coverage checks the exact CPU adapter contract, row-aligned distances, semantic-cutoff candidate sets, accepted internal permutations, rejected cross-cutoff permutations, bounded tie resolution, unclosed tie-band refusal, missing GPU support, shared GPU resource-pool ownership, refinement accounting, full benchmark integration, NumPy decision/rerank/retention equivalence, and GPU-memory artifact creation. The conditional real-FAISS test must execute rather than skip on the cluster before the CPU/GPU milestone passes.
- Added coverage includes cross-platform policy-float canonicalization, exact `h_j(y)` term-by-term agreement with the orthogonal conditional law, geometric rank-strata population conservation and approximation error, root residuals, the infinite-root/unit-retention boundary, budget monotonicity, LID-to-budget monotonicity, saturation, analytic/empirical interface compatibility, and tune-only scalar safety correction.
- Compiled-policy coverage checks dense linear/geometric LID values, both sides of every adjacent-float64 transition, invalid/out-of-domain fallback, deterministic serialization, artifact round-trip loading, and tamper rejection.
- Attribution coverage verifies that actual squared-distance ratios reproduce the rank-model prediction when the assumed power law is exact, that attribution modes produce complete query-level artifacts, and that different synthetic seeds create disjoint stable ID namespaces.
- The pre-Milestone-4 baseline also passed all then-current 22 tests on Slurm job `371035`, node `genoa05`, using Python `3.9.23`, NumPy `1.26.4`, and SciPy `1.13.0` from micromamba environment `tri-rag`.
- The extended sweep passed all 39 tests on Slurm job `371643`, node `genoa04`, commit `d5ec795abf0ca604c90ac2b5300708232874ef32`, using Python `3.9.23`, NumPy `1.26.4`, and SciPy `1.13.0`.
- The FAISS A100 milestone passed all 51 tests on Slurm job `373268`, node `a100-0`, commit `05ccf91`, using Python `3.9.23`, NumPy `1.26.4`, SciPy `1.13.0`, FAISS GPU `1.10.0` for CUDA `12.1.1`, and PyTorch `2.5.1` with CUDA `12.1`.
- The corrected 1M FAISS milestone passed all 54 tests on the same job and node at commit `0d387a1`; the real-FAISS CPU conformance test executed rather than skipping.
- The repaired SciFact v2 A100 gate passed all 66 then-current tests on job `373564`, node `a100-1`, commit `d776404`; the real-FAISS test executed rather than skipping.
- The BEIR SciFact adapter suite has six tests covering byte reproducibility, pinned config/checksum refusal, missing-document qrels, development/test ID overlap, development/test text exclusion, normalized-text grouping, and cross-split text disjointness.
- Six text-embedding tests cover the pinned dataset/model request, exact E5 prefix preservation, fake-provider normalization, cache reuse without model loading, source-artifact mutation refusal, request mutation refusal, array-tamper refusal, and partial-cache suppression on invalid provider output.
- Five real-original-baseline tests cover tune-only enforcement, strict cache tamper refusal, graded nDCG, stable tie breaking, deterministic artifacts, pinned real identities, and exclusion of timings from result identity.
- Five real-dimension-sweep tests cover protected-split refusal, immutable common-cost semantics, exact query-projection accounting, stable projected ranking conformance, terminal full-corpus behavior, query-level auditability, and deterministic tune-only artifacts.
- Eight real-policy tests cover immutable tune-only/determinism configuration, the observed Mac/Genoa LID-tail fixture, compiled/scientific identity separation, protected-split and common-cost refusal, terminal fallback, exact coordinate accounting, cached/analytic full-corpus boundary equivalence, and query-aligned decision reduction. Two additional Tri-Predict regressions prevent finite-budget floating-point saturation from satisfying an exact unit target and preserve deterministic unit retention at a complete-corpus budget. The full local suite now contains 86 tests: 85 pass and one optional real-FAISS test skips.

## Current artifacts

The historical v1 Stage-1 audit archive is
`scifact-stage1-20260827-091050-audit.tar.gz`, with local SHA-256
`f6932dfe1a002c2c4f349a269f55fb56e85441106c9ead9b2e99e0192069b9f5`.
It contains the source ZIP, complete canonical dataset, config, documentation,
and preparation log. It remains ignored and is useful as the authoritative
source archive, but its prepared v1 split is quarantined. The historical E5
audit archive is `scifact-e5-373564-audit.tar.gz`, SHA-256
`3d87f889cc5a5937ea3666b1f8f7657d02bb14467fb23151daa70dc7fcfa6941`;
its model/runtime validation passes, but its v1-bound arrays are likewise
quarantined. Neither artifact may be used for retrieval selection or claims.

The accepted v2 archive is `scifact-e5-v2-373564-audit.tar.gz`, SHA-256
`dddd51c97d04171f253820131ca37feae450e1ba2b620ed83bf2e9de29e0dd63`.
It contains the repaired canonical dataset, accepted embedding arrays and
manifest, source ZIP, configs, documentation, and full Slurm log. It remains
ignored rather than committed; downstream real runs bind its dataset and
embedding fingerprints.

The accepted Genoa original-baseline archive is
`scifact-original-tune-373780-audit.tar.gz`, SHA-256
`c91a402eea55de127381f82f21f3a988ced679959b88eed868604292fda1af6d`.
It contains both byte-identical deterministic baseline runs, their separate
timings, input manifests, code/config, and the complete Slurm log.

The accepted Genoa fixed-dimension archive is
`scifact-dimension-tune-373780-audit.tar.gz`, SHA-256
`99cf703cd384555305d8d526224ccc01208c76539cd79acb45b8ea600b737b21`.
It contains both byte-identical dimension runs, the original baseline identity,
frozen selection artifacts, source/config/test files, and the complete Slurm
log. The real policy outputs are currently independently reproduced local
artifacts in `/tmp`; their deterministic identities are listed above and await
the cluster reproduction gate before an audit archive is accepted.

The first Genoa policy archive is
`scifact-policy-tune-373780-audit.tar.gz`, SHA-256
`a74ce1d5ad1a13f4c7851deccac9314bfbec378c5c085194d134eeac1fd3bb13`.
It is retained as the authoritative protocol-v1 cross-platform diagnostic: its
two Genoa runs are internally byte-identical and its scientific decisions match
the Mac, but its result fingerprint is not accepted because platform-bound LID
tail noise and compiled boundaries contaminated that identity. A fresh Genoa
protocol-v2 run is required before accepting the final frozen policy archive.

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

The 1M/d1024 follow-up completed in the same allocation at commit `37e60d5da57817efe3af8a7874b16206586f672c`. Reuse reduced projected work from two million to one million distances per query and preserved every paired decision and retention value. Mean/p95 latency fell from `194.9228/198.2416` to `146.7481/151.3177 ms`, reductions of 24.71%/23.67%. The one-scan projected stage now costs `85.9015 ms` (58.54% of reuse total), while Tri-Predict costs `59.2982 ms` (40.41%), demonstrating the scan/policy crossover. All 32 policies still saturate at `M=2048`; mean retention is `0.06875`, with zero retention on 17 queries. Peak RSS was `4,685,352,960` bytes and the complete benchmark process took 99.88 seconds. The exact systems baseline passes; the Gaussian policy-quality result fails.

The FAISS 100k CPU/GPU comparison from job `373268` uses the same 64 queries and frozen compiled policy on both backends. Original fixed search improves from `12.4316` to `1.9796 ms/query`, while projected fixed improves from `2.7146` to `1.8744 ms/query`. Tri-Predict reuse improves only from `3.7243` to `3.4357 ms/query`; at this scale, the GPU does not make the adaptive path faster than either GPU fixed baseline. All adaptive queries still saturate at `M=1024` with mean retention `0.096875`, so this is a passed exact-backend systems gate and a negative adaptive-quality/latency result.

The separately frozen `M_max=1984` FAISS 1M comparison also passes every exact systems gate. It establishes that one A100 can hold both 1M-vector flat indexes with a shared resource pool and execute all four paths without device-memory growth. It does not establish useful adaptation: every query saturates, retention remains low, and both fixed GPU paths are faster than Tri-Predict reuse. The complete analysis and interpretation limits are in `docs/RETRIEVAL_LATENCY.md`.

## Next task

Run protocol v2 once on Genoa and require scientific result fingerprint
`2c31279a8f8038eebb049b0630548b2edd533ee5e1e01adb6cbd0a41e7e9bcb8`
and selection fingerprint
`2eaed81134b1621e9f2fd2f072a3c800cb6eaa8610bcaeb02f0a8465c34509f1`.
The Genoa compiled deployment fingerprint may differ from the local value, but
must reference analytic policy
`7838e1be673932f38c5b4db9d1cea06e168565b0b7feba3014bef618f89d4423`
and report zero validation mismatches. Archive the full query-level result and
separate timings. Only after that gate, implement the certification-only runner,
load the frozen fixed, monotone-binned, and Genoa compiled Tri-Predict artifacts,
and evaluate untouched `query_cert` exactly once. A failure is terminal and
must not trigger retuning on those queries.

## Known deviations and risks

- Configuration is JSON rather than YAML, and query-level output is JSONL rather than Parquet, to keep the first pass runnable with only the already available NumPy/SciPy stack. The artifacts remain machine-readable and auditable.
- Real E5 inference and the repaired v2 cache passed independent audit. The first v1 cache remains quarantined because its dataset split allowed duplicate query text across tune/cert/test. The public E5 model card already reports SciFact performance, so the off-the-shelf model choice is not a fully blind model-family selection even though no local cert/test retrieval outcome was inspected. Tune-only evidence evaluation, fixed-dimension selection, and policy fitting are complete, but no real certification/test retrieval outcome or answer generation has been performed.
- The real tune pilot/oracle clipped-LID gap is large (`14.8902` MAE), but it does not by itself attribute the negative analytic-policy efficiency solely to pilot error. The selected uncorrected Tri-Predict threshold must rise to `0.99995`, and its mean budget already exceeds fixed; oracle-LID and actual-distance counterfactuals remain diagnostic work to separate LID error from rank-model/mean-field error without touching certification data.
- The selected real-data `56.48%` coordinate-work reduction is an arithmetic proxy under one exact full projected scan and exact reranking. It is not measured latency saving and omits fixed overheads, memory hierarchy effects, batching, and backend-specific kernels. Genoa reproduction validates result portability, not serving performance.
- E5 is English-only and truncates inputs above 512 tokens. The embedding manifest exposes separate corpus/query truncation counts, but any effect on SciFact retrieval remains unknown until the frozen cache exists.
- GPU and CPU transformer kernels may differ slightly even under fixed packages, float32, deterministic algorithms, eager attention, disabled TF32, and a fixed cuBLAS workspace. The generated array hashes—not an assumption of cross-device bit identity—become the frozen downstream experiment identity.
- The current synthetic pilot LID differentiates the hardest fitted bin, but the allocation is not efficient relative to the certified fixed baseline. The negative result is a dataset/policy outcome, not hidden by retuning certification data.
- The current synthetic certification split has been inspected repeatedly during implementation. Its artifacts validate code paths but must not be presented as a fresh research claim or reused to choose new hyperparameters. Real-data policy selection and certification require newly frozen independent splits.
- The new global sweep avoids that old split and enforces tune-only selection in code. Its positive `11.23%` result is candidate-count efficiency, not a latency claim; `m_prime=24` increases projected-search arithmetic relative to `m_prime=16`.
- The extended sweep shows that maximizing relative saving against a dimension-specific certified fixed baseline is unstable across dimensions and seeds. Its selected `m_prime=8` policy passes independently, but the `31.42%` saving is amplified by a coarse-grid certification cliff and is not a global cost optimum.
- On Genoa, scalar Tri-Predict root solving dominates measured retrieval latency (`5.9988` of `6.0325 ms/query`) on the 160-item synthetic corpus. Candidate-count saving must not be presented as latency saving; vectorization, lookup-table caching, or a validated approximation is required before serving claims.
- The compiled policy preserves only decision fields. Analytic predicted-retention values remain diagnostics produced by the reference policy; they are not reconstructed or interpolated online. Adjacent-float64 transition locations depend on the SciPy/platform special-function implementation, so the compiled lookup is a deployment artifact bound to—but intentionally excluded from—the analytic scientific policy identity.
- A deployment must load the archived compiled artifact rather than silently recompiling it. The certification manifest must bind both the cross-platform analytic policy fingerprint and the exact platform deployment fingerprint used for lookup.
- The retrieval latency fixture uses normalized Gaussian vectors with realistic shapes and memory traffic, not embeddings from a text model. It is a systems benchmark only; semantic retrieval conclusions require the real external-query adapter.
- FAISS is optional and intentionally absent from the base NumPy/SciPy dependency set. The isolated A100 environment has compatible real CPU/GPU FAISS and PyTorch interop, but its packages and exact versions must remain part of every run manifest/log because they are not installed by the base project metadata.
- FAISS `IndexFlatL2` is exact in float32 but does not guarantee stable candidate identity at an exact top-k boundary tie. The adapter deterministically refines a bounded one-scan overfetch pool and aborts if that guard does not close the raw tie band; refinement therefore adds host work and latency even when the GPU scan is fast.
- `nvidia-smi` memory snapshots are whole-device observations. They are useful on an exclusive node but are not process allocator ownership measurements.
- At 100k on A100, GPU Tri-Predict reuse is only 1.08x faster than FAISS CPU and is slower than both GPU fixed paths. Exact original search benefits most (6.28x); small projected scans expose fixed upload/download, boundary-refinement, LID, lookup, and reranking costs.
- FAISS GPU exact k-selection is capped at 2048 in the tested build. The stable-boundary contract consumes `M + overfetch`, not only `M`; therefore the original 1M `M_max=2048` configuration is incompatible with a 64-neighbor guard. The new `M_max=1984` FAISS config is a distinct frozen operating point, not a silent edit or continuation of the old policy.
- The passed 1M FAISS result remains negative for adaptive serving: all queries saturate at `M=1984`, mean retention is 6.5625%, and GPU reuse is slower than both fixed GPU paths. GPU acceleration is workload-dependent—9.49x for original fixed but only 1.36x for the complete reuse path.
- The Genoa 100k run passes the systems gate but not the policy gate: all queries saturate at `M=1024`, mean top-10 retention is `0.096875`, and Tri-Predict accounts for 91.32% of reuse-path latency. The 1M run can establish scaling behavior only.
- The Genoa 1M run also passes only the systems gate. Reuse saves 24.71% mean latency, but all queries saturate at `M=2048` and mean retention falls to `0.06875`. The checked-in 100k and 1M configurations change `N`, `d`, `m_prime`, pilot size, and budget together, so their ratio is not a controlled single-variable scaling law.
- Tri-Predict's exact rank summation is intentionally correctness-oriented and currently costs several milliseconds per synthetic query. Large real corpora should use and validate the deterministic rank approximation before performance claims.
- Runtime timestamps and timing measurements are intentionally nondeterministic. Policy, metric, certificate, candidate, and reranked-ID values reproduce under the same manifest and seeds.
- The repository is now connected to GitHub; Slurm runs remain user-executed and their logs should be retained alongside commit IDs and environment versions.
