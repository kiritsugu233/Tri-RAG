# Local CPU exact-distance reuse probe v1

Approved scope: on 2026-10-26 the user replied to the four-path reuse proposal:
“那就先做本地CPU的距离缓存＋延迟选取基准，暂不接预测器。检查结果一致性、完整耗时
和内存，并给固定策略相同优化。” This authorizes the new probe, its CPU regressions,
profiler and this contract; it does not authorize changing existing scientific sources.

## Frozen diagnostic scope

- Synthetic external queries only, no model, SSH, GPU or archived real-data roles.
- N=1,000,000; original dimension384; projected dimension128; float64 arrays.
- Corpus/query seeds55251/55253, dense Gaussian projection seed55117, variance1/128.
  Normalize original corpus/queries once; do not renormalize projections.
- Exact direct-difference squared L2 and stable string-ID ties use the existing
  `ExactScan`, `stable_top`, `PrefixSession` and projection helpers unchanged.
- Grid32/64/128/256, block8192, k_ctx5. Each scripted path stops at one grid endpoint.
  Endpoints are predetermined labels, not decisions or an oracle quality claim.
- Twelve queries, three repeats; one excluded warmup per case. Shuffle the28
  cases independently per query/repeat with seed55259. Warm projected input and
  the reference top256 original rows for each query before interleaved timing.
- Threads: OPENBLAS/OMP/MKL/VECLIB=1. Full pipeline timing includes query projection,
  scan, cache allocation/write, selection, candidate-prefix handling, new-only
  original scoring, reranking and session cleanup. Fixture generation, hashes,
  common ID maps and correctness validation occur outside this timer for every method.
- Single-process hot-memory structural benchmark, not server/I/O/cold-start latency.
  Query identity is the statistical block; report raw repeats, medians, means,
  empirical p95 and paired query-block bootstrap intervals (2,000 replicates,
  seed55261). Small-sample p95 is descriptive, not a tail guarantee.

## Seven cases per endpoint

1. `stream_fixed`: unchanged blockwise exact search directly to finalM; score once.
2. `stream_max`: unchanged blockwise search to256, then the32..M prefix schedule.
3. `cache_reselect_fixed`: one full distance cache, one global stable selection toM.
4. `cache_reselect_lazy`: same cache, global stable selection at every32..M step.
5. `cache_partition_fixed`: same cache and a row-order buffer, one partition toM.
6. `cache_partition_lazy`: reuse the row-order buffer; partition only the not-yet-
   selected suffix at each expansion. Resolve cutoff ties by stable IDs before
   sorting the new prefix. This does not promise that suffix processing is sublinear.
7. `cache_max`: global cached selection to256 once, then prefix schedule32..M.

Fixed and adaptive variants share identical distance/selection implementations.
All methods compute N projected distances once; all score finalM original vectors.
No predictor, Tri feature, label, realized recall or query-specific endpoint choice.
The two maximum-first controls isolate eager selection from caching/backend effects.

## Correctness and access invariants

- Match reference candidate rows, distance bit patterns, reranked/context IDs and
  all original-distance bit patterns at every scripted prefix, not just finalM.
- Verify exact distance count=N, scans=1, original score count=finalM; expanded
  schedules retain prefixes. No cached projected vector distance is recomputed.
- Test duplicate/equal distances, nonsorted string IDs, zero, near ties, nonfinite
  inputs, bad/repeated/decreasing budgets, failed state reuse and query mutation.
- A session owns one copied query. Only current-prefix copies are returned; no
  feature/controller API is present. Another query requires a new session.
- Prefix buffers returned to callers are read-only and cannot corrupt private
  cache/order state. Data/model/cache bindings of all earlier studies stay unchanged.

## Memory and accounting

Latency samples do not run tracemalloc. In separate fresh child processes, measure
each of the seven cases to256 (the longest scripted schedule), with identical
preloaded inputs. Record baseline and peak process `ru_maxrss` with platform units,
and newly traced peak allocation including NumPy buffers during the complete call.
Calibrate NumPy tracing with a known8,000,000-byte allocation before trusting it.
The difference of lifetime RSS high-water marks is an imperfect incremental RSS
measure; report the absolute values and do not present a zero delta as zero cost.
Traced allocation excludes the preexisting common corpus, projected array and ID map.
Record retained-buffer byte counts separately; include row buffers, partition/gather
temporaries, Python objects and cleanup in measured peak/time, rather than claiming
the8MB distance buffer is total overhead. Concurrency is one; no extrapolated
concurrent peak is measured.

Inputs are generated once into a new temporary directory; the manifest includes
all array SHA-256 values and the configuration/source binding. Refuse output reuse,
source/config mismatches, nonfinite results or equality failures; retain failures.
No old scientific bindings or outputs may be refreshed.

## Interpretation and delivery

This asks whether exact reuse/lazy selection has structural benefit after fair
fixed controls, before paying for a predictor. It does not estimate retention,
choose a deployment policy, reopen certification or establish Tri's paid value.
An improvement common to fixed and adaptive paths is a backend optimization,
not evidence for query adaptivity. If lazy paths lose to the same optimized fixed
endpoint, report the expansion overhead. No automatic follow-on experiment.

New paths: `mechanism/tls_rag_step5_reuse_probe.py` and its regression file are L0;
this protocol is L0; `scripts/profile_tls_rag_step5_reuse_probe.py` is L1 and inherits
the protocol. The original approval is recorded verbatim in the protection registry.
Run the new CPU tests and existing relevant scientific regressions, protection and
diff checks. Preserve raw local receipts plus a versioned summary/evidence; commit
and push the continuing Step5 branch and verify the GitHub HEAD.
