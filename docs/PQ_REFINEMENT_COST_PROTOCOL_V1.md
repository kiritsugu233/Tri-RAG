# C1 PQ refinement cost protocol v1

2026-11-05. Scientific namespace: `pq-refinement-cost-v1`.

Actual user instruction: **“服务器已开启，继续”**. This resumes the previously
presented and deferred exact eight-path C1 scope in design-plan section11.
The earlier actual answers “暂缓C1，保留C0结果” / “暂不开机” remain historical;
they are not rewritten as approval. Only that same C1 scope resumes. Gate2,
Tri, training, fresh query roles, new representations and GPU work remain closed.

## Fixed inputs and dependencies

Use the original S0 source, full SHA/bytes/layout, thirteen products and closed
index SHA listed in `configs/tls_rag_step5_pq_refinement_cost_v1.json`. Never use
partial-bulk-transfer. Map only contiguous native train bytes read-only, after
one full source SHA check; do not open learn_neighbors/test/cert. Saved queries,
roles and candidates are already observed inputs; no new quality labels are read.

Match the C0 hash-selected16 discovery+16 check CPU query identities exactly.
Keep OPQ64's actual IVF index/residual code, original transform, IP search,
nprobe64/256, upstreamM6400, nativeFP32 clip(1-dot,0,2), k100 and explicit-row
ties. The original S0 search/rerank functions and four source bindings remain
unchanged. Do not refit, rebuild, switch codebooks or normalize again.

Pin existing NumPy1.26.4/SciPy1.13.0/h5py3.11.0/FAISS1.8.0. Record hardware,
CPU allocation and binary identities; FAISS14, BLAS1, batch1. Warm resident
original vectors; charge source verification, placement/prepayment separately.
Do not drop caches or create artificial SSD/remote/GPU workloads.

## Operations

1. `baseline`: actual original query transform/search and frozen rerank.
2. `observed`: same operation order and native functions with disjoint block
   clocks: ID validation/unique, candidate gather+copy, full-dimensional score
   and clip, stable selection, output100 re-gather and score. Release buffers
   at the original points. Compare full output to the baseline, disclose the
   observed-minus-baseline whole-call difference. Phase sums do not replace
   measured whole-call clocks.
3. `prepaid_input`: gather a candidate buffer beforehand, then score/select and
   return exact native distances from that buffer. Query/candidate/source-object
   binding is checked inside the callback. This is a paid-input oracle diagnostic,
   not a deployment gain or zero-memory-traffic implementation.
4. `prepaid_scores`: prepay candidate scores plus native output scores, then
   validate/select/return. All prepayment time/workspace is recorded. The two
   oracle savings are not additive read/compute identities.
5. Nine `subset_*` modes: prefix, spread and salted-ID-hash positions, with
   1600/3200/4800 survivors; preserve upstream order. Salt2026110411. Position
   preparation is separately paid. Timed callbacks include actual subset-ID
   access/validation and original rerank. Search still emits the original6400.
   These address/batch diagnostics do not claim preserved top100 or new search
   budgets. No truth, predicted usefulness or check quality selects survivors.

Every actual full candidate sequence and full output must match sealed S0 CPU
records exactly, including native distances; reject and retain disagreement.
Original functions independently reference each diagnostic subset as well.
Do not alter tolerances, scores or ties to match a new CPU/library. An exact
check failure ends the one scientific run; no automatic retry.

## Clock/sample/accounting contract

Two warmups and seven formal calls per query/mode. Randomize modes within each
repeat and query order using seed2026110413; keep all modes paired on the same
queries. Baseline, observed and two oracles plus nine subset modes =13 modes.
Complete run:2 probes*32 queries*13 modes*7 repeats=5824 formal calls, plus
warmups and exact checks. Do not combine repeats as independent queries.

Hash checks, comparison/audit reads, /proc observations and output writes occur
outside the whole-call clocks, and their real run cost is retained separately.
Upstream search remains inside every measured full callback. Index loading,
prepayment and position preparation are never called free deployment costs.

For each role/probe, average seven calls per query, then summarize16 queries.
Use10000 paired query-cluster percentile draws, seed2026110417; roles separate,
no simultaneous-coverage or fresh-certification claim. Report mean/median/p95
of query means, actual whole-call savings and gamma5%/10%/20% fee windows.
Do not linearly infer survivor cost from its count. A diagnostic's numerical
speedup is not a validated quality-preserving cancellation policy.

Record source gather batch/row counts, unique IDs, logical payload and4KiB/64KiB
virtual page coverage separately from physical traffic. /proc/self/io and
minor/major faults are process observations, not DRAM bytes. No attributed
DRAM-byte counter is configured by this CPU probe; retain null rather than
manufacture hardware bytes from payload or cache misses. This limits claims
about physical transport, even if gather/copy wall time is measurable.

## Supervision, failure and closure

One serial worker; no GPU query or new data download. Expected CPU experiment
15–30 minutes, not an automatic cutoff. Manual supervision has300s observations;
elapsed time, memory/disk observations or observation failures do not auto-abort.
Explicit stop targets only the controller's owned live Popen child, with token,
UID, parent PID and Linux start tick. Malformed/unowned stop requests do not
stop compute. Control/compute errors preserve partial records and close evidence.

Run roots must be new. Record sources/config/commit/environment/input SHA and
source metadata. Keep all closed outputs, errors and failed comparisons.
After worker/log closure, index and rehash every file. Never overwrite S0 or
delete its archives. Source SHA is verified once on entry; read-only mapping,
post-run size/mtime and code hashes are additionally checked.

First gate outcomes may remain unresolved: real gather/copy and survivor cost
windows alone do not establish a safe online controller, actual DRAM savings,
or a positive Tri/RAG result. Report available cost evidence and missing parts;
do not start gate2/3 automatically. Source changes after real inspection require
new scope/evaluation as applicable; software checks do not imply scientific gain.

## Execution

Local tiny and full CPU tests precede commit/push/verified Linux synchronization.
Use the existing server Python with the committed wrapper; it sets thread
environment before importing NumPy/FAISS.

```sh
/root/autodl-fs/gaussian-benchmark-space-v1/env/bin/python \
  scripts/run_tls_rag_step5_pq_refinement_cost_v1.py run --root NEW_ABSOLUTE_RUN_ROOT
# Separate terminal, only when the user explicitly wants to stop:
/root/autodl-fs/gaussian-benchmark-space-v1/env/bin/python \
  scripts/run_tls_rag_step5_pq_refinement_cost_v1.py stop --root SAME_RUN_ROOT
```
