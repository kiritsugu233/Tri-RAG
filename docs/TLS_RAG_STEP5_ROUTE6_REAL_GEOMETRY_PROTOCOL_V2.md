# R6-B0 v2 frozen real-geometry audit

2026-11-03. Namespace `tls-rag-route6-real-geometry-v2`.
Actual consent to exploration §51.3–51.4: **“批准第51节九路径、保留并停止原轮、测试同步及一次v2”**.
The original CPU run already stopped naturally at its compute deadline; preserve
its complete/partial artifacts and failure. No operator termination is now needed.
This is one new GPU audit, not a retry that changes the old receipt.

## Purpose and unchanged inputs

Audit general, heterogeneous displacement geometry on the same complete stored
Hotpot–Harrier source and the exact original128 learn development queries.
Do not assume independent errors or equicorrelation. No Gaussian projection,
Monte Carlo, ANN, budget decision, evidence evaluation or official test payload.
The earlier CPU result was24/128 discovery queries, not a completed audit.

Frozen config is [the v2 JSON](../configs/tls_rag_step5_route6_real_geometry_v2.json).
It binds source bytes/SHA, original57-file closed index SHA, roles SHA,
CPU scientific commit9f963f9 and24 reference queries. No input download,
copy, re-encoding, query reselection or old-run write. Before computation verify
every indexed old artifact, then verify the complete source SHA before/after.
Roles remain64 geometry_discovery /64 geometry_check, all development only.
The inspected learn pool cannot serve as future independent certification data.

Stored float32 vectors are cast exactly to FP64. No renormalization, jitter,
duplicate removal or sign folding. Keep k100, original seeds, dense G100/B148,
five far strata with16/12/12/12/12 samples, all populations and joint inclusion
probabilities. Reuse the v1 pure sampling, signed Gram, query-direction residual
and local diagnostic functions without changing their bytes or interpretation.
Original row-ID ordering is stable string-ID ordering within the same dataset.
Original-space FP64 IP is a separate diagnostic, not official FP32 truth.

## Batch calculation and comparison repair

Read the corpus once, verify stored norms within tau=2^-16 and finite values,
and hold its `[5233329,640]` FP64 tensor on the single existing A800.
The tensor uses26,794,644,480 bytes. Query batch16, corpus block8192.
Calculate L2 using direct coordinate differences and squared-sum in FP64;
calculate negative IP separately in FP64. Do not substitute norm-dot L2.
Hold only one batch of host L2/IP arrays (about1.34GB combined); do not save
the full128-by-corpus distance matrix. Release the batch before the next one.

Freeze absolute CPU/GPU comparison envelope **E=2^-36** for both metrics.
For each full score vector, sort the GPU values and join adjacent entries whose
gap is <=2E. Refine *every* connected component, including transitive chains,
on CPU using exact FP32-to-FP64 source coordinates retained on the GPU.
Within a component sort CPU scores then row ID. Gaps >2E cannot reverse if the
assumed per-score error envelope holds. Check every recomputed discrepancy;
any error >E, nonfinite score, device/precision mismatch or resource violation
stops the run. Do not increase E or silently fall back to approximate ranking.
Preserve zero distances and duplicate/tied rows. Recompute reporting cutoff
values and independently check both metrics on all selected local points.

### Conservative floating point rationale, with explicit limits

Let u=2^-53 and gamma_t=tu/(1-tu), d=640, ||x||,||q||<=1+tau.
There is no overflow: |x_j|,|q_j|<=1+tau. Exact FP32 inputs and their nonzero
differences/products lie within the normal FP64 exponent range, so gradual
underflow is not a limiting case for these operations. Under ordinary IEEE64
round-to-nearest subtraction/multiplication/reduction (or correctly rounded
FMA), a deliberately conservative direct L2 error bound for *each* CPU/GPU
evaluation is gamma_(d+4) ||x-q||². Thus their mutual discrepancy is at most
`2 gamma_644 * 4(1+tau)^2 < 5.73e-13`, below E≈1.4552e-11.
For signed IP use an absolute bound, including cancellation:
`2 gamma_644 sum_j |x_j q_j| <= 2 gamma_644 (1+tau)^2 < 1.44e-13`.
The operation-count bound allows a serial reduction, so a shorter reduction
tree does not demand a smaller envelope. E is fixed before inspecting v2 ranks.

These are bounds *conditional on the stated arithmetic model*, not a formal
verification of NumPy/BLAS/CUDA kernels, compiler transforms, drivers or all
hardware behavior. All products therefore set `roundoff_certified=false`.
Freeze existing NumPy1.26.4/SciPy1.13.0/h5py3.11.0/Torch2.8.0+cu128, Python,
Torch extension SHA, CUDA version, physical GPU UUID/name/driver, deterministic
algorithm and IEEE/TF32 settings, cuBLAS workspace and five BLAS thread limits.
The parent and owned worker must match the same lock. Configuration, roles and
runtime lock enter the scientific fingerprint; source maps also bind all code,
tests, protocol, registry and inherited helpers. No old cache migration.

## Validation and scope of evidence

Before real computation: full CPU/offline repository regression including
unchanged old suites, protection inventory/hash check, canonical commit/push
with remote HEAD verification, clean ff-only node sync, Linux new CPU tests.
Then one deterministic tiny *actual CUDA* comparison against the v1 CPU full
L2/IP order, including zero, duplicate and near-tied coordinates and a partial
query batch. Its source/config/runtime-bound receipt is required to launch.
CPU unit tests use an explicitly named synthetic fixture backend; it is never
exposed as a production CLI fallback and does not establish CUDA correctness.

On all24 previously completed CPU queries require identical selected G/B/F
IDs, full-corpus ranks and strata; compare U, signed Gram, D, query loadings and
residual Gram with absolute tolerance2^-40. No additional CPU science run or
selection of an easier reference subset. Matrix comparison supports only this
released prefix, not every unseen full-rank relation or macro omission tail.
Input/test-label traps, SHA mutation, interval chains/ties, fixed roles, batch
tail, source-bound tiny receipt and real owned-child stop are CPU regressions.

## One-run supervision and costs

New exclusive directory on `/root/autodl-tmp`, old source/results remain on
`/root/autodl-fs`. Reuse the existing environment, no installation or downloads.
One physical/visible A800, no unrelated GPU compute processes, >=40GiB free
device memory, >=24GiB free disk at start. Limits: compute1200s, total1800s,
aggregate process-tree RSS4GiB, GPU process memory40GiB, artifacts1GiB,
new disk20GiB; CPU allocation14, five BLAS thread variables all1.
The compute timer includes input reference checking, both complete-source SHA
passes, source read/norm/H2D, kernels, synchronization, D2H, sorting/refinement,
geometry and saved products. The total timer additionally bounds supervision.
Resource sampling is discrete; record possible stop/measurement overshoot.
Only the owned worker session is terminated on failure/limit; no node shutdown.

Record corpus preparation and each batch's shared kernel/transfer costs once.
Per-query distance time is explicitly amortized; its ordering/local processing
time is separate. Batch wall time and global elapsed include comparison, JSON
and storage work not fully represented in per-query timing fields. These are
offline geometry-audit costs, not RAG online or projection-serving latency.
Retain failures, progress and all closed products; write delivery SHA index
only after logs, progress and final execution receipt close. No automatic retry,
query replacement, resource enlargement or additional scientific experiment.

Replay after setting a reviewed v2 commit, existing source and output variables:

```bash
cd /root/autodl-tmp/Tri-RAG-git
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
route6_python=/root/autodl-fs/gaussian-benchmark-space-v1/env/bin/python
"$route6_python" scripts/check_code_protection.py
PYTHONPATH=.:src "$route6_python" -m unittest discover -s tests -p test_tls_rag_step5_route6_real_geometry_v2.py
"$route6_python" scripts/run_tls_rag_step5_route6_real_geometry_v2.py device-check --device-receipt "$route6_device_receipt"
"$route6_python" scripts/run_tls_rag_step5_route6_real_geometry_v2.py run --device-receipt "$route6_device_receipt" --run-dir "$route6_output"
```

Completion means128 saved query records,64/64 roles and24/24 old CPU matches,
final source identity, five geometry diagnostic SVGs/CSV, complete result and
closed receipt/index under the frozen caps. A failed target remains a failure.
This does not validate factor independence, omission probabilities, online joint
information, static dimension selection, lightweight Tri formulas or real RAG gain.
