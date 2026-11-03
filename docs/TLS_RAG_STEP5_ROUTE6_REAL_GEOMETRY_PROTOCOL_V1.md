# R6-B0 real-geometry protocol v1

Frozen implementation scope: exploration §49.2–49.7. Actual user consent:

> 批准第49节九路径、环境准备、测试同步及一次B0

This protocol authorizes one bounded existing-node B0 after full local regression,
GitHub source synchronization and Linux tests. It does not reopen M0/R6-A or old
cert/test roles. The nine approved paths are the two new real_geometry/real_audit
modules, this protocol, config/test/runner, test membership and two protection
registers. L1 handoff documents report actual progress separately.

## Source and roles

Use only the complete HotpotQA–Harrier file at HF revision
`a91cc76a3ab66b0353032e8427b9e34dcee22ded`,13,688,159,632 bytes,
SHA256 `08f1c68cc8bb1f24a29cb64d9ffc20ccc6a91c06d286538e4e659d4afb648114`.
Require train float32[5,233,329,640], metric `normalized`, dimension640.
Copy the preserved10,703,863,808-byte partial into a new exclusive source file;
range-fetch only its remainder,206/Content-Range identity required. Original and
failure receipts remain unchanged. A valid complete SHA precedes any payload use.
Transport fallback is allowed only while opening an official/fixed mirror range;
body failures stop without automatic retry or a full-download fallback.

Payload allowlist is train/learn. Other dataset shape/dtype metadata may be read;
test, neighbors, distances and learn_neighbors payloads must stay closed.
The whole inspected learn pool becomes development data. Select128 distinct
canonical FP32 vector-SHA groups (signed zero unified), representative lowest row,
ordered by SHA256(str(role_seed)+'|'+group_sha). First64 discovery, remaining64
check. Seed2026110301. This is vector duplicate isolation, not semantic independence.
Original text IDs/evidence mapping are unavailable; no semantic or evidence claim.
Stable strings bind the full file SHA, dataset and eight-digit row. Explicit
source-domain ID-map and selected row/string maps accompany all exported arrays.

No vector renormalization or data mutation. Stored norm error
`abs(||x||_2-1)` must be at most2^-16 for inspected train/learn vectors.
Nonfinite/out-of-range sources fail. Zero displacement, duplicates and ties are
recorded as degeneracies, without epsilon/jitter or replacement queries.

## Exact geometry and sampling

Scan the complete train in blocks65536 with FP64 direct squared differences.
Each query has one N-vector of distances, not a Q-by-N matrix. Sort by L2 then
stable row-ID order. Compute separate FP64 raw-IP diagnostic ranks; official
FP32 IP truth is preserved and not relabeled as FP64 L2. Native/L2 differences
are reported. B0 alone does not validate official machine recall.

G includes all100 true L2 neighbors. B includes ranks101..116 and M±16 for
M200/400/800/1600:148 distinct points. F covers every remaining competitor:
16 from gaps through rank1616;12 each from1617..floor(.01N), then to.1N,
then.5N, thenN. Uniform sampling without replacement, seed2026110302 derived
per stable query ID.312 points total, cap320. Insufficient populations stop.
Save n_h,s_h and single/joint inclusion probabilities. Same-stratum pair rate
is s_h(s_h-1)/[n_h(n_h-1)], not the product of marginal rates. Use design weights
for population descriptions; never turn weighted means into omission guarantees.

Save displacements, D, signed Gram, query-direction loading, residual Gram,
pair inclusion, selected row-ID/rank/stratum maps, block distributions and costs.
For e=q/||q||, u=x-q, a=u·e and r=u-ae, G=aa^T+H exactly, H=rr^T.
Unit vectors give a=-D/2; use actual a for floating stored data. Pi e is independent
of all Pi r; residual points remain generally correlated through H. No conditional
Bernoulli/Poisson-binomial independence follows. Report residual signed angles.

For general G, Cov(S_i,S_j)=2G_ij²/m and
Var(S_j-S_i)=2(D_i²+D_j²-2G_ij²)/m. Distance-error correlation is not flip correlation.
Report local correlation spectrum and predeclared rank1/2/4/8/16 plus diagonal
residual diagnostics, including near-boundary variance errors. These are local,
boundary-enriched descriptive fits, not full-corpus spectrum or macro-risk models.
Do not jitter singular matrices, discard signed geometry or silently repair data.
Floating results/PSD eigendiagnostics are not roundoff certified.

## Limits, environment and delivery

Acquisition1800s/new received bytes4GiB; compute1200s; total3600s.
CPU at most14 threads (runner uses one BLAS thread for reproducibility),
process-tree RSS4GiB, new disk20GiB, preflight free24GiB,
scientific/log products1GiB excluding the separately bound input file.
Local recovery at most512MiB; full source remains on server.
No GPU calls, Gaussian samples, MC, ANN construction, policy fitting, m selection
or formal certification. All offline reads/hash/copy/sort/geometry/storage costs
are included. Polling/termination can overshoot; receipt reports measured peaks.

Reuse `/root/autodl-fs/gaussian-benchmark-space-v1/env/bin/python` with
NumPy1.26.4/SciPy1.13.0/h5py3.11.0; no server installation/version changes.
Local h5py3.11.0 uses only the approved hash-verified cp39/macOS arm64 wheel in
`/private/tmp/route6-b0-h5py-20261103`, no dependencies installed or upgraded.
Full CPU/offline tests include tiny HDF5, forbidden payload traps, independent
direct-math/order references, range/SHA failures and actual owned-child stopping.

Run only from a clean GitHub-synchronized canonical Linux checkout. Source map,
environment/config and owner PID are frozen before the child starts. Output root
must not exist; no overwrite/reuse. On failure retain partial output and explicit
failure/monitor status; never substitute completed queries for128. Final delivery
index is built after progress/log/execution close, includes source_freeze and all
products, excludes itself and separately input-bound source/. No stale append hashes.

Replay after tests and preflight (use a new run directory):

```sh
cd /root/autodl-tmp/Tri-RAG-git
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
/root/autodl-fs/gaussian-benchmark-space-v1/env/bin/python scripts/run_tls_rag_step5_route6_real_geometry_v1.py preflight
/root/autodl-fs/gaussian-benchmark-space-v1/env/bin/python scripts/run_tls_rag_step5_route6_real_geometry_v1.py run --run-dir /root/autodl-fs/route6-B0-v1-NEW-COMMIT-TIMESTAMP
```

Quality/cost headroom, correlated macro-tail accuracy, deployable joint inputs,
static dimension selection and lightweight scalar advice remain unvalidated.
