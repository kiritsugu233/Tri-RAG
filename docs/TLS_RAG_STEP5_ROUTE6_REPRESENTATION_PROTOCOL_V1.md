# R6-S0 representation and integration screen v1

Namespace `tls-rag-route6-representation-screen-v1`. Frozen scientific contract
for the exact exploration §55.7–55.8 approval on 2026-11-03:

> 批准九路径、测试同步及一次S0（推荐）

The six introduced files and three narrow registry/test edits are the approved
scope. Old code, frozen protocols, test assertions and scientific artifacts do
not change. This screen deliberately explores representations outside the old
single-Gaussian MVP. It does not change that MVP or its scientific claims.
The JSON config is the complete enumerated configuration/seed/resource table.

## Inputs, roles and truth

Reuse the original, read-only Hotpot–Harrier HDF5, full5,233,329×640 storedFP32
train vectors and original128 learn queries. Source is13,688,159,632 bytes,
SHA256 `08f1c68cc8bb1f24a29cb64d9ffc20ccc6a91c06d286538e4e659d4afb648114`,
HF revision `a91cc76a3ab66b0353032e8427b9e34dcee22ded`.
Original roles SHA256
`f1a83a8c9aa008e63b52ae29a55fb45f60d200ec2c4c228866f21ad0885b4e08`.
Paths are frozen in the config. No source copy, download or embedding change.
Full SHA before and after scientific work is charged to this run.
Native HDF5 attribute is `distance=normalized`; its fixed VIBE distance function
is normalized cosine. Read public snapshots frozen at VIBE
`7ac76c1f8320a411ce7c9cb3791bdd7f2349b30c` under the preparation evidence.

Train and query embeddings already satisfy the2^-16 norm gate. Do not normalize
a second time. Native target is stored-FP32 `clip(1-dot(q,x),0,2)`, raw Faiss IP,
with final reranking by that same distance. Stored-vector FP64 L2 is a separate
audit and cannot replace this metric. Do not normalize projected embeddings.

Open only the128 original learn rows and corresponding learn_neighbors rows,
plus the full corpus. No test/neighbors/distances/avg_distances payload reads.
Metadata is allowed. Published learn IDs must be unique, in range and have k100.
Learn has no published distances. Recompute distances of those100 IDs and use
the maximum as cutoff. Primary recall counts returned100 items with distance
<=cutoff+1e-3 and divides by100. Strict published top100 ID retention is separate.
This is the same VIBE formula on published learn truth, a development metric,
not the official1000-query test score. Actual truth-generation backend is unknown.
Native FP32 full-corpus IP and the closed B0 FP64 full-corpus top100 are compared
with published truth. Invalid identity or cutoff disagreement aboveepsilon stops
with evidence. Keep both identities; do not silently replace published truth.

The prior FP64 reference is reused, not newly independent: B0 v2 index SHA256
`4f8493b4a95e61adc171e925da8ace813b4adfacc671fd4564e1871470d5f36d`.
Verify every indexed product and each exact role, full-corpus G ranks1..100 and
old IP/L2 top100 agreement. Its312-point local Gram is never substituted for the
full corpus. Stable data-boundary IDs use sourceSHA/dataset/zero-padded row;
compact candidate arrays explicitly reference the saved corpus ID-map artifact.

First64 original geometry_discovery select all configurations, budgets and
backend parameters. Last64 geometry_check audit frozen choices. All were opened
for previous geometry development, and the entire learn pool is development;
there is no independent certification or text-level independence claim. Check
records never tune or choose a different reported winning configuration.

## Representations and training

Sixteen main configurations: raw640; Gaussian128/256; samplePCA128/256;
PCA-retained plus Gaussian-residual32+96/64+192; rawPQ32/64 bytes;
OPQ640→PQ32/64; PCA256→PQ32/64; Gaussian256→PQ32/64; hybrid64+192→PQ64.
Four extra Gaussian/hybrid128/256 second-seed cases are quality-only diagnostics;
never select the favorable seed. Four more byte-matched cases Gaussian8/16 and
PCA8/16 have32/64-byte FP32 payloads and real costs. Total24. Equal payload does
not imply equal total memory. k100, M grid100/200/400/800/1600/3200/6400.
No per-query dimension/representation change.

All training uses the same65,536 corpus-only rows, PCG64 seed2026110351,
sampled without replacement and sorted, explicit frozen row-map/hash. Fit no
representation on queries or labels. PCA uses sample FP64 centered covariance,
descending eigenvectors with deterministic pivot-sign convention, no whitening.
This is samplePCA, not full-corpus PCA or a nearest-neighbor theorem.
Gaussian uses PCG64 standard256×640 draws, shared dimension prefixes scaled
by1/sqrt(m), main seed2026110353, stability2026110354. Hybrid independently draws
a standard192×640 residual matrix from PCG64(SeedSequence([Gaussian seed,1])),
a domain-separated residual stream independent of the retained deterministic coordinates, then stores
A=[U_r^T; G_s(I-U_r U_r^T)] and meanmu. Total coordinates r+s. The retained and
residual parts share original input geometry; no independence/equicorrelation
model is assumed. The whole hybrid is not called TriLaw(beta,rho,r+s).
All transform matrices deploy as storedFP32, centered as specified, and actual
model/matrix/codebook bytes and hashes are saved. Reference audits operate on
these stored coordinates, not ideal transforms.

PQ8-bit subcodes and coarse Faiss1.8 training niter25,nredo1, seeds2026110355
and2026110357. OPQ outer50/firstPQ40/laterPQ4, initial random rotation1234
per frozen Faiss1.8 source. Final PQ training25. OPQ is a full rotation, not an
iid Gaussian matrix; it has no deployment centering bias. Save each actual
trained model. Never weaken OPQ iterations to make Gaussian look better.
A training component exceeding1200s ends the run with incomplete evidence.

## Reference and real backends

Full-corpus FP64 reference: raw IP; other stored linear coordinates L2; PQ ADC
with exactly the fitted codebooks/codes. Independently compare FP64 ADC with
small decoded chunks. CUDA references use bounded FP64 reconstruction and
full scans, CPU fixtures independently sum ADC tables/direct distances.
Squared norms/dots are finite precision and not interval-certified; no B0
unit-vector error envelope is reused after transformation. References are audit
costs and decoded reference vectors are not a compressed deployment backend.

Actual Faiss CPU1.8 Flat/PQ, Torch2.8 CUDAFP32 full linear scans, and CPU IVF
Flat/IVFPQ nlist1024, nprobe16/64/256/1024, max_codes0. No GPU Faiss is installed;
Torch is identified as Torch. Original PQ and OPQ main distance is IP, with
same-codebook L2 ADC diagnostic. Other transformed coded cases use L2.
All return M then original-space top100 reranking. Stable cutoff ties use
canonical row IDs and expand the actual backend request beyond M+1 if needed.
All competitors participate. No HNSW or20k-subset shortcut.

Select <=7 IVF identities from discovery: always raw; best qualifying PCA;
best non-Gaussian32/64-byte coded cases; <=1 each uncoded Gaussian, hybrid,
and Gaussian-family coded case. No qualifying M means grid failed, not budget
expansion. Among qualified points use minimum mean complete latency, then lower
resident bytes and lexical ID. Primary .95/.99 targets; strict-ID frontier is
separate. Keep all completed points. Same-byte frontiers use actual32/64-byte
payload instances. Hardware profiles remain separate.
Each IVF's FP64 reference reconstructs its own coarse+residual codebook inFP64;
its actual FP32 fullprobe is another saved audit. Never compare IVFPQ against
an unrelated global IndexPQ and call the difference pure ANN loss. Persist
coded indices. For Flat indices save coarse centroids, list offsets/explicit IDs;
rebuild their float payload from source+matrix rather than persist raw replicas.
Oracle rebuild must verify coarse centroids; rebuild cost is audit cost.

Per-query saved candidates and G membership transitions retain lost/rescued IDs.
Different topM sets need not be nested. Signed recall changes are not clipped
into a fictional additive decomposition of nonnegative losses.

## Timing, memory and lifecycle

Batch1 main deployment. CPU profile Faiss14, five BLAS env limits1. All methods
share that profile. GPU complete linear pipelines include transform, transfer,
search, ID return, original rerank and synchronization. CPU PQ vs GPU Gaussian
is a hardware comparison, not a representation-only advantage. Corpus placement
is separate from per-query transfer. No label enters the deployment search API.

Quality can scan maxM once then evaluate prefixes. Timings always make a new
real call for each M, including query transform; never charge free cached prefix
expansion. Each qualified representation/backend selects at most two M values
from discovery (.95/.99). A grid with no qualified M gets M100/6400 cost-only
diagnostics, which cannot become eligible quality claims. Two warmups and three
actual repeats per query/M; fixed seed2026110359 mixed query/M schedule and
predeclared shuffled configuration visit order after raw truth gate. Repeated
returned IDs must match the corresponding quality interface or fail with evidence.
Seed-only diagnostics have no latency/selection eligibility. CPU1 sensitivity
uses the16 discovery IDs with smallest SHA256(seed2026110361,ID), is supplementary,
and cannot affect selection. Save raw samples, mean/median/p95, never fastest only.

Logical deployment bytes include raw13,397,322,240B for final reranking plus
search payload, matrices/codebooks/IDs/IVF lists and actually required copies.
Existing raw disk is not zero. Physical peak tree RSS/GPU includes audit buffers
and duplicates; it is reported separately from logical deployment memory.
No complete full-float corpus representation archive. Save model/code/list/IDs
and compact per-query arrays within8GiB. Cold input/load and source-verification,
sample/PCA fit, representation/PQ fit, corpus transform/encode, index build,
write/load, and audit reference are reported separately. Input acquisition/B0
cost is historical sunk, while new S0 work is paid. Report P+Q*T for Q=1,1e3,
1e5,1e6 and positive break-even intersections; these are measured-cost accounting,
not a serving-scale benchmark or an unlimited-amortization claim.

## Free-budget oracle, not Tri deployment

Select <=3 Gaussian-family CPU pipelines using discovery only; representation,
dimension, seed, backend/nprobe stay fixed. Measure all seven M actual calls,
two warmups/three repeats, including transform/search/rerank/transfers. Pilot
and controller are free only in this offline diagnostic. No interpolated or
free-prefix t(q,M). Save coefficient tables separately for discovery/check and
primary recall/strict ID. No label is passed to actual search.

Minimize mean sum x(q,M)t(q,M), constraints sum_M x(q,M)=1 and mean quality>=R.
LP permits fractional x and is a randomized clairvoyant optimistic lower bound,
not a deterministic/deployable policy. SciPy1.13 milp uses binary x; save status,
feasibility, primal objective, dual/gap. Solver timeout is unresolved, not an
optimum. <=60s each optimizer, <=300s total; save incomplete solver status on
cap. Also solve query-independent fixed mixtures and compare eligible frozen
fixed controls on matched queries/metric/hardware. A mean constraint is not a
per-query guarantee. Fixed-control eligibility is frozen on discovery.
Positive measured headroom does not prove online inputs can recover it. <5%
gaps and timing variability remain small/unresolved. No formal superiority or
certification; no bootstrap significance claim. If the optimistic lower bound
cannot help, that measured action space does not justify a more complex budget
predictor. Missing coefficients or incomplete runs prove no negative conclusion.

## Execution and delivery

Use existing pinned environment (NumPy1.26.4/SciPy1.13.0/h5py3.11.0/Faiss1.8.0/
Torch2.8.0+cu128 on A800). No installs/upgrades. All local CPU tests/full old
suite must pass, then canonical commit/push/remote HEAD verification, clean Linux
ff-only sync, new CPU tests and an actual tiny CUDA receipt. Receipt freezes
source/config/environment and binary identity; no CPU fallback for the real run.

Stdlib-only parent starts global clock before heavy imports/preflight, creates
an exclusive new output, starts one owned worker/session and monitors <=1s
nominal cadence, with bounded external-tool timeouts. No heavy parent CUDA context.
Limits: scientific7200s, global9000s including preflight/verification/closure;
RSS64GiB, owned GPU48GiB, CPU14, artifact8GiB/new disk12GiB, free disk>=16GiB.
All sources/config/versions/binaries, hardware/precision/thread settings are
fingerprinted. All random seeds explicit. Every bounded scientific write has
prebudget checks. A sampling/shutdown overshoot is possible and reported; no
soft-limit result is called full completion. Monitor failure kills only the
owned session and never reports success. Preserve every failed/partial artifact.
No automatic retry, resource expansion or cherry-picked configuration/query.
Before handoff close all logs/receipts, index every file (index itself excluded),
verify inventory/bytes/SHA. Final status distinguishes completion and unresolved
missing stages. Node remains on; only owned science exits. Nothing authorizes
new Gaussian MC, online pilots, macro-tail solvers, light Tri formulas, official
test, independent RAG gains, third-stage controller training or a second S0.
