# PQ refinement geometry protocol v1

2026-11-05; namespace `pq-refinement-geometry-v1`. Actual consent:
**“批准第14节八路径，按顺序准备并执行”**, responding to the exact eight-path
proposal in design-plan section14. Direction: prepare locally, pass tests and
GitHub synchronization, then notify the user to power on a CPU server. No SSH
or real experiment before that handoff. Gate3 Tri and adapter/predictor training
remain closed. Original C1/source protocols and artifacts remain unchanged.

## Inputs and scientific identity

The new config binds20 original closed S0 products, the full source size/SHA and
native train mmap layout, the actual OPQ64 residual-IVFPQ index and original
source/config/test/runner/protocol hashes. Reuse the identical C0/C1 hash-selected
16 discovery+16 check queries; no new role or HDF5 truth/test payload. Saved
published truth/cutoffs/native global top100 are only evaluation inputs.
Primary nprobe64, high-quality nprobe256, upstreamM6400 and k100 remain fixed.
CPU FAISS14/BLAS1/batch1 and the existing four pinned package versions remain.
No new library, model download, GPU, main-codebook training or index rebuild.

Extract only requested IDs/codes from THIS index's inverted lists. Verify all
list IDs against original list_rows/list_offsets and the closed index hash.
Decode its own coarse centers plus8bit residual codes in native FP32 addition
order; one first requested row per list is checked against FAISS reconstruction.
The global FlatPQ codebook is never substituted. No direct-map mutation/writeback.

With forward OPQ matrix A, store shared FP64 inverse B, condition<=2 and
max-entry |A*B-I|<=1e-10. Canonical yhat is the actual FP32 decode-plus-coarse;
canonical real reconstruction is B*yhat+mean_FP32. The computational FP64
reconstruction approximates that real operator with an explicit error envelope.
Define e=x-xhat directly; no assumption that A transpose is its inverse or that
ADC equals q dot xhat. Online LUT uses B transpose*q and unrounded code sums,
enclosing the difference from native coarse-add rounding. That extra query
transform/LUT cost is paid. Original PQ candidate order remains fixed.

## Offline geometry and side summaries

Read only the union of candidates and the first32768 closed fit_rows corpus IDs.
Original values/reconstructions are processed in4096-row blocks; keep a temporary
RAM residual matrix, not a persisted full decoded corpus or Gram. Record every
candidate's native/real/ADC/reconstructed score, correction, radius, signed gap,
normalized gap, cutoff ties, zero residual and clip cases. Boundary bands are
absolute normalized gap<=.01/.1/1, with quantiles and per-query records retained.

Four-byte upward FP32 residual radius is shared norm baseline. Gaussian side
summaries are FP32 m8/16/32 plus the radius,36/68/132B per vector. Four seeds
2026110401–04; first is primary. Generate standard32-row G then explicitly scale
each R_m=G[:m]/sqrt(m), store each resulting FP32 R_m*e separately. No adaptive m,
best-seed selection, projection renormalization or multi-seed deployment fusion.

Same-payload strong controls: additional residualPQ32/64/128 one-byte subcodes
plus4B remaining-residual radius. Only the32768 existing corpus fit IDs train
these NEW side codebooks,20iterations/one redo, seeds2026110501/02/03. Main IVF
centroids/codes and query-independent training-row identities never change.
The remaining radius covers reconstruction arithmetic and the additional PQ
error, including its FP32 training/encoding input. All shared inverse, matrices,
side codebooks/scalars, IDs/workspaces, original vectors and laboratory resident
comparators are counted separately. Candidate-union encoding is not a measured
full-N build: report full5,233,329-row payload algebra, full-N encoding unmeasured.
FP64 LUT codebook copies and the explicit packed row-map are also charged. A
packed union can have a smaller cache working set than full-N summaries; timing
is a local-summary proxy, not a measured full-N deployment. ID-to-summary mapping
is recomputed from actual candidate IDs inside every controlled callback.
This Python prototype also copies original64B codes,4B list labels,8B row IDs
and8B build offsets per packed row. Their actual local bytes and conservative
full-N prototype equivalents are charged in addition to the4/36/68/132B summary
payload. A possible dense code/label layout is only an unimplemented accounting
alternative, not free reuse of the original IVF codes. Codebook FP32 and FP64
copies are counted separately. Equal summary payload does not imply equal total
memory, and no complete-system win may omit these copies.

## Bounds, probability and numerical scope

Norm control: real score upper q*xhat+||q||*r. SidePQ control adds its LUT score
and uses its remaining radius. Gaussian: Q=||q||>0,u=q/Q,h=R*u,T=||h||²,
ahat=h dot (R*e)/T. For ideal real Gaussian R, conditional on h,
ahat~Normal(a,(r_true²-a²)/(m*T)), a=u dot e. The nominal upper uses
q*xhat+Q*min(r,ahat+z_(1-alpha)*r/sqrt(m*T)), alpha=.01/100.
No candidate-independence assumption is made. Top100 union reasoning concerns
a fixed candidate set independent of R, over random R, not a fixed matrix's
unknown production query distribution. This is a baseline, not Tri novelty.

Actual code uses stored radii/sketches, norm.isf(alpha), and conservative
bounded-error envelopes. Unit roundoffs are u32=2^-24,u64=2^-53 and
gamma_n(u)=n*u/(1-n*u). Let S_y bound decoded coarse+code norms using maximum
coarse norm plus sqrt(sum of maximum subcentroid squared norms), and F_B=||B||F.
Residual arithmetic envelope is16*gamma_d(u64)*(2+F_B*(1+2*u32)*S_y+1);
stored radius additionally covers8*gamma_d(u64)*(computed_norm+1), then rounds
upward with nextafter FP32. Raw original row norm must not exceed2.

The LUT base envelope includes FP32 decode-add rounding and FP64 inverse-query
table/sum errors: Q_up*F_B*S_y*(2*u32+16*gamma_d(u64)), where
Q_up=||q||_computed*(1+8*gamma_d(u64)). SidePQ adds its FP64 LUT envelope.
Gaussian adds a sketch/storage envelope from ||R||F, the residual envelope,
stored radius and FP32/FP64 roundoffs before applying the nominal radius term.
The implementation retains the exact per-query envelope values. The final
native-score allowance covers gamma_d(u32)*Q_up*2 and FP32 subtraction/clip;
convert to native lower distances with outward nextafter. Equal bounds are read.

These formulas assume ordinary finite IEEE dot operations without overflow or
unbounded underflow effects. Synthetic high-precision/reference checks and real
native violations are audits, not a proof of vendor BLAS behavior or exact
Gaussian sampling from finite PRNG/rounding. **formal_certification=false**.
Do not fit guard tolerances to observed violations. Zero Q, T<1e-12, invalid
radii/nonfinite bounds fall back to full reads and their cost is retained.
Invalid query/source/ill-conditioned matrix is an error. Deterministic bound
violations or native full-read mismatches stop and retain evidence; Gaussian
tail events are counted as statistical outcomes, not silently repaired.

## Reading and full-clock comparisons

Primary scan reads first100 PQ candidates, then contiguous256-candidate blocks,
using lower distance bounds vs current native kth threshold. Read equality;
gather/score all chosen rows within each block, maintain stable row-ID ties.
No unread raw vectors, exact scores or true identities enter that controller.
Final100 output vectors are re-read/re-scored as in the unchanged baseline.
Each source ID is first-read at most once, apart from the explicit output read.

The separately named oracle supplies true candidate-top100 pilot identities,
actually pays those100 reads and the bound calculation, then uses the final
threshold and one remaining batch. Its scheduling is favorable and labeled;
it only diagnoses this rule, not a deployment policy/information-theoretic bound.
All37 modes remain: full baseline, forced full-read block control, three fixed
prefix1600/3200/4800 controls, norm+12Gaussian+3sidePQ scans and their16 oracles.
No post-check mode/seed/width tuning or automatic Tri stage.

Every measured call repeats the original full upstream search. The entire
callback pays candidate validation, mapping checks, dual query/LUT/sketch work,
summary access, bounds, scheduling, actual original gathers/scores/selection and
output. Exact masks are never prepaid for a deployed scan. Record bound-only
aux and scan clocks separately, with total callback authoritative. Saving fewer
rows is not inferred proportional latency/traffic. Trace construction inside
the callback is paid; persistence/hashing/quality and independent conformance
reads are outside, with overall experiment elapsed retained.

Two warmups/seven formal calls per query/mode, schedule seed2026110511. Whole
production grid gives16576 formal calls, not16576 independent queries. First
formal trace is saved; every repeated read/output signature must agree. Calls
retain actual unique row/batch/payload counts, returned IDs/distances and times.
Physical DRAM/SSD/requests remain null. Use the same query-cluster10000-draw
bootstrap seed2026110513 and separate roles, gamma5%/10%/20%; marginal intervals,
not fresh certification or simultaneous coverage. p95 is of query means.
Report candidate-top100 losses and benchmark/strict/native-global quality
separately. Never call a speedup a benefit without matching its quality/cost.

## Readiness, supervision and delivery

Local new tests, full old689 regression, FAISS synthetic tiny, protection/diff
checks and verified GitHub push precede notification to power on. CPU-only
server>=32GiB RAM (64GiB preferable), original14CPU quota and existing data.
Estimated real CPU30–60min, not an elapsed kill threshold. Two-GiB outputs are a
planning reference, not a reason to delete evidence or alter science. Original
C1 ownership/manual-stop/closure machinery is imported unchanged; its inherited
`C1_running` observation label is informational, while namespace/freeze/progress
and result identify gate2.300s observations; no telemetry/resource-age abort.

After explicit user power-on, verify clean reviewed source, related Linux tests
and CPU preflight, then one real run in a NEW absolute directory. No automatic
retry, third gate, GPU query, remote data download or server power operation.
Compute/correctness failures and manual stop preserve partial outputs. Seal and
rehash every closed file after child/log closure. Verify source hashes/stat
unchanged. Report geometry/quality/work cancellation/net clock/strong-control
outcomes independently; Python overhead only limits this implementation.

```sh
# Local synthetic only; new output directory required.
/private/tmp/route6-b0-local-env-20261103/bin/python \
  scripts/run_tls_rag_step5_pq_refinement_geometry_v1.py tiny --root NEW_LOCAL_ROOT
# After the user opens the CPU server and the reviewed commit is synced:
/root/autodl-fs/gaussian-benchmark-space-v1/env/bin/python \
  scripts/run_tls_rag_step5_pq_refinement_geometry_v1.py preflight
/root/autodl-fs/gaussian-benchmark-space-v1/env/bin/python \
  scripts/run_tls_rag_step5_pq_refinement_geometry_v1.py run --root NEW_SERVER_ROOT
# Separate terminal, only on explicit user stop:
/root/autodl-fs/gaussian-benchmark-space-v1/env/bin/python \
  scripts/run_tls_rag_step5_pq_refinement_geometry_v1.py stop --root SAME_SERVER_ROOT
```
