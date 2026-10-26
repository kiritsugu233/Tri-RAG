# Step 5 baseline/headroom development screen v1

Approved 2026-10-26: the user accepted the four exact paths and read-only SSH
input scope in `TLS_RAG_STEP5_BASELINE_HEADROOM_PROPOSAL.md`, replying
“批准上述范围，稍后无卡启动服务器”. The preceding goal was “可以，先补齐1，2步吧”.
This contract implements those two steps only; no predictor/Tri fit or new certification.

## Frozen inputs and access

Reuse the real 1M normalized MiniLM384 corpus, archived float64 Gaussian128
projection (seed55117), original.faiss and opened tune truth/qrels. Reconstruct
the same Gaussian matrix with the unchanged pinned helper. Original data/source
bindings and all existing scientific files remain unchanged.
Parent sampling SHA is `52e22ee02d6da7746f202a430a34c0624a3bbf23cf841db97099029b8b85d6ae`.
Use exactly its Selection256 and Evaluation512. Both have already been observed;
evaluation is isolated from this selection, not fresh independent certification.
Original cert/latency/test IDs, arrays, labels and files remain closed.

Read-only SSH exports only those queries/truth/qrels and descriptors for corpus,
IDs, projected vectors and original index. Verify anchored parent bundle/index/
tune manifests, opened-role receipts and legacy source/protocol metadata. The
server does not receive modified repository code, a new index or an experiment.
Local transfer verifies every large-file SHA before use; no downloads/encoding/
HNSW rebuild. Credentials never enter files, commands, logs or receipts.

## Six methods

- `original_exact`: direct float64 original-space distances once for all N,
  cached global stable top10, context first5; reuse those exact distances, no
  redundant scoring or query projection.
- `original_ann64`: archived original HNSW, FAISS1.8.0, one thread, efSearch256,
  M64; use unchanged CandidateSession float64 refinement/reranking.
- `projected_fixed:{32,64,128,256}`: unchanged optimized cache_reselect_fixed
  probe, including query projection, one full projected scan, cache write,
  stable global selection and original-space reranking. No projected renormalization.

Same caching/global selection optimization for original exact and projected
fixed. Compare exact/exact and the practical ANN control separately, all at the
same quality constraints. No graph-search settings or frozen grid tuning.
N=1M, k_gt10, k_ctx5, direct squaredL2, stable string ties, block8192.

## Selection, evaluation and measurement

For each group compute actual returned sets/quality for all methods. In this
offline quality pass, projected exact prefixes may share one scan; this is not
latency. Save query-level candidate/context IDs, retention, Hit and original truth.
Original exact must match archived top10 and Hit; retain mismatch evidence and stop.

For actual timing choose64 queries per group by SHA-256 canonical JSON
[`step5-baseline-headroom-v1`,55273,group,query_id], breaking ties by ID. Three
repeats, independently shuffled six-method order with seed55279, four excluded
warmup queries/group. Warm both original/projected arrays and the index storage
before timing; report this hot-memory setup explicitly. Cache allocation, distance
computation, selection, accesses, new-only scoring, rerank and large-buffer release
are inside each call timer. Hashes, shared ID map, validation and setup are outside.
Verify every timed call's returned IDs/quality against that method's quality pass.

Quality target: mean retention>=.90 and mean Hit difference from exact>=-.02.
Select the fastest eligible projected fixed M using Selection quality and action
mean cost; tie by smallerM. Write an immutable source/input/results/choice seal
before accessing Evaluation via the group API. No eligible M is a valid outcome.
Report the frozen choice on Evaluation even if it fails; never substitute anotherM.
All fixed curves and post-hoc empirical eligible controls are diagnostics only.

## Perfect information cost model

For Selection and Evaluation separately use each method's group mean complete
call cost, constant for all queries. Do not select incidental per-query fast times.
Use actual per-query top10 retention counts and binary Hit with the unchanged
read-only MILP/LP helpers; preserve solver infeasibility/gaps.

Compute projected fixed controls and query-independent mixtures, with original
exact/ANN controls when eligible. Perfect oracle selects one projected fixed
action before any search; all information, pilot, features and decisions free.
This is an optimistic model, not a deployable policy or measured adaptive latency.
Report headroom versus each control, budget counts, quality and free-cost budget
at0/5/10/20% improvement sensitivities (engineering sensitivities, not new acceptance
thresholds). Separately match original ANN's actual quality; never replace the
.90 main target with that sensitivity. No target/grid enlargement after failures.

## Memory, resources and receipts

One numerical thread via OPENBLAS/OMP/MKL/VECLIB. RSS<=8GiB, local execution<=45min
including verification/setup (transfer separately logged), new input/output<=8GiB,
free disk>=12GiB. Stop preserving evidence on breach; no automatic resize/retry.
Measure original exact, ANN and projected fixed256 memory in separate processes,
calibrating NumPy tracing, reporting newly allocated peak and absolute RSS/history
highwater limitations. No latency measurement runs with allocation tracing.
Mean/median/empirical p95 and nominal paired query-block bootstrap intervals use
2000 replicates, seed55283;64 query blocks with repeats kept together. Descriptive
results are not population guarantees or tail certification.

Output must be a new directory; refuse overwrites. Record new source/input hashes,
sampling/seal, quality sets, every timing repeat, memory, failures, cost models and
formal_certification=false. Preserve raw inputs/receipts; version compact evidence
and results, update STATUS/IMPLEMENTATION_PLAN, run offline regressions/protection/
diff checks and push only the continuing Step5 branch. No automatic follow-on run.
