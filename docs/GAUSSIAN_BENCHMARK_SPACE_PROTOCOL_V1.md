# Gaussian benchmark space v1 — approved frozen protocol

User authorization: after the nine-path proposal in
[gap audit §5](GAUSSIAN_RAG_BENCHMARK_GAP_AUDIT.md), the user replied **“批准”**
on 2026-11-02. This authorizes those paths, isolated h5py3.11.0 installation,
full regression, source synchronization and the bounded Linux experiment.
It does not authorize old route4/5 resumption, synthetic A0, Tri training or
statistical certification. Existing scientific bytes and results remain fixed.

## Source and roles

The source is the full VIBE HotpotQA–Harrier640 file at revision
`a91cc76a3ab66b0353032e8427b9e34dcee22ded`, 13,688,159,632bytes,
SHA256 `08f1c68cc8bb1f24a29cb64d9ffc20ccc6a91c06d286538e4e659d4afb648114`.
Transport may differ, identity may not. Verify entire size/hash before HDF5 access.
Require train `(5233329,640)`, learn `(96845,640)`, test `(1000,640)`, float32,
native metric `normalized`, and supplied neighbor shapes. Never sample competitors,
renormalize native vectors or alter supplied ground truth. Stable string IDs are
`vibe-hotpotqa:{corpus|learn|test}:{row}`; persisted numeric rows refer to explicit maps.

Development uses the first256 learn rows under ascending SHA256 of
`63001:learn:{row}`. Other learn rows are reserved. Officialtest arrays are read
only after all development actions, selection and a source/config-bound nonempty
seal are written. No qualified deployment point means officialtest stays closed.
This is exploratory systems measurement, not independent formal certification.
Observed development queries cannot be reused as fresh certification data.

Native distance is `clip(1-dot,0,2)`. Recall@100 is the fraction of100 returned
unique candidates at distance at most supplied kth distance +0.001. Strict ID
retention is separately reported. Learn thresholds are derived from supplied
kth neighbor IDs and native vectors, explicitly labeled as derived. First32
development queries undergo full native exact reference and native-vs-L2 overlap
audit. A kth distance mismatch exceeding1e-5 stops execution; this gate diagnoses
input provenance and does not modify native epsilon/labels. Full corpus/query
squared-norm ranges are recorded, with no corrective renormalization.
Tie convention for new exact/selection operations: ascending corpus row ID.
Native benchmark permissive scoring and strict ID overlap may differ at ties.

## Fixed representation and comparators

The config `configs/gaussian_benchmark_space_v1.json` is the complete frozen grid.
Dense Gaussian entries use standard deviation1/sqrt(m); m64/128/256/512,
seeds63013/63017. Each representation is fixed across queries; no projected
renormalization. Exact projected squared-L2 search uses query batches16 and corpus
blocks32768, with original native reranking. M100/200/400/800/1600/3200/6400.
Float32 search arithmetic and source norm deviations are recorded limitations,
not an exact-real-arithmetic theorem. Exact search is a representation diagnostic,
not a claimed deployment latency winner.

Full corpus HNSW original degree16/32 and projected degree32, efConstruction200,
graph seed63011. Original HNSW requests100 candidates and varies efSearch over
the budget grid. Projected HNSW requestsM and varies efSearch=M/2M. Choose the
smallestm with both seeds' development maximum mean native recall>=0.95 and
its next gridm (if any); absent any qualifyingm, use512 as negative control.
Do not change this selection after officialtest inspection.

IVF-PQ FastScan uses inner product,4096 lists,160 subquantizers,4bit codes,
100k corpus-only uniform training rows without replacement, seed63007 for sampling
and FAISS clustering; nprobe8/16/32/64/128, exact native refinement of100/400/1600.
Unsupported compiled interfaces stop, not silently substitute. FAISS CPU1.8.0,
NumPy1.26.4, SciPy1.13.0 and h5py3.11.0 are required. This is not a replication of
the current VIBE leaderboard or evidence of superiority to all modern methods.
FAISS parallel graph construction may be nondeterministic despite its seed:
save the actual built indexes, use those same bytes for reporting/replay, and
never claim byte-identical multithreaded rebuilds. Query replay must preserve quality.

## Measurements and inference

Development3 repeats and sealedtest5 repeats rotate action order by query/repeat.
Full retrieval time includes query projection/copy, backend search/settings,
candidate validation/selection and original native reranking. The disjoint stage
clocks sum to total; scoring, truth lookup and logging occur after the clock.
Pilot/risk/expansion costs are zero because these mechanisms are absent in phase1.
Backend search time is also available separately, without calling it full latency.
Offline FAISS up to8 threads; latency FAISS1 and BLAS1. No GPU or host scheduling changes.
Build and projection construction time, index bytes, retained original vectors
and projected construction arrays are recorded separately. The supervisor records
observed process peak RSS, elapsed time and output bytes; index size alone is not
system memory. Development candidate/return rows, metrics and timing are saved per
query/action/repeat. Exact candidates, projection matrices, indexes, seals and roles
are retained. No reuse of an old namespace/cache.

For GT G, exact candidates E and ANN candidates A, report all signed terms:
`loss(A)=loss(E)+|G∩(E\A)|/k−|G∩(A\E)|/k`.
ANN can recover original neighbors outside E; exact failure is scoped to that
fixed ranking/budget family, not all ANN outputs.

Oracle matrices stay within a fixed representation/build. Quality feasibility is
mean of per-query best quality. For targets0.95/0.99, solve the randomized LP
minimizing actual measured mean cost under mean quality, one probability simplex
per query. Independently evaluate the dual lower bound
`lambda*target + mean_q min_a(cost[q,a]−lambda*quality[q,a])`, lambda>=0.
Save primal quality/cost, dual lambda/bound and gap. A randomized lower bound is
not an implemented deterministic policy, and neither oracle uses deployable inputs.
No per-query representation/seed switching and no candidate-count latency proxy.

For each family/target, seal the lowest mean development-cost qualified action;
break cost ties by build key then action index. Report only these sealed points on
1000 officialtest queries; do not retune. If a point fails quality on test, report
failure. Primary output is the matched-quality frontier, allowing no-quality-space,
no-cost-gain, fixed-only gain, possible oracle budget headroom, and inconclusive
resource stops. None establishes Tri incremental value or answer correctness.

## Limits, tests and replay

One experimental invocation:4h wall,96GiB observed RSS,100GiB output,8GiB temporary
files. External watchdog samples every0.25s and terminates the worker process group;
up to5s TERM grace then KILL. Sampling/write overshoot is possible and must be
reported from actual output size. Data download is separately bounded to15minutes
per declared source attempt; preserve partials/failures, do not silently extend.
Outputs must be new directories, never overwrite. No automatic scientific resume
or resource enlargement. Partial failure is not a completed frontier.

New offline tests exercise source/roles/metrics, full corpus shape checks,
projection geometry, independent brute-force references and ties, signed ANN
recovery, global-mean oracle and dual certificate, actual HNSW/FastScan and exact
refinement, full clock decomposition, resource checks, immutable receipts and
small HDF5 end-to-end sealed reporting. Existing full suite is still required.
Old native libraries are unchanged. Local and Linux results are reported separately.

After reviewed source is committed/pushed and Linux is fast-forwarded to that
exact commit without overwriting local changes:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /root/autodl-fs/gaussian-benchmark-space-v1/env/bin/python \
  scripts/run_gaussian_benchmark_space_v1.py \
  --source /path/to/sha256-verified/data.hdf5 \
  --output /root/autodl-fs/gaussian-benchmark-space-v1/run-UNIQUE
```

A source-bound clean checkout, full hash verification and dependency verification
are mandatory. No officialtest operation or system experiment has occurred merely
because this protocol exists. Actual results belong in STATUS and new run receipts.
