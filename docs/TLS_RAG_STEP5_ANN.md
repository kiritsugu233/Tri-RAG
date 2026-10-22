# TLS-RAG Step 5 ANN v1

## Authorization and scope

On 2026-10-22, after the exact-path proposal, the user instructed:
“再仔细检查一遍，看看设计基线有没有问题，没问题就开始执行吧，先在本地改，改好之后同步到github，然后给我服务器端的指令同步和运行测试，告诉我预计需要运行多久。”
This authorizes the reviewed new Step 5 implementation, local validation and normal
GitHub synchronization. Initial implementation used `codex/tls-rag-step4-retention-v3`.
On 2026-10-23 the user explicitly requested a new Step 5 branch and cleanup of
redundant earlier branches. All subsequent Step 5 work/synchronization uses
`codex/tls-rag-step5-ann` in the same canonical checkout. It does not reopen historical roles or change
the old MVP's constraints. No real data, model or benchmark is run in local tests.

New scientific files are `tls_rag_step5_{ann,data,policy,protocol,benchmark}.py`
under `src/tri_rag_harness/`, `configs/tls_rag_step5_ann_v1.json`,
`tests/test_tls_rag_step5_ann.py`, and this document. The benchmark module is
registered **L0** because it also contains tune selection and role orchestration.
Existing scientific source, dependency pins, frozen configurations, tests and
historical source bindings are unchanged. Protection registration only appends
these files and records the actual consent; the human-readable register is extended.

## Review corrections before implementation

1. The old all-pairs candidate features scale quadratically. V1 uses 32 evenly
   spaced projected ranks, the current top-5 context and up to 8 newly accepted
   shell items (at most 45 IDs), with exact sampled diversity and observed-pair
   features. This is a **new feature schema**, not unchanged Row 3 or its old fit.
   LID uses original-space distances of the current accepted pool; pilot is the
   first stage. Invalid features force expansion. No plan proxy or churn is used.
2. Single-call fixed ANN and staged fixed ANN are both measured. Only original
   scores and accepted membership are reused. Every ANN expansion restarts graph
   search; raw returned sets may be nonnested. Keep old members and accept nearest
   new raw candidates by float64 projected distance/string ID until size M.
   Previously admitted members can displace closer later raw results. Neither the
   accepted pool nor the risk scores are claimed to be exact projected prefixes.
3. The 8,192-query certificate may be underpowered. Failed point targets terminate;
   passing point targets with failing bounds are `inconclusive` and also terminal.
   No late sample enlargement, threshold change, replacement policy or role retry.
4. Canonical vectors use float64 normalization/storage to preserve the repository's
   reference geometry. The 1M original/projected arrays therefore use about 4.10 GB,
   plus roughly 2.05 GB float32 vector payloads and graphs in the two HNSW indexes.
   The earlier 2.05 GB estimate covered only float32 arrays, not this full footprint.
5. The GPU exact control uses a conservative PyTorch float64 blocked direct-L2
   implementation with host stable merging, not a tuned GPU FAISS performance
   frontier. Batch-64 results are serial complete pipelines grouped into blocks,
   not native batched ANN or concurrent serving. These limits are explicit outputs.

## Frozen experiment

The registered JSON is the only executable real protocol; alternate real config
files are rejected. Gaussian projection: fixed m'=128, seed 55117, variance 1/128,
no projected renormalization; dimension 384; k_gt=10 and context size 5.
Candidate ladder: 64/256/1024/4096. CPU FAISS 1.8.0 `IndexHNSWFlat`, graph
connections=32, efConstruction=200, one build/search thread, fixed insertion order
and index seed 55121. `efSearch=max(256,2*M)` is predetermined, not a second learned
control. No vector compression, dimension allocation, churn or answer generation.

New stagewise standardized ridge models (regularization .01) fit continuous exact
top-10 retention on cal-model-fit. Independent cal-bound-fit prediction-minus-
retention residuals give nonnegative `higher` quantiles .5/.8/.9. Nine Tri-feature
candidates use thresholds .9/.925/.95; nine base-feature candidates are diagnostics.
Invalid states or missing fits expand. Maximum budget stops without asserting quality.
These residual adjustments are not confidence bounds or per-query guarantees.

Tune first selects the fastest measured single-call projected fixed comparator
meeting mean retention >=.90 and Hit@5 drop <=.02 relative to exact original.
Original-space ANN gets its own quality-matched fixed point. Primary policy
selection considers only the nine Tri-feature candidates satisfying the same
quality targets, retention loss <=.01 against the frozen projected comparator,
and at least 10% original-distance reduction. Select minimum mean original
distances; break ties by stable policy ID. No eligible policy or comparator means
`no_tune_candidate`; cert/latency/test stay closed. Tune cached trajectories are
offline selection records and cannot stand in for measured online latency.

Cert/test also retain direct original ANN, staged fixed, exact projected fixed,
and exact projected adaptive transfer diagnostics. The last uses the ANN-fitted
policy unchanged; it is not a separately calibrated or selected exact policy.

## Source, roles and audit

Use the official [MS MARCO passage v1 source list](https://github.com/microsoft/msmarco/blob/master/Datasets.md):
8,841,823 passages, collection archive approximately 2.9 GB; training qrels are
sparse. Its [terms](https://microsoft.github.io/msmarco/) restrict use to
non-commercial research and do not provide an open-source data license.
No train triples/top1000 downloads are needed. Preparation accepts local TSVs only.
It hashes the actual collection, queries and train-qrels files and records row count;
it does not invent an upstream immutable revision or claim source authenticity from
row count alone. Source hashes are frozen before fitting; they require source audit
before any unconditional provenance claim.

Choose the million passages with smallest seeded SHA-derived PID ranks. Eligible
queries must have a positive train qrel remaining in this subset and cannot copy
a normalized corpus text. Group lexical variants by NFKC/case/whitespace
normalization and a sorted punctuation-free token signature with numbers replaced.
Choose one seeded representative per family, then seeded disjoint role allocation:

| Role | Count |
|---|---:|
| cal_model_fit | 8192 |
| cal_bound_fit | 4096 |
| tune | 2048 |
| cert | 8192 |
| latency | 2048 |
| test | 2048 |

This targets training lexical-family representatives with positives in the hash
subcorpus, not all original queries. Insufficient eligible families fails before
encoding. Official dev/test labels are not inputs. Sparse qrels define judged
Hit@5; unjudged does not mean irrelevant or insufficient evidence.

An optional pre-approved historical exclusion JSON contains exactly:
`text_hashes`, `family_hashes`, `source_ids` (lists), and `complete` (boolean).
Hashes use this module's canonical functions; source IDs are prefixed
`msmarco-passage-v1:`. It must be supplied without opening old protected archives.
The run records whether completeness was asserted, not independently proven.
No manifest means **history exclusion remains unestablished**. Lexical grouping
does not prove semantic family independence; `formal_certification=false` remains
explicit even when conditional numerical gates pass.

The model is all-MiniLM-L6-v2 at declared revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, max length 256. A local model
directory is required; preparation binds actual file hashes and encoded arrays.
The CLI revision is an operator declaration, not independently verified upstream
provenance. Model identity verification and truncation effects remain audit limits.

## Boundaries, statistics and failure rules

Each role is an explicit command. Exclusive directory reservation precedes work;
decisions are serialized and hash-closed before role labels or exact top-k joins.
Runtime latency cannot open labels. Input, source code, environment, indexes,
fitted models, residuals, selection, decisions, per-query metrics and final artifacts
are fingerprinted. Every ancestor stage must retain its seal and pass. A crash
leaves an exclusive reservation; do not delete it to retry an opened role. These
are audit controls, not filesystem permissions or proof of unrecorded noninspection.

Conditional quality gates use the unchanged empirical-Bernstein implementation:
retention lower bound >=.90; paired Hit@5 delta lower bound >=-.02; paired
retention delta against frozen fixed lower bound >=-.01. Differences in [-1,1]
are mapped to [0,1] and bounds transformed back. Three quality claims and three
systems claims each receive alpha=.05/6. Historical outcomes never select this policy.

Label-free systems gates require upper bounds on mean original-distance ratio
<=.90, mean total latency ratio <=.90, and p95 latency ratio <=1.05, all against
the tune-frozen projected single-call comparator. Use 5 paired repeats, 32 warmup
queries per method and 5,000 seeded query-block bootstrap resamples preserving
repeats/method pairing. Percentile bootstrap is empirical uncertainty, not a
distribution-free finite-sample guarantee. No trimming, failed-request removal,
test-based comparator selection, or reinterpretation of candidate reduction as latency.

Online timing begins with an already encoded normalized query and includes
projection, every restarted search, projected candidate refinement, candidate
merge, original-vector gathering/scoring, bounded features, controller, reranking
and context IDs. Fixed controls pay their own operations. Raw per-stage timers and
complete wall time are saved. Encoding, offline ground truth, index build/load,
bundle hashing and artifact writing are separately excluded. CPU/GPU exact original
controls are timed; the GPU path includes query upload, synchronization, result
download and host merging and audits eight latency-query top-k identities against
the CPU reference. It does not certify numerical identity on all queries.

Report p50/p95, mean, serial QPS, batch-block throughput, RSS, GPU identity, timings,
and failure status. GPU memory allocator profiling and OS cold-cache eviction are
not implemented; cold start is index setup/load metadata, not a cold-cache claim.
General serving superiority remains false: passing the CPU adaptive-vs-fixed gate
does not prove superiority over original ANN, optimized GPU exact, native batching,
another graph backend or a full RAG system.

## Server synchronization and tests — first action

Use `/root/autodl-tmp/Tri-RAG-git`. Preserve dirty files: do not reset/clean.
Compare the final delivered commit with the fetched HEAD. The following runs only
offline tests and a tiny synthetic real-HNSW self-test; it downloads no dataset or
model, builds no real corpus index and opens no real experimental role.

```bash
set -euo pipefail
cd /root/autodl-tmp/Tri-RAG-git
git status --short --branch
git diff --exit-code
git diff --cached --exit-code
git fetch origin codex/tls-rag-step5-ann
git switch codex/tls-rag-step5-ann
git pull --ff-only origin codex/tls-rag-step5-ann
git rev-parse HEAD

# Clone once; preserve the Step 4 environment and all old run roots.
if [ ! -d /root/autodl-tmp/conda-envs/tri-rag-step5 ]; then
  /root/miniconda3/bin/conda create -y \
    -p /root/autodl-tmp/conda-envs/tri-rag-step5 \
    --clone /root/autodl-tmp/conda-envs/tri-rag-step4
fi
STEP5_PY=/root/autodl-tmp/conda-envs/tri-rag-step5/bin/python
"$STEP5_PY" -m pip install --no-deps faiss-cpu==1.8.0
"$STEP5_PY" -c 'import numpy, scipy, faiss; print(numpy.__version__, scipy.__version__, faiss.__version__); assert (numpy.__version__, scipy.__version__, faiss.__version__) == ("1.26.4", "1.13.0", "1.8.0")'

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
STEP5_CHECK=$(mktemp -d /root/autodl-tmp/tls-rag-step5-check-XXXXXX)
"$STEP5_PY" scripts/check_code_protection.py
"$STEP5_PY" -m unittest discover -s tests -v 2>&1 | tee "$STEP5_CHECK/tests.log"
"$STEP5_PY" -m tri_rag_harness.tls_rag_step5_benchmark self-test \
  --require-faiss --output "$STEP5_CHECK/hnsw-selftest" \
  2>&1 | tee "$STEP5_CHECK/selftest.log"
echo "$STEP5_CHECK"
```

Expected: all tests pass, optional historical GPU test may skip depending on
the installed backend; the explicit self-test must say
`ann_integration_exercised=true`, `real_data_opened=false`. A synthetic
`no_tune_candidate` is expected and preserves closed cert/latency/test roles;
the separate permissive synthetic fixture tests all six success transitions.
Do not treat either fixture as a real scientific result.

Budget: warm environment tests/self-test approximately **1–3 minutes** on the A800
host CPU, conservatively allow **5 minutes**; first Conda clone/FAISS wheel setup
roughly **3–15 additional minutes**, network/disk dependent. No A800 experiment time
has been measured by this local implementation.

## Real experiment entry points (separate from tests)

After source/model/historical-exclusion audit and input preparation, the local-file
producer is discoverable with:

```bash
python -m tri_rag_harness.tls_rag_step5_data --help
python -m tri_rag_harness.tls_rag_step5_benchmark run-stage --help
```

Preparation requires `--collection-tsv`, `--queries-tsv`, `--qrels-train`,
`--model-dir`, `--model-revision`, `--output` (absent), and optionally
`--history-manifest`; use `CUBLAS_WORKSPACE_CONFIG=:4096:8`, offline Hugging Face
mode and `--device cuda`. It does not fetch sources or a model implicitly.
Then call `run-stage --bundle BUNDLE --run RUN --role ROLE` separately in the
table order; all roles share the same binding and run directory. Non-passing real
stages exit 2, preserve evidence and lock successors. Do not run a shell loop that
ignores failures. A real latency stage additionally requires CUDA PyTorch.

Planning allowance: 40 GB free disk, at least 32 GB host RAM (28 GiB RSS guard),
one A800 80 GB. The million-vector encoding/index/scan workload is **not locally
benchmarked**. Allow an initial 6–12 hour experiment window, with a cumulative
12-hour checked compute budget; this is a stop limit, not a promise of completion.
Downloaded-source time and embedding preparation are additional and network/model
dependent. Checks occur between blocks/queries and cannot interrupt a long native
index build; an external job timeout is needed for a hard wall-time ceiling.
Retain any negative/underpowered/resource-limited run and assess measured stage
rates before scheduling further work; never change the frozen sample or budgets
to make the same opened evaluation finish or pass.
