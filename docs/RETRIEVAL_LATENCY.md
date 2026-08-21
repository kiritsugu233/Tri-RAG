# Retrieval-only latency benchmark

Updated: 2026-08-20

## Purpose

`tri_rag_harness.retrieval_benchmark` measures retrieval latency at realistic
embedding dimensions and corpus sizes without invoking an LLM. It is separate
from statistical certification: latency observations must not select or repair
a certified policy.

The benchmark uses deterministic, external, normalized Gaussian query and
corpus vectors stored as `float32` memmaps. Their dimensions and memory traffic
are realistic, but they are not embeddings from a text model and support no
semantic-quality claim.

## Compared paths

Every measured query runs four paths:

1. `original_space_fixed_m`: one streaming exact scan in original space;
2. `projected_space_fixed_m`: query projection, one projected scan to fixed `M`,
   then exact original-space candidate reranking;
3. `projected_tri_predict_reuse`: one projected scan retaining top `M_max`, a
   pilot prefix, original pilot distances, LID, Tri-Predict, cached expansion,
   and exact reranking;
4. `projected_tri_predict_double_scan`: the legacy control with an independent
   pilot scan and expansion scan.

The reuse path computes exactly `N` projected distances per query. The legacy
control computes `2N`. Both must choose the same `M(q)`, candidates, and final
retention under exact search. Original pilot distances are cached and reused by
reranking in both paths.

The main synthetic certification harness now uses the same one-scan invariant:
it scans to `M_max` once, obtains the pilot from the prefix, and slices the
cached ranking after the policy decision.

## Memory-bounded exact backend

`StreamingExactSquaredL2Index` scans an ndarray or memmap in fixed row blocks
and keeps only global top-k candidates. It never materializes a
query-by-corpus distance matrix. Each result records:

- scan count;
- vector distance evaluations;
- distance-coordinate work;
- vector bytes scanned;
- measured search latency.

Top-k ties use stable corpus row number, corresponding to the benchmark's
implicit stable `doc-{row}` IDs.

## Configurations

### Smoke

`configs/retrieval_latency_smoke.json` uses `N=10,000`, `d=128`, and
`m_prime=32`. It is for local validation, not reporting.

### Genoa baseline

`configs/retrieval_latency_100k_d768.json` uses:

- `N=100,000`;
- 64 measured and 4 warmup queries;
- `d=768`, `m_prime=96`;
- fixed `M=512`;
- `M_grid=[32,64,128,256,512,1024]`;
- approximately 307.2 MB original and 38.4 MB projected corpus storage.

### Genoa scale-up

`configs/retrieval_latency_1m_d1024.json` uses:

- `N=1,000,000`;
- 32 measured and 2 warmup queries;
- `d=1024`, `m_prime=128`;
- fixed `M=1024`;
- `M_grid=[64,128,256,512,1024,2048]`;
- approximately 4.096 GB original and 512 MB projected corpus storage.

The 100k run is the first gate. Run the 1M configuration only after the 100k
artifacts pass scan-count and memory checks.

## Recorded artifacts

- `manifest.json`: config, hashes, environment, generation durations, projection
  metadata, thread environment, and reproducibility identity;
- `summary.json`: per-method latency percentiles, quality, budget, work, memory,
  and reuse comparison;
- `per_query.jsonl`: query-level stage timings and work counters;
- `memory.json`: memmap sizes, query/projection sizes, cache size, and peak RSS;
- `tri_predict_policy.json`;
- `report.md`.

Stages include query projection, original scan, pilot scan, original pilot
distance computation, LID estimation, Tri-Predict, expansion, original rerank,
and end-to-end latency. Summary latency includes mean, p50, p95, and p99.

## Reproducible CPU command

Set BLAS thread counts before Python starts. The first baseline is deliberately
single-threaded so per-query latency and memory traffic are interpretable:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
python3 -m tri_rag_harness.retrieval_benchmark \
  --config configs/retrieval_latency_100k_d768.json \
  --output runs/retrieval_latency-100k
```

The generated memmaps are intentionally inside the run directory and ignored
by Git. Audit bundles should include JSON/JSONL/Markdown/log artifacts but omit
the `data/` directory.

## Local structural result

The 100k/d768 configuration completed locally and confirmed the expected work
invariant: reuse performed 100,000 projected distances and scanned 38.4 MB per
query, versus 200,000 distances and 76.8 MB for the legacy control. It selected
the same budgets and retention. Mac latency is not a Genoa result and is not
used for serving claims.

## Interpretation limits

- Exact NumPy streaming search is the correctness baseline, not the final
  production backend.
- Gaussian vectors reproduce dimensions and memory traffic but not real text
  embedding anisotropy, LID, or neighbor structure.
- Different methods may use different candidate budgets; latency and retention
  must be read together.
- Page cache, BLAS implementation, CPU frequency, NUMA placement, and thread
  count are part of the benchmark environment.
- Candidate-count saving is not latency saving. Tri-Predict root solving,
  projected scans, and memory traffic are all reported independently.
- FAISS CPU/GPU comparisons come only after the exact Genoa baseline passes.
