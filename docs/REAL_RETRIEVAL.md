# Real SciFact retrieval baseline

## Scope and frozen inputs

The first real retrieval gate is an exact original-space quality reference on
`query_tune` only. It does not project vectors, choose `m_prime`, fit a policy,
run certification, or inspect retrieval outcomes for `query_cert` or
`query_test`. The config loader rejects either protected split.

`configs/real_scifact_original_exact_tune.json` freezes:

- repaired dataset fingerprint
  `4a73586d3a29a0567287e501ac3c06c998af661cdc74dbc589e7525a7924f903`;
- accepted E5 cache fingerprint
  `2ec53ce38e226129ba0feffcd28ba1da1081e0627ad8e54f4a60e430c341e914`;
- embedding config and request fingerprints;
- normalized original-space squared L2 in NumPy float64;
- lexicographic stable-document-ID tie breaking;
- cutoffs 1, 5, and 10, with `k_ctx=5` and `k_gt=10`;
- query batch size 32 and the frozen 403-query tune ID hash.

The baseline-config fingerprint is
`ff675fed06fc6506ed68a83426a021ee53a701f06af4144351b2172c2dbc19f6`.
Before search, the runner revalidates the dataset artifacts and the complete
embedding cache without loading the text model, then checks ordered corpus and
query IDs against the prepared JSONL rows.

## Command

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.real_original_baseline \
  --config configs/real_scifact_original_exact_tune.json \
  --dataset data/prepared/scifact-dedup-v2 \
  --embedding-config configs/real_scifact_e5_base_v2_embeddings.json \
  --embedding-cache data/embeddings/scifact-e5-base-v2-dedup-v2 \
  --output runs/scifact-original-exact-tune
```

The result directory contains:

- `per_query.jsonl`: query ID, positive qrels, exact top-10 document IDs, and
  evidence hit/recall/nDCG at every frozen cutoff;
- `summary.json`: aggregate tune-only evidence metrics;
- `manifest.json`: complete input identities, result artifact hashes, and a
  deterministic result fingerprint;
- `timings.json`: non-result index/search timing and work counts;
- `report.md`: concise interpretation-limited table.

Timings are intentionally excluded from the result fingerprint. This permits
byte comparison of retrieval results across machines without treating hardware
latency as scientific identity.

## Local correctness run

The independently accepted A100 arrays were consumed from the audit archive on
the Mac without copying them into the repository. Two fresh output directories
matched byte for byte for `manifest.json`, `per_query.jsonl`, `summary.json`,
and `report.md`. The deterministic result fingerprint is
`2921f39dc051bc3331da8bf9b0ddc6c584dcd1f043099d8dda353653a1926b1c`.

On 403 tune queries, the exact original-space reference produced:

| cutoff | evidence hit | evidence recall | nDCG |
| ---: | ---: | ---: | ---: |
| 1 | 0.630273 | 0.606493 | 0.630273 |
| 5 | 0.803970 | 0.786849 | 0.713828 |
| 10 | 0.866005 | 0.850124 | 0.735501 |

These are tune-only labeled-evidence results for the frozen E5 model. They are
not embedding-neighbor retention, a policy certificate, a test result, or an
answer-quality claim. A Genoa rerun should reproduce the result fingerprint;
its timings are reported separately as a systems observation.

## Genoa reproduction

Slurm job `373780` on Genoa reproduced the deterministic result fingerprint
`2921f39dc051bc3331da8bf9b0ddc6c584dcd1f043099d8dda353653a1926b1c`
twice at commit `37f68fc`. The two runs are byte-identical for the manifest,
403 query-level records, summary, and report. Their non-identity timings were
`0.5372` and `0.5336 ms/query`; this difference is normal measurement noise.
All 71 then-current tests passed apart from the expected optional real-FAISS
skip in the NumPy environment. The returned audit archive has SHA-256
`c91a402eea55de127381f82f21f3a988ced679959b88eed868604292fda1af6d`.

An independent local audit recomputed every result-artifact hash, the result
fingerprint, and every evidence aggregate from `per_query.jsonl`. The archive
code matches commit `37f68fc`; only Python `3.9.6` locally versus `3.9.23` on
Genoa changes the non-result manifest fingerprint.

## Fixed-dimension tune sweep

The next gate is implemented by
`configs/real_scifact_fixed_dimension_tune.json` and
`tri_rag_harness.real_dimension_sweep`. Its config fingerprint is
`3265e303c5249a6b90868f5234d333eca3f1fc4bc28c12cdb710382e2b71eabd`.
Before observing any projected real-data result, it froze:

- `query_tune` as the only accessible split;
- dense Gaussian entries with variance `1/m_prime`, base seed `27011`, and no
  projected-vector renormalization;
- nested, same-seed candidate projections at dimensions
  `[16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]`;
- `k_ctx=5`, `k_gt=10`, `M_pilot=32`, and the common grid
  `[32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072,
  4096, 5183]`;
- a tune empirical-Bernstein selection score with `alpha=0.05` and target
  `0.95`, explicitly not a certificate;
- the common absolute coordinate-work objective
  `(N+d)*m_prime + d*M`.

The objective counts one query projection (`d*m_prime`), one projected
full-corpus scan (`N*m_prime`), and exact original-space reranking (`d*M`). It
does not compare candidate saving against a different denominator at each
dimension. The terminal full-corpus budget makes failure visible rather than
silently expanding a candidate grid. Evidence labels and evidence metrics do
not enter selection.

Two local real-array runs produced byte-identical manifest, per-query,
selection, frozen-projection, summary, and report artifacts. An independent
reduction of all 4,836 query/dimension records reproduced every one of the 192
dimension/budget bounds and the final ordering. The selected tune operating
point is:

- `m_prime=192`, fixed reference `M=768`;
- mean tune embedding retention `0.985360`;
- tune lower bound `0.958051` against target `0.95`;
- coordinate work `1,732,416` versus `3,980,544` for an original full scan,
  a theoretical reduction of `56.48%`;
- selection fingerprint
  `093588a27e0d588b9407d02fe5c5ed7e46f6a5fdc02a1881738abbde4eda01fb`;
- frozen-projection fingerprint
  `8a9a1148527db16c43bc3fedf6da1ac79ae00c0d76f2a31321cd6d9fe049809e`;
- deterministic result fingerprint
  `5dcb0a5f17cc1f2f1684a38c71ede5c8dfc1de709f98496156959e14fbea7558`.

These are tune selection results, not an independent retention certificate,
measured latency saving, evidence result, test result, or answer-quality claim.
The runner saves monotone per-budget overlap and retention for every query and
dimension, plus a full-ranking row hash bound to the accepted corpus ID map.
Timings are saved separately and excluded from result identity.

Run the sweep with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.real_dimension_sweep \
  --config configs/real_scifact_fixed_dimension_tune.json \
  --dataset data/prepared/scifact-dedup-v2 \
  --embedding-config configs/real_scifact_e5_base_v2_embeddings.json \
  --embedding-cache data/embeddings/scifact-e5-base-v2-dedup-v2 \
  --original-baseline runs/scifact-original-exact-tune \
  --output runs/scifact-fixed-dimension-tune
```

## Next gate

Reproduce the fixed-dimension selection on Genoa and require the deterministic
result and selection fingerprints above. After that reproduction, use only
`query_tune` to fit the fixed-budget, monotone-binned, and Tri-Predict policies
at the frozen projection. Write all policy artifacts before evaluating the
untouched `query_cert` split. A failed certificate remains terminal and must
not trigger retuning on the same certification queries.
