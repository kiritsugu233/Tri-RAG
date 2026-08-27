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

## Next gate

Before any projected retrieval outcome is observed, predeclare the fixed
projection seed, candidate dimensions, `M_grid`, and a cross-dimension compute
objective on `query_tune`. Dimension selection must not maximize candidate
saving against a different fixed baseline per dimension; the objective must
account for projected scan work and original-space reranking work on a common
scale.
