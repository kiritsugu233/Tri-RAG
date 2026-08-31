# FiQA PDCTP text preparation and E5 cache gate

## Scope

This gate prepares only the corpus and external-query text needed for the new
FiQA E5 cache, then independently audits the cache. It does not materialize or
read qrel pairs or relevance values, run retrieval, fit or select a method,
open any of the five protocol roles, run an LLM, or use an approximate index.

The v2-only dataset adapter is `pdctp_fiqa_text_only_v1`. The existing BEIR and
Raw Tri-Predict v1 paths remain unchanged. Unlike the legacy retrieval dataset
profile, this profile requires `corpus.jsonl`, `queries.jsonl`, `splits.json`,
`empty_documents.json`, and `formatted_text_hashes.json`, and explicitly
forbids `qrels.jsonl`.

## Canonical preparation

The preparation runner validates the frozen protocol, all-closed role
assignments, source-audit file identity, full FiQA archive identity, and the
corpus/query member hashes before reading text. It opens only the corpus and
query ZIP members. The archive hash binds all other bytes, but qrel members are
not opened by this gate.

It preserves all 57,638 corpus IDs, maps the 38 empty source items to the
frozen `[EMPTY_DOCUMENT]` marker, and writes all 6,648 external queries in the
predeclared cal/tune/cert/latency/test role order. Every role remains closed;
the role field is used only for frozen row alignment and is not an inference
feature.

The exact local command is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m \
  tri_rag_harness.pdctp_fiqa_dataset \
  --protocol-freeze artifacts/pdctp_fiqa_real_protocol_v1/protocol_freeze.json \
  --role-assignments artifacts/pdctp_fiqa_real_protocol_v1/role_assignments.json \
  --source-audit artifacts/pdctp_fiqa_source_audit_v1/source_audit.json \
  --archive data/fiqa.zip \
  --output data/pdctp-fiqa-text-v1
```

Two independent local output directories match for all six files byte for
byte. The canonical dataset-manifest fingerprint is
`bfc25daad8d2d382390a0a42c3aa03b96e965965ba17c2065aaf8bef00903240`.
The empty-document ordered-ID hash is
`a8a32f58fc05b4a5edb7a83e78e56458a6c3455c64eb820771beb35be33fcca7`.
The ordered formatted corpus/query pair hashes are
`4cfdc2995dc2e6a96f933e8e4ffd99a10641b74aa928facf2711fd722be428ef`
and
`e1f63c6dcf61293f3a734aef0f3c7aef35953f00bd630d91fe946268d3c3bc25`.

## Frozen E5 request

`configs/pdctp_fiqa_e5_base_v2_embeddings.json` binds that dataset fingerprint
to `intfloat/e5-base-v2` revision
`f52bf8ec8c7124536f0efb74aca902b2995e5bcd`. It retains the v1-continuity
model snapshot file list and exact runtime package versions, E5 passage/query
prefixes, 768 dimensions, a 512-token maximum, float32 output, deterministic
algorithms, eager attention, disabled TF32, and canonical L2 normalization.
Its config fingerprint is
`dce9c5f590c0348672dc3ab6f90a8e07e5b170c2174a5c2aab5b9eaeabc8bc78`.

## Independent cache audit

`tri_rag_harness.pdctp_embedding_audit` does not load the model or reuse the
cache builder's acceptance function. It independently rehashes every prepared
artifact and embedding file; reconstructs E5 formatted-text hashes; verifies
exact query-role and cache-row alignment; recomputes array fingerprints,
shapes, dtypes, finiteness, and normalization error; checks the pinned model
snapshot and deterministic CUDA/runtime metadata; and validates corpus/query
token-length and truncation summaries.

A successful audit emits `embedding_audit.json` and `report.md` with decision
`READY_TO_OPEN_QUERY_CAL`. It does not mutate the protocol state. Any mismatch
fails before output publication.

The real cache has not yet been built or accepted. The next action is one
manual A100 Slurm run that reproduces the text artifacts, builds and reuses the
cache from the existing pinned local model snapshot, runs the audit twice, and
compares the two audit outputs byte for byte. Calibration remains forbidden
until those artifacts are returned and independently reduced.

## Tests

The complete local CPU suite reports 139 passes, one expected optional
real-FAISS skip, and zero failures across 140 tests. New tests cover real
request identities, deterministic qrel-free preparation, empty-marker
retention, changed-archive refusal before publication, text-only cache build,
deterministic audit output, and independent array-tamper refusal.
