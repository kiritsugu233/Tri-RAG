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

The real cache is accepted. Slurm allocation `375414` on `a100-0` at commit
`1671b2b` reproduced all six preparation artifacts byte for byte against the
Mac result, built the cache once, reused it once without loading the model, and
produced two byte-identical audits. The embedding-request, model-snapshot,
embedding-manifest, and audit fingerprints are respectively
`1ee3e2333a27b993e632f76274b05edc04bbde587ecf80045dda457ae387b903`,
`000e13bbbec6825eb1c94ddd1f01e47071b45a1f6c8c749cc97306308ad0c874`,
`079545ef7c6af8ab27a5c8382dbd8174905f1bb537df59e94d572b6c2f2b04c1`,
and
`54af315d5b94b43a81be71ea29ab860635f0748a97108e0cda120a510947dd71`.

The corpus/query arrays have shapes `57638 x 768` and `6648 x 768`, SHA-256
values
`07dc56f7c458de152f52fe2e890e2e3606a6dca5a8f048c0b8afdfcb3fcf676e`
and
`27d452a96664fd92ccd54a1a91c0cf234859671b13fe3a5c445e3371a6b7fb13`,
and maximum absolute L2-norm errors `2.9996434e-08` and `2.9913058e-08`.
The raw 512-token audit records 2,446 truncated corpus inputs out of 57,638
(`0.0424373`) and zero truncated queries out of 6,648. The long-token warning
comes from the deliberate untruncated length pass; the embedding pass applies
the frozen 512-token limit.

The first attempt in the general `tri-rag` micromamba environment stopped
before cache publication because its inference stack was absent and its Torch
version was `2.8.0`, not the frozen `2.5.1`. The accepted run used the existing
`tri-rag-faiss` micromamba environment and passed every strict package/runtime
check. The protocol and config were not relaxed.

The returned 192 MB archive has SHA-256
`87288fd7e913930474c9f764017780b729ab00404e93d88e2fb9ffc0359c1133`.
Its size is expected: the two float32 arrays alone occupy 197,486,592 bytes
before small NPY headers and do not compress materially. Local transfer
verification passed; extraction occupied approximately 240 MB. A fresh local
audit rescanned all files and arrays and matched both A100 audit artifacts byte
for byte. The accepted small audit artifacts are checked in under
`artifacts/pdctp_fiqa_e5_v1`.

This closes only the embedding gate. The next gate may open `query_cal` to fit
the preregistered calibrator candidates; tune/cert/latency/test remain closed.

## Tests

The complete local CPU suite reports 140 passes, one expected optional
real-FAISS skip, and zero failures across 141 tests. New tests cover real
request identities, deterministic qrel-free preparation, empty-marker
retention, changed-archive refusal before publication, text-only cache build,
deterministic audit output, independent array-tamper refusal, and the accepted
real audit identity.
