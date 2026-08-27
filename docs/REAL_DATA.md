# Real external-query dataset

## Frozen Stage-1 choice

The first real-data target is the BEIR distribution of SciFact. It is small
enough for exact retrieval and has external claim queries with document-level
relevance judgments. This stage prepares data only; it does not select an
embedding model, projection dimension, candidate budget, or policy.

The checked-in configuration freezes:

- archive URL:
  `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`;
- publisher-listed archive MD5:
  `5f7d1de60b170fc8027bb7898e2efca1`;
- adapter schema `beir_zip_v1` and ID namespace `beir-scifact`;
- the official train qrels as the development pool and the official test qrels
  as the untouched `query_test` split;
- split seed `41017`, with the development query IDs ordered by
  `sha256(seed || NUL || source_query_id)` and divided 50/50 into
  `query_tune` and `query_cert`.

The split procedure uses query IDs only, not qrel labels, document identities,
retrieval results, LID, or retention. Corpus IDs and query IDs are explicitly
namespaced as `beir-scifact:doc:*` and `beir-scifact:query:*`, so they cannot be
confused even when the upstream IDs have the same textual value.

## Source and license metadata

The BEIR dataset index provides the archive and checksum. SciFact attributes
claims and evidence annotations under CC BY 4.0 and the abstract corpus derived
from S2ORC under ODC-By 1.0. These declarations and their source URLs are stored
in both `configs/real_scifact_dataset.json` and every prepared dataset manifest.
They are metadata for auditability, not a legal interpretation.

## Preparation command

No Slurm allocation or GPU is needed. Run this on a login node or local machine:

```bash
cd ~/Tri-RAG

mkdir -p data/source data/prepared

curl -L --fail --retry 3 \
  -o data/source/scifact.zip \
  https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip

md5sum data/source/scifact.zip

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
python3 -m tri_rag_harness.beir_dataset \
  --config configs/real_scifact_dataset.json \
  --archive data/source/scifact.zip \
  --output data/prepared/scifact
```

The adapter refuses an unexpected archive checksum, a corrupt ZIP, missing
members, duplicate IDs/qrels, qrels pointing to absent queries/documents,
overlapping development/test query IDs, or an existing output directory.

Inspect the immutable identity and split sizes with:

```bash
jq '{
  fingerprint,
  config_fingerprint,
  source: .source.archive,
  counts,
  splits,
  ids,
  split_rule,
  artifacts
}' data/prepared/scifact/dataset_manifest.json
```

Prepared text and source archives live below ignored `data/`; only the empty
directory marker is versioned. The manifest records SHA-256 for the archive,
each consumed ZIP member, and each canonical output artifact. It intentionally
contains no timestamp or machine path, so rerunning against identical input is
byte-reproducible.

## Outputs

- `corpus.jsonl`: stable document ID, upstream document ID, title, and text;
- `queries.jsonl`: stable query ID, upstream query ID, text, and frozen split;
- `qrels.jsonl`: stable/upstream query and document IDs, relevance, and split;
- `splits.json`: ordered stable query IDs for tune, cert, and test;
- `dataset_manifest.json`: source, license, split, count, and artifact identity.

The next gate is to validate a real prepared manifest and then implement a
pluggable, revision-pinned text-embedding cache. No policy may inspect
`query_cert` or `query_test` while choosing the embedding, projection,
`m_prime`, budget grid, threshold, or safety correction.

## Validated Stage-1 artifact

The real archive was prepared on 2026-08-27 with Python `3.9.23` at commit
`aff63e4`. The returned audit archive has local SHA-256
`f6932dfe1a002c2c4f349a269f55fb56e85441106c9ead9b2e99e0192069b9f5`.
Independent local checks verified every artifact hash, every qrel/query/document
reference, split union/disjointness, corpus/query ID separation, pair
uniqueness, and a byte-for-byte regeneration from the archived source ZIP.

The frozen dataset manifest fingerprint is
`6f54d75d95c40569f7382270e833c8602afd317042e2a791118e4a15992038df`.
It contains 5,183 corpus documents, 1,109 external queries, and 1,258 positive
qrels. The split sizes are 404 tune, 405 cert, and 300 official test queries;
their qrel counts are 467, 452, and 339 respectively. This closes the real-data
preparation gate. The next stage is specified in `docs/REAL_EMBEDDINGS.md`.
