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
- adapter schema `beir_zip_v2` and ID namespace `beir-scifact`;
- the official train qrels as the development pool and the official test qrels
  as the untouched `query_test` split;
- query-text normalization by NFKC, Unicode case folding, and whitespace
  collapse before splitting;
- removal from development of any normalized query text already present in
  official test, while retaining official test unchanged;
- split seed `41017`, with remaining development queries grouped by normalized
  text, groups ordered by `sha256(seed || NUL || normalized_text)`, and whole
  groups assigned 50/50 to `query_tune` and `query_cert`.

The split procedure uses query text and IDs only, not qrel labels, document
identities, retrieval results, LID, or retention. It guarantees that a
normalized query text cannot occur in more than one of tune, cert, and test.
Corpus IDs and query IDs are explicitly namespaced as `beir-scifact:doc:*` and
`beir-scifact:query:*`, so they cannot be confused even when the upstream IDs
have the same textual value.

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
  --output data/prepared/scifact-dedup-v2
```

The adapter refuses an unexpected archive checksum, a corrupt ZIP, missing
members, duplicate IDs/qrels, qrels pointing to absent queries/documents,
overlapping development/test query IDs, normalized query text crossing frozen
splits, or an existing output directory.

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
  exclusions,
  artifacts
}' data/prepared/scifact-dedup-v2/dataset_manifest.json
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

No policy may inspect `query_cert` or `query_test` while choosing the embedding,
projection, `m_prime`, budget grid, threshold, or safety correction.

## Split-leakage audit and repaired identity

The original `beir_zip_v1` archive was prepared on 2026-08-27 with Python
`3.9.23` at commit `aff63e4`. The returned audit archive has local SHA-256
`f6932dfe1a002c2c4f349a269f55fb56e85441106c9ead9b2e99e0192069b9f5`.
Independent local checks verified every artifact hash, every qrel/query/document
reference, split union/disjointness, corpus/query ID separation, pair
uniqueness, and a byte-for-byte regeneration from the archived source ZIP.
Those ID-level checks passed, but a later text-level audit found four duplicate
normalized query-text groups. Three crossed a frozen boundary: source IDs
`1291/1292` and `871/870` crossed tune/test, while `90/89` crossed tune/cert.
The old dataset fingerprint
`6f54d75d95c40569f7382270e833c8602afd317042e2a791118e4a15992038df`
is therefore retained only as an audit trail and is prohibited from downstream
retrieval selection, certification, or test claims.

Adapter v2 was regenerated locally from the independently archived source ZIP,
without inspecting retrieval results or labels during splitting. Its config
fingerprint is
`9a05e8e23d3a09b55916d429fe1f80385d0947e6467cd9b8061e19532272285f`
and its dataset-manifest fingerprint is
`4a73586d3a29a0567287e501ac3c06c998af661cdc74dbc589e7525a7924f903`.
It retains all 5,183 documents, 1,107 of 1,109 source queries, and 1,256 of
1,258 positive qrels. Development IDs `1291` and `871` are excluded because
their normalized text occurs in official test; the test records are unchanged.
The repaired split sizes are 403 tune, 404 cert, and 300 test, with 453, 464,
and 339 qrels. All three normalized-text intersections are empty. The two
remaining duplicate groups (`85/86` and `89/90`) each remain wholly inside one
split. A cluster byte-reproduction of this repaired identity is the next gate.
