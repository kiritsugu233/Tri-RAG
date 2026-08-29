# FiQA source and capacity audit

## Scope

This is the stop/go source gate for the provisional Pilot-Distance Calibrated
Tri-Predict v2 dataset. It verifies only archive identity, ZIP/member integrity,
qrel referential integrity, normalized query-text grouping, and whether a
duplicate-safe five-role assignment can satisfy the checked-in 1,567-query
certification power requirement. It does not create embeddings, run retrieval
or a policy, measure latency, inspect method outcomes, or authorize protected
role access.

## Pinned source and use restriction

- BEIR archive:
  `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip`
- Official BEIR MD5: `17918ed23cd04fb15047f73e6c3bd9d9`
- Independently measured SHA-256:
  `32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02`
- Exact archive size: `17,948,027` bytes
- Dataset version/config fingerprint:
  `17b561b18b4c721f9bec843afe5351a5da038a4b928a97f5c41ad8c797b92487`
- Upstream FiQA-2018 Task 2 page:
  `https://sites.google.com/view/fiqa/home`
- BEIR redistribution disclaimer:
  `https://github.com/beir-cellar/beir#disclaimer`

The upstream page states that both Task 2 training and testing data are
available only for non-commercial use. BEIR explicitly says that redistribution
does not establish the user's permission. The audit therefore records
`commercial_use_permitted=false`; it does not reinterpret that restriction as
an open-source license.

## Exact command

After downloading the pinned archive to ignored local storage:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m \
  tri_rag_harness.pdctp_dataset_audit \
  --config configs/pdctp_fiqa_source_audit_v1.json \
  --archive data/fiqa.zip \
  --output artifacts/pdctp_fiqa_source_audit_v1
```

The command is deterministic and network-free once the archive exists. A
second local run matched both output files byte for byte. The complete offline
suite then ran 130 tests: 129 passed, one optional real-FAISS conformance test
skipped because FAISS is not installed, and zero failed.

The independent cluster audit ran in Slurm allocation `374320` on `genoa06`
using the micromamba `tri-rag` environment at commit `d5c31a3`. It ran the same
130 tests in 30.641 seconds with 129 passes, one optional FAISS skip, and zero
failures. Both generated JSON files matched the checked-in artifacts byte for
byte. The returned audit archive has SHA-256
`098910e034dfb1790913699a3c3ad9e4c13106852722821dde8218539e31f46e`
and passed independent local transfer verification.

## Findings

The archive identity, ZIP CRC, required members, and all qrel query/document
references pass. Counts at relevance at least one are:

| Item | Count |
| --- | ---: |
| Corpus items | 57,638 |
| Source queries | 6,648 |
| Train eligible queries / positive qrels | 5,500 / 14,166 |
| Dev eligible queries / positive qrels | 500 / 1,238 |
| Test eligible queries / positive qrels | 648 / 1,706 |

NFKC + casefold + whitespace-collapse normalization finds no duplicate query
text within or across the native train/dev/test qrel pools. No non-test query
therefore requires exclusion for matching test text.

There are 38 corpus items with empty title and text. All 38 are referenced by a
positive qrel: 35 in train, two in dev, and one in test. They remain part of the
pinned source identity; neither the audit nor later preparation may silently
delete them. The protocol-freeze gate must predeclare their deterministic text
representation and report their effect as a source-data limitation.

## Five-role feasibility witness

The witness is constructive but not yet a protected protocol freeze:

1. keep native test IDs in `query_test`;
2. exclude any non-test normalized-text matches to test;
3. give native dev groups priority for label-free `query_latency`;
4. order train-only normalized-text groups by
   `sha256(seed + NUL + normalized_text)` with seed `62419`;
5. allocate at least 1,567 queries to `query_cert`, then split the remainder
   approximately equally between `query_cal` and `query_tune` without splitting
   a text group.

The resulting role counts are:

| Role | Count |
| --- | ---: |
| `query_cal` | 1,966 |
| `query_tune` | 1,967 |
| `query_cert` | 1,567 |
| `query_latency` | 500 |
| `query_test` | 648 |

IDs and normalized texts are disjoint across all five roles. Subdivision uses
native membership and a stable text-group hash, never relevance magnitude.
The witness fingerprint is
`dfccd57b074d8faa11a410f5d94a970133e29fcadac9b08e4972d747249fcaff`.

## Decision and artifacts

Decision: **GO_TO_PROTOCOL_FREEZE**. FiQA has enough duplicate-safe eligible
queries for the frozen worst-case power requirement. This GO permits only the
next preregistration/freeze gate; it does not permit method evaluation or
protected-role outcome access.

Checked-in artifacts:

- `artifacts/pdctp_fiqa_source_audit_v1/source_audit.json`, SHA-256
  `dd65862cf8c87b0a40c3dcc2fae3971ec26bc7ad44b0db00803e063dfedc72d9`;
- `artifacts/pdctp_fiqa_source_audit_v1/role_feasibility_witness.json`, SHA-256
  `afe83d33062b3c7afb0da68e76fd7b8a3db7aed96a58d1c1364f3302f3940099`.

The next gate must freeze the actual five-role IDs together with the complete
real-data protocol, including the empty-document rule, evidence targets,
candidate grids, fresh projection seed, `m_prime=192`, budget grid, tolerances,
alpha allocation, hardware blocks, and every stochastic seed. Until that gate
is reviewed, no embedding or method run is authorized.
