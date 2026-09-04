# FiQA query_cert certification gate

This gate performs the one-time scientific certification of the frozen PDCTP
suite. It is implemented by `tri_rag_harness.pdctp_fiqa_query_cert` and frozen
by `configs/pdctp_fiqa_query_cert_v1.json`, whose fingerprint is
`c6357a748f7f3262f481c1f597d3f25acb76892a9dbfc621735effe6b0bd8143`.

## Scope and access order

Before certification access, the runner validates the protocol, five-role
assignment, source audit, qrel-free dataset and embedding cache, accepted
query_tune audit, all 16 returned tune-file hashes, terminal selection, six
frozen policies, component registry, Bonferroni hypotheses, power plan,
projection, and exact post-tune guard state. It reconstructs every policy and
replays the complete cal/tune state transition. Any mismatch stops before a
certification token is issued.

Only after those checks and the complete FiQA archive hash pass may the runner
open all 1,567 frozen `query_cert` IDs in their original order. FiQA's combined
train qrel member is filtered by its first column; document IDs and relevance
values are parsed only for cert-role rows. Cal, tune, latency, and test qrel
outcomes are skipped before interpretation.

The runner never fits a calibrator, selects or modifies a policy, expands or
clips a budget, measures latency, accesses `query_test`, invokes an approximate
index, or runs an LLM. `oracle_exact` LID, exact top-k identities, qrels, and
realized outcomes are excluded from every policy decision.

## Frozen evaluation

The exact normalized E5 cache and dense Gaussian matrix are reused with
`m_prime=192`, seed `83047`, variance `1/192`, and NumPy scale
`1/sqrt(192)`. Projected vectors are not renormalized. Original and projected
distances are float64 squared L2 with lexicographic document-ID tie breaking.

Each query performs one projected full-corpus scan. Its top-64 prefix is
reranked in original space to construct the deployable feature vector and
canonical 10-decimal pilot LID. All six frozen policies decide from that same
label-free observation before exact top-10 and relevance outcomes are reduced.
The projected ranking is then reused for each selected candidate prefix and
exact original-space reranking to `k_ctx=5`.

One query-level record stores pilot distances and features, every policy
decision and fingerprint, selected `M`, exact-top-10 projected ranks,
cert-positive qrel ranks, candidate evidence recall, final evidence recall,
reranked context IDs, and separate pilot/expansion/rerank work. These records
are sufficient to reconstruct every aggregate without reopening the source.

## Certification family

The six hypotheses and Bonferroni allocation `alpha=0.05/6` are consumed
unchanged from the tune artifact. Full PDCTP is tested for:

- absolute embedding-retention lower bound at least `0.95`;
- candidate and final evidence noninferiority to frozen fixed `M=768`, each
  with margin `-0.02`; and
- strictly negative upper paired normalized-budget bounds versus fixed,
  monotone-binned, and Raw Tri-Predict.

Every bound is an empirical-Bernstein bound on paired per-query differences
and retains the complete left/right/difference vector. The runner immediately
reconstructs each bound from those values. A PASS requires all six hypotheses;
otherwise the result is a terminal FAIL. Both outcomes close certification and
forbid a repeat, refit, retune, comparator substitution, or budget expansion.

The tune data already show PDCTP using more work than fixed, so the frozen
budget-superiority claim is at risk. That observation cannot justify changing
the certification contract.

## Outputs

A completed run writes eight artifacts:

- `query_cert_access.json`;
- `query_cert_qrel_access.json`;
- `projection.json`;
- `query_cert_records.jsonl`;
- `certification.json`;
- `protocol_state_after_query_cert.json`;
- `manifest.json`; and
- `report.md`.

The implementation has only been exercised with synthetic cert fixtures and a
metadata-only replay of the accepted real tune state. The real `query_cert`
role remains unopened until the committed runner is pulled to the cluster and
executed once.

The CLI also accepts `--preflight-only`. That mode performs every upstream,
cache, returned-file, policy, hypothesis, state, and source-archive validation
used by the real command, then asserts that the post-tune state is unchanged
and exits without creating an output directory or opening query_cert. Run this
once on the cluster before removing the flag for the single protected run.

## Runner command

Run from the repository root inside the existing `tri-rag-faiss` micromamba
environment:

```bash
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

python3 -m tri_rag_harness.pdctp_fiqa_query_cert \
  --config configs/pdctp_fiqa_query_cert_v1.json \
  --real-protocol-config configs/pdctp_fiqa_real_protocol_freeze_v1.json \
  --protocol-freeze artifacts/pdctp_fiqa_real_protocol_v1/protocol_freeze.json \
  --role-assignments artifacts/pdctp_fiqa_real_protocol_v1/role_assignments.json \
  --source-audit artifacts/pdctp_fiqa_source_audit_v1/source_audit.json \
  --fiqa-archive data/fiqa.zip \
  --embedding-audit artifacts/pdctp_fiqa_e5_v1/embedding_audit.json \
  --embedding-config configs/pdctp_fiqa_e5_base_v2_embeddings.json \
  --prepared data/pdctp-fiqa-text-375414-a \
  --embedding-cache data/embeddings/pdctp-fiqa-e5-375414 \
  --query-tune-audit artifacts/pdctp_fiqa_query_tune_v1/query_tune_audit.json \
  --query-tune-run runs/pdctp-fiqa-query-tune-376924 \
  --power-plan artifacts/pdctp_network_free/power_plan_v1.json \
  --output runs/pdctp-fiqa-query-cert-<job-id>
```

Append `--preflight-only` to that command for the safe validation pass. Its
success message must say that query_cert remains closed. The subsequent real
command must use the same inputs and output path without the flag, and must not
be repeated after it opens the protected role.

The local network-free suite contains 161 tests: 160 pass and the optional
real-FAISS conformance test is skipped because FAISS is absent. Six new cert
tests cover the exact config and accepted tune identity, first-column qrel
filtering, deterministic one-scan records, deployable-only decisions, all six
query-reconstructable paired bounds, and terminal one-time role behavior.
