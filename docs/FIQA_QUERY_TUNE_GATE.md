# FiQA query_tune selection gate

This is the second protected-outcome gate in the fresh PDCTP protocol. It is
implemented by `tri_rag_harness.pdctp_fiqa_query_tune` and frozen by
`configs/pdctp_fiqa_query_tune_v1.json` (fingerprint
`06c647625bb01192b54ae0698e9e4150fe4fec0d2b4407858de74c763573d7d0`).

## Preconditions and role scope

Before opening any tune outcome, the runner validates the protocol, role,
source, qrel-free dataset, embedding, accepted query-calibration audit, all
eight returned query-calibration files, both candidate bundles, and the power
plan. It reconstructs all four LID candidates, 675 residual base models, and
all 2,025 residual operating points, then exactly replays the post-calibration
five-role state. The accepted query-calibration audit fingerprint is
`e3cd09d125a868b685df02b23f0706926fa5786752d863f17bf13d6293de7884`.

Only then may the guard open the complete 1,967-ID `query_tune` role in its
frozen order. `query_cert`, `query_latency`, and `query_test` remain closed.
The runner cannot refit or replace a calibrator, run certification, measure
latency, invoke an LLM, or use an approximate index.

FiQA stores query_cal, query_tune, and query_cert qrels together in the native
train member. The tune loader therefore reads the first TSV column first. It
parses document IDs and relevance only when that stable query ID belongs to
`query_tune`; every other row is skipped before its outcome fields are parsed.
The full source archive and train-member identities remain bound to the
accepted source audit.

## Retrieval and per-query evidence

The runner reuses the frozen dense Gaussian matrix (`m_prime=192`, seed
`83047`, variance `1/192`) and refuses any matrix-identity difference from the
query_cal run. Inputs are normalized before projection, projected vectors are
never renormalized, and original/projected search uses exact float64 squared
L2 with lexicographic document-ID ties.

Each query performs one projected distance scan. Its top-64 prefix supplies
pilot reranking and deployable features; the same projected order supplies all
21 expansion budgets. Exact original-space top-10 identities label embedding
retention. Positive tune qrels label candidate evidence recall, and exact
original reranking of each candidate prefix to `k_ctx=5` labels final evidence
recall. Query records preserve every identity, rank, curve, and final reranked
top-k needed for independent reconstruction.

## Frozen family selection

The runner evaluates exactly 2,086 preregistered candidates:

| method family | candidates |
| --- | ---: |
| fixed | 21 |
| monotone-binned | 15 |
| Raw Tri-Predict | 5 |
| LID-calibration-only | 20 |
| budget-residual-only | 405 |
| full PDCTP | 1,620 |

The fixed reference is the smallest frozen budget whose query_tune
empirical-Bernstein retention lower bound reaches 0.95. Relative to that
reference, every other candidate must reach the same retention lower-bound
target and stay within the predeclared 0.02 candidate/final evidence
noninferiority margins. Each method family is optimized independently among
its eligible candidates by minimum common coordinate work, then lower mean
budget, then canonical candidate fingerprint. This avoids weakening a
comparator by forcing it to reuse the full PDCTP hyperparameters.

The complete `2086 x 1967` int32 candidate-budget matrix is retained alongside
query-level outcome curves, so every aggregate and selection decision can be
recomputed. On success, a deduplicated component registry and reconstructable
six-method suite are frozen. The preregistered Bonferroni hypotheses are then
frozen before, but do not open, query_cert. The seed-83059 shuffled-profile
diagnostic is tune-only and explicitly excluded from fit, selection, and
certification.

If any method family has no eligible candidate, the gate writes a terminal
failure and stops. Retuning, changing a threshold, expanding the budget grid,
or substituting a weaker comparator is forbidden.

## Outputs

A successful run writes 16 artifacts, including:

- tune access and qrel-access records;
- the projection and 1,967 query-level retrieval/evidence records;
- all candidate outcomes and the int32 per-query budget matrix;
- the frozen selection, six policies, and deduplicated selected components;
- per-query records for all six selected methods;
- the shuffled-profile diagnostic and frozen certification hypotheses;
- the post-tune protocol state, manifest, and report.

Portable artifacts contain no timestamps or measured latency. The success
decision is `QUERY_TUNE_SELECTION_FROZEN_READY_FOR_CERT_IMPLEMENTATION`; it
authorizes implementation and audit of the certification runner, not access to
query_cert.

## Runner command

From the repository root, substitute the existing source archive, preparation,
embedding cache, query_cal run, and a fresh output name:

```bash
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

python3 -m tri_rag_harness.pdctp_fiqa_query_tune \
  --config configs/pdctp_fiqa_query_tune_v1.json \
  --real-protocol-config configs/pdctp_fiqa_real_protocol_freeze_v1.json \
  --protocol-freeze artifacts/pdctp_fiqa_real_protocol_v1/protocol_freeze.json \
  --role-assignments artifacts/pdctp_fiqa_real_protocol_v1/role_assignments.json \
  --source-audit artifacts/pdctp_fiqa_source_audit_v1/source_audit.json \
  --fiqa-archive <fiqa.zip> \
  --embedding-audit artifacts/pdctp_fiqa_e5_v1/embedding_audit.json \
  --embedding-config configs/pdctp_fiqa_e5_base_v2_embeddings.json \
  --prepared <prepared-text-directory> \
  --embedding-cache <embedding-cache-directory> \
  --query-cal-audit artifacts/pdctp_fiqa_query_cal_v1/query_cal_audit.json \
  --query-cal-run <returned-query-cal-run-directory> \
  --power-plan artifacts/pdctp_network_free/power_plan_v1.json \
  --output runs/pdctp-fiqa-query-tune-<job-id>
```

The real query_tune role has not been opened. It remains closed until this
implementation, tests, configuration, and documentation are committed and the
user starts the one-time cluster run.
