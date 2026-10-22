# TLS-RAG Step 4: closed A800 replay

Start at [AGENTS.md](../AGENTS.md) and [START_HERE.md](../START_HERE.md).
Canonical checkout: `/Users/guanghongxu/Query-Adaptive-Tri-RAG`;
branch: `codex/tls-rag-step4-retention-v3`. No new branch or history rewriting.

The A800 v1-03 replay passed at experiment commit
`a2c2a274cd2009aab9384e0288d41e5ad7292722`; closeout was recorded at
`4a6efebfeaf2e9f9d40d66a8c140b1753baff9b7`. Inputs/results and verified archive
locations are in [STATUS.md](../STATUS.md). Historical probe outcomes were already
observed; this is not a new independent validation, certificate or latency result.

The only current preparation/replay entry is
`scripts/run_tls_rag_step4_repaired_acceptance.sh`. Read the
[A800 contract](TLS_RAG_STEP4_A800_REPLAY.md) and corresponding source/test before
any separately requested replay. Closeout does not authorize a repeated experiment.

For algorithm questions use the [retention-v3 contract](TLS_RAG_STEP4_RETENTION_V3.md),
its config/source/tests, then exact imported Step 2/3 definitions as needed.
The [reviewed-source compatibility contract](TLS_RAG_STEP4_REPAIRED_ACCEPTANCE.md)
and [numerical repair](TRI_LAW_NUMERICAL_FIX.md) explain historical source binding.
Old direct wrappers were retired; [cleanup index](MARKDOWN_REVIEW.md) preserves
fixed Git links. Do not change old loader hashes to accept incompatible inputs.

The separate synthetic v1 [brief](TLS_RAG_STEP4_BRIEF.md) and
[readiness report](TLS_RAG_STEP4_READINESS.md) remain available;
`scripts/run_tls_rag_step4_readiness.sh` still tests synthetic reproducibility.
Its closed real-data gate applies to v1, not the completed v3 replay.

The [Step 5 handoff](TLS_RAG_STEP5_HANDOFF.md) is planning-only. Creating its
agent is deferred until the user's next instruction; implementation requires
approval of a concrete experiment proposal.
