# TLS-RAG Step 4 current entry

Updated 2026-09-07. Begin with [root AGENTS](../AGENTS.md) and
[START_HERE](../START_HERE.md). Use the canonical checkout and inspect actual
branch, HEAD and changes. The current implementation
starts from the HEAD of `codex/tls-rag-step4-retention-v3`, including reviewed baseline
`62377d854ddd183925c4b62ca81653e9c14d9c34` and the approved Tri-Law numerical v2 repair.
Keep `codex/repo-review-code-protection` at that baseline as a backup; use the same
Step 4 branch for subsequent updates without creating new branches. `248c29e` is only the historical experiment
baseline. Read [the repair record](TRI_LAW_NUMERICAL_FIX.md): old v2 source-bound
bundles remain rejected by the unchanged old loaders; historical direct replay
requires the original commit. The approved compatibility entry below supports
reviewed-source recomputation with explicit historical validation, not rehashing.
Continue new development on the current branch.

## Current repaired-source acceptance (2026-09-10)

The user approved the separate [compatibility entry](TLS_RAG_STEP4_REPAIRED_ACCEPTANCE.md).
Read that document, `src/tri_rag_harness/tls_rag_step4_repaired_acceptance.py`,
its matching test and `scripts/run_tls_rag_step4_repaired_acceptance.sh` for the
current runnable command. It checks historical provenance and recomputes features;
it does not modify old loaders, caches or retention-v3 policy semantics.
The older direct v3 command below is historical and cannot accept old bindings
on this repaired checkout. The user will commit/push this milestone manually.

## Frozen retention-v3 engine reference

Read in this order when the task concerns the latest experiment:

1. [Code protection registry](CODE_PROTECTION.md), relevant file entries only.
2. [Canonical workflow](TLS_RAG_LOCAL_WORKFLOW.md).
3. [Retention-v3 brief](TLS_RAG_STEP4_RETENTION_V3.md).
4. `configs/tls_rag_step4_retention_v3.json`.
5. `src/tri_rag_harness/tls_rag_step4_retention_v3.py` and
   `tests/test_tls_rag_step4_retention_v3.py`.
6. `scripts/run_tls_rag_step4_retention_v3.sh`.

For precise parent-interface questions, read
[TLS-RAG v2 probe](TLS_RAG_STEP4_REAL_PROBE.md), its source/config/tests,
`tls_rag_step4_nfcorpus.py`, and the exact Step 2/3/shared definitions those
files import. For a numerical-core change, also read the relevant review
finding and obtain L0 approval before editing. This list explicitly allows
root governance; older narrower startup allowlists do not override it.

V3 reuses the specified v2 development roles and requires the same actually
unopened probe. The prior implementation authorization and documented exact
cluster command persist; this review neither executes that command nor records
a new real result. No automatic probe retuning, new protected-data access,
formal certificate, latency claim or Step 5 follows from a passing unit test.

## V1 readiness and historical provenance

The initial readiness protocol is a separate, completed milestone. Its literal
`m_prime=2`, `[3,6,12]` budgets and closed real-data gate remain unchanged.
For a requested v1 replay, read its [brief](TLS_RAG_STEP4_BRIEF.md),
[readiness report](TLS_RAG_STEP4_READINESS.md), frozen config/source/test, and
`scripts/run_tls_rag_step4_readiness.sh`. The read-only replay command is:

```bash
sh scripts/run_tls_rag_step4_readiness.sh /private/tmp/tls-rag-readiness-NEW
```

Use an absent output parent. This script checks synthetic reproducibility; its
"real-data gate remains closed" message refers to v1. The separate v2/v3
experiment permissions do not open v1's gate or authorize Step 5.

[The original Step 3-to-Step 4 handoff](TLS_RAG_STEP3_TO_STEP4_HANDOFF.md)
records ancestry and the original v1 startup boundary; it is not the current
successor assignment. Historical STATUS/plan access remains bounded to the
marked Step 4 sections unless the user's current task explicitly requires more.
