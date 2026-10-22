# Query-Adaptive Tri-RAG: current entry

Updated 2026-10-22. Local checkout: `/Users/guanghongxu/Query-Adaptive-Tri-RAG`.
Continue on `codex/tls-rag-step4-retention-v3`; do not create a new branch.
The review backup `codex/repo-review-code-protection` remains untouched at
`62377d854ddd183925c4b62ca81653e9c14d9c34`.

## Current state and next assignment

Step 4 A800 reconstructed-data replay is closed: `replay_targets_met`, with
all three gates passed. This is not independent validation or a latency,
evidence-sufficiency or answer-quality claim. See the latest [status](STATUS.md).

[Step 5 handoff](docs/TLS_RAG_STEP5_HANDOFF.md) is prepared for a future agent.
The user deferred creating/dispatching that agent until further instruction.
Its first task will be a short ANN experiment proposal, followed by user approval;
implementation, dependency installation and benchmark execution are not yet authorized.
ANN remains outside the old MVP; the proposal must specify a narrow new-stage extension.

## Read in order

1. [AGENTS.md](AGENTS.md): authoritative scope, canonical checkout and L0 approval.
2. This entry and the latest relevant section of [STATUS.md](STATUS.md).
3. Relevant [protection entries](docs/CODE_PROTECTION.md); run the read-only check.
4. Only the task-specific brief and code needed for the authorized work.

```bash
cd /Users/guanghongxu/Query-Adaptive-Tri-RAG
python3 scripts/check_code_protection.py
sh scripts/run_tests.sh
```

## Preserved references

- [Step 4 entry](docs/TLS_RAG_STEP4_START_HERE.md), [A800 contract](docs/TLS_RAG_STEP4_A800_REPLAY.md),
  [reviewed-source acceptance](docs/TLS_RAG_STEP4_REPAIRED_ACCEPTANCE.md).
- [Tri-Law specification](docs/TRI_LAW_SPEC.md), [numerical repair](docs/TRI_LAW_NUMERICAL_FIX.md),
  [repository review](docs/REPOSITORY_REVIEW.md).
- [Raw Tri-Predict negative baseline](docs/RAW_TRI_PREDICT_V1_BASELINE.md),
  [calibrated protocol](docs/CALIBRATED_TRI_PREDICT_PROTOCOL.md),
  [v3 diagnosis](docs/CALIBRATED_TRI_PREDICT_V3_DIAGNOSIS.md).
- [Cleanup and historical-document index](docs/MARKDOWN_REVIEW.md).

Historical briefs, reports and unchecked plan items do not authorize reopening
protected roles. TLS-RAG retention v3 and Calibrated Tri-Predict v3 are different
families. Preserve source-bound inputs; never refresh old bindings after source
changes. `agent.md` is only an alias for AGENTS.md.
