# Query-Adaptive Tri-RAG: current entry

Updated 2026-10-25. Local checkout: `/Users/guanghongxu/Query-Adaptive-Tri-RAG`.
Continue all Step 5 work and synchronization on `codex/tls-rag-step5-ann`.
The user explicitly authorized this new branch and redundant-branch cleanup;
do not create another branch for subsequent Step 5 version updates.
The review backup `codex/repo-review-code-protection` remains untouched at
`62377d854ddd183925c4b62ca81653e9c14d9c34`.

## Current state and next assignment

The authorized [mechanism A/B study](docs/TLS_RAG_STEP5_MECHANISM_PROTOCOL.md) is
complete: triplet features improve prediction MAE by5.68%, but the selected policy
uses more candidates and has higher measured latency. The tested acceleration
route is stopped; no follow-on real-data retrieval/policy-fitting experiment is
authorized. A subsequent user-authorized local diagnostic prototype batches the
unchanged law: T construction improves about7.6x on synthetic Mac inputs, with
exact tested feature/decision equality. It is not installed in frozen sources or
a server speedup result; read the latest STATUS.md for its scope and policy-oracle limits.
New code stays under
mechanism/ to preserve legacy bindings. Read STATUS.md for actual tables, verified
cloud/local artifacts and the controls' evaluation-quality limitation. Data were
already-open cal/tune queries, with no fresh independent certification claim.

Step 4 A800 reconstructed-data replay is closed: `replay_targets_met`, with
all three gates passed. This is not independent validation or a latency,
evidence-sufficiency or answer-quality claim. See the latest [status](STATUS.md).

[Step 5 ANN v1](docs/TLS_RAG_STEP5_ANN.md) is implemented in a new namespace after
the user's 2026-10-22 proposal-review/implementation authorization. The A800
synchronization, 337 offline tests and required real-HNSW synthetic self-test
passed on code `40728dd`; see STATUS.md for actual logs and timings.
The subsequent authorized real run ended with `no_tune_candidate`: calibration
passed, but no Tri candidate met the declared cost gate. Cert/latency/test stayed
closed. Results and inputs are verified on cloud storage; duplicate hot copies
were replaced with cloud links. Read the latest storage/diagnosis and closeout in
STATUS.md. History/semantic independence remain unestablished; no formal claim exists.
ANN is a narrow new-stage extension; the old MVP's scope and Step 4 bindings remain frozen.
The [earlier handoff](docs/TLS_RAG_STEP5_HANDOFF.md) records the pre-authorization boundary.

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
