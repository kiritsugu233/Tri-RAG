# Query-Adaptive Tri-RAG: current entry

Updated 2026-09-07. Local checkout:
`/Users/guanghongxu/Query-Adaptive-Tri-RAG`.

The maintained code is **the reviewed repository with Tri-Law numerical v2**
on the single continuing Step 4 branch `codex/tls-rag-step4-retention-v3`.
[GitHub current code](https://github.com/kiritsugu233/Tri-RAG/tree/codex/tls-rag-step4-retention-v3).
The review backup `codex/repo-review-code-protection` remains at
`62377d854ddd183925c4b62ca81653e9c14d9c34`; subsequent implementation does not update it.
The existing Step 4 branch includes that reviewed baseline and keeps its name
across future versions. This supersedes the earlier review-branch workflow.
All subsequent authorized implementation starts from that branch's latest HEAD;
do not automatically create another branch for each step. Local directory stays
`/Users/guanghongxu/Query-Adaptive-Tri-RAG`.

The experimental family remains TLS-RAG Step 4 retention v3. Commit `248c29e`
is its historical pre-review baseline; `c40b7b0` is the review-only snapshot.
Neither is the new development starting point. The approved numerical changes,
checks and source-binding consequences are in
[Tri-Law repair](docs/TRI_LAW_NUMERICAL_FIX.md). Inspect actual branch/HEAD before
working; do not switch to an old branch based on a historical document.

V3 implementation tests passed. The user explicitly confirmed on 2026-09-07
that the v3 real acceptance code has not been run. The v2 NFCorpus tune failure
is user-reported. Neither implies a positive v3 scientific result, formal
certificate, serving latency improvement, or authorization for Step 5.

## Current acceptance entry (2026-09-10)

Use [reviewed-source acceptance](docs/TLS_RAG_STEP4_REPAIRED_ACCEPTANCE.md).
The approved compatibility entry validates the pinned historical v2 bundle and
rebuilds every numerical feature using reviewed Tri-Law. Old loaders still reject
old/current source mismatches; never rewrite historical hashes. The retention-v3
policy and thresholds remain frozen. Real acceptance has not run locally.
For this milestone the user will commit and push; leave implementation changes
in the canonical checkout for that handoff, without creating another branch.

## Read before working

1. [AGENTS.md](AGENTS.md): scope, preservation and prior approval for L0 changes.
2. [Code protection](docs/CODE_PROTECTION.md): per-file levels and review limits.
   Run `python3 scripts/check_code_protection.py` before editing and at handoff.
3. The brief named in the user's task. For the current experiment, read
   [TLS-RAG Step 4 start here](docs/TLS_RAG_STEP4_START_HERE.md).

The [review report](docs/REPOSITORY_REVIEW.md) retains original defect evidence;
[the repair record](docs/TRI_LAW_NUMERICAL_FIX.md) closes R1/R2 for the tested
numerical scope. Consult both before changing the scientific core.
[Markdown review](docs/MARKDOWN_REVIEW.md) identifies every tracked Markdown
file's role. It is an index, not a request to preload historical documents.

Do not preload the large historical status, implementation plan, artifacts,
runs, or archives for an ordinary next-step task. Use the bounded current
sections when recording a milestone. The user's explicit repository-wide
review authorizes reviewing repository source and Markdown; it does not turn
untracked experiment archives into fresh evaluation data.

## Version boundaries

| Family | Recorded state | Appropriate entry |
| --- | --- | --- |
| Raw Tri-Predict v1 | Terminal negative baseline, tag `raw-tri-predict-v1-terminal-negative` (`fb09c00`) | [Historical baseline](docs/RAW_TRI_PREDICT_V1_BASELINE.md) |
| Calibrated Tri-Predict v2/v3 | Frozen historical calibration/diagnosis work; no positive successor claim | [Historical v2 protocol](docs/CALIBRATED_TRI_PREDICT_PROTOCOL.md), [v3 diagnosis](docs/CALIBRATED_TRI_PREDICT_V3_DIAGNOSIS.md) |
| TLS-RAG Steps 1–3 | Completed design and synthetic implementation | Historical briefs only when specifically needed |
| TLS-RAG Step 4 v1 | Completed synthetic protocol/readiness; its real gate remains closed | [V1 readiness](docs/TLS_RAG_STEP4_READINESS.md) |
| TLS-RAG Step 4 v2 | Real retrieval-proxy probe; user reported no qualifying tune candidate | [V2 probe](docs/TLS_RAG_STEP4_REAL_PROBE.md) |
| TLS-RAG Step 4 retention v3 with reviewed Tri-Law | Compatibility entry tested; real acceptance pending | [Reviewed-source acceptance](docs/TLS_RAG_STEP4_REPAIRED_ACCEPTANCE.md) |

Calibrated Tri-Predict v3 and TLS-RAG retention v3 are different methods.
An old prompt, unchecked historical plan item, or reported past gate does not
authorize rerunning protected roles or implementing another step.

## Offline validation

From the canonical checkout, using the existing NumPy/SciPy environment:

```bash
python3 scripts/check_code_protection.py
sh scripts/run_tests.sh
```

For a fresh synthetic v1 reproducibility check, use an absent output parent:

```bash
sh scripts/run_tls_rag_step4_readiness.sh /private/tmp/tls-rag-readiness-NEW
```

These commands do not run the real experiment. The unchanged old loaders still
reject pre-repair source-bound bundles. Historical direct replay requires its
original commit. The explicitly approved repaired-acceptance entry instead
validates the pinned historical provenance, reuses verified vectors, recomputes
features and creates a separate reviewed-source binding without rewriting old
hashes. Follow the new acceptance brief; actual cluster success requires an actual result.

The original walking skeleton already exists. Its runnable config is
`configs/synthetic_mvp.json`; the old proposed `configs/mvp_scifact.yaml`
was never the implementation CLI. Historical architecture and experiment
documents describe their respective method families, not the current task list.
