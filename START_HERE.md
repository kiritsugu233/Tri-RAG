# Query-Adaptive Tri-RAG: current entry

Updated 2026-10-30. Canonical checkout:
`/Users/guanghongxu/Query-Adaptive-Tri-RAG`.
Maintained branch: **`codex/tls-rag-step5-ann`**; continue Step 5 updates there.
The reviewed/Tri-Law-repaired backup remains `codex/repo-review-code-protection`
at `62377d854ddd183925c4b62ca81653e9c14d9c34`.

## Current stage

TLS-RAG Step 5, route2 continuation v2 A→B→C is complete on scientific code
`08ed4ce`. The learned/Tri stopping paths fail overall net latency gain;
trained-support/OOD fallback remains unimplemented. Preserve the resumable
backend and recorded failures. No new algorithm experiment is assigned by
this entry. Routes3/4 await their own instructions; route4 is a plan only.

Read [current STATUS](STATUS.md) for the latest result and cleanup checkpoint.
The [living exploration journal](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md),
sections27–31, retains the complete v2 process/results. Earlier screens and
milestones are indexed in the [repository map](docs/REPOSITORY_MAP.md).

## Read before work

1. [AGENTS.md](AGENTS.md), then this file and [STATUS.md](STATUS.md).
2. The relevant [protection entries](docs/CODE_PROTECTION.md), with
   `docs/code_protection.json` as the machine inventory.
3. The user's task-specific brief/code. Use the
   [repository map](docs/REPOSITORY_MAP.md) to locate it.
4. Run `python3 scripts/check_code_protection.py` before editing and handoff.

The current repository review/cleanup is in
[REPOSITORY_CLEANUP_REVIEW](docs/REPOSITORY_CLEANUP_REVIEW.md).
Its four-path guard/register repair A was approved and implemented; see STATUS
for verification. [Test simplification B](docs/TEST_SUITE_SIMPLIFICATION_PROPOSAL.md)
was subsequently approved by the user's “执行” and implemented within its exact scope.
All independent coverage remains; further protected changes need their own scope.
Old prompts, plan items and historical approvals are not a fresh execution request.

## Validation and history

```bash
python3 scripts/check_code_protection.py
sh scripts/run_tests.sh
# Optional daily subset; full is still required before implementation handoff:
sh scripts/run_tests.sh step5
```

The full Step 5/native suite needs the existing fixed-version environment and
native libraries; see [validation](docs/REPOSITORY_CLEANUP_REVIEW.md#validation).
`TRI_RAG_TEST_PYTHON` selects the interpreter without changing dependencies.
No argument or `full` runs429 methods; `step5` runs231 and `legacy` runs198.
The two subsets partition the full suite; unselected tests are not passed/skipped.
Shared-core/governance/dependency/historical changes and every implementation
delivery still require full validation. New test files need explicit classification.
Keep historical source-bound artifacts at their original source identity.

Detailed old startup narration is retained at
[the pre-cleanup snapshot](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/START_HERE.md).
Scientific contracts and the exploration journal keep their existing paths;
use [the map](docs/REPOSITORY_MAP.md) rather than preloading all historical documents.
