# Query-Adaptive Tri-RAG: current entry

Updated 2026-11-01. Canonical checkout:
`/Users/guanghongxu/Query-Adaptive-Tri-RAG`.
Maintained branch: **`codex/tls-rag-step5-ann`**; continue Step 5 updates there.
The reviewed/Tri-Law-repaired backup remains `codex/repo-review-code-protection`
at `62377d854ddd183925c4b62ca81653e9c14d9c34`.

## Current stage

TLS-RAG Step 5, route2 continuation v2 A→B→C is complete on scientific code
`08ed4ce`. The learned/Tri stopping paths fail overall net latency gain;
trained-support/OOD fallback remains unimplemented. Preserve the resumable
backend and recorded failures. The user's 2026-10-30 direction now assigns
**route4 follow-up research** to the continuing Step 5 agent: prioritize an
offline joint `(m,M)` advisor plus an independent candidate-pool Gaussian
screen, with independent-view union conditional on their results. Start with
theory/prior-work review and a concrete scoped implementation proposal in
[journal sections32–34](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md).
P0 is complete: the exact marginal matches the 2013 sign-flip law after
polarization; the loss/counting bridge is valid but need not be tight. Section34
proposes P1 fixed-pool projection/cost screening with five new files, explicit
test-suite membership and narrow registration changes. The user's actual reply
was “保留P0，暂缓P1实现”: retain P0; P1 is deferred, not approved for execution.
No new route4 algorithm experiment has run; exact L0 implementation/data scope
was not approved at that checkpoint. On 2026-11-01 the user explicitly said
“继续P1”, resuming the exact section34 eight-path scope. The new isolated
implementation passes442 full tests (13 new, zero skips); real-input P1 is next,
bounded to local CPU20k/96 historical queries and45 minutes. Read section36.
The original route3 rescue remains unassigned; route5 is still documentation only.

On 2026-10-31 the user requested **route5 documentation only**: offline HNSW
neighbor selection/pruning with Gaussian comparisons and exact-distance fallback.
See [journal section35](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#35-方案5离线建图时的邻居选择与剪枝2026-10-31).
Its cost screen, fixed-state diagnostics, comparator, full build and external-query
validation are proposed/not run; the conditional-risk and graph-quality bridges
remain unresolved. This record does not resume route4 P1 or authorize new runs.

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
No argument or `full` now runs442 methods; the collected cohorts are step5=244,
legacy=198. The previous429 count is the pre-P1 checkpoint.
The two subsets partition the full suite; unselected tests are not passed/skipped.
Shared-core/governance/dependency/historical changes and every implementation
delivery still require full validation. New test files need explicit classification.
Keep historical source-bound artifacts at their original source identity.

Detailed old startup narration is retained at
[the pre-cleanup snapshot](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/START_HERE.md).
Scientific contracts and the exploration journal keep their existing paths;
use [the map](docs/REPOSITORY_MAP.md) rather than preloading all historical documents.
