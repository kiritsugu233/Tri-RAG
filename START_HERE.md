# Query-Adaptive Tri-RAG: current entry

Updated 2026-11-02. Canonical checkout:
`/Users/guanghongxu/Query-Adaptive-Tri-RAG`.
Maintained branch: **`codex/tls-rag-step5-ann`**; continue Step 5 updates there.
The reviewed/Tri-Law-repaired backup remains `codex/repo-review-code-protection`
at `62377d854ddd183925c4b62ca81653e9c14d9c34`.

## Current stage

The current user requested closure verification for **route4**, then beginning
**route5 offline HNSW construction**. Route4 is closed for the current method:
the no-candidate/no-net-benefit results, resource_stop and missing checks are
retained; do not resume it. Archive/index and eight review payloads were verified
again on2026-11-02, with GitHub HEAD matching the local closure record.
Read journal section41 for the route5 source audit and concrete **R5-A cost-only
proposal**. The user's subsequent actual reply was **“仅保留提案，暂不实施”**:
route5 is deferred/not_run. Retain the proposal; do not implement, restore its
environment, read its scientific inputs or run R5-A–E under the earlier general
direction. Resume only after a new explicit instruction covering the scope.
The old temporary FAISS environment and HNSW headers remain incomplete; any
later authorized implementation must restore the same pinned versions/verified source.

The user has handed the separate **joint-Gaussian rank-tail theory study** to a
new chat, managed independently from the Step5 systems experiments. Read the
[T0 literature audit, symbolic derivation and sole claim card](docs/JOINT_RANK_THEORY_T0.md)
and the [original framework](docs/TRI_LAW_JOINT_RANK_RESEARCH_FRAMEWORK.md).
T0 identifies a signed-cycle candidate: identical proper-subset inversion laws
can coexist with a positive fixed-target rank-tail gap, including on the unit
sphere. The cycle identifiability mechanism has close prior work; publication
novelty is **not established**, and the derivation awaits independent review.
The six-path deterministic T1 proposal is not approved; no new scientific code,
protocol or simulation has run. Keep the old Step5 journal read-only for this
study; record new theory work in the independent note/namespace. The previous
system stops and closed data roles below remain in force.

TLS-RAG Step 5, route2 continuation v2 A→B→C is complete on scientific code
`08ed4ce`. The learned/Tri stopping paths fail overall net latency gain;
trained-support/OOD fallback remains unimplemented. Preserve the resumable
backend and recorded failures. The user's historical 2026-10-30 direction assigned
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
Exact L0 implementation/data scope was not approved at that checkpoint.
On 2026-11-01 the user explicitly said
“继续P1”, resuming the exact section34 eight-path scope. The new isolated
implementation now passes445 full tests (16 P1, zero skips), after repairing
a pre-input monitor startup failure. P1 is now complete on scientific source
`b38ad51`:20k corpus,32 advice+32 select queries evaluated, eight design seeds,
157 full timing configurations. No Gaussian configuration meets quality; all12
free-M oracles are infeasible. Report32 stays closed; do not start P2 training,
retune this grid, reopen report or rerun. Read section36 for results and audit.
The original route3 rescue remains unassigned; route5's current preparation is in section41.

Section38's two independent application scenarios were explicitly approved:
“批准第38节全部范围，按A→B执行”. The isolated five-file implementation and
narrow test/register additions now pass467 full tests (22 new, zero skips).
After source commit/push verification, temporary-bank lifecycle A then paid
offline-risk decisions B ran locally, within the declared45min/4GiB/2GiB limits.
Read journal section39 for the actual artifact-limit stop and overshoot.
This is a new namespace, not a P1 rerun; original report/cert/test stay closed.
The run on source`1880435` has now stopped at349.469s due to its2GiB
artifact budget: A is complete with no Gaussian lifecycle winner; all seven
B advisor choices are sealed, with Tri costing~10.79x direct-three and selecting
byte-identical deployment models. Extra-seed checks completed5/7; the sixth
was truncated and the seventh was not run. Preserve resource_stop and do not
resume/retune automatically. Read journal39.3–39.7 and STATUS for actual limits.

On 2026-10-31 the user requested **route5 documentation only**: offline HNSW
neighbor selection/pruning with Gaussian comparisons and exact-distance fallback.
See [journal section35](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#35-方案5离线建图时的邻居选择与剪枝2026-10-31).
Its cost screen, fixed-state diagnostics, comparator, full build and external-query
validation remain not run; the new direction/proposal is recorded in section41.
The conditional-risk and graph-quality bridges
remain unresolved. That route5 record did not authorize route4 P1 or a route5
run; P1's separate authorization and completed negative result are above.

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
No argument or `full` now runs467 methods; the collected cohorts are step5=269,
legacy=198. The previous429 count is the pre-P1 checkpoint.
The two subsets partition the full suite; unselected tests are not passed/skipped.
Shared-core/governance/dependency/historical changes and every implementation
delivery still require full validation. New test files need explicit classification.
Keep historical source-bound artifacts at their original source identity.

Detailed old startup narration is retained at
[the pre-cleanup snapshot](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/START_HERE.md).
Scientific contracts and the exploration journal keep their existing paths;
use [the map](docs/REPOSITORY_MAP.md) rather than preloading all historical documents.
