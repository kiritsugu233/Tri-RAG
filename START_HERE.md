# Query-Adaptive Tri-RAG: current entry

Updated 2026-11-03. Canonical checkout:
`/Users/guanghongxu/Query-Adaptive-Tri-RAG`.
Maintained branch: **`codex/tls-rag-step5-ann`**; continue Step 5 updates there.
The reviewed/Tri-Law-repaired backup remains `codex/repo-review-code-protection`
at `62377d854ddd183925c4b62ca81653e9c14d9c34`.

## Current stage

**Latest task: R6-B0 full source verified; CPU audit in progress, batch completion proposed (2026-11-03).**
Scientific commit9f963f9 is on GitHub and the clean node. Local563 tests and
Linux26 tests pass. The only authorized B0 started at09:57:54Z; full source
size/SHA passed after1267.661s, with the historical partial unchanged.
Actual train is5,233,329x640; learn96,845x640; test payload stays closed.
Read [section51](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#51-b0首轮完整来源取得cpu吞吐问题与批量a800完成提案2026-11-03待批准).
The per-query CPU scan takes42–56s, incompatible with128 queries in1200s.
This is an audit implementation throughput defect, not Gaussian quality evidence.
At10:28:40Z11 queries were complete; the original limits remain active.
A separate nine-path v2 FP64 A800 batch/CPU near-order proposal needs explicit
consent for GPU use and another run. No v2 code/GPU has run at this checkpoint.

**Implementation checkpoint: R6-B0 local validation (2026-11-03).**
The user answered section49's exact question: “批准第49节九路径、环境准备、测试同步及一次B0”.
Six new files and narrow test/protection updates implement complete-source,
learn-only, general real-geometry auditing. Full563 CPU/offline tests pass with
zero skips (68.270s), including26 new tests. Old scientific files/contracts stay
fixed;269 registered files pass protection. Read
[section50](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#50-r6-b0获批实施与本地完整验收2026-11-03).
Next: GitHub/clean Linux synchronization, Linux tests, then the single approved
bounded source preparation/geometry run. No real query payload or source resume
has run at this checkpoint. Do not auto retry or broaden the approved budget.

**Preparation checkpoint: R6-B0 source/node scope (2026-11-03).**
The user said to start and confirmed the SSH endpoint. Node access now works:
A80080GB idle,14 CPU quota,120GiB RAM; retained partial is10,703,863,808 bytes.
The complete frozen source must be13,688,159,632 bytes with its published SHA.
The retained HDF5 environment is usable without a server install. Read
[section49's nine-path proposal](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#49-r6-b0只读准备实际来源与精确九路径提案2026-11-03).
Official default k100 requires a320-point cap (312 planned), rather than the
earlier256. Proposed128 development queries come only from `learn`; `test`
payloads stay closed. Lost original text-ID mapping prevents a semantic
independence claim. Native inner-product and stored-vector L2 orders must be
audited separately. No source resume, query sampling or new scientific code has
run at the preparation checkpoint. Exact L0 scope was subsequently approved above.

**Previous plan: general-correlation real geometry (2026-11-03).**
The user requires general dependence under one shared Gaussian matrix. Independent
and equicorrelated cases remain controls; future models must be justified by real
query displacement angles, distance margins and signed Gram structure. Read
[section48's nonfrozen plan](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#48-一般相关下的真实几何审计与算法设计计划非冻结提案2026-11-03).
Proposed R6-B0 first audits complete real-corpus geometry on fresh development
queries; later shared-projection checks distinguish local structure, full-corpus
omission and actual backend headroom. Conditional residual dependence and online
input/selection costs remain explicit. No new scientific code, data transfer,
sampling or node run is authorized/executed by this planning turn. Data/source,
roles, resource limits and exact new L0 paths precede any implementation/run.

**Latest checkpoint: R6-M0 completed; partial numerical resolution (2026-11-03).**
The user explicitly approved the nine paths/tests/sync and one CPU M0. Read
[section47's results and provenance limitation](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#47-r6-m0获批实现完整验收与一次cpu参考计算2026-11-03)
and the [frozen protocol](docs/TLS_RAG_STEP5_ROUTE6_MACRO_REFERENCE_PROTOCOL_V1.md).
Scientific source48e2af5 evaluated all12 models/108 events in642.842s, with zero
new MC/CUDA. Independent models resolve54 events to2^-14; all54 equicorrelated
events remain unresolved at32768 nodes/model. FP64 rounding is uncertified.
All108 brackets overlap old Gram CP diagnostically. The old aggregation can be
vastly loose, yet all36 ell1 events still have high omission risk; this does not
establish Gaussian RAG headroom, a lightweight predictor or static dimension selection.
The original delivery index has a stale progress hash and omits source_freeze.
It is retained; an independent full19-file index and all local SHA/source checks
pass. The approved runner/test scope now fixes closed-product indexing; final
full537 tests pass, zero skips (77.023s), plus21 Linux tests (7.564s). Repair
commitb71273f is verified on GitHub and the node; TLS failures required a verified
GitHub bundle relay, with origin unchanged. No second M0 run occurred. Preserve the
original run/source identity and failure evidence. Do not auto retry/expand M0
or reopen R6-B/C/D, VIBE, old A0 or routes4/5. Nodes remain on; computation exited.

**Latest checkpoint: R6-A numerical revision1 completed on A800 (2026-11-03).**
Scientific source `bdd6cc8`; local full516/516 and Linux35 tests pass, with no skips.
After a fresh lock/GPU check, all60 units/120 paths completed in350.464s:
7,864,320 shared-projection samples,14,472 catalogue events,509 artifact files.
Independent mask/rank/count/CP and source/SHA audits passed; no numerical gray
bits, marginal/omission anomaly candidates or S2 input warnings were observed.
Read [section45 results and limitations](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#45-r6-a数值修订1完整a800结果与收尾2026-11-03).
At risk0.01/0.001, best S1 bounds pass28/6 of252 cells; S2 with extra offline
joint-CP inputs passes37/14. All gains are single-target. In108 macro cells,
independent Gram CP passes16/10 but every bound passes0: the macro aggregation
gap remains. This is finite diagnostic completion, not RAG headroom or a formal
all-domain theorem. Full raw bins remain on the server; all metadata/indexes
were recovered and verified locally. The original failed run/lock and interrupted
copy remain intact. Computation exited; the node was not shut down.
R6-A is closed for this fixed design; do not retune/rerun or begin R6-B/C/D/VIBE/
old A0/routes4/5 automatically. A new application gap/scope or macro theory
proposal is required. The checkpoint paragraphs below retain the actual history.

**Previous checkpoint: first R6-A A800 run failed; scoped numerical repair approved
(2026-11-03).** The user opened the node. Existing dependencies were reused,
the clean Linux checkout was synchronized through GitHub/ff-only to `dde6720`,
32 Linux tests and an independent tiny CUDA check passed. The first fixed run
then stopped with `worker_failed` after153.482226s:30 units sampled on both paths,
29 units analyzed. No resource or monitor limit was hit.
In `analyze_unit`, subtracting reordered floating-point sums produced eta
`-2.22e-16` for an empty complement. The negative-input guard correctly rejected it.
Read [section44's reproducible failure and exact five-path proposal](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#44-r6-a-a800首轮失败与数值修复提案待批准2026-11-03).
The user's actual reply to the exact five-path repair and fresh-run question was
“批准五路径修复、测试同步及新一轮R6-A（推荐）”. The complement is now summed
directly, preserving the negative guard/thresholds and the failed run. All
**516 offline tests pass, zero skips** (72.108s), including35 Scheme6 tests.
After source synchronization, create a new lock and GPU check before the approved
sample0 repair1 run (a separately approved3600s maximum, unchanged other caps).
Do not reuse the old environment lock, mix old/new counts or infer a completed
A800 measurement from the local regression. Section44.4 records actual approval
and tests; complete finite scientific acceptance is still pending.
R6-B/C/D and previously closed tasks remain stopped.

**Latest user task: Scheme6 R6-A local code and tests (2026-11-02).** The user
explicitly approved the section42.9 nine paths, local tests and synchronization:
“批准上述九路径及本地测试、同步”. Read the
[implementation/validation handoff in section43](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#43-方案6-r6-a获批实现与本地验收a800待开机2026-11-02)
and the [frozen R6-A protocol](docs/TLS_RAG_STEP5_ROUTE6_PROTOCOL_V1.md).
The boundary/conditional/moment bridges, stable scalar equivalent, finite shared
Gaussian sampler and supervised A800 entry are implemented in a new namespace.
Full offline regression passes **513/513, zero skips**, including32 new R6-A tests
(79.706s test body). The CPU self-test is an eight-sample/path integration fixture;
no complete60-unit scientific sample or CUDA check has run. No dependencies were
installed. A800 connection, device/environment lock and scientific execution wait
for the user to open the node; do not infer GPU success from local tests.
R6-B/C/D remain unimplemented and require their own data/scope approval.
The [original section42 proposal](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#42-方案6边界聚焦的候选遗漏界离线静态选维与轻量风险计算2026-11-02)
is retained byte-for-byte as the earlier checkpoint.
The withdrawn VIBE work was reverted by `3fef5d8` to the pre-benchmark file tree;
server partial data/provenance remain, and its checkout rollback was not verified
because SSH closed before authentication. Do not resume VIBE, old A0 or routes4/5.
The older current-stage paragraphs below are historical checkpoints, subordinate
to this latest instruction and the current STATUS.

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
novelty is **not established**. The user approved the six-path deterministic T1
scope and then explicitly approved the minimal test-environment restoration.
Three new scientific files and14 deterministic tests are implemented; exact
certificates and an independent Schur-complement audit agree. The complete
offline suite now passes481/481 with zero skips (70.081s test body).
Only faiss-cpu1.8.0 and scikit-learn1.6.1 were restored in the separate temporary
environment, with official wheel hashes verified; four existing native libraries
and their binary identities are preserved. Initial environment failures remain
recorded. T1 deterministic acceptance is complete; no joint-rank T2 sampling,
new data access or system experiment is authorized. The subsequent
[prior-work and proof audit](docs/JOINT_RANK_THEORY_T0.md#12-后续先行与证明审计保留机制笔记停止当前候选升级2026-11-02)
finds the same signed-cycle matrix pair explicitly in Urschel et al.2017 Lemma2.
The rank-tail bridge remains valid under the stated assumptions, but its
publication increment has not been established beyond a specialization of known
tools. Retain JRC-CYCLE-1 as a mechanism note; stop its T2 escalation under the
predeclared contribution gate (`covered_or_corollary_only`, corollary-only case).
This does not claim that prior work states the identical rank-gap lower bound
or solves the broader Gaussian identifiability question. Keep the old Step5 journal
read-only for this study; record new theory work in the independent note/namespace. The previous
system stops and closed data roles below remain in force.

The user's subsequent clarification explicitly resumes **Gaussian plus Tri-Law
RAG application research**, even when the cycle construction is known. Read the
[application research note and exact A0 proposal](docs/GAUSSIAN_RAG_RANK_BRIDGE_RESEARCH.md).
The cycle novelty stop above applies only to that foundational contribution;
it does not stop this application task. The new symbolic draft retains the shared
target length, applies known Bernoulli marginal bounds conditionally, and then
integrates. It gives a valid mathematical bridge and a no-weaker-bound comparison,
while keeping oracle geometry separate from deployable pilot inputs. No new code,
sampling or data access has been approved: section8 proposes the exact eight-path
synthetic A0 scope. Existing T1 acceptance and all prior systems results remain.

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
export TRI_RAG_TEST_PYTHON=/private/tmp/joint-rank-t1-test-env-20261102/bin/python
export STEP5_ROUTE1_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/route1.dylib
export STEP5_COMPARATORS_V2_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/comparators_v2.dylib
export STEP5_CONTINUATION_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/continuation_v1.dylib
export STEP5_CONTINUATION_V2_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/continuation_v2.dylib
sh scripts/run_tests.sh
```

The full Step 5/native suite needs the fixed-version environment and native
libraries above; the older temporary Python environment is incomplete. See
[current replay and environment provenance](docs/JOINT_RANK_THEORY_T0.md#11-环境恢复批准与t1完整验收2026-11-02).
`TRI_RAG_TEST_PYTHON` selects the interpreter without changing dependencies.
The current full suite collects and passes481 methods (step5=283, legacy=198),
including14 new independent-theory methods in the shared daily cohort.
The previous467 checkpoint was step5=269, legacy=198.
The previous429 count is the pre-P1 checkpoint.
The two subsets partition the full suite; unselected tests are not passed/skipped.
Shared-core/governance/dependency/historical changes and every implementation
delivery still require full validation. New test files need explicit classification.
Keep historical source-bound artifacts at their original source identity.

Detailed old startup narration is retained at
[the pre-cleanup snapshot](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/START_HERE.md).
Scientific contracts and the exploration journal keep their existing paths;
use [the map](docs/REPOSITORY_MAP.md) rather than preloading all historical documents.
