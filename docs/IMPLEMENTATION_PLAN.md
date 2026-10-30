# Implementation milestones

Updated 2026-10-30. Current stage: TLS-RAG Step 5 on `codex/tls-rag-step5-ann`.
Read [START_HERE](../START_HERE.md) and [STATUS](../STATUS.md) before work.
Unchecked historical items are not execution authorization.

## Repository review and cleanup

- [x] Check canonical branch/HEAD/status and protection before editing.
- [x] Inventory all242 tracked baseline files; parse118 Python files, inspect
  native include/dependency paths, configs, test coverage and source bindings.
- [x] Trace current Step 5 and retained historical families; publish a per-file
  [repository map](REPOSITORY_MAP.md) and [cleanup findings](REPOSITORY_CLEANUP_REVIEW.md).
- [x] Remove repeated historical narration from startup/status/plan while retaining
  its exact Git snapshot; correct obsolete workflow/next-agent instructions.
- [x] Keep all scientific sources/tests/configs/source bindings unchanged.
  No whole source file qualifies for automatic deletion in this review.
- [x] Run429 tests in the existing fixed-version/native environment without skips;
  document default-environment missing-FAISS failure rather than hiding it.
- [x] Make the L1 test launcher independent of cwd with explicit interpreter choice;
  final full rerun from `/private/tmp`:429/429,0 skips. Protection/link/binding checks pass.
- [x] Obtain exact four-path L0 approval for checker/register repair (review A).
- [x] Repair native/ignored mechanism discovery and synchronize 44 missing table entries.
- [x] Preserve all 10 governance tests; add one parameterized native/Git-state regression.
- [x] Full fixed-environment suite:430/430,0 skips; governance11/11;234 levels preserved.
- [x] Document A800 guard-source identity consequence; five Step5 maps unchanged.
- [x] Complete table/link/hash verification for same-branch Step5 synchronization.

## 测试精简方案 B：已批准并实施

- [x] 对430方法逐项列出处置、实际断言/调用依据及来源绑定影响。
- [x] 真实unittest只收集确认：日常候选232＋历史198=430，无重叠/漏项。
- [x] 提出M01四场景等价合并及明确四路径审批，保留所有唯一历史断言。
- [x] 用户以“执行”批准 [中文方案 B](TEST_SUITE_SIMPLIFICATION_PROPOSAL.md) 的四路径范围。
- [x] M01等价合并，保留四场景、其他五方法及其余45测试文件；无永久删除。
- [x] 默认/full保持完整；新增step5/legacy，全集收集错误和未知文件明确失败。
- [x] 收集429=231+198且互斥；仅两个旧ID→一个新ID；临时三场景故障均检出。
- [x] 固定环境完整429/429、0跳过；Step5与历史子集单独验收，详见STATUS。
- [x] 六组来源绑定与全部科学源码/配置/依赖不变；仅两个获批L0内容哈希更新。
- [x] 同步实际批准、235项人工/机器登记及启动/交接命令。

## Scientific milestone index

| Family/milestone | Recorded state | Detailed record |
| --- | --- | --- |
| Raw Tri-Predict v1 | Terminal negative; frozen | [Baseline](RAW_TRI_PREDICT_V1_BASELINE.md) |
| Calibrated Tri-Predict v2/v3 | Historical; frozen | [Protocol](CALIBRATED_TRI_PREDICT_PROTOCOL.md), [diagnosis](CALIBRATED_TRI_PREDICT_V3_DIAGNOSIS.md) |
| TLS-RAG Steps2/3 | Synthetic implementation complete | [Step2](TLS_RAG_STEP2_SYNTHETIC.md), [Step3](TLS_RAG_STEP3_SYNTHETIC.md) |
| Step4 repaired/A800 replay | Replay gates passed; no independent effectiveness/latency claim | [A800 contract](TLS_RAG_STEP4_A800_REPLAY.md), [historical status](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/STATUS.md) |
| Step5 ANN v1 | No tune candidate; protected later roles closed | [Contract](TLS_RAG_STEP5_ANN.md), [historical status](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/STATUS.md) |
| Step5 exact mechanism/reuse/baselines | Studies complete; conclusions version scoped | [Repository map](REPOSITORY_MAP.md) |
| Route1 v1/v2 | Negative net-speed results | [Journal §§14/19](TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md) |
| Route2 continuation v1/v2 | A→B→C complete; current learned/Tri paths fail net gain | [Journal §§24/27–31](TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md) |
| Route3 independent-view rescue | Not run | Same journal §7; separate scope needed |
| Route4 offline geometry/index configuration | Plan only | Same journal §21; separate scope needed |

## Pending decisions and safety limits

- [x] User approved the [guard/register repair A](REPOSITORY_CLEANUP_REVIEW.md#proposal-a-protected-guard-and-register-repair).
- [x] Exact [proposal B](TEST_SUITE_SIMPLIFICATION_PROPOSAL.md) approved and implemented;
  further consolidation or permanent assertion retirement needs a separate proposal.
- [ ] Trained-support/OOD fallback remains unimplemented for continuation v2.
  A future repair needs its own L0 scope/version and evaluation; do not relabel
  the historical study as safe deployment or rerun protected data automatically.
- [ ] Fresh independent confirmation, new corpus/1M, adapter/routes3/4 and formal
  certification are separate future work, not assigned by this checklist.

## Full historical plan

The complete prior1521-line ledger, original approvals, unchecked milestones,
commands and compatibility constraints remain available at
[the cleanup baseline](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/docs/IMPLEMENTATION_PLAN.md). Offline retrieval:

```bash
git show 50defc35bc9f42f5b24d2dedf5f45a89e51257ad:docs/IMPLEMENTATION_PLAN.md
```

The source-bound scientific contracts and append-only research journal remain
at their existing local paths. This index does not replace those contracts.
