# Implementation milestones

Updated 2026-11-01. Current stage: TLS-RAG Step 5 on `codex/tls-rag-step5-ann`.
Read [START_HERE](../START_HERE.md) and [STATUS](../STATUS.md) before work.
Unchecked historical items are not execution authorization.

## 方案4 P1恢复（2026-11-01）

- [x] 实际指示“继续P1”恢复第34节精确八路径；追加实际批准与第36节过程记录。
- [x] 五个新文件、测试入口归类及窄幅人工/机器登记；旧科学文件保持。
- [x] 新13项和全仓442项通过、无跳过；真实PID树资源停止及封存回归通过。
- [ ] 冻结源码并按本地CPU20k/96/45分钟范围完成唯一真实输入P1。
- [ ] 核对逐query/seed、完整成本、风险界、免费M上限、报告门及封存，记录真实负/正结果。
- [ ] 根据证据决定是否值得另提P2；本次不训练顾问或自动推进其他方案。

## 方案5：离线建图的邻居选择/剪枝（仅设计记录）

- [x] 按用户实际要求，将侧边讨论追加为[探索日志第35节](TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#35-方案5离线建图时的邻居选择与剪枝2026-10-31)。
- [x] 区分离线研究、真正建图及原空间线上查询；定位已知A/待算B的三点比较。
- [x] 记录条件风险/自适应建图缺口、完整成本、强对照、图质量验证和停止规则。
- [ ] R5-A：取得相应范围授权后，先测剪枝在完整建图中的成本份额及上限。
- [ ] R5-B：固定候选现场的几何/投影反转与完整邻居选择重放。
- [ ] R5-C：低维判定/精算回退的可用信息、同输入无Tri消融及配置价值。
- [ ] R5-D：精确L0范围批准后才接新命名空间的实际建图；局部结论不冒充整图保证。
- [ ] R5-E：合法外部查询上比较匹配质量的搜索成本，建图与查询收益分别判定。

全部科学阶段尚未执行；本次仅文档写入，不恢复方案4暂缓的P1。

## 方案4后续：已指派 Step 5 接续研究

- [x] 根据用户本次实际指示，将下一轮研究追加为[探索日志第32节](TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md)。
- [x] 明确优先顺序：离线联合`(m,M)`顾问＋独立候选池筛选；独立视图并集条件后续。
- [x] 记录2013精确翻转概率的变量代换线索、核心文献、成本账本和停止条件。
- [x] P0：Step5独立理论/先行工作复核、输入身份审计及精确P1提案；见同日志第33–34节。
- [x] 378组边界/常规表达式与2880项集合检查；记录相同边际、不同排名尾部的可实现反例。
- [x] 记录实际答复“保留P0，暂缓P1实现”；保留P0并停止于此，不自动恢复P1。
- [ ] 第34节五个新文件、测试入口归类及窄幅登记范围取得用户精确批准；当前暂缓/未批准。
- [ ] P1：取得相应范围批准后，验证固定候选池上纯投影规律、计数界及完整成本。
- [ ] P2：比较Tri配置、同输入间隔/角度模型和同预算直接投影试验；无增量则停止/降级。
- [ ] P3：冻结具体配置及矩阵后，按未参与选择的查询角色验证；旧分组不冒充fresh独立性。
- [ ] P4：仅在前置条件通过且范围明确后接ANN/扩大规模/比较同预算多视图。
- [ ] 各阶段记录逐query/seed结果、尾部失配和负结果；本轮交接不标记任何算法阶段完成。

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
| Route4 offline geometry/index configuration | P0 complete; user deferred P1; no algorithm experiment run | Same journal §§21/32–34 |
| Route5 offline graph neighbor selection/pruning | Design documented only; no implementation or build run | Same journal §35 |

## Pending decisions and safety limits

- [x] User approved the [guard/register repair A](REPOSITORY_CLEANUP_REVIEW.md#proposal-a-protected-guard-and-register-repair).
- [x] Exact [proposal B](TEST_SUITE_SIMPLIFICATION_PROPOSAL.md) approved and implemented;
  further consolidation or permanent assertion retirement needs a separate proposal.
- [ ] Trained-support/OOD fallback remains unimplemented for continuation v2.
  A future repair needs its own L0 scope/version and evaluation; do not relabel
  the historical study as safe deployment or rerun protected data automatically.
- [ ] Fresh independent confirmation, new corpus/1M, adapter/original route3 and
  formal certification remain separate future work. Route4 research is assigned
  above; its new implementation/data scope still requires exact approval.

## Full historical plan

The complete prior1521-line ledger, original approvals, unchecked milestones,
commands and compatibility constraints remain available at
[the cleanup baseline](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/docs/IMPLEMENTATION_PLAN.md). Offline retrieval:

```bash
git show 50defc35bc9f42f5b24d2dedf5f45a89e51257ad:docs/IMPLEMENTATION_PLAN.md
```

The source-bound scientific contracts and append-only research journal remain
at their existing local paths. This index does not replace those contracts.
