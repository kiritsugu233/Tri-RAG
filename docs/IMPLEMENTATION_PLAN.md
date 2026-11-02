# Implementation milestones

Updated 2026-11-02. Current stage: TLS-RAG Step 5 on `codex/tls-rag-step5-ann`.
Read [START_HERE](../START_HERE.md) and [STATUS](../STATUS.md) before work.
Unchecked historical items are not execution authorization.

## 当前任务：方案4关闭，方案5先做R5-A

- [x] 再核对方案4原status、封存索引、八项审查回执及GitHub同步；当前方法收尾。
- [x] 保留B仅5/7额外核查和resource_stop；未完成部分不再自动补跑。
- [x] 只读核对FAISS1.8建图调用路径，形成[第41节R5-A精确提案](TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#41-方案4收尾核对与方案5-r5-a成本预检提案2026-11-02)。
- [ ] 第41节八路径获批后，恢复原固定环境，实施等价建图成本探针并冻结源码。
- [ ] 本地CPU完成完整费用、剪枝份额/上限及投影准备代价；无空间/不确定均如实保留。
- [ ] 依R5-A证据决定是否提出R5-B；本轮不自动进入比较器或近似图实验。

## 共同Gaussian投影的排名尾部：理论研究框架

- [x] 按用户请求提供[完整研究框架与Agent启动说明](TRI_LAW_JOINT_RANK_RESEARCH_FRAMEWORK.md)。
- [x] 固定k=1概率对象及信息层级；区分已有边际/反例、经典矩界与待研究Gaussian约束。
- [x] 给出条件基准、归一化/可实现性、唯一主命题卡、实施提案路径和停止条件。
- [x] 将本次文档交付追加到探索日志第40节；科学源/旧结果不变，登记仅新增一个L1条目。
- [ ] T0：系统核对先行结果，选择一个可检验的增量，形成精确L0实现/协议/预算提案。
- [ ] T1：对应范围获批后复核数学基准与几何可实现性；不把框架中的式子当已验收。
- [ ] T2：冻结同摘要/不同联合结构的受控比较，保存逐投影记录及统计不确定性。
- [ ] T3：完成主命题证明或严格反例/不可辨识界；没有结论则如实停止该命题。
- [ ] T4：独立复核与先行对照，交付明确适用范围；不自动扩到ANN或其他系统接口。

## 两个应用场景验证（2026-11-01）

- [x] 只读复核旧成本/质量/风险证据，指出生命周期费用缺口与“高上界不能证明危险”。
- [x] 提出七库临时生命周期A、三态风险/直接probe价值B；精确八路径见探索日志第38节。
- [x] 实际答复“批准第38节全部范围，按A→B执行”；五新文件/测试入口/窄幅登记完成。
- [x] 完整467项、零跳过，22项新场景回归及真实worker链；245项保护通过。
- [x] 源1880435提交/推送/核对后冻结，执行A完整、B七库决策；额外核查5/7后资源终止。
- [x] 独立核对448真值/77,920精排/104,320时钟/2533封存文件，记录A/B分别结果与限制。
- [x] A无生命周期赢家；B与直接三轮相同模型却约10.79倍顾问费用，标8坏动作不省整轮工作。
- [x] 保留resource_stop/截断文件/缺测，报告实际质量与未决状态，不冒充完整B通过。
- [ ] B剩余两库稳定性及20k独立B-select未完成；作为收尾缺测保留，不再作为自动执行待办。

## 方案4 P1原因分析（2026-11-01）

- [x] 只复核既有advice/select结果，区分上游C覆盖、Gaussian排序噪声及聚合界松弛。
- [x] SVD/独立Gram分解已观察遗漏点对；核对数据谱、边界间隔和已有成本可竞争配置。
- [x] 记录可修复的机制、正交/数据感知残差的理论边界及未验证部分；见探索日志第37节。
- [x] 区分本轮额外损失标准、整体召回、独立认证和等质量净收益；保留原no_candidate。
- [ ] 如继续，先为新表示/动作空间提出独立版本、精确L0范围与成本止损条件。
  本轮分析不授权扩大m/M、正交化重检索、残差表示、收紧界实现或重新打开report。

## 方案4 P1恢复（2026-11-01）

- [x] 实际指示“继续P1”恢复第34节精确八路径；追加实际批准与第36节过程记录。
- [x] 五个新文件、测试入口归类及窄幅人工/机器登记；旧科学文件保持。
- [x] 新13项和全仓442项通过、无跳过；真实PID树资源停止及封存回归通过。
- [x] 保留105e2f9读取输入前的监控启动失败；修复真实监控测试/共享截止时间，完整445项通过。
- [x] 修正源b38ad51同步后完成唯一有效真实P1，137.777秒；原预算扣留60秒，没有并发旧run。
- [x] 独立核对64条真值/39,584质量/25,120计时及452项科学封存；记录风险、质量与成本负结果。
- [x] 科学门槛no_candidate，12个免费M诊断全不可行；report32保持关闭，未冒充独立通过。
- [x] 按停止条件不进入本动作空间的P2，不训练顾问或自动推进其他方案。

## 方案5：离线建图的邻居选择/剪枝

- [x] 按用户实际要求，将侧边讨论追加为[探索日志第35节](TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#35-方案5离线建图时的邻居选择与剪枝2026-10-31)。
- [x] 区分离线研究、真正建图及原空间线上查询；定位已知A/待算B的三点比较。
- [x] 记录条件风险/自适应建图缺口、完整成本、强对照、图质量验证和停止规则。
- [x] 2026-11-02用户指示开始方案5；只读准备及具体R5-A提案完成，第41节实现范围待批。
- [ ] R5-A：取得相应范围授权后，先测剪枝在完整建图中的成本份额及上限。
- [ ] R5-B：固定候选现场的几何/投影反转与完整邻居选择重放。
- [ ] R5-C：低维判定/精算回退的可用信息、同输入无Tri消融及配置价值。
- [ ] R5-D：精确L0范围批准后才接新命名空间的实际建图；局部结论不冒充整图保证。
- [ ] R5-E：合法外部查询上比较匹配质量的搜索成本，建图与查询收益分别判定。

全部方案5科学阶段尚未执行；当前准备不重开已收尾的方案4。

## 方案4后续：已指派 Step 5 接续研究

- [x] 根据用户本次实际指示，将下一轮研究追加为[探索日志第32节](TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md)。
- [x] 明确优先顺序：离线联合`(m,M)`顾问＋独立候选池筛选；独立视图并集条件后续。
- [x] 记录2013精确翻转概率的变量代换线索、核心文献、成本账本和停止条件。
- [x] P0：Step5独立理论/先行工作复核、输入身份审计及精确P1提案；见同日志第33–34节。
- [x] 378组边界/常规表达式与2880项集合检查；记录相同边际、不同排名尾部的可实现反例。
- [x] 记录10月30日实际答复“保留P0，暂缓P1实现”；该检查点未擅自恢复。
- [x] 11月1日“继续P1”恢复第34节五文件/测试入口/窄幅登记范围，实际执行与结果见上。
- [x] P1：11月1日获批并执行固定候选池验证，no_candidate；详见第36节，后续原因复核见第37节。
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
| Route4 offline geometry/index configuration | P1 completed, no candidate; report closed, P2 not started | Same journal §§21/32–36 |
| Route5 offline graph neighbor selection/pruning | R5-A source audit/proposal ready; exact implementation scope pending, no build run | Same journal §§35/41 |
| Joint-Gaussian rank-tail theory | Framework delivered; T0–T4 not run in this task | [Framework](TRI_LAW_JOINT_RANK_RESEARCH_FRAMEWORK.md), same journal §40 |

## Pending decisions and safety limits

- [x] User approved the [guard/register repair A](REPOSITORY_CLEANUP_REVIEW.md#proposal-a-protected-guard-and-register-repair).
- [x] Exact [proposal B](TEST_SUITE_SIMPLIFICATION_PROPOSAL.md) approved and implemented;
  further consolidation or permanent assertion retirement needs a separate proposal.
- [ ] Trained-support/OOD fallback remains unimplemented for continuation v2.
  A future repair needs its own L0 scope/version and evaluation; do not relabel
  the historical study as safe deployment or rerun protected data automatically.
- [ ] Fresh independent confirmation, new corpus/1M, adapter/original route3 and
  formal certification remain separate future work. Route4's current method is
  closed; route5 R5-A's exact implementation/data scope is proposed above.

## Full historical plan

The complete prior1521-line ledger, original approvals, unchecked milestones,
commands and compatibility constraints remain available at
[the cleanup baseline](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/docs/IMPLEMENTATION_PLAN.md). Offline retrieval:

```bash
git show 50defc35bc9f42f5b24d2dedf5f45a89e51257ad:docs/IMPLEMENTATION_PLAN.md
```

The source-bound scientific contracts and append-only research journal remain
at their existing local paths. This index does not replace those contracts.
