# 测试精简审批方案 B（中文，待批准）

日期：2026-10-30。代码/测试审阅基线：
`fd91423880c0d504c531dce4388ae9e4ba86b628`。
工作目录 `/Users/guanghongxu/Query-Adaptive-Tri-RAG`，分支
`codex/tls-rag-step5-ann`。本文件是独立审批方案，**尚未执行**。
此前方案 A 的批准不覆盖本方案。本轮只生成文档和新增 L1 文档登记，
未修改任何测试、执行选择或保护哈希。

## 一、请批准的明确范围

推荐：**分层运行 + 一组有明确等价依据的合并；不永久删除独立断言。**

| 原有方法处置 | 当前方法数 | 批准后的执行方式 |
| --- | ---: | --- |
| 保留 | 230 | 日常 Step5 和完整套件都执行，原方法/断言保持 |
| 合并 M01 | 2 | 同一缓存校验函数的两个方法合为一个参数化方法；三个错误身份场景及一个合法场景保留 |
| 退役（日常） | 198 | 退出日常 Step5，原文件/方法/断言仍完整保留；完整/历史套件继续执行 |
| 永久删除 | **0** | 尚未证明可永久删除且不损失独立覆盖，因此本方案不申请删除 |

数量结果：日常 Step5 **232→231** 个方法，完整套件 **430→429** 个方法，
历史套件 **198** 个方法。不能把日常231项宣传成“整个仓库只剩231项测试”。
两个方法合并只减少一个方法入口，不声称减少了实际科学检查数量。
本方案不承诺运行耗时按数量同比下降，批准实施后实测报告。

“退役”在本文严格表示**从日常执行集合退役**，不是删除文件、断言或停止所有验证。
如果希望永久删掉独立的历史回归，需要另一个明确列出覆盖损失的审批，不能由本文推导。

## 二、为什么采用这个范围

430项来自46个测试文件。此前各科研版本累计新增，但入口始终无差别运行全仓。
本次逐项列出全部430个方法及实际断言/调用；方法数量不是覆盖率。
已用当前固定环境的真实 unittest loader **仅收集、不执行** 验证：
232个日常候选 + 198个历史候选 = 430，互不重叠、没有漏项。
基线430项完整通过的实际记录见 [方案 A 验收](REPOSITORY_CLEANUP_REVIEW.md)。

- Tri-Law、投影尺度、禁止投影后归一化、精确排序/平局、LID、缓存身份、
  认证公式、治理保护、续搜状态一致性和当前角色隔离保留。
- Step2/3虽然较旧，但当前 Step5 从它们复用状态/observed-pair 特征，因此保留日常验证。
- 旧 PDCTP/FiQA、SciFact/通用 BEIR/E5 和 Step4 专用流程归入历史集合。
  它们的独立协议断言**没有被 Step5 全部替代**；完整套件仍必须验证这些断言。
- 跨版本配置/原生一致性测试不能只因名称相似合并。核查发现多个 Step5 测试文件
  本身在来源清单中，因此本次不改动这些测试文件，不以刷新旧实验哈希换取合并。
- 唯一申请的 M01 是同一个函数、同一基础元数据、同一种异常校验，只是变更字段不同。

## 三、唯一实际合并 M01

L0文件：[tests/test_config_projection.py](../tests/test_config_projection.py)。
基线文件SHA-256：`5be6e9bb3e540e27d52388bc1ab9e9847fb4d9e311b33926922599ecebd135fa`。

| 原方法/场景 | 新方法中的覆盖位置 |
| --- | --- |
| `ConfigProjectionTests.test_projection_metadata_change_invalidates_cache`：seed1→2 | `test_projection_cache_identity_changes_are_rejected` / `subTest(field="seed")`，必须抛 `CacheMetadataMismatch` |
| 同一原方法：相同缓存身份合法通过 | 新方法保留该成功调用，不得被异常用例替代 |
| `ConfigProjectionTests.test_corpus_or_model_change_invalidates_projection_cache`：model@1→model@2 | 新方法 / `subTest(field="embedding_model")`，同一异常要求 |
| 同一原方法：corpus-a→corpus-b | 新方法 / `subTest(field="corpus_hash")`，同一异常要求 |

保留基准元数据 dimension32/m_prime16/seed1/normalization=True/model@1/corpus-a。
每个错误场景重新从该基准构造，不共享上一个场景的修改。该文件其他五个方法原样保留。
不改投影实现、数值容差、随机种子或历史预期。旧的精确方法名会被新名称替代；
仓库执行代码中未找到对这两个旧方法名的直接引用，外部按方法名运行的命令需要迁移。

## 四、套件入口和触发规则

只在现有 `scripts/run_tests.sh` 内实现分组/清单核对，不新增测试模块/新的实现文件。
继续支持已有 `TRI_RAG_TEST_PYTHON`、仓库外调用、固定原生库环境。

| 命令（批准后的拟议接口） | 执行范围 |
| --- | --- |
| `sh scripts/run_tests.sh` | 保持兼容：仍为完整套件，预计429项 |
| `sh scripts/run_tests.sh full` | 显式完整套件，同上；不把默认入口静默改成只跑部分 |
| `sh scripts/run_tests.sh step5` | 日常 Step5＋共享核心＋Step2/3，预计231项 |
| `sh scripts/run_tests.sh legacy` | 历史集合198项，和日常集合互斥 |

入口必须打印所选套件、实际收集/选中数量、未选中集合及“是否完整验证”。
未选中历史集合不能记成通过，也不使用 skip 装饰器伪装精简。
collection/import错误必须失败，不能因过滤集合而隐藏。新增测试文件未归类必须报告错误，
不能悄悄漏掉；已有完整发现仍保留全部方法。

适用规则：

1. 开发当前 Step5的中间迭代可显式用 `step5`；只改文档不强制重跑科研测试。
2. 触碰共享数学/检索/嵌入/统计/角色/缓存/来源绑定/依赖/治理或历史实现时，必须用 `full`。
3. **本次 B 及以后的任何实现变更交付提交/推送前仍用完整套件**，不降低根规则要求。
   `step5` 成功不能被记录成“全仓测试通过”。
4. 历史复现使用 `legacy` 或完整套件，并保持该实验原提交/身份；不重新打开旧真实角色。
5. `scripts/run_tls_rag_step4_readiness.sh` 仍通过无参数入口跑完整套件；不修改该脚本。

21个日常文件和25个历史文件的明确名单见第五节；第六节给出430个方法的逐项清单。
现有默认430方法全部收集的事实不因本方案发生改变。

## 五、精确文件范围和兼容影响

### 需要审批的实现/治理变更

| 路径 | 当前级别 | 获批后允许的变化 |
| --- | --- | --- |
| `tests/test_config_projection.py` | L0 | 仅M01两个方法→一个参数化方法，保留四个场景及其他五个方法 |
| `scripts/run_tests.sh` | L1，执行选择继承L0审批 | 增加full/step5/legacy分组与收集完整性/错误检查，无参数仍完整；不改任何测试跳过条件 |
| `docs/CODE_PROTECTION.md` | L0 | 记录实际批准与套件规则，补本文的L1文档行；现有等级/审批边界不变 |
| `docs/code_protection.json` | L0治理 | 记录实际批准及仅上述两个有L0内容哈希文件的获批变化；不刷新任何科学源/实验哈希 |

必要的L1交接同步路径：`README.md`、`START_HERE.md`、`NEXT_AGENT_PROMPT.md`、
`STATUS.md`、`docs/IMPLEMENTATION_PLAN.md`、`docs/TLS_RAG_LOCAL_WORKFLOW.md`、
`docs/REPOSITORY_MAP.md`、`docs/REPOSITORY_CLEANUP_REVIEW.md`及本提案。
只更新命令、套件范围、实测结果和审批状态，不改历史科学结论。

### 明确保持的边界

- 其余45个测试文件不移动、不合并、不删除、不改断言；包括全部来源绑定的Step5测试。
- 原有保护等级不降低；不修改AGENTS、检查器、科学源码、native、依赖或实验配置。
- 核对现有来源清单：M01文件与通用测试入口不在已检查的科研来源绑定路径中；
  包级全Python绑定也不涉及它们。审批后仍须逐路径比较，不能只相信本提案。
- 不应重现方案A的A800来源身份变化：本次不改其来源清单中的保护脚本。
  若验证发现任何未预期来源变化，保留现场，停止额外修改并说明原因；不自动刷新旧包。
- 测试方法数变化是维护组织变化，不等于新算法、科学证书或新数据结果。

### 文件级处置总表

下表中“退役”为退出日常、完整保留；其余文件的“保留”不是宣称每条断言永远不可改。
今后仍可按真实覆盖证据另提精简范围。

| 测试文件 | 原方法数 | 日常处置 | 完整套件与覆盖依据 |
| --- | ---: | --- | --- |
| [test_attribution.py](../tests/test_attribution.py) | 2 | 保留 | 保留2项；合成公开诊断入口；区分 LID/秩/其余误差及种子命名空间 |
| [test_beir_dataset.py](../tests/test_beir_dataset.py) | 6 | 退役（日常） | 仍执行6项；旧 BEIR/SciFact 准备器的归档校验、外部查询、文本去重与划分 |
| [test_certification.py](../tests/test_certification.py) | 3 | 保留 | 保留3项；共享 empirical-Bernstein 手算值、证书身份和样本量 |
| [test_code_protection.py](../tests/test_code_protection.py) | 11 | 保留 | 保留11项；当前治理检查；只读、哈希、原生发现、路径安全及登记规则 |
| [test_config_projection.py](../tests/test_config_projection.py) | 7 | 合并M01＋其余保留 | 7→6项；四个缓存身份场景保留；共享投影方差、禁止投影后归一化、缓存身份及数组读入 |
| [test_end_to_end.py](../tests/test_end_to_end.py) | 1 | 保留 | 保留1项；仍由打包命令提供的合成 harness 完整输出、划分和复现 |
| [test_index_retrieval.py](../tests/test_index_retrieval.py) | 10 | 保留 | 保留10项；共享精确平方 L2/流式/FAISS 接口、稳定平局与 k 边界 |
| [test_lid_policy.py](../tests/test_lid_policy.py) | 5 | 保留 | 保留5项；LID 欧氏转换、无效距离、预算网格、单调性及策略身份 |
| [test_mprime_sweep.py](../tests/test_mprime_sweep.py) | 2 | 保留 | 保留2项；原合成维度扫描 CLI；选模与认证分离、冻结结果 |
| [test_pdctp_calibration.py](../tests/test_pdctp_calibration.py) | 7 | 退役（日常） | 仍执行7项；历史 PDCTP 校准器的约束求解、参数格点、序列化与残差修正 |
| [test_pdctp_dataset_audit.py](../tests/test_pdctp_dataset_audit.py) | 5 | 退役（日常） | 仍执行5项；历史 FiQA 源审计、归档/标签完整性与五角色容量 |
| [test_pdctp_features.py](../tests/test_pdctp_features.py) | 7 | 退役（日常） | 仍执行7项；历史 PDCTP pilot 特征、距离排序/格点/无效状态及防标签输入 |
| [test_pdctp_fiqa_embedding_gate.py](../tests/test_pdctp_fiqa_embedding_gate.py) | 6 | 退役（日常） | 仍执行6项；历史 FiQA/E5 嵌入闸门、角色清单、缓存及模型身份 |
| [test_pdctp_fiqa_query_cal.py](../tests/test_pdctp_fiqa_query_cal.py) | 6 | 退役（日常） | 仍执行6项；历史 FiQA cal-only 拟合、候选重建和单次精确扫描 |
| [test_pdctp_fiqa_query_cert.py](../tests/test_pdctp_fiqa_query_cert.py) | 6 | 退役（日常） | 仍执行6项；历史 FiQA 六假设认证、提前封存及失败终态 |
| [test_pdctp_fiqa_query_tune.py](../tests/test_pdctp_fiqa_query_tune.py) | 8 | 退役（日常） | 仍执行8项；历史 FiQA 独立候选选择、冻结策略、预算及访问前校验 |
| [test_pdctp_foundation.py](../tests/test_pdctp_foundation.py) | 1 | 退役（日常） | 仍执行1项；历史 PDCTP 五角色合成 walking skeleton 的整套复现 |
| [test_pdctp_policies.py](../tests/test_pdctp_policies.py) | 4 | 退役（日常） | 仍执行4项；历史 PDCTP 各消融统一输入、回退及 Raw v1 行为保持 |
| [test_pdctp_protocol_statistics.py](../tests/test_pdctp_protocol_statistics.py) | 7 | 退役（日常） | 仍执行7项；历史五角色协议与成对界、总体 alpha 分配及 power |
| [test_pdctp_real_protocol.py](../tests/test_pdctp_real_protocol.py) | 5 | 退役（日常） | 仍执行5项；历史 FiQA 协议冻结、输入篡改与角色闭合 |
| [test_pdctp_v3.py](../tests/test_pdctp_v3.py) | 6 | 退役（日常） | 仍执行6项；历史有效 Tri-LID/曲线修正、缓存身份及候选一致性 |
| [test_real_dimension_sweep.py](../tests/test_real_dimension_sweep.py) | 5 | 退役（日常） | 仍执行5项；历史 SciFact 实数据维度扫描、投影成本、平局及 tune-only 选择 |
| [test_real_original_baseline.py](../tests/test_real_original_baseline.py) | 5 | 退役（日常） | 仍执行5项；历史 SciFact 原空间基线、证据指标、缓存及 tune 身份 |
| [test_real_policy_certify.py](../tests/test_real_policy_certify.py) | 4 | 退役（日常） | 仍执行4项；历史 SciFact 证书入口、冻结策略、防认证后选模及终态 |
| [test_real_policy_test.py](../tests/test_real_policy_test.py) | 4 | 退役（日常） | 仍执行4项；历史 SciFact 终态证书绑定与仅描述性 test 记录 |
| [test_real_policy_tune.py](../tests/test_real_policy_tune.py) | 8 | 退役（日常） | 仍执行8项；历史 SciFact tune-only 策略选择、完整语料边界及成本 |
| [test_real_tune_diagnostics.py](../tests/test_real_tune_diagnostics.py) | 3 | 退役（日常） | 仍执行3项；历史 SciFact tune 归因/证据诊断及禁止重新认证 |
| [test_retrieval_benchmark.py](../tests/test_retrieval_benchmark.py) | 6 | 保留 | 保留6项；共享系统基准、同值/切点语义、实际检索工作及 GPU k 上限 |
| [test_text_embeddings.py](../tests/test_text_embeddings.py) | 6 | 退役（日常） | 仍执行6项；历史通用文本嵌入生产器；固定模型、输入格式、规范化与缓存校验 |
| [test_tls_rag_step2.py](../tests/test_tls_rag_step2.py) | 13 | 保留 | 保留13项；Step5 仍间接依赖的合成状态、精确前缀、证据隔离和工作计量 |
| [test_tls_rag_step3.py](../tests/test_tls_rag_step3.py) | 23 | 保留 | 保留23项；Step5 仍直接复用的 observed-pair 风险/失真特征及历史状态契约 |
| [test_tls_rag_step4.py](../tests/test_tls_rag_step4.py) | 26 | 退役（日常） | 仍执行26项；历史 Step4 readiness 冻结协议、访问/收据链、控制器分支与合成复现 |
| [test_tls_rag_step4_a800_replay.py](../tests/test_tls_rag_step4_a800_replay.py) | 17 | 退役（日常） | 仍执行17项；历史 A800 回放的来源/镜像/载荷校验、闭合及复现 |
| [test_tls_rag_step4_probe.py](../tests/test_tls_rag_step4_probe.py) | 15 | 退役（日常） | 仍执行15项；历史 NFCorpus probe 的分组、标签闸门、源/缓存和失败闭合 |
| [test_tls_rag_step4_repaired_acceptance.py](../tests/test_tls_rag_step4_repaired_acceptance.py) | 19 | 退役（日常） | 仍执行19项；历史 reviewed-source 兼容入口、旧哈希、重建与禁止重复 probe |
| [test_tls_rag_step4_retention_v3.py](../tests/test_tls_rag_step4_retention_v3.py) | 12 | 退役（日常） | 仍执行12项；历史 retention-v3 连续目标、校准/网格、父缓存与 probe 闭合 |
| [test_tls_rag_step5_ann.py](../tests/test_tls_rag_step5_ann.py) | 30 | 保留 | 保留30项；Step5 公共 ANN/候选缓存/角色/校准/计时入口及真实 HNSW 合成接口 |
| [test_tls_rag_step5_ann_comparators_v2.py](../tests/test_tls_rag_step5_ann_comparators_v2.py) | 16 | 保留 | 保留16项；冻结比较器 v2 的概率/数值/缓存/原生路径；测试本身参与来源绑定 |
| [test_tls_rag_step5_ann_continuation_v1.py](../tests/test_tls_rag_step5_ann_continuation_v1.py) | 14 | 保留 | 保留14项；当前 v2 继承的续搜 v1 状态/生命周期/调用序列；测试参与来源绑定 |
| [test_tls_rag_step5_ann_continuation_v2.py](../tests/test_tls_rag_step5_ann_continuation_v2.py) | 15 | 保留 | 保留15项；当前续搜 v2 的分组、特征、残差、原生模型和 worker；测试参与来源绑定 |
| [test_tls_rag_step5_ann_enhancement.py](../tests/test_tls_rag_step5_ann_enhancement.py) | 13 | 保留 | 保留13项；仍提供复用 helper 的 route1；原生/过滤/信息闸门，测试参与来源绑定 |
| [test_tls_rag_step5_baseline_headroom.py](../tests/test_tls_rag_step5_baseline_headroom.py) | 12 | 保留 | 保留12项；当前对照复用的精确/ANN/成本空间与输入闸门；测试参与来源绑定 |
| [test_tls_rag_step5_mechanism.py](../tests/test_tls_rag_step5_mechanism.py) | 14 | 保留 | 保留14项；仍被后续调用的 ExactScan/PrefixSession 和实验审计；测试参与来源绑定 |
| [test_tls_rag_step5_reuse_probe.py](../tests/test_tls_rag_step5_reuse_probe.py) | 8 | 保留 | 保留8项；仍被基线调用的缓存/稳定选择/平局/隔离；测试参与来源绑定 |
| [test_tri_law.py](../tests/test_tri_law.py) | 10 | 保留 | 保留10项；精确 Tri-Law 公式、R1/R2 边界、高精度、小尾部、广播和 Monte Carlo |
| [test_tri_predict.py](../tests/test_tri_predict.py) | 17 | 保留 | 保留17项；冻结 Tri-Predict 近似、数值聚合、完整语料边界和保存/加载行为 |

## 六、430个方法的逐项覆盖处置

这是**审批附录**，普通下一步工作不要求预读全部条目。
每行定位基线中真实方法、执行集合和实际断言/目标调用，编号B001—B430。
“断言位置”是静态语法位置数，不是运行次数或覆盖率；参数化循环可执行多次。
对未修改的428个方法，覆盖依据是保留其完整原测试，并只控制何时执行。
退役行不会把不同版本的检查声称为等价替代。M01的精确四场景对应以第三节为准。

### test_attribution.py（2项）

覆盖契约：合成公开诊断入口；区分 LID/秩/其余误差及种子命名空间。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B001 [AttributionTests.test_attribution_artifacts_separate_lid_rank_and_remaining_error](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_attribution.py#L15) | 保留 | 日常＋完整，方法原样 | 调用：`run_attribution`、`load_config`；原5处断言位置：`self.assertEqual`、`self.assertIn`；完整断言不删 |
| B002 [AttributionTests.test_synthetic_seed_namespaces_make_fresh_split_ids_distinct](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_attribution.py#L45) | 保留 | 日常＋完整，方法原样 | 调用：`load_config`、`generate_synthetic_dataset`；原2处断言位置：`self.assertTrue`；完整断言不删 |

### test_beir_dataset.py（6项）

覆盖契约：旧 BEIR/SciFact 准备器的归档校验、外部查询、文本去重与划分。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B003 [BEIRDatasetTests.test_prepare_is_reproducible_external_and_disjoint](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_beir_dataset.py#L106) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_beir_dataset_config`、`prepare_beir_dataset`；原15处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertTrue`；完整断言不删 |
| B004 [BEIRDatasetTests.test_archive_checksum_mismatch_is_rejected_before_output](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_beir_dataset.py#L169) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_beir_dataset_config`、`prepare_beir_dataset`；原2处断言位置：`self.assertRaisesRegex`、`self.assertFalse`；完整断言不删 |
| B005 [BEIRDatasetTests.test_missing_qrel_document_is_rejected_before_output](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_beir_dataset.py#L182) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_beir_dataset_config`、`prepare_beir_dataset`；原2处断言位置：`self.assertRaisesRegex`、`self.assertFalse`；完整断言不删 |
| B006 [BEIRDatasetTests.test_development_test_query_overlap_is_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_beir_dataset.py#L203) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_beir_dataset_config`、`prepare_beir_dataset`；原2处断言位置：`self.assertRaisesRegex`、`self.assertFalse`；完整断言不删 |
| B007 [BEIRDatasetTests.test_normalized_duplicate_text_cannot_cross_splits](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_beir_dataset.py#L216) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_beir_dataset_config`、`prepare_beir_dataset`；原7处断言位置：`self.assertNotIn`、`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B008 [BEIRDatasetTests.test_checked_in_scifact_config_is_pinned](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_beir_dataset.py#L275) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_beir_dataset_config`；原5处断言位置：`self.assertEqual`；完整断言不删 |

### test_certification.py（3项）

覆盖契约：共享 empirical-Bernstein 手算值、证书身份和样本量。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B009 [CertificationTests.test_hand_computed_fixture](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_certification.py#L13) | 保留 | 日常＋完整，方法原样 | 调用：`empirical_bernstein`；原4处断言位置：`self.assertAlmostEqual`；完整断言不删 |
| B010 [CertificationTests.test_certificate_identity_is_frozen](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_certification.py#L25) | 保留 | 日常＋完整，方法原样 | 调用：`make_certificate`、`validate_certificate_identity`；原2处断言位置：`self.assertRaises`；完整断言不删 |
| B011 [CertificationTests.test_sample_size_plan_satisfies_requested_width](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_certification.py#L45) | 保留 | 日常＋完整，方法原样 | 调用：`plan_sample_size`；原2处断言位置：`self.assertGreater`、`self.assertLessEqual`；完整断言不删 |

### test_code_protection.py（11项）

覆盖契约：当前治理检查；只读、哈希、原生发现、路径安全及登记规则。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B012 [CodeProtectionTests.test_unchanged_core_passes_without_writing](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_code_protection.py#L36) | 保留 | 日常＋完整，方法原样 | 原3处断言位置：`self.assertEqual`；完整断言不删 |
| B013 [CodeProtectionTests.test_changed_core_fails_without_resetting_it](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_code_protection.py#L42) | 保留 | 日常＋完整，方法原样 | 原2处断言位置：`self.assertTrue`、`self.assertEqual`；完整断言不删 |
| B014 [CodeProtectionTests.test_missing_core_fails_without_recreating_it](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_code_protection.py#L48) | 保留 | 日常＋完整，方法原样 | 原2处断言位置：`self.assertTrue`、`self.assertFalse`；完整断言不删 |
| B015 [CodeProtectionTests.test_unclassified_implementation_is_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_code_protection.py#L53) | 保留 | 日常＋完整，方法原样 | 原1处断言位置：`self.assertIn`；完整断言不删 |
| B016 [CodeProtectionTests.test_duplicate_and_invalid_classification_are_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_code_protection.py#L57) | 保留 | 日常＋完整，方法原样 | 原2处断言位置：`self.assertTrue`；完整断言不删 |
| B017 [CodeProtectionTests.test_external_path_is_rejected_before_reading](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_code_protection.py#L64) | 保留 | 日常＋完整，方法原样 | 原1处断言位置：`self.assertTrue`；完整断言不删 |
| B018 [CodeProtectionTests.test_symlink_cannot_replace_registered_core](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_code_protection.py#L68) | 保留 | 日常＋完整，方法原样 | 原1处断言位置：`self.assertTrue`；完整断言不删 |
| B019 [CodeProtectionTests.test_ignored_new_source_is_still_discovered](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_code_protection.py#L75) | 保留 | 日常＋完整，方法原样 | 原1处断言位置：`self.assertEqual`；完整断言不删 |
| B020 [CodeProtectionTests.test_native_inventory_respects_git_states_and_payload_exclusions](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_code_protection.py#L86) | 保留 | 日常＋完整，方法原样 | 原5处断言位置：`self.assertIn`、`self.assertEqual`、`self.assertNotIn`等；完整断言不删 |
| B021 [CodeProtectionTests.test_registry_self_hash_exception_is_explicit](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_code_protection.py#L124) | 保留 | 日常＋完整，方法原样 | 原2处断言位置：`self.assertIn`、`self.assertEqual`；完整断言不删 |
| B022 [CodeProtectionTests.test_controlled_file_does_not_need_a_frozen_digest](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_code_protection.py#L134) | 保留 | 日常＋完整，方法原样 | 原1处断言位置：`self.assertEqual`；完整断言不删 |

### test_config_projection.py（7项）

覆盖契约：共享投影方差、禁止投影后归一化、缓存身份及数组读入。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B023 [ConfigProjectionTests.test_default_config_validates](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_config_projection.py#L23) | 保留 | 日常＋完整，方法原样 | 调用：`load_config`；原9处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B024 [ConfigProjectionTests.test_invalid_projection_scale_related_budget_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_config_projection.py#L45) | 保留 | 日常＋完整，方法原样 | 调用：`load_config`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B025 [ConfigProjectionTests.test_projection_entries_have_expected_scale](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_config_projection.py#L54) | 保留 | 日常＋完整，方法原样 | 调用：`dense_gaussian_projection`；原2处断言位置：`self.assertAlmostEqual`；完整断言不删 |
| B026 [ConfigProjectionTests.test_projected_vectors_are_not_renormalized](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_config_projection.py#L60) | 保留 | 日常＋完整，方法原样 | 调用：`normalize_rows`、`dense_gaussian_projection`、`project_rows`；原2处断言位置：`self.assertGreater`、`self.assertFalse`；完整断言不删 |
| B027 [ConfigProjectionTests.test_model_free_embedding_array_ingestion](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_config_projection.py#L69) | 保留 | 日常＋完整，方法原样 | 调用：`load_embedding_array`；原2处断言位置：`self.assertEqual`、`np.testing.assert_array_equal`；完整断言不删 |
| B028 [ConfigProjectionTests.test_projection_metadata_change_invalidates_cache](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_config_projection.py#L79) | 合并M01 | 新方法：日常＋完整 | 同一 `validate_projection_cache_metadata`；seed错误＋合法同值 |
| B029 [ConfigProjectionTests.test_corpus_or_model_change_invalidates_projection_cache](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_config_projection.py#L100) | 合并M01 | 新方法：日常＋完整 | 同一校验器；model/corpus错误分别保留 |

### test_end_to_end.py（1项）

覆盖契约：仍由打包命令提供的合成 harness 完整输出、划分和复现。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B030 [EndToEndTests.test_complete_run_is_auditable_disjoint_and_reproducible](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_end_to_end.py#L30) | 保留 | 日常＋完整，方法原样 | 调用：`run_harness`；原22处断言位置：`self.assertTrue`、`self.assertEqual`、`self.assertIn`；完整断言不删 |

### test_index_retrieval.py（10项）

覆盖契约：共享精确平方 L2/流式/FAISS 接口、稳定平局与 k 边界。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B031 [ExactIndexTests.test_exact_top_k_matches_brute_force](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_index_retrieval.py#L69) | 保留 | 日常＋完整，方法原样 | 调用：`normalize_rows`、`ExactSquaredL2Index`；原2处断言位置：`np.testing.assert_array_equal`、`np.testing.assert_allclose`；完整断言不删 |
| B032 [ExactIndexTests.test_ties_are_broken_by_stable_id](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_index_retrieval.py#L83) | 保留 | 日常＋完整，方法原样 | 调用：`ExactSquaredL2Index`；原1处断言位置：`self.assertEqual`；完整断言不删 |
| B033 [ExactIndexTests.test_candidate_overlap_equals_exact_rerank_overlap](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_index_retrieval.py#L89) | 保留 | 日常＋完整，方法原样 | 调用：`normalize_rows`、`ExactSquaredL2Index`；原1处断言位置：`self.assertEqual`；完整断言不删 |
| B034 [ExactIndexTests.test_streaming_top_k_matches_brute_force_and_counts_scans](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_index_retrieval.py#L102) | 保留 | 日常＋完整，方法原样 | 调用：`normalize_rows`、`StreamingExactSquaredL2Index`；原5处断言位置：`np.testing.assert_array_equal`、`np.testing.assert_allclose`、`self.assertEqual`；完整断言不删 |
| B035 [ExactIndexTests.test_faiss_cpu_adapter_matches_streaming_contract](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_index_retrieval.py#L123) | 保留 | 日常＋完整，方法原样 | 调用：`normalize_rows`、`StreamingExactSquaredL2Index`、`FaissExactSquaredL2Index`；原9处断言位置：`np.testing.assert_array_equal`、`np.testing.assert_allclose`、`self.assertEqual`等；完整断言不删 |
| B036 [ExactIndexTests.test_faiss_adapter_resolves_bounded_ties_and_refuses_unclosed_tie_band](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_index_retrieval.py#L150) | 保留 | 日常＋完整，方法原样 | 调用：`FaissExactSquaredL2Index`；原2处断言位置：`self.assertEqual`、`self.assertRaises`；完整断言不删 |
| B037 [ExactIndexTests.test_faiss_adapter_refuses_missing_gpu_support](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_index_retrieval.py#L167) | 保留 | 日常＋完整，方法原样 | 调用：`FaissExactSquaredL2Index`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B038 [ExactIndexTests.test_faiss_gpu_indexes_can_share_one_resource_pool](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_index_retrieval.py#L177) | 保留 | 日常＋完整，方法原样 | 调用：`FaissExactSquaredL2Index`；原4处断言位置：`self.assertIs`、`self.assertEqual`、`self.assertFalse`等；完整断言不删 |
| B039 [ExactIndexTests.test_faiss_gpu_adapter_rejects_k_plus_guard_above_selection_limit](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_index_retrieval.py#L200) | 保留 | 日常＋完整，方法原样 | 调用：`FaissExactSquaredL2Index`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B040 [ExactIndexTests.test_real_faiss_cpu_conformance_when_installed](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_index_retrieval.py#L214) | 保留 | 日常＋完整，方法原样 | 调用：`normalize_rows`、`StreamingExactSquaredL2Index`、`FaissExactSquaredL2Index`；原2处断言位置：`np.testing.assert_array_equal`、`np.testing.assert_allclose`；完整断言不删 |

### test_lid_policy.py（5项）

覆盖契约：LID 欧氏转换、无效距离、预算网格、单调性及策略身份。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B041 [LIDPolicyTests.test_valid_lid_uses_euclidean_distances_and_excludes_boundary](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_lid_policy.py#L22) | 保留 | 日常＋完整，方法原样 | 原2处断言位置：`self.assertTrue`、`self.assertAlmostEqual`；完整断言不删 |
| B042 [LIDPolicyTests.test_lid_rejects_zero_duplicate_and_insufficient_distances](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_lid_policy.py#L29) | 保留 | 日常＋完整，方法原样 | 原3处断言位置：`self.assertFalse`、`self.assertEqual`；完整断言不删 |
| B043 [LIDPolicyTests.test_budget_policies_are_grid_safe_and_monotone](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_lid_policy.py#L42) | 保留 | 日常＋完整，方法原样 | 调用：`MonotoneBinnedPolicy.fit`、`FixedBudgetPolicy`；原5处断言位置：`self.assertTrue`、`self.assertIn`、`self.assertGreaterEqual`等；完整断言不删 |
| B044 [LIDPolicyTests.test_policy_fingerprint_canonicalizes_cross_platform_float_noise](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_lid_policy.py#L73) | 保留 | 日常＋完整，方法原样 | 调用：`MonotoneBinnedPolicy`；原2处断言位置：`self.assertEqual`；完整断言不删 |
| B045 [LIDPolicyTests.test_frozen_monotone_policy_loader_rejects_tampering](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_lid_policy.py#L91) | 保留 | 日常＋完整，方法原样 | 调用：`MonotoneBinnedPolicy`、`MonotoneBinnedPolicy.from_serialized(artifact).serialize`、`MonotoneBinnedPolicy.from_serialized`；原2处断言位置：`self.assertEqual`、`self.assertRaisesRegex`；完整断言不删 |

### test_mprime_sweep.py（2项）

覆盖契约：原合成维度扫描 CLI；选模与认证分离、冻结结果。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B046 [MPrimeSweepTests.test_rejects_certification_records_during_selection](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_mprime_sweep.py#L31) | 保留 | 日常＋完整，方法原样 | 调用：`_validate_tune_only`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B047 [MPrimeSweepTests.test_sweep_freezes_selection_before_fresh_certification](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_mprime_sweep.py#L40) | 保留 | 日常＋完整，方法原样 | 调用：`run_mprime_sweep`、`fingerprint`；原12处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertIn`等；完整断言不删 |

### test_pdctp_calibration.py（7项）

覆盖契约：历史 PDCTP 校准器的约束求解、参数格点、序列化与残差修正。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B048 [PDCTPCalibrationTests.test_lid_fit_enforces_nonnegative_pilot_coefficient_and_cal_role](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_calibration.py#L40) | 退役（日常） | 历史＋完整，方法原样 | 调用：`LIDCalibrationRecord`、`PilotLIDCalibrator.fit`；原3处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertRaisesRegex`；完整断言不删 |
| B049 [PDCTPCalibrationTests.test_lid_artifact_round_trip_domain_and_tamper_validation](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_calibration.py#L75) | 退役（日常） | 历史＋完整，方法原样 | 调用：`LIDCalibrationRecord`、`PilotLIDCalibrator.fit`、`PilotLIDCalibrator.from_serialized`；原6处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertRaises`；完整断言不删 |
| B050 [PDCTPCalibrationTests.test_quantile_objective_and_hand_computed_median_optimum](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_calibration.py#L118) | 退役（日常） | 历史＋完整，方法原样 | 调用：`quantile_pinball_loss`、`BudgetResidualRecord`、`TriBudgetResidualCalibrator.fit`；原3处断言位置：`self.assertAlmostEqual`、`self.assertEqual`；完整断言不删 |
| B051 [PDCTPCalibrationTests.test_residual_grid_ceiling_lower_bound_fallback_and_ties](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_calibration.py#L169) | 退役（日常） | 历史＋完整，方法原样 | 原8处断言位置：`self.assertEqual`、`self.assertGreaterEqual`、`self.assertTrue`；完整断言不删 |
| B052 [PDCTPCalibrationTests.test_residual_corrections_can_be_positive_low_and_negative_high](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_calibration.py#L190) | 退役（日常） | 历史＋完整，方法原样 | 原4处断言位置：`self.assertGreater`、`self.assertEqual`、`self.assertLess`；完整断言不删 |
| B053 [PDCTPCalibrationTests.test_residual_round_trip_and_tamper_rejection](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_calibration.py#L199) | 退役（日常） | 历史＋完整，方法原样 | 调用：`TriBudgetResidualCalibrator.from_serialized`、`BudgetResidualRecord`、`TriBudgetResidualCalibrator.fit`；原4处断言位置：`self.assertEqual`、`self.assertRaises`、`self.assertRaisesRegex`；完整断言不删 |
| B054 [PDCTPCalibrationTests.test_residual_parameters_snap_platform_solver_noise](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_calibration.py#L227) | 退役（日常） | 历史＋完整，方法原样 | 原4处断言位置：`self.assertEqual`、`self.assertTrue`；完整断言不删 |

### test_pdctp_dataset_audit.py（5项）

覆盖契约：历史 FiQA 源审计、归档/标签完整性与五角色容量。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B055 [PDCTPDatasetAuditTests.test_audit_is_reproducible_and_proves_five_role_capacity](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_dataset_audit.py#L119) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_dataset_audit_config`、`audit_pdctp_beir_source`；原16处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertFalse`等；完整断言不删 |
| B056 [PDCTPDatasetAuditTests.test_archive_sha256_mismatch_is_rejected_before_output](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_dataset_audit.py#L176) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_dataset_audit_config`、`audit_pdctp_beir_source`；原2处断言位置：`self.assertRaisesRegex`、`self.assertFalse`；完整断言不删 |
| B057 [PDCTPDatasetAuditTests.test_missing_qrel_document_is_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_dataset_audit.py#L191) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_dataset_audit_config`、`audit_pdctp_beir_source`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B058 [PDCTPDatasetAuditTests.test_checked_in_fiqa_config_pins_source_license_and_power](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_dataset_audit.py#L204) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_dataset_audit_config`；原6处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B059 [PDCTPDatasetAuditTests.test_checked_in_fiqa_audit_is_self_consistent_and_outcome_free](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_dataset_audit.py#L223) | 退役（日常） | 历史＋完整，方法原样 | 调用：`fingerprint`；原8处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |

### test_pdctp_features.py（7项）

覆盖契约：历史 PDCTP pilot 特征、距离排序/格点/无效状态及防标签输入。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B060 [PDCTPFeatureTests.test_hand_computed_feature_profile](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_features.py#L39) | 退役（日常） | 历史＋完整，方法原样 | 原13处断言位置：`self.assertTrue`、`self.assertAlmostEqual`、`self.assertEqual`；完整断言不删 |
| B061 [PDCTPFeatureTests.test_ratio_features_are_scale_invariant_but_radius_is_not](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_features.py#L67) | 退役（日常） | 历史＋完整，方法原样 | 原2处断言位置：`self.assertAlmostEqual`；完整断言不删 |
| B062 [PDCTPFeatureTests.test_feature_values_use_the_frozen_cross_platform_lattice](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_features.py#L80) | 退役（日常） | 历史＋完整，方法原样 | 调用：`PilotDistanceObservation.from_arrays`；原3处断言位置：`self.assertEqual`、`self.assertTrue`；完整断言不删 |
| B063 [PDCTPFeatureTests.test_squared_l2_is_converted_after_stable_original_sort](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_features.py#L97) | 退役（日常） | 历史＋完整，方法原样 | 调用：`stable_sort_pilot_distances`；原3处断言位置：`self.assertEqual`；完整断言不删 |
| B064 [PDCTPFeatureTests.test_invalid_states_are_deterministic_and_trigger_fixed_fill](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_features.py#L107) | 退役（日常） | 历史＋完整，方法原样 | 调用：`PilotDistanceObservation.from_arrays`；原9处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertIsNotNone`；完整断言不删 |
| B065 [PDCTPFeatureTests.test_observation_contract_cannot_accept_forbidden_inference_fields](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_features.py#L141) | 退役（日常） | 历史＋完整，方法原样 | 调用：`PilotDistanceObservation`；原2处断言位置：`self.assertTrue`、`self.assertRaises`；完整断言不删 |
| B066 [PDCTPFeatureTests.test_feature_spec_round_trip_and_tamper_rejection](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_features.py#L153) | 退役（日常） | 历史＋完整，方法原样 | 调用：`PilotDistanceFeatureSpec.from_serialized`；原2处断言位置：`self.assertEqual`、`self.assertRaisesRegex`；完整断言不删 |

### test_pdctp_fiqa_embedding_gate.py（6项）

覆盖契约：历史 FiQA/E5 嵌入闸门、角色清单、缓存及模型身份。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B067 [PDCTPFiQAEmbeddingGateTests.test_checked_in_real_embedding_audit_is_self_consistent](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_embedding_gate.py#L259) | 退役（日常） | 历史＋完整，方法原样 | 调用：`fingerprint`；原9处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertFalse`；完整断言不删 |
| B068 [PDCTPFiQAEmbeddingGateTests.test_checked_in_embedding_request_pins_real_text_manifest_and_e5](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_embedding_gate.py#L284) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_text_embedding_config`；原8处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertFalse`；完整断言不删 |
| B069 [PDCTPFiQAEmbeddingGateTests.test_text_only_preparation_is_deterministic_and_keeps_empty_marker](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_embedding_gate.py#L306) | 退役（日常） | 历史＋完整，方法原样 | 原7处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertTrue`；完整断言不删 |
| B070 [PDCTPFiQAEmbeddingGateTests.test_preparation_rejects_changed_archive_before_publication](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_embedding_gate.py#L331) | 退役（日常） | 历史＋完整，方法原样 | 原2处断言位置：`self.assertRaisesRegex`、`self.assertFalse`；完整断言不删 |
| B071 [PDCTPFiQAEmbeddingGateTests.test_embedding_build_accepts_text_only_profile_and_audit_is_deterministic](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_embedding_gate.py#L342) | 退役（日常） | 历史＋完整，方法原样 | 调用：`build_or_load_text_embedding_cache`、`audit_pdctp_embedding_cache`；原3处断言位置：`self.assertEqual`、`self.assertTrue`；完整断言不删 |
| B072 [PDCTPFiQAEmbeddingGateTests.test_independent_audit_rejects_tampered_cache_array](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_embedding_gate.py#L373) | 退役（日常） | 历史＋完整，方法原样 | 调用：`build_or_load_text_embedding_cache`、`audit_pdctp_embedding_cache`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |

### test_pdctp_fiqa_query_cal.py（6项）

覆盖契约：历史 FiQA cal-only 拟合、候选重建和单次精确扫描。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B073 [PDCTPFiQAQueryCalTests.test_checked_in_query_cal_audit_is_self_consistent_and_fit_only](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_cal.py#L103) | 退役（日常） | 历史＋完整，方法原样 | 调用：`fingerprint`；原15处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertFalse`；完整断言不删 |
| B074 [PDCTPFiQAQueryCalTests.test_checked_config_binds_accepted_embedding_gate](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_cal.py#L141) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_query_cal_config`；原5处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B075 [PDCTPFiQAQueryCalTests.test_compact_quantile_solver_round_trip_and_legacy_equivalence](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_cal.py#L155) | 退役（日常） | 历史＋完整，方法原样 | 调用：`PilotDistanceFeatureSpec`、`PilotFeatureVector`、`BudgetResidualRecord`；原5处断言位置：`self.assertEqual`、`self.assertAlmostEqual`、`self.assertNotEqual`；完整断言不删 |
| B076 [PDCTPFiQAQueryCalTests.test_exact_query_cal_records_are_deterministic_and_single_scan](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_cal.py#L198) | 退役（日常） | 历史＋完整，方法原样 | 调用：`build_query_cal_records`；原8处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B077 [PDCTPFiQAQueryCalTests.test_all_candidates_fit_without_selection_and_reconstruct](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_cal.py#L235) | 退役（日常） | 历史＋完整，方法原样 | 调用：`build_query_cal_records`、`fit_query_cal_candidates`、`reconstruct_residual_candidate`；原7处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertRaisesRegex`；完整断言不删 |
| B078 [PDCTPFiQAQueryCalTests.test_upstream_tamper_is_rejected_while_all_roles_are_closed](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_cal.py#L269) | 退役（日常） | 历史＋完整，方法原样 | 调用：`FiveRoleAssignments`、`FiveRoleProtocolGuard`、`fingerprint`；原3处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertRaisesRegex`；完整断言不删 |

### test_pdctp_fiqa_query_cert.py（6项）

覆盖契约：历史 FiQA 六假设认证、提前封存及失败终态。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B079 [PDCTPFiQAQueryCertTests.test_checked_config_freezes_one_time_certification_only](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_cert.py#L179) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_query_cert_config`；原7处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertRaisesRegex`；完整断言不删 |
| B080 [PDCTPFiQAQueryCertTests.test_accepted_tune_audit_is_still_pre_cert_and_terminally_frozen](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_cert.py#L203) | 退役（日常） | 历史＋完整，方法原样 | 调用：`fingerprint`、`_validate_query_tune_file_inventory`；原8处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertRaisesRegex`；完整断言不删 |
| B081 [PDCTPFiQAQueryCertTests.test_qrel_parser_skips_every_non_cert_outcome_before_parsing](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_cert.py#L235) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_query_cert_qrels`；原4处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertRaisesRegex`；完整断言不删 |
| B082 [PDCTPFiQAQueryCertTests.test_certification_is_one_time_terminal_and_keeps_later_roles_closed](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_cert.py#L264) | 退役（日常） | 历史＋完整，方法原样 | 调用：`FiveRoleAssignments`、`FiveRoleProtocolGuard`；原8处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertFalse`等；完整断言不删 |
| B083 [PDCTPFiQAQueryCertTests.test_exact_cert_records_are_deterministic_label_safe_and_one_scan](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_cert.py#L304) | 退役（日常） | 历史＋完整，方法原样 | 调用：`build_query_cert_records`、`PDCTPDecisionInput`；原14处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertTrue`；完整断言不删 |
| B084 [PDCTPFiQAQueryCertTests.test_all_six_paired_bounds_reconstruct_and_failure_is_terminal](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_cert.py#L383) | 退役（日常） | 历史＋完整，方法原样 | 调用：`build_query_cert_records`、`make_query_certification`、`validate_paired_bound`；原9处断言位置：`self.assertTrue`、`self.assertFalse`、`self.assertEqual`；完整断言不删 |

### test_pdctp_fiqa_query_tune.py（8项）

覆盖契约：历史 FiQA 独立候选选择、冻结策略、预算及访问前校验。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B085 [PDCTPFiQAQueryTuneTests.test_checked_config_freezes_query_tune_only_and_all_family_counts](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_tune.py#L152) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_query_tune_config`；原6处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertRaisesRegex`；完整断言不删 |
| B086 [PDCTPFiQAQueryTuneTests.test_checked_in_query_tune_audit_accepts_only_the_frozen_selection](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_tune.py#L176) | 退役（日常） | 历史＋完整，方法原样 | 调用：`fingerprint`；原16处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertFalse`等；完整断言不删 |
| B087 [PDCTPFiQAQueryTuneTests.test_qrel_parser_skips_non_tune_outcomes_before_parsing](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_tune.py#L218) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_query_tune_qrels`；原4处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertRaisesRegex`；完整断言不删 |
| B088 [PDCTPFiQAQueryTuneTests.test_exact_tune_records_are_deterministic_reconstructable_and_one_scan](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_tune.py#L247) | 退役（日常） | 历史＋完整，方法原样 | 调用：`build_query_tune_records`；原11处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertNotIn`；完整断言不删 |
| B089 [PDCTPFiQAQueryTuneTests.test_all_candidate_families_select_independently_and_suite_reconstructs](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_tune.py#L325) | 退役（日常） | 历史＋完整，方法原样 | 调用：`select_query_tune_policies`、`reconstruct_frozen_policy_suite`；原10处断言位置：`self.assertTrue`、`self.assertEqual`、`np.testing.assert_array_equal`等；完整断言不删 |
| B090 [PDCTPFiQAQueryTuneTests.test_tri_predict_profiles_are_reused_across_threshold_candidates](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_tune.py#L379) | 退役（日常） | 历史＋完整，方法原样 | 调用：`select_query_tune_policies`；原3处断言位置：`self.assertTrue`、`self.assertLessEqual`、`self.assertLess`；完整断言不删 |
| B091 [PDCTPFiQAQueryTuneTests.test_missing_eligible_family_is_terminal_without_policy_substitution](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_tune.py#L417) | 退役（日常） | 历史＋完整，方法原样 | 调用：`select_query_tune_policies`；原5处断言位置：`self.assertFalse`、`self.assertEqual`、`self.assertIn`等；完整断言不删 |
| B092 [PDCTPFiQAQueryTuneTests.test_upstream_tamper_is_rejected_before_tune_token](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_fiqa_query_tune.py#L456) | 退役（日常） | 历史＋完整，方法原样 | 调用：`FiveRoleAssignments`、`FiveRoleProtocolGuard`、`fingerprint`；原3处断言位置：`self.assertEqual`、`self.assertIsNone`、`self.assertRaisesRegex`；完整断言不删 |

### test_pdctp_foundation.py（1项）

覆盖契约：历史 PDCTP 五角色合成 walking skeleton 的整套复现。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B093 [PDCTPFoundationEndToEndTests.test_five_role_walking_skeleton_is_deterministic_and_auditable](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_foundation.py#L17) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_foundation_config`、`run_pdctp_foundation`、`validate_paired_bound`；原29处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertFalse`等；完整断言不删 |

### test_pdctp_policies.py（4项）

覆盖契约：历史 PDCTP 各消融统一输入、回退及 Raw v1 行为保持。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B094 [PDCTPPolicyTests.test_raw_v1_serialization_loading_and_decisions_remain_unchanged](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_policies.py#L92) | 退役（日常） | 历史＋完整，方法原样 | 调用：`RawTriPredictPDCTPPolicy`、`TriPredictPolicy.from_serialized`；原4处断言位置：`self.assertEqual`；完整断言不删 |
| B095 [PDCTPPolicyTests.test_all_ablations_share_one_decision_interface](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_policies.py#L105) | 退役（日常） | 历史＋完整，方法原样 | 调用：`MonotoneBinnedPolicy`、`FixedPDCTPPolicy`、`MonotonePDCTPPolicy`；原7处断言位置：`self.assertEqual`、`self.assertIn`、`self.assertGreaterEqual`；完整断言不删 |
| B096 [PDCTPPolicyTests.test_invalid_feature_state_has_terminal_fallback](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_policies.py#L151) | 退役（日常） | 历史＋完整，方法原样 | 调用：`CalibratedTriPredictPolicy`；原2处断言位置：`self.assertEqual`、`self.assertTrue`；完整断言不删 |
| B097 [PDCTPPolicyTests.test_decision_contract_excludes_all_forbidden_fields](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_policies.py#L174) | 退役（日常） | 历史＋完整，方法原样 | 调用：`PDCTPDecisionInput`；原3处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertRaises`；完整断言不删 |

### test_pdctp_protocol_statistics.py（7项）

覆盖契约：历史五角色协议与成对界、总体 alpha 分配及 power。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B098 [PDCTPProtocolStatisticsTests.test_checked_in_versioned_config_has_all_foundation_schemas](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_protocol_statistics.py#L43) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_foundation_config`；原10处断言位置：`self.assertEqual`、`self.assertGreater`；完整断言不删 |
| B099 [PDCTPProtocolStatisticsTests.test_config_rejects_role_selection_and_latency_mutations](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_protocol_statistics.py#L56) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_foundation_config`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B100 [PDCTPProtocolStatisticsTests.test_duplicate_text_and_identity_crossing_are_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_protocol_statistics.py#L76) | 退役（日常） | 历史＋完整，方法原样 | 调用：`FiveRoleAssignments`；原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B101 [PDCTPProtocolStatisticsTests.test_five_role_state_machine_refuses_early_or_labeled_access](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_protocol_statistics.py#L91) | 退役（日常） | 历史＋完整，方法原样 | 调用：`FiveRoleProtocolGuard`；原8处断言位置：`self.assertRaisesRegex`、`self.assertFalse`、`self.assertTrue`；完整断言不删 |
| B102 [PDCTPProtocolStatisticsTests.test_paired_bound_is_hand_reconstructable_and_tamper_evident](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_protocol_statistics.py#L138) | 退役（日常） | 历史＋完整，方法原样 | 调用：`make_paired_bound`、`validate_paired_bound`；原3处断言位置：`self.assertEqual`、`self.assertRaisesRegex`；完整断言不删 |
| B103 [PDCTPProtocolStatisticsTests.test_budget_superiority_uses_upper_paired_bound](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_protocol_statistics.py#L164) | 退役（日常） | 历史＋完整，方法原样 | 调用：`make_paired_bound`；原2处断言位置：`self.assertLess`、`self.assertTrue`；完整断言不删 |
| B104 [PDCTPProtocolStatisticsTests.test_familywise_allocation_and_power_plan_are_deterministic](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_protocol_statistics.py#L181) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_foundation_config`、`bonferroni_allocation`、`make_power_plan`；原5处断言位置：`self.assertAlmostEqual`、`self.assertEqual`、`self.assertGreater`；完整断言不删 |

### test_pdctp_real_protocol.py（5项）

覆盖契约：历史 FiQA 协议冻结、输入篡改与角色闭合。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B105 [PDCTPRealProtocolTests.test_checked_in_config_freezes_real_contract](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_real_protocol.py#L31) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_real_protocol_config`、`derive_fiqa_budget_grid`；原9处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B106 [PDCTPRealProtocolTests.test_freeze_is_reproducible_power_supported_and_all_roles_closed](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_real_protocol.py#L51) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_real_protocol_config`、`freeze_pdctp_real_protocol`；原12处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertTrue`等；完整断言不删 |
| B107 [PDCTPRealProtocolTests.test_checked_in_freeze_artifacts_are_self_consistent](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_real_protocol.py#L102) | 退役（日常） | 历史＋完整，方法原样 | 调用：`fingerprint`；原6处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B108 [PDCTPRealProtocolTests.test_tampered_input_is_rejected_before_output](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_real_protocol.py#L131) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_real_protocol_config`、`freeze_pdctp_real_protocol`；原2处断言位置：`self.assertRaisesRegex`、`self.assertFalse`；完整断言不删 |
| B109 [PDCTPRealProtocolTests.test_mutations_of_projection_grid_roles_and_stop_gate_are_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_real_protocol.py#L144) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_pdctp_real_protocol_config`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |

### test_pdctp_v3.py（6项）

覆盖契约：历史有效 Tri-LID/曲线修正、缓存身份及候选一致性。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B110 [PDCTPV3Tests.test_cache_identity_uses_exact_lid_bits_and_full_numerical_problem](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_v3.py#L117) | 退役（日常） | 历史＋完整，方法原样 | 调用：`TriPredictPredictionGridCacheV3`；原5处断言位置：`self.assertNotEqual`、`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B111 [PDCTPV3Tests.test_checked_config_is_v3_query_cal_diagnostic_only](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_v3.py#L160) | 退役（日常） | 历史＋完整，方法原样 | 原7处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B112 [PDCTPV3Tests.test_fit_is_query_cal_only_and_artifact_is_new_cache_free_schema](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_v3.py#L182) | 退役（日常） | 历史＋完整，方法原样 | 调用：`EffectiveTriLIDCalibratorV3.fit`、`EffectiveTriLIDCalibratorV3.from_serialized`；原6处断言位置：`self.assertRaisesRegex`、`self.assertEqual`、`self.assertNotIn`；完整断言不删 |
| B113 [PDCTPV3Tests.test_two_regime_ablation_changes_only_curve_shape_and_fits_better](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_v3.py#L215) | 退役（日常） | 历史＋完整，方法原样 | 调用：`EffectiveCurveTriPredictPolicyV3`、`PDCTPDecisionInput`；原5处断言位置：`self.assertEqual`、`self.assertIsNotNone`、`self.assertLess`；完整断言不删 |
| B114 [PDCTPV3Tests.test_complete_candidate_suite_is_identical_cached_and_uncached](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_v3.py#L258) | 退役（日常） | 历史＋完整，方法原样 | 调用：`TriPredictPredictionGridCacheV3`、`EffectiveCurveTriPredictPolicyV3`、`fingerprint`；原4处断言位置：`self.assertEqual`、`self.assertLessEqual`；完整断言不删 |
| B115 [PDCTPV3Tests.test_inference_input_stays_narrow_and_invalid_features_terminate](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_pdctp_v3.py#L322) | 退役（日常） | 历史＋完整，方法原样 | 调用：`EffectiveCurveTriPredictPolicyV3`；原4处断言位置：`self.assertEqual`、`self.assertTrue`；完整断言不删 |

### test_real_dimension_sweep.py（5项）

覆盖契约：历史 SciFact 实数据维度扫描、投影成本、平局及 tune-only 选择。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B116 [RealDimensionSweepTests.test_checked_in_protocol_freezes_common_cost_and_tune_scope](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_dimension_sweep.py#L41) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_real_dimension_sweep_config`；原10处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertTrue`；完整断言不删 |
| B117 [RealDimensionSweepTests.test_config_rejects_protected_scope_and_cost_formula_mutation](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_dimension_sweep.py#L59) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_real_dimension_sweep_config`；原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B118 [RealDimensionSweepTests.test_common_coordinate_work_includes_query_projection](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_dimension_sweep.py#L77) | 退役（日常） | 历史＋完整，方法原样 | 调用：`_coordinate_work`；原5处断言位置：`self.assertEqual`、`self.assertAlmostEqual`；完整断言不删 |
| B119 [RealDimensionSweepTests.test_projected_ranking_matches_exact_index_and_stable_id_ties](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_dimension_sweep.py#L87) | 退役（日常） | 历史＋完整，方法原样 | 调用：`_exact_projected_rankings`、`ExactSquaredL2Index`；原2处断言位置：`np.testing.assert_array_equal`、`self.assertEqual`；完整断言不删 |
| B120 [RealDimensionSweepTests.test_tiny_runner_is_tune_only_auditable_and_reproducible](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_dimension_sweep.py#L98) | 退役（日常） | 历史＋完整，方法原样 | 调用：`write_json`、`stable_id_hash`、`fingerprint`；原8处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertIn`；完整断言不删 |

### test_real_original_baseline.py（5项）

覆盖契约：历史 SciFact 原空间基线、证据指标、缓存及 tune 身份。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B121 [RealOriginalBaselineTests.test_tune_only_exact_baseline_is_auditable_and_reproducible](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_original_baseline.py#L223) | 退役（日常） | 历史＋完整，方法原样 | 调用：`run_real_original_baseline`；原12处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertNotIn`；完整断言不删 |
| B122 [RealOriginalBaselineTests.test_config_rejects_certification_or_test_scope](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_original_baseline.py#L270) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_real_original_baseline_config`、`run_real_original_baseline`；原3处断言位置：`self.assertRaisesRegex`、`self.assertFalse`；完整断言不删 |
| B123 [RealOriginalBaselineTests.test_graded_ndcg_penalizes_reversed_relevance](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_original_baseline.py#L298) | 退役（日常） | 历史＋完整，方法原样 | 调用：`_ndcg_at_k`；原1处断言位置：`self.assertAlmostEqual`；完整断言不删 |
| B124 [RealOriginalBaselineTests.test_tampered_embedding_is_rejected_before_output](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_original_baseline.py#L309) | 退役（日常） | 历史＋完整，方法原样 | 调用：`run_real_original_baseline`；原2处断言位置：`self.assertRaisesRegex`、`self.assertFalse`；完整断言不删 |
| B125 [RealOriginalBaselineTests.test_checked_in_config_freezes_tune_scope_and_real_identities](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_original_baseline.py#L330) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_real_original_baseline_config`；原6处断言位置：`self.assertEqual`；完整断言不删 |

### test_real_policy_certify.py（4项）

覆盖契约：历史 SciFact 证书入口、冻结策略、防认证后选模及终态。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B126 [RealPolicyCertificationTests.test_checked_in_protocol_freezes_cert_scope_and_genoa_bundle](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_certify.py#L418) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_real_policy_certification_config`；原8处断言位置：`self.assertEqual`；完整断言不删 |
| B127 [RealPolicyCertificationTests.test_config_rejects_noncert_scope_and_posthoc_selection](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_certify.py#L435) | 退役（日常） | 历史＋完整，方法原样 | 调用：`write_json`、`load_real_policy_certification_config`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B128 [RealPolicyCertificationTests.test_frozen_policy_tamper_is_rejected_before_protected_data_access](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_certify.py#L458) | 退役（日常） | 历史＋完整，方法原样 | 调用：`write_json`、`load_real_policy_certification_config`、`run_real_policy_certification`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B129 [RealPolicyCertificationTests.test_tiny_certification_is_auditable_terminal_and_reproducible](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_certify.py#L482) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_real_policy_certification_config`、`run_real_policy_certification`、`make_certificate`；原15处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertFalse`；完整断言不删 |

### test_real_policy_test.py（4项）

覆盖契约：历史 SciFact 终态证书绑定与仅描述性 test 记录。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B130 [RealPolicyTestTests.test_checked_in_protocol_freezes_test_and_terminal_cert_identities](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_test.py#L161) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_real_policy_test_config`、`load_real_policy_certification_config`；原7处断言位置：`self.assertEqual`；完整断言不删 |
| B131 [RealPolicyTestTests.test_config_rejects_non_test_scope_selection_certificate_and_retuning](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_test.py#L193) | 退役（日常） | 历史＋完整，方法原样 | 调用：`write_json`、`load_real_policy_test_config`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B132 [RealPolicyTestTests.test_certification_tamper_is_rejected_before_test_data_access](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_test.py#L222) | 退役（日常） | 历史＋完整，方法原样 | 调用：`write_json`、`load_real_policy_test_config`、`run_real_policy_test`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B133 [RealPolicyTestTests.test_tiny_test_is_descriptive_auditable_and_reproducible](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_test.py#L248) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_real_policy_test_config`、`run_real_policy_test`、`stable_id_hash`；原23处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertTrue`等；完整断言不删 |

### test_real_policy_tune.py（8项）

覆盖契约：历史 SciFact tune-only 策略选择、完整语料边界及成本。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B134 [RealPolicyTuneTests.test_checked_in_protocol_is_tune_only_and_frozen](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_tune.py#L24) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_real_policy_tune_config`；原13处断言位置：`self.assertEqual`、`self.assertIn`；完整断言不删 |
| B135 [RealPolicyTuneTests.test_config_rejects_protected_scope_and_cost_mutation](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_tune.py#L45) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_real_policy_tune_config`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B136 [RealPolicyTuneTests.test_genoa_lid_tail_noise_is_canonicalized_before_policy_use](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_tune.py#L71) | 退役（日常） | 历史＋完整，方法原样 | 调用：`_canonical_lid_float`；原1处断言位置：`self.assertEqual`；完整断言不删 |
| B137 [RealPolicyTuneTests.test_compiled_lookup_is_not_a_scientific_result_artifact](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_tune.py#L86) | 退役（日常） | 历史＋完整，方法原样 | 原1处断言位置：`self.assertNotIn`；完整断言不删 |
| B138 [RealPolicyTuneTests.test_tune_scope_guard_rejects_certification_records](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_tune.py#L91) | 退役（日常） | 历史＋完整，方法原样 | 调用：`_validate_tune_only`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B139 [RealPolicyTuneTests.test_coordinate_work_uses_the_predeclared_common_objective](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_tune.py#L106) | 退役（日常） | 历史＋完整，方法原样 | 调用：`_coordinate_work`；原5处断言位置：`self.assertEqual`、`self.assertAlmostEqual`；完整断言不删 |
| B140 [RealPolicyTuneTests.test_cached_tri_decision_uses_exact_complete_corpus_boundary](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_tune.py#L119) | 退役（日常） | 历史＋完整，方法原样 | 调用：`TriPredictPolicy`、`_choose_tri_from_raw`；原5处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B141 [RealPolicyTuneTests.test_decision_evaluation_reads_only_selected_tune_budget](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_policy_tune.py#L147) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_real_policy_tune_config`、`PolicyDecision`、`_evaluate_decisions`；原3处断言位置：`self.assertEqual`；完整断言不删 |

### test_real_tune_diagnostics.py（3项）

覆盖契约：历史 SciFact tune 归因/证据诊断及禁止重新认证。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B142 [RealTuneDiagnosticsTests.test_checked_in_config_is_tune_only_and_frozen](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_tune_diagnostics.py#L109) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_real_tune_diagnostics_config`、`load_real_policy_certification_config`；原6处断言位置：`self.assertEqual`；完整断言不删 |
| B143 [RealTuneDiagnosticsTests.test_config_rejects_protected_scope_selection_and_recertification](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_tune_diagnostics.py#L122) | 退役（日常） | 历史＋完整，方法原样 | 调用：`write_json`、`load_real_tune_diagnostics_config`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B144 [RealTuneDiagnosticsTests.test_tiny_diagnostics_are_auditable_tune_only_and_reproducible](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_real_tune_diagnostics.py#L151) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_real_tune_diagnostics_config`、`run_real_tune_diagnostics`、`stable_id_hash`；原20处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertIn`；完整断言不删 |

### test_retrieval_benchmark.py（6项）

覆盖契约：共享系统基准、同值/切点语义、实际检索工作及 GPU k 上限。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B145 [RetrievalBenchmarkTests.test_conformance_accepts_only_internal_cutoff_permutations](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_retrieval_benchmark.py#L67) | 保留 | 日常＋完整，方法原样 | 调用：`_compare_search_results`；原5处断言位置：`self.assertTrue`、`self.assertFalse`；完整断言不删 |
| B146 [RetrievalBenchmarkTests.test_conformance_rejects_a_row_crossing_a_semantic_cutoff](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_retrieval_benchmark.py#L83) | 保留 | 日常＋完整，方法原样 | 调用：`_compare_search_results`；原3处断言位置：`self.assertFalse`、`self.assertTrue`；完整断言不删 |
| B147 [RetrievalBenchmarkTests.test_tiny_benchmark_reuses_one_scan_and_matches_double_scan_decisions](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_retrieval_benchmark.py#L126) | 保留 | 日常＋完整，方法原样 | 调用：`run_retrieval_benchmark`；原15处断言位置：`self.assertEqual`、`self.assertTrue`；完整断言不删 |
| B148 [RetrievalBenchmarkTests.test_gpu_k_limit_is_rejected_before_output_or_fixture_generation](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_retrieval_benchmark.py#L199) | 保留 | 日常＋完整，方法原样 | 调用：`load_retrieval_benchmark_config`、`run_retrieval_benchmark`；原2处断言位置：`self.assertRaisesRegex`、`self.assertFalse`；完整断言不删 |
| B149 [RetrievalBenchmarkTests.test_faiss_1m_config_leaves_exactly_one_guard_below_gpu_limit](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_retrieval_benchmark.py#L212) | 保留 | 日常＋完整，方法原样 | 调用：`load_retrieval_benchmark_config`；原2处断言位置：`self.assertEqual`；完整断言不删 |
| B150 [RetrievalBenchmarkTests.test_tiny_faiss_cpu_benchmark_conforms_to_numpy](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_retrieval_benchmark.py#L224) | 保留 | 日常＋完整，方法原样 | 调用：`run_retrieval_benchmark`；原15处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertIsNone`等；完整断言不删 |

### test_text_embeddings.py（6项）

覆盖契约：历史通用文本嵌入生产器；固定模型、输入格式、规范化与缓存校验。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B151 [TextEmbeddingTests.test_fake_provider_builds_normalized_cache_and_reuses_it](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_text_embeddings.py#L163) | 退役（日常） | 历史＋完整，方法原样 | 调用：`build_or_load_text_embedding_cache`；原15处断言位置：`self.assertFalse`、`self.assertEqual`、`np.testing.assert_allclose`等；完整断言不删 |
| B152 [TextEmbeddingTests.test_changed_dataset_artifact_is_rejected_before_provider_call](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_text_embeddings.py#L209) | 退役（日常） | 历史＋完整，方法原样 | 调用：`build_or_load_text_embedding_cache`；原2处断言位置：`self.assertRaisesRegex`、`self.assertEqual`；完整断言不删 |
| B153 [TextEmbeddingTests.test_changed_request_refuses_existing_cache](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_text_embeddings.py#L228) | 退役（日常） | 历史＋完整，方法原样 | 调用：`build_or_load_text_embedding_cache`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B154 [TextEmbeddingTests.test_tampered_array_refuses_cache_reuse](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_text_embeddings.py#L250) | 退役（日常） | 历史＋完整，方法原样 | 调用：`build_or_load_text_embedding_cache`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B155 [TextEmbeddingTests.test_zero_provider_output_does_not_publish_partial_cache](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_text_embeddings.py#L271) | 退役（日常） | 历史＋完整，方法原样 | 调用：`build_or_load_text_embedding_cache`；原2处断言位置：`self.assertRaisesRegex`、`self.assertFalse`；完整断言不删 |
| B156 [TextEmbeddingTests.test_checked_in_config_pins_model_and_dataset](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_text_embeddings.py#L286) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_text_embedding_config`；原10处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |

### test_tls_rag_step2.py（13项）

覆盖契约：Step5 仍间接依赖的合成状态、精确前缀、证据隔离和工作计量。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B157 [TlsRagStep2Tests.test_external_ids_normalization_projection_scale_and_no_renormalization](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step2.py#L60) | 保留 | 日常＋完整，方法原样 | 调用：`dense_gaussian_projection`；原6处断言位置：`self.assertFalse`、`np.testing.assert_allclose`、`self.assertAlmostEqual`等；完整断言不删 |
| B158 [TlsRagStep2Tests.test_squared_l2_stable_string_ties_in_original_and_projected_spaces](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step2.py#L83) | 保留 | 日常＋完整，方法原样 | 原5处断言位置：`self.assertLess`、`self.assertEqual`、`self.assertGreaterEqual`；完整断言不删 |
| B159 [TlsRagStep2Tests.test_one_projected_scan_prefix_reuse_cached_original_rerank_and_context](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step2.py#L114) | 保留 | 日常＋完整，方法原样 | 调用：`run_phase_a`；原10处断言位置：`self.assertEqual`、`self.assertTrue`、`np.testing.assert_allclose`；完整断言不删 |
| B160 [TlsRagStep2Tests.test_fixed_controller_has_exactly_two_actions_and_never_skips_grid](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step2.py#L164) | 保留 | 日常＋完整，方法原样 | 原7处断言位置：`self.assertEqual`、`self.assertIsNone`、`self.assertIs`；完整断言不删 |
| B161 [TlsRagStep2Tests.test_controller_schema_and_recursive_forbidden_field_rejection](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step2.py#L191) | 保留 | 日常＋完整，方法原样 | 调用：`assert_deployable_only`；原5处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertRaises`等；完整断言不删 |
| B162 [TlsRagStep2Tests.test_phase_a_is_immutable_closed_before_labels_and_join_preserves_fingerprint](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step2.py#L213) | 保留 | 日常＋完整，方法原样 | 调用：`run_step2`；原7处断言位置：`self.assertEqual`、`self.assertRaises`、`self.assertTrue`等；完整断言不删 |
| B163 [TlsRagStep2Tests.test_hand_computed_evidence_coverage_sufficiency_and_contradiction](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step2.py#L238) | 保留 | 日常＋完整，方法原样 | 调用：`EvidencePlan`、`FacetSlot`、`EvidenceLabelStore`；原8处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertTrue`；完整断言不删 |
| B164 [TlsRagStep2Tests.test_candidate_context_and_later_gain_counterexamples_are_distinct](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step2.py#L270) | 保留 | 日常＋完整，方法原样 | 原11处断言位置：`self.assertTrue`、`self.assertFalse`、`self.assertNotEqual`；完整断言不删 |
| B165 [TlsRagStep2Tests.test_degenerate_terminal_and_nonattainment_cases_are_explicit](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step2.py#L291) | 保留 | 日常＋完整，方法原样 | 原10处断言位置：`self.assertTrue`、`self.assertEqual`、`self.assertIn`等；完整断言不删 |
| B166 [TlsRagStep2Tests.test_observed_pair_beta_rho_preserve_angle_and_exact_tri_law_api](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step2.py#L309) | 保留 | 日常＋完整，方法原样 | 调用：`same_distance_different_angle_fixture`、`tri_law_probability`；原9处断言位置：`self.assertAlmostEqual`、`self.assertGreater`、`self.assertNotAlmostEqual`等；完整断言不删 |
| B167 [TlsRagStep2Tests.test_separate_complete_work_and_timing_counters](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step2.py#L343) | 保留 | 日常＋完整，方法原样 | 原13处断言位置：`self.assertEqual`、`self.assertTrue`；完整断言不删 |
| B168 [TlsRagStep2Tests.test_two_run_portable_artifacts_are_identical_and_aggregates_reconstruct](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step2.py#L369) | 保留 | 日常＋完整，方法原样 | 调用：`run_step2`；原10处断言位置：`self.assertEqual`、`self.assertNotIn`、`self.assertAlmostEqual`等；完整断言不删 |
| B169 [TlsRagStep2Tests.test_runner_scope_is_offline_exact_synthetic_and_dependency_free](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step2.py#L417) | 保留 | 日常＋完整，方法原样 | 原3处断言位置：`self.assertNotIn`、`self.assertEqual`、`self.assertFalse`；完整断言不删 |

### test_tls_rag_step3.py（23项）

覆盖契约：Step5 仍直接复用的 observed-pair 风险/失真特征及历史状态契约。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B170 [TlsRagStep3Tests.test_step2_frozen_schedule_and_semantic_compatibility_are_unchanged](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L132) | 保留 | 日常＋完整，方法原样 | 调用：`run_step2_phase_a`、`FixedScheduleController`、`join_phase_b`；原7处断言位置：`self.assertEqual`、`self.assertTrue`；完整断言不删 |
| B171 [TlsRagStep3Tests.test_step2_compatibility_fallback_covers_both_phases_and_rejects_drift](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L167) | 保留 | 日常＋完整，方法原样 | 调用：`run_step2_phase_a`、`FixedScheduleController`、`join_phase_b`；原6处断言位置：`self.assertFalse`、`self.assertTrue`、`self.assertRaisesRegex`；完整断言不删 |
| B172 [TlsRagStep3Tests.test_step2_semantic_fingerprint_tolerates_only_frozen_float_lattice](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L220) | 保留 | 日常＋完整，方法原样 | 调用：`run_step2_phase_a`、`FixedScheduleController`、`step2_semantic_trajectory_fingerprint`；原2处断言位置：`self.assertEqual`、`self.assertNotEqual`；完整断言不删 |
| B173 [TlsRagStep3Tests.test_step2_semantic_supervision_rebinds_phase_a_and_rejects_label_change](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L248) | 保留 | 日常＋完整，方法原样 | 调用：`run_step2_phase_a`、`FixedScheduleController`、`join_phase_b`；原2处断言位置：`self.assertEqual`、`self.assertNotEqual`；完整断言不删 |
| B174 [TlsRagStep3Tests.test_hand_computed_beta_rho_and_unchanged_tri_law_value](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L284) | 保留 | 日常＋完整，方法原样 | 调用：`tri_law_probability`；原5处断言位置：`self.assertEqual`、`self.assertAlmostEqual`；完整断言不删 |
| B175 [TlsRagStep3Tests.test_pair_builder_is_current_prefix_only_and_refuses_unseen_candidates](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L294) | 保留 | 日常＋完整，方法原样 | 原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B176 [TlsRagStep3Tests.test_boundary_frontier_summaries_quantiles_and_previous_deltas](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L305) | 保留 | 日常＋完整，方法原样 | 原9处断言位置：`self.assertEqual`、`self.assertAlmostEqual`、`self.assertTrue`等；完整断言不删 |
| B177 [TlsRagStep3Tests.test_tied_duplicate_zero_nonfinite_collinear_and_empty_counts](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L326) | 保留 | 日常＋完整，方法原样 | 调用：`build_observed_pair_risk_profile`；原9处断言位置：`self.assertEqual`、`self.assertGreater`、`self.assertFalse`；完整断言不删 |
| B178 [TlsRagStep3Tests.test_same_distances_different_angles_change_profile](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L358) | 保留 | 日常＋完整，方法原样 | 调用：`same_distance_different_angle_fixture`、`tri_law_probability`；原3处断言位置：`self.assertAlmostEqual`、`self.assertNotAlmostEqual`；完整断言不删 |
| B179 [TlsRagStep3Tests.test_realized_projection_changes_distortion_not_ex_ante_pair_probability](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L376) | 保留 | 日常＋完整，方法原样 | 原2处断言位置：`self.assertEqual`、`self.assertNotEqual`；完整断言不删 |
| B180 [TlsRagStep3Tests.test_rows_two_to_four_have_exact_one_factor_feature_differences](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L386) | 保留 | 日常＋完整，方法原样 | 调用：`feature_names_for_row`；原5处断言位置：`self.assertEqual`；完整断言不删 |
| B181 [TlsRagStep3Tests.test_synthetic_partitions_are_stable_disjoint_and_not_real_roles](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L397) | 保留 | 日常＋完整，方法原样 | 调用：`stable_synthetic_partition`；原5处断言位置：`self.assertTrue`、`self.assertFalse`；完整断言不删 |
| B182 [TlsRagStep3Tests.test_model_fit_labels_do_not_reach_bound_or_inference_objects](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L409) | 保留 | 日常＋完整，方法原样 | 调用：`CalibratedStep3Controller`；原8处断言位置：`self.assertTrue`、`self.assertGreaterEqual`、`self.assertEqual`等；完整断言不删 |
| B183 [TlsRagStep3Tests.test_clopper_pearson_hand_values_and_family_wise_alpha_allocation](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L436) | 保留 | 日常＋完整，方法原样 | 调用：`clopper_pearson_one_sided`；原6处断言位置：`self.assertEqual`、`self.assertAlmostEqual`；完整断言不删 |
| B184 [TlsRagStep3Tests.test_candidate_stage_bin_reachability_uses_unique_query_ids_and_reconstructs](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L452) | 保留 | 日常＋完整，方法原样 | 原5处断言位置：`self.assertEqual`、`self.assertTrue`；完整断言不删 |
| B185 [TlsRagStep3Tests.test_empty_or_underpowered_cells_are_vacuous_and_prohibit_stop](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L472) | 保留 | 日常＋完整，方法原样 | 调用：`make_deployable_state`；原4处断言位置：`self.assertTrue`、`self.assertIs`、`self.assertFalse`；完整断言不删 |
| B186 [TlsRagStep3Tests.test_dual_bounds_and_every_validity_flag_are_required_for_stop](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L484) | 保留 | 日常＋完整，方法原样 | 调用：`make_deployable_state`；原5处断言位置：`self.assertTrue`、`self.assertIs`；完整断言不删 |
| B187 [TlsRagStep3Tests.test_all_invalid_states_expand_next_only_and_terminate_explicitly](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L498) | 保留 | 日常＋完整，方法原样 | 调用：`CalibratedStep3Controller`、`make_deployable_state`；原4处断言位置：`self.assertIs`、`self.assertEqual`、`self.assertIn`；完整断言不删 |
| B188 [TlsRagStep3Tests.test_recursive_forbidden_fields_and_no_label_store_reachability](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L520) | 保留 | 日常＋完整，方法原样 | 调用：`make_deployable_state`、`assert_deployable_only`；原2处断言位置：`self.assertNotIn`、`self.assertRaises`；完整断言不删 |
| B189 [TlsRagStep3Tests.test_phase_a_serializes_before_evaluation_join_and_fingerprint_survives](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L533) | 保留 | 日常＋完整，方法原样 | 调用：`run_step3`；原4处断言位置：`self.assertTrue`、`self.assertGreater`；完整断言不删 |
| B190 [TlsRagStep3Tests.test_complete_separate_work_timing_and_aggregate_reconstruction](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L553) | 保留 | 日常＋完整，方法原样 | 原8处断言位置：`self.assertTrue`、`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B191 [TlsRagStep3Tests.test_two_runs_have_byte_identical_portable_artifacts_and_denied_claims](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L571) | 保留 | 日常＋完整，方法原样 | 原4处断言位置：`self.assertEqual`、`self.assertNotIn`、`self.assertIn`；完整断言不删 |
| B192 [TlsRagStep3Tests.test_source_and_manifest_prohibit_network_real_approximate_gpu_llm_answer_and_step4](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step3.py#L585) | 保留 | 日常＋完整，方法原样 | 原3处断言位置：`self.assertNotIn`、`self.assertTrue`、`self.assertEqual`；完整断言不删 |

### test_tls_rag_step4.py（26项）

覆盖契约：历史 Step4 readiness 冻结协议、访问/收据链、控制器分支与合成复现。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B193 [TlsRagStep4Tests.test_frozen_protocol_rejects_every_top_level_change](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L45) | 退役（日常） | 历史＋完整，方法原样 | 调用：`digest`、`parse_protocol`；原3处断言位置：`self.assertEqual`、`self.assertRaisesRegex`、`self.assertRaises`；完整断言不删 |
| B194 [TlsRagStep4Tests.test_duplicate_keys_nonfinite_and_type_substitution_are_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L58) | 退役（日常） | 历史＋完整，方法原样 | 调用：`parse_protocol`；原2处断言位置：`self.assertRaises`；完整断言不删 |
| B195 [TlsRagStep4Tests.test_protocol_storage_cannot_be_changed_through_returned_mapping](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L67) | 退役（日常） | 历史＋完整，方法原样 | 调用：`build_synthetic_fixture`、`Protocol`、`canonical`；原3处断言位置：`self.assertEqual`、`self.assertRaises`；完整断言不删 |
| B196 [TlsRagStep4Tests.test_wrong_pin_rejects_before_any_path_read_or_fixture_generation](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L75) | 退役（日常） | 历史＋完整，方法原样 | 调用：`load_protocol`、`run_readiness`；原5处断言位置：`self.assertRaisesRegex`、`read.assert_not_called`、`self.assertRaises`等；完整断言不删 |
| B197 [TlsRagStep4Tests.test_bad_upstream_identity_rejects_before_fixture_or_output](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L88) | 退役（日常） | 历史＋完整，方法原样 | 调用：`run_readiness`；原3处断言位置：`self.assertRaisesRegex`、`builder.assert_not_called`、`self.assertFalse`；完整断言不删 |
| B198 [TlsRagStep4Tests.test_real_protected_and_arbitrary_path_requests_fail_without_io](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L99) | 退役（日常） | 历史＋完整，方法原样 | 调用：`run_readiness`；原3处断言位置：`self.assertRaises`、`opened.assert_not_called`、`inspected.assert_not_called`；完整断言不删 |
| B199 [TlsRagStep4Tests.test_cli_has_no_real_data_or_config_override_switch](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L114) | 退役（日常） | 历史＋完整，方法原样 | 调用：`main`；原2处断言位置：`self.assertRaises`、`self.assertEqual`；完整断言不删 |
| B200 [TlsRagStep4Tests.test_synthetic_roles_are_complete_disjoint_and_external](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L122) | 退役（日常） | 历史＋完整，方法原样 | 调用：`validate_fixture`、`build_synthetic_fixture`；原5处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B201 [TlsRagStep4Tests.test_duplicate_query_and_family_rejected_within_and_across_roles](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L132) | 退役（日常） | 历史＋完整，方法原样 | 调用：`validate_fixture`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B202 [TlsRagStep4Tests.test_unknown_role_nonstring_ids_missing_queries_and_corpus_overlap_reject](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L141) | 退役（日常） | 历史＋完整，方法原样 | 调用：`validate_fixture`；原2处断言位置：`self.assertRaises`；完整断言不删 |
| B203 [TlsRagStep4Tests.test_role_order_prior_receipt_and_pin_fail_closed](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L156) | 退役（日常） | 历史＋完整，方法原样 | 原3处断言位置：`self.assertRaises`、`self.assertTrue`、`self.assertRaisesRegex`；完整断言不删 |
| B204 [TlsRagStep4Tests.test_labels_cannot_open_before_all_role_decisions_close](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L167) | 退役（日常） | 历史＋完整，方法原样 | 原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B205 [TlsRagStep4Tests.test_phase_a_rejects_labels_extra_payload_and_grid_growth](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L180) | 退役（日常） | 历史＋完整，方法原样 | 原2处断言位置：`self.assertRaises`、`self.assertTrue`；完整断言不删 |
| B206 [TlsRagStep4Tests.test_phase_a_receipt_binds_records_and_survives_label_opening](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L189) | 退役（日常） | 历史＋完整，方法原样 | 调用：`digest`；原5处断言位置：`self.assertEqual`、`self.assertRaisesRegex`；完整断言不删 |
| B207 [TlsRagStep4Tests.test_latency_never_opens_labels_even_with_valid_receipt](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L208) | 退役（日常） | 历史＋完整，方法原样 | 原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B208 [TlsRagStep4Tests.test_failed_tune_or_cert_is_terminal_without_later_roles_or_retry](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L218) | 退役（日常） | 历史＋完整，方法原样 | 调用：`ROLES.index`；原3处断言位置：`self.assertRaisesRegex`、`self.assertTrue`；完整断言不删 |
| B209 [TlsRagStep4Tests.test_simulated_receipt_chain_and_all_phase_orders_reconstruct](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L231) | 退役（日常） | 历史＋完整，方法原样 | 调用：`simulate_roles`、`digest`；原5处断言位置：`self.assertEqual`；完整断言不删 |
| B210 [TlsRagStep4Tests.test_state_guard_rejects_nested_labels_unknown_features_and_subclasses](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L251) | 退役（日常） | 历史＋完整，方法原样 | 调用：`mock_state`、`guard_deployable_state`；原3处断言位置：`self.assertRaises`；完整断言不删 |
| B211 [TlsRagStep4Tests.test_state_guard_rejects_malformed_grid_ids_flags_and_feature_identity](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L265) | 退役（日常） | 历史＋完整，方法原样 | 调用：`mock_state`、`guard_deployable_state`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B212 [TlsRagStep4Tests.test_input_guard_does_not_change_upstream_decisions](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L280) | 退役（日常） | 历史＋完整，方法原样 | 调用：`mock_state`、`mock_controller`、`canonical`；原2处断言位置：`self.assertEqual`；完整断言不删 |
| B213 [TlsRagStep4Tests.test_nonfinite_features_are_rejected_without_imputation_or_controller_call](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L290) | 退役（日常） | 历史＋完整，方法原样 | 调用：`mock_controller`、`mock_state`、`guard_deployable_state`；原2处断言位置：`self.assertRaisesRegex`、`choose.assert_not_called`；完整断言不删 |
| B214 [TlsRagStep4Tests.test_all_controller_branches_rows_and_terminal_failures](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L302) | 退役（日常） | 历史＋完整，方法原样 | 调用：`synthetic_controller_probes`、`feature_names_for_row`；原11处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertTrue`等；完整断言不删 |
| B215 [TlsRagStep4Tests.test_internal_and_formal_alpha_budgets_and_claims_are_separate](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L320) | 退役（日常） | 历史＋完整，方法原样 | 原9处断言位置：`self.assertEqual`、`self.assertAlmostEqual`、`self.assertFalse`等；完整断言不删 |
| B216 [TlsRagStep4Tests.test_existing_output_not_enumerated_or_overwritten](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L334) | 退役（日常） | 历史＋完整，方法原样 | 调用：`run_readiness`；原2处断言位置：`self.assertRaises`、`self.assertEqual`；完整断言不删 |
| B217 [TlsRagStep4Tests.test_symlink_output_is_rejected_without_writing_target](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L344) | 退役（日常） | 历史＋完整，方法原样 | 调用：`run_readiness`；原2处断言位置：`self.assertRaisesRegex`、`self.assertFalse`；完整断言不删 |
| B218 [TlsRagStep4Tests.test_offline_two_runs_are_byte_identical_and_all_artifacts_rehash](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4.py#L355) | 退役（日常） | 历史＋完整，方法原样 | 调用：`run_readiness`、`digest`；原8处断言位置：`self.assertEqual`、`self.assertIn`、`self.assertFalse`；完整断言不删 |

### test_tls_rag_step4_a800_replay.py（17项）

覆盖契约：历史 A800 回放的来源/镜像/载荷校验、闭合及复现。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B219 [A800ReplayTests.test_valid_new_binding_and_old_loader_rejection](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L61) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.validate_rebuilt_bundle`、`v2.load_bundle`；原3处断言位置：`self.assertEqual`、`self.assertRaisesRegex`；完整断言不删 |
| B220 [A800ReplayTests.test_reject_old_schema_or_false_independence](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L68) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.validate_rebuilt_bundle`；原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B221 [A800ReplayTests.test_tampering_and_source_changes](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L74) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.validate_rebuilt_bundle`；原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B222 [A800ReplayTests.test_symlink_payload_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L80) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.validate_rebuilt_bundle`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B223 [A800ReplayTests.test_root_mount_link_is_supported](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L85) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.validate_rebuilt_bundle`；成功路径要求不抛异常；原调用保留 |
| B224 [A800ReplayTests.test_complete_replay_closure_and_no_independence_claim](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L89) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.guard.check_closed`、`a.run_replay`、`v2.sha256`；原9处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertFalse`等；完整断言不删 |
| B225 [A800ReplayTests.test_failed_tune_does_not_open_probe](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L108) | 退役（日常） | 历史＋完整，方法原样 | 调用：`v2.fingerprint`；原2处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B226 [A800ReplayTests.test_existing_output_and_nested_output_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L115) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.run_replay`；原2处断言位置：`self.assertRaises`、`self.assertRaisesRegex`；完整断言不删 |
| B227 [A800ReplayTests.test_interruption_preserves_attempt](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L121) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.run_replay`；原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B228 [A800ReplayTests.test_script_explicit_exit_handling](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L127) | 退役（日常） | 历史＋完整，方法原样 | 原2处断言位置：`self.assertNotIn`、`self.assertEqual`；完整断言不删 |
| B229 [A800ReplayTests.test_preparation_uses_cpu_pinned_model_and_new_schema](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L133) | 退役（日常） | 历史＋完整，方法原样 | 调用：`v2.sha256`、`a.prepare_rebuilt_bundle`、`a.transport_record('https://hf-mirror.com', self.data['embedding'])['source_urls'].values`；原10处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertFalse`等；完整断言不删 |
| B230 [A800ReplayTests.test_endpoint_allowlist_and_mapping](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L180) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.transport_record`、`a.adapter.SOURCES.items`、`a.declared_transport`；原3处断言位置：`self.assertEqual`、`self.assertRaises`、`self.assertRaisesRegex`；完整断言不删 |
| B231 [A800ReplayTests.test_transport_tampering_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L192) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.transport_record`、`a.validate_rebuilt_bundle`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B232 [A800ReplayTests.test_raw_source_and_model_hashes_checked](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L198) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.validate_rebuilt_bundle`；原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B233 [A800ReplayTests.test_model_manifest_accepts_hidden_and_nested_files](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L207) | 退役（日常） | 历史＋完整，方法原样 | 调用：`v2.sha256`、`a.validate_rebuilt_bundle`、`a.checked_model_manifest`；原1处断言位置：`self.assertEqual`；完整断言不删 |
| B234 [A800ReplayTests.test_model_manifest_rejects_unsafe_paths_digests_and_symlinks](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L216) | 退役（日常） | 历史＋完整，方法原样 | 调用：`v2.sha256`、`a.checked_model_manifest`；原4处断言位置：`self.assertRaisesRegex`、`self.assertRaises`；完整断言不删 |
| B235 [A800ReplayTests.test_config_is_frozen](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_a800_replay.py#L230) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.load_protocol`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |

### test_tls_rag_step4_probe.py（15项）

覆盖契约：历史 NFCorpus probe 的分组、标签闸门、源/缓存和失败闭合。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B236 [ProbeTests.test_protocol_is_pinned_and_component_behavior_preserved](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L34) | 退役（日常） | 历史＋完整，方法原样 | 调用：`p.load_protocol`、`p.component_config`；原4处断言位置：`self.assertEqual`、`self.assertRaisesRegex`；完整断言不删 |
| B237 [ProbeTests.test_duplicate_texts_stay_in_one_family_and_order_independent](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L47) | 退役（日常） | 历史＋完整，方法原样 | 调用：`p.assign_roles`；原4处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B238 [ProbeTests.test_insufficient_units_fail_without_reusing_queries](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L59) | 退役（日常） | 历史＋完整，方法原样 | 调用：`p.assign_roles`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B239 [ProbeTests.test_qrel_staging_keeps_zero_grade_query_and_rejects_unknown_doc](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L63) | 退役（日常） | 历史＋完整，方法原样 | 调用：`adapter.partition_qrels`；原2处断言位置：`self.assertEqual`、`self.assertRaisesRegex`；完整断言不删 |
| B240 [ProbeTests.test_qrel_parser_validates_schema_duplicates_and_grades](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L71) | 退役（日常） | 历史＋完整，方法原样 | 调用：`adapter.read_train_qrels`；原3处断言位置：`self.assertEqual`、`self.assertRaisesRegex`；完整断言不删 |
| B241 [ProbeTests.test_source_allowlist_excludes_official_dev_test_labels](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L82) | 退役（日常） | 历史＋完整，方法原样 | 调用：`adapter.SOURCES.values`；原3处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertFalse`；完整断言不删 |
| B242 [ProbeTests.test_projection_is_fixed_gaussian_and_not_renormalized](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L87) | 退役（日常） | 历史＋完整，方法原样 | 原3处断言位置：`np.testing.assert_array_equal`、`self.assertFalse`；完整断言不删 |
| B243 [ProbeTests.test_bad_embeddings_and_overlapping_roles_fail](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L94) | 退役（日常） | 历史＋完整，方法原样 | 调用：`p.make_environment`；原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B244 [ProbeTests.test_pipeline_closes_every_role_before_labels_and_reproduces](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L107) | 退役（日常） | 历史＋完整，方法原样 | 调用：`p.make_environment`、`p.s3.ScorePrediction`、`p.run_probe`；原14处断言位置：`self.assertTrue`、`self.assertIsNotNone`、`self.assertEqual`等；完整断言不删 |
| B245 [ProbeTests.test_four_plus_four_underpowered_bins_keep_probe_closed](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L173) | 退役（日常） | 历史＋完整，方法原样 | 调用：`p.s3.ScorePrediction`、`p.run_probe`；原7处断言位置：`self.assertEqual`、`self.assertIsNone`、`self.assertFalse`等；完整断言不删 |
| B246 [ProbeTests.test_no_eligible_tune_candidate_keeps_probe_unopened](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L205) | 退役（日常） | 历史＋完整，方法原样 | 调用：`p.run_probe`；原3处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B247 [ProbeTests.test_proxy_gain_depends_on_future_context_not_candidate_coverage](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L219) | 退役（日常） | 历史＋完整，方法原样 | 调用：`p.supervision`；原2处断言位置：`self.assertEqual`；完整断言不删 |
| B248 [ProbeTests.test_bundle_identity_and_late_label_tampering_are_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L227) | 退役（日常） | 历史＋完整，方法原样 | 调用：`p.save`、`p.save_rows`、`p.fingerprint`；原4处断言位置：`self.assertEqual`、`self.assertRaisesRegex`；完整断言不删 |
| B249 [ProbeTests.test_unknown_bundle_paths_rejected_before_access](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L257) | 退役（日常） | 历史＋完整，方法原样 | 调用：`p.load_protocol`、`p.save`、`p.fingerprint`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B250 [ProbeTests.test_preexisting_output_is_never_overwritten](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_probe.py#L268) | 退役（日常） | 历史＋完整，方法原样 | 调用：`p.run_probe`；原1处断言位置：`self.assertRaises`；完整断言不删 |

### test_tls_rag_step4_repaired_acceptance.py（19项）

覆盖契约：历史 reviewed-source 兼容入口、旧哈希、重建与禁止重复 probe。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B251 [RepairedAcceptanceTests.test_pinned_historical_hashes_match_git_reference](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L72) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.HISTORICAL_CODE_SHA256.items`、`v2.sha256`；原2处断言位置：`self.assertEqual`、`self.assertNotEqual`；完整断言不删 |
| B252 [RepairedAcceptanceTests.test_old_loader_still_rejects_but_explicit_historical_validation_accepts](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L79) | 退役（日常） | 历史＋完整，方法原样 | 调用：`v2.load_bundle`、`a.load_environment`；原3处断言位置：`self.assertRaisesRegex`、`self.assertEqual`；完整断言不删 |
| B253 [RepairedAcceptanceTests.test_mixed_or_current_source_hashes_are_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L86) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.read_json`、`v2.sha256`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B254 [RepairedAcceptanceTests.test_bundle_payload_and_old_feature_tampering_are_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L95) | 退役（日常） | 历史＋完整，方法原样 | 原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B255 [RepairedAcceptanceTests.test_model_metadata_and_evaluation_identity_mismatch_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L104) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.read_json`；原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B256 [RepairedAcceptanceTests.test_historical_status_and_selection_dependencies_are_verified](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L119) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.read_json`；原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B257 [RepairedAcceptanceTests.test_changed_source_or_closure_stops_before_label_read](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L137) | 退役（日常） | 历史＋完整，方法原样 | 原4处断言位置：`self.assertRaisesRegex`、`self.assertFalse`；完整断言不删 |
| B258 [RepairedAcceptanceTests.test_unsafe_manifests_and_symlinks_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L155) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.checked_manifest`；原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B259 [RepairedAcceptanceTests.test_duplicate_json_keys_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L167) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.read_json`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B260 [RepairedAcceptanceTests.test_any_historical_probe_artifact_or_manifest_entry_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L173) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.read_json`；原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B261 [RepairedAcceptanceTests.test_late_input_changes_and_probe_opening_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L186) | 退役（日常） | 历史＋完整，方法原样 | 原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B262 [RepairedAcceptanceTests.test_rebuild_never_copies_old_numerical_features](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L196) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.load_environment`、`a.rebuild_features`、`v3.parse_features`；原3处断言位置：`self.assertEqual`、`self.assertNotEqual`；完整断言不删 |
| B263 [RepairedAcceptanceTests.test_pipeline_rebuilds_all_roles_and_enforces_selection_and_closure](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L213) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.check_closed`、`v2.sha256`、`a.read_json`；原14处断言位置：`self.assertIsNotNone`、`self.assertTrue`、`self.assertEqual`等；完整断言不删 |
| B264 [RepairedAcceptanceTests.test_failing_tune_keeps_probe_unprepared_and_unreserved](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L248) | 退役（日常） | 历史＋完整，方法原样 | 调用：`v3.load_protocol`、`v2.fingerprint`；原4处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B265 [RepairedAcceptanceTests.test_crash_after_probe_reservation_blocks_retry](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L258) | 退役（日常） | 历史＋完整，方法原样 | 原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B266 [RepairedAcceptanceTests.test_concurrent_probe_reservation_blocks_before_prepare](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L269) | 退役（日常） | 历史＋完整，方法原样 | 原2处断言位置：`self.assertRaisesRegex`、`self.assertFalse`；完整断言不删 |
| B267 [RepairedAcceptanceTests.test_reproducible_scientific_artifacts_and_cli](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L280) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.main`；原2处断言位置：`self.assertEqual`；完整断言不删 |
| B268 [RepairedAcceptanceTests.test_existing_or_nested_output_refused_without_mutation](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L292) | 退役（日常） | 历史＋完整，方法原样 | 调用：`a.run_acceptance`；原3处断言位置：`self.assertRaises`、`self.assertRaisesRegex`、`self.assertFalse`；完整断言不删 |
| B269 [RepairedAcceptanceTests.test_frozen_acceptance_and_policy_grid_are_unchanged](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_repaired_acceptance.py#L301) | 退役（日常） | 历史＋完整，方法原样 | 调用：`v3.load_protocol`、`v3.policy_grid`、`v3.gates(row, protocol).values`；原4处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertFalse`；完整断言不删 |

### test_tls_rag_step4_retention_v3.py（12项）

覆盖契约：历史 retention-v3 连续目标、校准/网格、父缓存与 probe 闭合。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B270 [RetentionV3Tests.test_protocol_has_27_candidates_and_unchanged_acceptance](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_retention_v3.py#L54) | 退役（日常） | 历史＋完整，方法原样 | 调用：`r.load_protocol`、`r.policy_grid`；原4处断言位置：`self.assertEqual`、`self.assertRaisesRegex`；完整断言不删 |
| B271 [RetentionV3Tests.test_continuous_retention_targets_are_not_boolean_hit_labels](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_retention_v3.py#L70) | 退役（日常） | 历史＋完整，方法原样 | 调用：`r.retention_targets`、`r.fit_models`；原3处断言位置：`self.assertTrue`、`self.assertAlmostEqual`、`self.assertEqual`；完整断言不删 |
| B272 [RetentionV3Tests.test_low_retention_score_expands_even_with_valid_context](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_retention_v3.py#L81) | 退役（日常） | 历史＋完整，方法原样 | 调用：`r.Controller`、`r.Policy`；原2处断言位置：`self.assertEqual`；完整断言不删 |
| B273 [RetentionV3Tests.test_stage_adaptation_margin_and_explicit_terminal_failure](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_retention_v3.py#L92) | 退役（日常） | 历史＋完整，方法原样 | 调用：`r.Controller`、`r.Policy`；原4处断言位置：`self.assertEqual`；完整断言不删 |
| B274 [RetentionV3Tests.test_no_calibration_and_forbidden_or_nonfinite_inputs](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_retention_v3.py#L106) | 退役（日常） | 历史＋完整，方法原样 | 调用：`r.Controller`、`r.Policy`；原4处断言位置：`self.assertEqual`、`self.assertRaises`；完整断言不删 |
| B275 [RetentionV3Tests.test_quantile_margin_and_insufficient_calibration](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_retention_v3.py#L119) | 退役（日常） | 历史＋完整，方法原样 | 调用：`r.calibrate`；原3处断言位置：`self.assertAlmostEqual`、`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B276 [RetentionV3Tests.test_duplicate_missing_or_extra_cache_states_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_retention_v3.py#L130) | 退役（日常） | 历史＋完整，方法原样 | 调用：`r.parse_features`；原2处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B277 [RetentionV3Tests.test_unchanged_gates_reject_v2_collapse_and_no_saving](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_retention_v3.py#L141) | 退役（日常） | 历史＋完整，方法原样 | 调用：`r.load_protocol`、`r.gates`；原2处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B278 [RetentionV3Tests.test_complete_pipeline_is_reproducible_and_closes_probe_before_labels](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_retention_v3.py#L147) | 退役（日常） | 历史＋完整，方法原样 | 调用：`r.run`；原8处断言位置：`self.assertTrue`、`self.assertIsNotNone`、`self.assertEqual`等；完整断言不删 |
| B279 [RetentionV3Tests.test_no_tune_candidate_keeps_probe_features_and_labels_closed](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_retention_v3.py#L169) | 退役（日常） | 历史＋完整，方法原样 | 调用：`r.run`；原3处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B280 [RetentionV3Tests.test_parent_binding_cache_reuse_and_opened_probe_rejection](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_retention_v3.py#L185) | 退役（日常） | 历史＋完整，方法原样 | 调用：`v2.run_probe`、`r.verify_parent`、`v2.sha256`；原6处断言位置：`self.assertEqual`、`self.assertRaisesRegex`；完整断言不删 |
| B281 [RetentionV3Tests.test_existing_output_refused](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step4_retention_v3.py#L217) | 退役（日常） | 历史＋完整，方法原样 | 调用：`r.run`；原1处断言位置：`self.assertRaises`；完整断言不删 |

### test_tls_rag_step5_ann.py（30项）

覆盖契约：Step5 公共 ANN/候选缓存/角色/校准/计时入口及真实 HNSW 合成接口。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B282 [Step5AnnTests.test_stable_boundary_uses_string_ids](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L38) | 保留 | 日常＋完整，方法原样 | 调用：`stable_top`；原1处断言位置：`self.assertEqual`；完整断言不删 |
| B283 [Step5AnnTests.test_exact_reference_across_blocks_and_ties](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L41) | 保留 | 日常＋完整，方法原样 | 调用：`ExactCandidateIndex`；原1处断言位置：`self.assertEqual`；完整断言不删 |
| B284 [Step5AnnTests.test_exact_matches_direct_bruteforce](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L46) | 保留 | 日常＋完整，方法原样 | 调用：`ExactCandidateIndex`；原1处断言位置：`self.assertEqual`；完整断言不删 |
| B285 [Step5AnnTests.test_exact_rejects_invalid_ids_query_and_k](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L54) | 保留 | 日常＋完整，方法原样 | 调用：`ExactCandidateIndex`；原3处断言位置：`self.assertRaises`；完整断言不删 |
| B286 [Step5AnnTests.test_non_nested_expansion_keeps_previous_members](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L64) | 保留 | 日常＋完整，方法原样 | 调用：`expand_candidate_pool`；原1处断言位置：`self.assertEqual`；完整断言不删 |
| B287 [Step5AnnTests.test_missing_or_duplicate_ann_result_is_failure](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L69) | 保留 | 日常＋完整，方法原样 | 调用：`expand_candidate_pool`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B288 [Step5AnnTests.test_original_scores_reused_across_restarted_search](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L74) | 保留 | 日常＋完整，方法原样 | 调用：`CandidateSession`；原4处断言位置：`self.assertEqual`、`self.assertTrue`；完整断言不删 |
| B289 [Step5AnnTests.test_feature_reads_only_current_pool_and_is_bounded](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L90) | 保留 | 日常＋完整，方法原样 | 调用：`normalized`、`ExactCandidateIndex`、`CandidateSession`；原4处断言位置：`self.assertEqual`、`self.assertLessEqual`、`self.assertTrue`；完整断言不删 |
| B290 [Step5AnnTests.test_controller_rejects_label_dictionary](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L106) | 保留 | 日常＋完整，方法原样 | 调用：`choose`、`policy_grid`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B291 [Step5AnnTests.test_uncalibrated_expands_then_terminal_stop](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L110) | 保留 | 日常＋完整，方法原样 | 调用：`policy_grid`、`choose`；原3处断言位置：`self.assertEqual`；完整断言不删 |
| B292 [Step5AnnTests.test_controller_rejects_grid_and_policy_drift](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L117) | 保留 | 日常＋完整，方法原样 | 调用：`policy_grid`、`choose`；原2处断言位置：`self.assertRaises`；完整断言不删 |
| B293 [Step5AnnTests.test_new_fit_and_independent_residual_margin](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L124) | 保留 | 日常＋完整，方法原样 | 调用：`fit_models`、`calibrate`；原4处断言位置：`self.assertIsNotNone`、`self.assertGreater`、`self.assertIsNone`等；完整断言不删 |
| B294 [Step5AnnTests.test_text_and_lexical_family_canonicalization](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L134) | 保留 | 日常＋完整，方法原样 | 调用：`normalize_text`、`family_key`；原2处断言位置：`self.assertEqual`；完整断言不删 |
| B295 [Step5AnnTests.test_hash_sample_reproducible_and_order_independent](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L138) | 保留 | 日常＋完整，方法原样 | 调用：`sample_corpus`；原2处断言位置：`self.assertEqual`；完整断言不删 |
| B296 [Step5AnnTests.test_role_family_history_corpus_exclusion](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L148) | 保留 | 日常＋完整，方法原样 | 调用：`text_hash`、`freeze_roles`、`family_key`；原4处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B297 [Step5AnnTests.test_insufficient_families_stops_before_encoding](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L162) | 保留 | 日常＋完整，方法原样 | 调用：`freeze_roles`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B298 [Step5AnnTests.test_artifact_tampering_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L166) | 保留 | 日常＋完整，方法原样 | 调用：`seal`、`verify_seal`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B299 [Step5AnnTests.test_future_role_and_early_labels_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L175) | 保留 | 日常＋完整，方法原样 | 调用：`RunLedger`；原3处断言位置：`self.assertRaises`；完整断言不删 |
| B300 [Step5AnnTests.test_crash_reservation_cannot_be_reused](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L185) | 保留 | 日常＋完整，方法原样 | 调用：`RunLedger`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B301 [Step5AnnTests.test_pass_cannot_bypass_decision_and_label_closure](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L191) | 保留 | 日常＋完整，方法原样 | 调用：`RunLedger`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B302 [Step5AnnTests.test_closed_decisions_are_immutable](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L197) | 保留 | 日常＋完整，方法原样 | 调用：`RunLedger`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B303 [Step5AnnTests.test_failed_or_inconclusive_stage_locks_successors](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L206) | 保留 | 日常＋完整，方法原样 | 调用：`RunLedger`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B304 [Step5AnnTests.test_protocol_binding_rejects_changed_budget](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L214) | 保留 | 日常＋完整，方法原样 | 调用：`RunLedger`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B305 [Step5AnnTests.test_quality_bounds_range_scaling_and_power](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L222) | 保留 | 日常＋完整，方法原样 | 调用：`quality_bounds`；原4处断言位置：`self.assertTrue`、`self.assertFalse`、`self.assertLess`等；完整断言不删 |
| B306 [Step5AnnTests.test_paired_bootstrap_preserves_query_repeat_blocks](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L231) | 保留 | 日常＋完整，方法原样 | 调用：`paired_bootstrap`；原3处断言位置：`self.assertEqual`、`self.assertAlmostEqual`、`self.assertRaises`；完整断言不删 |
| B307 [Step5AnnTests.test_self_test_real_pipeline_and_labels_order](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L240) | 保留 | 日常＋完整，方法原样 | 调用：`self_test`、`verify_seal`、`synthetic_protocol`；原8处断言位置：`self.assertFalse`、`self.assertIn`、`self.assertTrue`等；完整断言不删 |
| B308 [Step5AnnTests.test_seeded_pipeline_reproduces_scientific_records](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L260) | 保留 | 日常＋完整，方法原样 | 调用：`self_test`；原2处断言位置：`self.assertEqual`；完整断言不删 |
| B309 [Step5AnnTests.test_six_stage_pipeline_with_explicit_synthetic_gate_fixture](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L276) | 保留 | 日常＋完整，方法原样 | 调用：`normalized`、`ExactCandidateIndex`、`write_bundle`；原5处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertRaises`；完整断言不删 |
| B310 [Step5AnnTests.test_real_hnsw_end_to_end_self_test](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L312) | 保留 | 日常＋完整，方法原样 | 调用：`self_test`；原2处断言位置：`self.assertTrue`、`self.assertEqual`；完整断言不删 |
| B311 [Step5AnnTests.test_real_hnsw_deterministic_build_and_ef_rule](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann.py#L318) | 保留 | 日常＋完整，方法原样 | 调用：`normalized`、`HnswCandidateIndex`、`sha256`；原4处断言位置：`self.assertEqual`；完整断言不删 |

### test_tls_rag_step5_ann_comparators_v2.py（16项）

覆盖契约：冻结比较器 v2 的概率/数值/缓存/原生路径；测试本身参与来源绑定。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B312 [ComparatorMathematics.test_only_approved_route_and_fixed_cpu_settings](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L21) | 保留 | 日常＋完整，方法原样 | 调用：`config`；原6处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B313 [ComparatorMathematics.test_polar_boundary_is_exact_f_tail](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L26) | 保留 | 日常＋完整，方法原样 | 调用：`polarization_threshold`；原1处断言位置：`self.assertAlmostEqual`；完整断言不删 |
| B314 [ComparatorMathematics.test_normal_boundary_is_exact_conditional_tail](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L33) | 保留 | 日常＋完整，方法原样 | 调用：`conditional_normal_threshold`；原1处断言位置：`self.assertAlmostEqual`；完整断言不删 |
| B315 [ComparatorMathematics.test_fixed_pair_monte_carlo_three_rules](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L40) | 保留 | 日常＋完整，方法原样 | 调用：`config`；原1处断言位置：`self.assertLess`；完整断言不删 |
| B316 [ComparatorMathematics.test_invalid_geometry_fails_instead_of_clipping](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L54) | 保留 | 日常＋完整，方法原样 | 调用：`polarization_threshold`、`conditional_normal_threshold`；原3处断言位置：`self.assertRaises`；完整断言不删 |
| B317 [ComparatorMathematics.test_student_table_monotone_floor_is_conservative](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L62) | 保留 | 日常＋完整，方法原样 | 调用：`config`、`conditional_student_table`、`table_lookup`；原6处断言位置：`self.assertTrue`、`self.assertLessEqual`、`self.assertIsNone`等；完整断言不删 |
| B318 [ComparatorMathematics.test_unit_boundary_minimizes_over_actual_norm_range](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L72) | 保留 | 日常＋完整，方法原样 | 调用：`unit_boundary`；原2处断言位置：`self.assertLessEqual`、`self.assertRaises`；完整断言不删 |
| B319 [ComparatorMathematics.test_report_requires_frozen_selection_and_source_identity](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L81) | 保留 | 日常＋完整，方法原样 | 调用：`seal_selection`；原4处断言位置：`self.assertRaises`、`self.assertEqual`；完整断言不删 |
| B320 [ComparatorNative.test_auxiliary_geometry_normalized_before_projection_only](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L114) | 保留 | 日常＋完整，方法原样 | 原5处断言位置：`np.testing.assert_array_equal`、`np.testing.assert_allclose`、`self.assertGreater`等；完整断言不删 |
| B321 [ComparatorNative.test_cache_refuses_changed_index_and_file](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L124) | 保留 | 日常＋完整，方法原样 | 调用：`Context`；原2处断言位置：`self.assertRaises`；完整断言不删 |
| B322 [ComparatorNative.test_native_noop_bitwise_matches_actual_faiss](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L132) | 保留 | 日常＋完整，方法原样 | 原4处断言位置：`np.testing.assert_array_equal`、`self.assertEqual`、`self.assertGreater`；完整断言不删 |
| B323 [ComparatorNative.test_all_shadow_rules_preserve_path_and_compute_once](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L139) | 保留 | 日常＋完整，方法原样 | 原6处断言位置：`np.testing.assert_array_equal`、`self.assertGreater`、`self.assertEqual`；完整断言不删 |
| B324 [ComparatorNative.test_native_polar_cutoff_matches_guarded_python](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L148) | 保留 | 日常＋完整，方法原样 | 调用：`unit_boundary`、`polarization_threshold`；原1处断言位置：`self.assertAlmostEqual`；完整断言不删 |
| B325 [ComparatorNative.test_active_batch_and_progressive_account_for_actual_work](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L159) | 保留 | 日常＋完整，方法原样 | 调用：`Context`；原8处断言位置：`self.assertTrue`、`self.assertEqual`、`self.assertGreater`等；完整断言不删 |
| B326 [ComparatorNative.test_future_information_and_invalid_inputs_forbidden](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L171) | 保留 | 日常＋完整，方法原样 | 原2处断言位置：`self.assertRaises`；完整断言不删 |
| B327 [ComparatorNative.test_independent_worker_and_screen_keep_query_level_evidence](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_comparators_v2.py#L176) | 保留 | 日常＋完整，方法原样 | 调用：`save`；原4处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertGreater`；完整断言不删 |

### test_tls_rag_step5_ann_continuation_v1.py（14项）

覆盖契约：当前 v2 继承的续搜 v1 状态/生命周期/调用序列；测试参与来源绑定。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B328 [ContinuationContracts.test_only_route2_first_trial_enabled](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v1.py#L18) | 保留 | 日常＋完整，方法原样 | 调用：`config`；原4处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B329 [ContinuationContracts.test_gap_policy_uses_only_past_observations](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v1.py#L24) | 保留 | 日常＋完整，方法原样 | 调用：`gap_policy`、`gap_methods`、`config`；原6处断言位置：`self.assertTrue`、`self.assertFalse`、`self.assertRaises`等；完整断言不删 |
| B330 [ContinuationContracts.test_quality_constrained_oracle_is_not_query_independent_mixture](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v1.py#L33) | 保留 | 日常＋完整，方法原样 | 调用：`config`、`stopping_oracle`；原5处断言位置：`self.assertAlmostEqual`、`self.assertFalse`、`np.testing.assert_allclose`；完整断言不删 |
| B331 [ContinuationContracts.test_oracle_prices_conservative_monotone_and_bad_cohort_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v1.py#L40) | 保留 | 日常＋完整，方法原样 | 调用：`stopping_oracle`、`config`；原4处断言位置：`self.assertEqual`、`self.assertRaises`；完整断言不删 |
| B332 [ContinuationContracts.test_new_report_requires_selection_and_source_seal](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v1.py#L46) | 保留 | 日常＋完整，方法原样 | 调用：`seal_selection`；原4处断言位置：`self.assertRaises`、`self.assertEqual`；完整断言不删 |
| B333 [ContinuationNative.test_uninterrupted_matches_faiss_bitwise](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v1.py#L75) | 保留 | 日常＋完整，方法原样 | 原4处断言位置：`np.testing.assert_array_equal`、`self.assertTrue`、`self.assertGreater`；完整断言不删 |
| B334 [ContinuationNative.test_many_pauses_preserve_entire_call_sequence_and_final_state](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v1.py#L81) | 保留 | 日常＋完整，方法原样 | 调用：`Session`；原4处断言位置：`self.assertEqual`、`np.testing.assert_array_equal`；完整断言不删 |
| B335 [ContinuationNative.test_snapshot_is_copy_and_does_not_reorder_live_heap](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v1.py#L94) | 保留 | 日常＋完整，方法原样 | 调用：`Session`；原5处断言位置：`self.assertEqual`、`self.assertTrue`、`np.testing.assert_array_equal`；完整断言不删 |
| B336 [ContinuationNative.test_absolute_budget_and_complete_batch_accounting](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v1.py#L102) | 保留 | 日常＋完整，方法原样 | 调用：`Session`；原4处断言位置：`self.assertTrue`、`self.assertEqual`；完整断言不删 |
| B337 [ContinuationNative.test_natural_end_cannot_be_reopened](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v1.py#L109) | 保留 | 日常＋完整，方法原样 | 调用：`Session`；原5处断言位置：`self.assertTrue`、`self.assertEqual`、`np.testing.assert_array_equal`；完整断言不删 |
| B338 [ContinuationNative.test_query_memory_is_owned_and_sessions_are_independent](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v1.py#L116) | 保留 | 日常＋完整，方法原样 | 调用：`Session`；原2处断言位置：`np.testing.assert_array_equal`；完整断言不删 |
| B339 [ContinuationNative.test_zero_distance_duplicates_and_gap_boundary](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v1.py#L124) | 保留 | 日常＋完整，方法原样 | 调用：`Session`；原4处断言位置：`self.assertEqual`、`self.assertIn`、`self.assertTrue`；完整断言不删 |
| B340 [ContinuationNative.test_invalid_input_closed_state_and_future_information_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v1.py#L129) | 保留 | 日常＋完整，方法原样 | 调用：`Session`；原6处断言位置：`self.assertRaises`、`hook.assert_not_called`；完整断言不删 |
| B341 [ContinuationNative.test_runtime_policy_and_independent_worker_match](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v1.py#L142) | 保留 | 日常＋完整，方法原样 | 调用：`save`、`sha`；原4处断言位置：`self.assertEqual`、`self.assertGreater`、`self.assertRaises`；完整断言不删 |

### test_tls_rag_step5_ann_continuation_v2.py（15项）

覆盖契约：当前续搜 v2 的分组、特征、残差、原生模型和 worker；测试参与来源绑定。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B342 [ContinuationV2Contracts.test_only_approved_cpu_scope](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L33) | 保留 | 日常＋完整，方法原样 | 调用：`config`；原5处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B343 [ContinuationV2Contracts.test_nested_query_isolation_all_checkpoints_share_identity](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L42) | 保留 | 日常＋完整，方法原样 | 调用：`nested_folds`、`config`；原5处断言位置：`self.assertEqual`、`self.assertRaises`；完整断言不删 |
| B344 [ContinuationV2Contracts.test_all_seals_required_for_outer_gate](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L53) | 保留 | 日常＋完整，方法原样 | 调用：`nested_folds`、`config`、`sources`；原4处断言位置：`self.assertRaises`、`self.assertEqual`；完整断言不删 |
| B345 [ContinuationV2Contracts.test_exact_tri_features_share_raw_geometry_and_ignore_future](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L70) | 保留 | 日常＋完整，方法原样 | 调用：`observed_features`、`config`、`tri_law_probability`；原5处断言位置：`np.testing.assert_array_equal`、`self.assertTrue`、`np.testing.assert_allclose`等；完整断言不删 |
| B346 [ContinuationV2Contracts.test_degenerate_geometry_has_explicit_mask](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L90) | 保留 | 日常＋完整，方法原样 | 调用：`observed_features`、`config`；原2处断言位置：`self.assertTrue`；完整断言不删 |
| B347 [ContinuationV2Contracts.test_calibration_preserves_late_gain_not_only_next_step](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L98) | 保留 | 日常＋完整，方法原样 | 调用：`calibrate`、`config`；原1处断言位置：`self.assertAlmostEqual`；完整断言不删 |
| B348 [ContinuationV2Contracts.test_decision_uses_only_current_frozen_scores_and_calibration](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L106) | 保留 | 日常＋完整，方法原样 | 调用：`decide`、`config`；原4处断言位置：`self.assertTrue`、`self.assertFalse`、`self.assertRaises`；完整断言不删 |
| B349 [ContinuationV2Contracts.test_replay_skips_coalesced_checkpoint](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L116) | 保留 | 日常＋完整，方法原样 | 调用：`replay`、`config`；原1处断言位置：`self.assertEqual`；完整断言不删 |
| B350 [ContinuationV2Contracts.test_zero_errors_do_not_certify_zero_risk](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L123) | 保留 | 日常＋完整，方法原样 | 调用：`paired_quality`、`config`；原2处断言位置：`self.assertEqual`、`self.assertGreater`；完整断言不删 |
| B351 [ContinuationV2Native.test_full_boundary_and_batch_resume_match_faiss_bitwise](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L148) | 保留 | 日常＋完整，方法原样 | 调用：`BoundarySession`；原4处断言位置：`self.assertEqual`、`np.testing.assert_array_equal`；完整断言不删 |
| B352 [ContinuationV2Native.test_pending_drain_observation_and_snapshot_do_not_mutate](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L165) | 保留 | 日常＋完整，方法原样 | 调用：`BoundarySession`；原6处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertLessEqual`；完整断言不删 |
| B353 [ContinuationV2Native.test_tree_export_matches_sklearn_with_float32_split_convention](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L179) | 保留 | 日常＋完整，方法原样 | 调用：`Predictor`、`export_model`；原1处断言位置：`np.testing.assert_allclose`；完整断言不删 |
| B354 [ContinuationV2Native.test_training_prediction_calibration_with_coalesced_states](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L189) | 保留 | 日常＋完整，方法原样 | 调用：`fit_models`、`predictions`、`predictors`；原3处断言位置：`self.assertEqual`、`self.assertIn`；完整断言不删 |
| B355 [ContinuationV2Native.test_invalid_closed_state_and_future_args_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L210) | 保留 | 日常＋完整，方法原样 | 调用：`BoundarySession`；原4处断言位置：`self.assertRaises`、`hook.assert_not_called`；完整断言不删 |
| B356 [ContinuationV2Native.test_real_policy_selective_gate_and_independent_workers](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_continuation_v2.py#L222) | 保留 | 日常＋完整，方法原样 | 调用：`save`、`sha`、`Runtime`；原6处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertRaises`；完整断言不删 |

### test_tls_rag_step5_ann_enhancement.py（13项）

覆盖契约：仍提供复用 helper 的 route1；原生/过滤/信息闸门，测试参与来源绑定。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B357 [Route1Contracts.test_only_route1_enabled](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_enhancement.py#L20) | 保留 | 日常＋完整，方法原样 | 调用：`config`；原4处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertTrue`；完整断言不删 |
| B358 [Route1Contracts.test_threshold_is_exact_inverse_of_unchanged_law](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_enhancement.py#L24) | 保留 | 日常＋完整，方法原样 | 调用：`config`、`critical_beta`、`tri_law_probability`；原3处断言位置：`self.assertAlmostEqual`、`self.assertEqual`、`np.testing.assert_allclose`；完整断言不删 |
| B359 [Route1Contracts.test_invalid_geometry_rejected_instead_of_clipped](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_enhancement.py#L32) | 保留 | 日常＋完整，方法原样 | 调用：`critical_beta`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B360 [Route1Contracts.test_role_partitions_are_fixed_disjoint_and_development_only](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_enhancement.py#L36) | 保留 | 日常＋完整，方法原样 | 调用：`split_groups`、`config`；原4处断言位置：`self.assertEqual`、`self.assertRaises`；完整断言不删 |
| B361 [Route1Contracts.test_report_requires_bound_selection_and_sources](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_enhancement.py#L44) | 保留 | 日常＋完整，方法原样 | 调用：`seal_selection`；原4处断言位置：`self.assertRaises`、`self.assertEqual`；完整断言不删 |
| B362 [Route1Contracts.test_original_rerank_and_labels_do_not_change_geometry](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_enhancement.py#L55) | 保留 | 日常＋完整，方法原样 | 调用：`rerank`、`metrics`；原5处断言位置：`self.assertEqual`、`self.assertRaises`；完整断言不删 |
| B363 [Route1Contracts.test_output_exclusive_and_pairing_required](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_enhancement.py#L63) | 保留 | 日常＋完整，方法原样 | 调用：`save`、`paired_ratio`、`config`；原3处断言位置：`self.assertRaises`、`self.assertEqual`；完整断言不删 |
| B364 [Route1Native.test_native_noop_bitwise_matches_faiss](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_enhancement.py#L86) | 保留 | 日常＋完整，方法原样 | 原4处断言位置：`np.testing.assert_array_equal`、`self.assertEqual`、`self.assertGreater`；完整断言不删 |
| B365 [Route1Native.test_shadow_overhead_and_free_oracle_preserve_output](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_enhancement.py#L94) | 保留 | 日常＋完整，方法原样 | 原7处断言位置：`np.testing.assert_array_equal`、`self.assertGreater`、`self.assertLess`；完整断言不删 |
| B366 [Route1Native.test_final_result_rejection_is_not_frontier_rejection](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_enhancement.py#L106) | 保留 | 日常＋完整，方法原样 | 原2处断言位置：`self.assertTrue`；完整断言不删 |
| B367 [Route1Native.test_independent_timing_worker_matches_correctness_candidates](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_enhancement.py#L116) | 保留 | 日常＋完整，方法原样 | 调用：`save`、`sources`、`rerank`；原3处断言位置：`self.assertEqual`、`self.assertGreater`；完整断言不删 |
| B368 [Route1Native.test_deployable_gate_separates_pending_and_counts_audit](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_enhancement.py#L138) | 保留 | 日常＋完整，方法原样 | 原6处断言位置：`self.assertTrue`、`self.assertEqual`、`self.assertLessEqual`；完整断言不删 |
| B369 [Route1Native.test_future_information_forbidden_in_active_mode_and_bad_storage_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_ann_enhancement.py#L147) | 保留 | 日常＋完整，方法原样 | 原4处断言位置：`self.assertRaises`；完整断言不删 |

### test_tls_rag_step5_baseline_headroom.py（12项）

覆盖契约：当前对照复用的精确/ANN/成本空间与输入闸门；测试参与来源绑定。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B370 [BaselineHeadroomTests.test_original_and_projected_use_same_reference_geometry](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_baseline_headroom.py#L42) | 保留 | 日常＋完整，方法原样 | 调用：`execute`、`project_rows`、`projected_fixed`；原5处断言位置：`self.assertEqual`；完整断言不删 |
| B371 [BaselineHeadroomTests.test_original_ann_actual_sets_and_no_truth_in_decision](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_baseline_headroom.py#L55) | 保留 | 日常＋完整，方法原样 | 调用：`execute`；原4处断言位置：`self.assertEqual`；完整断言不删 |
| B372 [BaselineHeadroomTests.test_truth_mismatch_stops_instead_of_refreshing](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_baseline_headroom.py#L65) | 保留 | 日常＋完整，方法原样 | 调用：`execute`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B373 [BaselineHeadroomTests.test_quality_hit_may_decrease_and_outside_qrels_are_valid](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_baseline_headroom.py#L69) | 保留 | 日常＋完整，方法原样 | 调用：`quality`；原1处断言位置：`self.assertEqual`；完整断言不删 |
| B374 [BaselineHeadroomTests.test_selection_only_eligible_fixed_no_eval_fallback](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_baseline_headroom.py#L73) | 保留 | 日常＋完整，方法原样 | 调用：`select_fixed`；原2处断言位置：`self.assertEqual`、`self.assertIsNone`；完整断言不删 |
| B375 [BaselineHeadroomTests.test_perfect_oracle_uses_mean_action_costs_and_keeps_infeasibility](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_baseline_headroom.py#L80) | 保留 | 日常＋完整，方法原样 | 调用：`perfect_information_bound`；原4处断言位置：`self.assertTrue`、`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B376 [BaselineHeadroomTests.test_eval_gate_and_selection_seal_are_binding](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_baseline_headroom.py#L97) | 保留 | 日常＋完整，方法原样 | 调用：`ObservedInputs.__new__`、`seal_selection`；原4处断言位置：`self.assertRaises`、`self.assertEqual`、`self.assertRaisesRegex`；完整断言不删 |
| B377 [BaselineHeadroomTests.test_sampling_identity_only_order_invariant_and_parent_sha_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_baseline_headroom.py#L110) | 保留 | 日常＋完整，方法原样 | 调用：`timing_ids`、`make_request`；原3处断言位置：`self.assertEqual`、`self.assertRaises`；完整断言不删 |
| B378 [BaselineHeadroomTests.test_loader_rejects_unapproved_request_before_arrays](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_baseline_headroom.py#L118) | 保留 | 日常＋完整，方法原样 | 调用：`save`、`ObservedInputs`；原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B379 [BaselineHeadroomTests.test_complete_small_input_loader_and_download_checksum_guard](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_baseline_headroom.py#L123) | 保留 | 日常＋完整，方法原样 | 调用：`save`、`load_protocol`、`sha`；原3处断言位置：`self.assertEqual`、`self.assertRaises`、`self.assertRaisesRegex`；完整断言不删 |
| B380 [BaselineHeadroomTests.test_existing_output_is_not_modified_even_on_rejected_run](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_baseline_headroom.py#L148) | 保留 | 日常＋完整，方法原样 | 调用：`save`；原2处断言位置：`self.assertNotEqual`、`self.assertEqual`；完整断言不删 |
| B381 [BaselineHeadroomTests.test_remote_export_only_observed_groups_and_no_closed_role_reads](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_baseline_headroom.py#L158) | 保留 | 日常＋完整，方法原样 | 调用：`fingerprint`、`save`、`sha`；原6处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |

### test_tls_rag_step5_mechanism.py（14项）

覆盖契约：仍被后续调用的 ExactScan/PrefixSession 和实验审计；测试参与来源绑定。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B382 [MechanismMathTests.test_exact_scan_matches_brute_force_ties_and_ranks](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_mechanism.py#L94) | 保留 | 日常＋完整，方法原样 | 调用：`ExactScan`；原5处断言位置：`self.assertEqual`；完整断言不删 |
| B383 [MechanismMathTests.test_decomposition_keeps_rescues_not_additive_error](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_mechanism.py#L105) | 保留 | 日常＋完整，方法原样 | 调用：`decompose`；原4处断言位置：`self.assertEqual`、`self.assertAlmostEqual`；完整断言不删 |
| B384 [MechanismMathTests.test_flat_cutoff_tie_resolved_over_entire_group](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_mechanism.py#L116) | 保留 | 日常＋完整，方法原样 | 调用：`flat_prefix`；原2处断言位置：`self.assertEqual`、`self.assertTrue`；完整断言不删 |
| B385 [MechanismMathTests.test_sampling_deterministic_identity_only_and_disjoint](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_mechanism.py#L123) | 保留 | 日常＋完整，方法原样 | 调用：`sample_groups`；原3处断言位置：`self.assertEqual`、`self.assertFalse`、`self.assertRaises`；完整断言不删 |
| B386 [MechanismMathTests.test_query_block_bootstrap_preserves_pairing_and_repeats](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_mechanism.py#L133) | 保留 | 日常＋完整，方法原样 | 调用：`interval`；原3处断言位置：`self.assertEqual`；完整断言不删 |
| B387 [MechanismBoundaryTests.test_reader_never_opens_forbidden_roles_and_keeps_eval_closed](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_mechanism.py#L152) | 保留 | 日常＋完整，方法原样 | 原5处断言位置：`self.assertRaises`、`self.assertFalse`、`self.assertEqual`；完整断言不删 |
| B388 [MechanismBoundaryTests.test_input_tampering_rejected_without_refreshing_binding](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_mechanism.py#L170) | 保留 | 日常＋完整，方法原样 | 原1处断言位置：`self.assertRaisesRegex`；完整断言不删 |
| B389 [MechanismBoundaryTests.test_no_retry_parent_reservation_and_seal_tampering](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_mechanism.py#L174) | 保留 | 日常＋完整，方法原样 | 调用：`Ledger`；原4处断言位置：`self.assertRaises`；完整断言不删 |
| B390 [MechanismBoundaryTests.test_resource_stop_preserves_reserved_stage](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_mechanism.py#L185) | 保留 | 日常＋完整，方法原样 | 原2处断言位置：`self.assertRaises`、`self.assertTrue`；完整断言不删 |
| B391 [MechanismBoundaryTests.test_new_namespace_does_not_change_legacy_source_binding](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_mechanism.py#L192) | 保留 | 日常＋完整，方法原样 | 调用：`source_binding`；原2处断言位置：`self.assertTrue`、`self.assertEqual`；完整断言不删 |
| B392 [MechanismBoundaryTests.test_features_match_reviewed_geometry_and_no_hidden_tri](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_mechanism.py#L198) | 保留 | 日常＋完整，方法原样 | 调用：`project_rows`、`ExactScan`、`PrefixSession`；原9处断言位置：`self.assertEqual`、`np.testing.assert_allclose`、`self.assertRaises`；完整断言不删 |
| B393 [MechanismBoundaryTests.test_fit_calibration_are_matched_and_invalid_expands](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_mechanism.py#L220) | 保留 | 日常＋完整，方法原样 | 调用：`fit_models`、`calibrate`、`adjusted`；原3处断言位置：`self.assertIsNotNone`、`self.assertFalse`、`self.assertRaises`；完整断言不删 |
| B394 [MechanismBoundaryTests.test_real_pipeline_uses_own_budget_and_scoring_cache_without_hidden_tri](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_mechanism.py#L234) | 保留 | 日常＋完整，方法原样 | 调用：`pipeline`；原7处断言位置：`self.assertEqual`、`self.assertGreater`、`self.assertGreaterEqual`；完整断言不删 |
| B395 [MechanismBoundaryTests.test_complete_two_phase_fixture_with_real_faiss_when_available](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_mechanism.py#L248) | 保留 | 日常＋完整，方法原样 | 调用：`phase_a`、`phase_b`、`sha256`；原6处断言位置：`self.assertTrue`、`self.assertFalse`、`self.assertEqual`；完整断言不删 |

### test_tls_rag_step5_reuse_probe.py（8项）

覆盖契约：仍被基线调用的缓存/稳定选择/平局/隔离；测试参与来源绑定。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B396 [ReuseProbeTests.test_random_prefixes_are_bitwise_reference_equal](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_reuse_probe.py#L37) | 保留 | 日常＋完整，方法原样 | 调用：`project_rows`；成功路径要求不抛异常；原调用保留 |
| B397 [ReuseProbeTests.test_cutoff_ties_nonsorted_ids_zero_and_near_ties](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_reuse_probe.py#L41) | 保留 | 日常＋完整，方法原样 | 成功路径要求不抛异常；原调用保留 |
| B398 [ReuseProbeTests.test_every_distance_equal_through_all_expansions](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_reuse_probe.py#L46) | 保留 | 日常＋完整，方法原样 | 成功路径要求不抛异常；原调用保留 |
| B399 [ReuseProbeTests.test_cache_owns_query_and_returns_readonly_prefix_copies](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_reuse_probe.py#L50) | 保留 | 日常＋完整，方法原样 | 调用：`ExactDistanceSession`；原6处断言位置：`self.assertEqual`、`self.assertRaises`；完整断言不删 |
| B400 [ReuseProbeTests.test_bad_budgets_do_not_consume_state](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_reuse_probe.py#L65) | 保留 | 日常＋完整，方法原样 | 调用：`ExactDistanceSession`、`project_rows`；原2处断言位置：`self.assertRaises`、`self.assertEqual`；完整断言不删 |
| B401 [ReuseProbeTests.test_bad_inputs_and_nonfinite_distances_fail](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_reuse_probe.py#L74) | 保留 | 日常＋完整，方法原样 | 调用：`ExactDistanceSession`；原2处断言位置：`self.assertRaises`；完整断言不删 |
| B402 [ReuseProbeTests.test_actual_vector_rows_read_once_and_not_on_expansion](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_reuse_probe.py#L83) | 保留 | 日常＋完整，方法原样 | 调用：`ExactDistanceSession`；原3处断言位置：`self.assertEqual`；完整断言不删 |
| B403 [ReuseProbeTests.test_all_complete_cases_match_every_prefix_and_reranking](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tls_rag_step5_reuse_probe.py#L101) | 保留 | 日常＋完整，方法原样 | 调用：`project_rows`、`ExactScan`、`run_case`；原9处断言位置：`np.testing.assert_array_equal`、`self.assertEqual`；完整断言不删 |

### test_tri_law.py（10项）

覆盖契约：精确 Tri-Law 公式、R1/R2 边界、高精度、小尾部、广播和 Monte Carlo。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B404 [TriLawTests.test_joint_near_tie_collinearity_regression](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_law.py#L29) | 保留 | 日常＋完整，方法原样 | 调用：`tri_law_threshold`、`tri_law_probability`；原3处断言位置：`self.assertAlmostEqual`、`self.assertGreater`；完整断言不删 |
| B405 [TriLawTests.test_threshold_matches_decimal_across_float64_domain](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_law.py#L39) | 保留 | 日常＋完整，方法原样 | 调用：`tri_law_threshold`；原5处断言位置：`self.assertGreaterEqual`、`self.assertTrue`、`self.assertEqual`等；完整断言不删 |
| B406 [TriLawTests.test_representable_tails_survive_threshold_overflow](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_law.py#L57) | 保留 | 日常＋完整，方法原样 | 调用：`tri_law_probability`；原4处断言位置：`self.assertGreater`、`self.assertTrue`、`self.assertLessEqual`等；完整断言不删 |
| B407 [TriLawTests.test_conditional_overflow_and_subnormal_regression](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_law.py#L75) | 保留 | 日常＋完整，方法原样 | 调用：`tri_law_conditional_orthogonal`；原2处断言位置：`np.testing.assert_allclose`、`self.assertEqual`；完整断言不删 |
| B408 [TriLawTests.test_boundary_broadcasting_and_probability_order](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_law.py#L89) | 保留 | 日常＋完整，方法原样 | 调用：`tri_law_threshold`、`tri_law_probability`；原6处断言位置：`self.assertEqual`、`self.assertTrue`、`np.testing.assert_array_equal`；完整断言不删 |
| B409 [TriLawTests.test_algebraic_identities_and_boundaries](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_law.py#L101) | 保留 | 日常＋完整，方法原样 | 调用：`tri_law_threshold`、`tri_law_probability`；原8处断言位置：`np.testing.assert_allclose`、`self.assertEqual`、`self.assertAlmostEqual`等；完整断言不删 |
| B410 [TriLawTests.test_orthogonal_specialization](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_law.py#L121) | 保留 | 日常＋完整，方法原样 | 调用：`tri_law_probability`；原1处断言位置：`self.assertAlmostEqual`；完整断言不删 |
| B411 [TriLawTests.test_conditional_marginalization](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_law.py#L127) | 保留 | 日常＋完整，方法原样 | 调用：`tri_law_conditional_orthogonal`、`tri_law_probability`；原2处断言位置：`self.assertLess`、`self.assertAlmostEqual`；完整断言不删 |
| B412 [TriLawTests.test_input_validation](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_law.py#L141) | 保留 | 日常＋完整，方法原样 | 调用：`tri_law_probability`、`tri_law_conditional_orthogonal`；原1处断言位置：`self.assertRaises`；完整断言不删 |
| B413 [TriLawTests.test_monte_carlo_conformance](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_law.py#L157) | 保留 | 日常＋完整，方法原样 | 调用：`tri_law_probability`；原1处断言位置：`self.assertLessEqual`；完整断言不删 |

### test_tri_predict.py（17项）

覆盖契约：冻结 Tri-Predict 近似、数值聚合、完整语料边界和保存/加载行为。

| 编号/基线方法 | 处置 | 执行集合 | 实际调用/断言依据 |
| --- | --- | --- | --- |
| B414 [TriPredictTests.test_exact_finite_rank_sum_matches_conditional_law_terms](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L23) | 保留 | 日常＋完整，方法原样 | 调用：`tri_law_conditional_orthogonal`、`tri_predict_h_j`；原1处断言位置：`self.assertAlmostEqual`；完整断言不删 |
| B415 [TriPredictTests.test_rank_quadrature_is_deterministic_and_conserves_population](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L45) | 保留 | 日常＋完整，方法原样 | 调用：`deterministic_rank_quadrature`；原6处断言位置：`self.assertTrue`、`np.testing.assert_array_equal`、`self.assertFalse`等；完整断言不删 |
| B416 [TriPredictTests.test_h_j_is_monotone_in_conditioning_scale](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L56) | 保留 | 日常＋完整，方法原样 | 调用：`tri_predict_h_j`；原3处断言位置：`self.assertTrue`、`self.assertEqual`；完整断言不删 |
| B417 [TriPredictTests.test_bounded_root_solves_mean_field_equation](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L72) | 保留 | 日常＋完整，方法原样 | 调用：`solve_y_star`、`tri_predict_h_j`；原1处断言位置：`self.assertAlmostEqual`；完整断言不删 |
| B418 [TriPredictTests.test_full_budget_boundary_has_infinite_roots_and_unit_retention](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L93) | 保留 | 日常＋完整，方法原样 | 调用：`solve_y_star`、`tri_predict_retention`；原2处断言位置：`self.assertTrue`、`self.assertEqual`；完整断言不删 |
| B419 [TriPredictTests.test_predicted_retention_is_budget_monotone_and_bounded](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L115) | 保留 | 日常＋完整，方法原样 | 调用：`tri_predict_retention_grid`；原2处断言位置：`self.assertTrue`；完整断言不删 |
| B420 [TriPredictTests.test_geometric_approximation_agrees_with_exact_sum_and_prediction](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L127) | 保留 | 日常＋完整，方法原样 | 调用：`tri_predict_h_j`、`tri_predict_retention`；原2处断言位置：`self.assertLess`；完整断言不删 |
| B421 [TriPredictTests.test_actual_distance_diagnostic_matches_rank_model_when_power_law_is_exact](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L164) | 保留 | 日常＋完整，方法原样 | 调用：`tri_predict_retention_grid`、`actual_distance_retention_grid`；原1处断言位置：`self.assertAlmostEqual`；完整断言不删 |
| B422 [TriPredictTests.test_analytic_policy_is_grid_safe_lid_monotone_and_logs_saturation](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L189) | 保留 | 日常＋完整，方法原样 | 调用：`TriPredictPolicy`；原5处断言位置：`self.assertTrue`、`self.assertEqual`；完整断言不删 |
| B423 [TriPredictTests.test_exact_unit_target_requires_complete_corpus](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L207) | 保留 | 日常＋完整，方法原样 | 调用：`TriPredictPolicy`；原4处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B424 [TriPredictTests.test_complete_corpus_is_not_reduced_by_safety_correction](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L224) | 保留 | 日常＋完整，方法原样 | 调用：`TriPredictPolicy`；原4处断言位置：`self.assertEqual`、`self.assertFalse`；完整断言不删 |
| B425 [TriPredictTests.test_compiled_policy_matches_dense_reference_and_adjacent_boundaries](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L244) | 保留 | 日常＋完整，方法原样 | 调用：`TriPredictPolicy`、`CompiledTriPredictPolicy.compile`、`CompiledTriPredictPolicy.from_serialized`；原13处断言位置：`self.assertEqual`、`compiled.assert_equivalent`、`self.assertFalse`等；完整断言不删 |
| B426 [TriPredictTests.test_frozen_analytic_policy_loader_rejects_tampering](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L296) | 保留 | 日常＋完整，方法原样 | 调用：`TriPredictPolicy`、`TriPredictPolicy.from_serialized(artifact).serialize`、`TriPredictPolicy.from_serialized`；原2处断言位置：`self.assertEqual`、`self.assertRaisesRegex`；完整断言不删 |
| B427 [TriPredictTests.test_compiled_policy_falls_back_outside_frozen_lid_domain](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L312) | 保留 | 日常＋完整，方法原样 | 调用：`TriPredictPolicy`、`CompiledTriPredictPolicy.compile`；原3处断言位置：`self.assertEqual`、`self.assertTrue`、`self.assertFalse`；完整断言不删 |
| B428 [TriPredictTests.test_safety_correction_is_fit_only_from_supplied_tune_records](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L333) | 保留 | 日常＋完整，方法原样 | 调用：`TriPredictPolicy`、`TriPredictPolicy.fit`；原3处断言位置：`self.assertAlmostEqual`、`self.assertEqual`；完整断言不删 |
| B429 [TriPredictTests.test_analytic_and_empirical_policies_share_decision_interface](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L371) | 保留 | 日常＋完整，方法原样 | 调用：`MonotoneBinnedPolicy`、`TriPredictPolicy`；原2处断言位置：`self.assertIn`、`self.assertFalse`；完整断言不删 |
| B430 [TriPredictTests.test_invalid_inputs_are_rejected](https://github.com/kiritsugu233/Tri-RAG/blob/fd91423880c0d504c531dce4388ae9e4ba86b628/tests/test_tri_predict.py#L392) | 保留 | 日常＋完整，方法原样 | 调用：`tri_predict_retention`、`tri_predict_retention_grid`、`deterministic_rank_quadrature`；原1处断言位置：`self.assertRaises`；完整断言不删 |

## 七、批准后的验收和停止条件

1. 修改前记录基线430个测试ID及全部L0/来源绑定，确认没有用户改动被覆盖。
2. M01的三个错误身份仍各自触发原异常，合法同值仍通过；用临时故障注入确认
   错误接受任一原场景会导致新测试失败；其他五个方法内容不变。
3. 实际收集证明：日常231、历史198、完整429；两个子集互斥且并集恰等于完整。
   从基线430到429仅有M01两个旧ID→一个新ID的可解释变化。
4. 用固定依赖/四个原生库运行完整429项，要求无失败、无跳过；同时验证两个显式子集。
   不能通过缺包跳过、改容差、删断言或改数据规模来取得成功。
5. 核对无参数与full结果相同；未知套件、未归类测试文件、导入错误都必须明确失败。
   只用临时目录/临时副本验证新增文件和错误场景，不修改科学源码。
6. 再比较全部科学来源绑定/代码/配置/历史数据哈希；仅更新真实获批的测试/治理哈希。
   保护检查、Markdown和diff检查通过，再提交/推送同一Step5分支并核实远端。
7. 报告实测时间和选择范围；不将日常少跑方法等同于整个仓库减少相同数量的断言。

以下情况不在本次批准中：永久删除198项、移动来源绑定的测试、改核心算法、放松阈值、
新增科研实验/数据访问、改分支或默认分支、修改AGENTS。若实施需要扩大这些范围，先说明原因。

## 八、审批状态与本轮交付

**状态：仅提案，未批准、未执行。** 用户此次只要求中文逐项审批方案。
本轮只读收集核对430方法和断言/依赖，未运行新的科研实验或修改测试。
完整测试最近一次结果仍归属于方案A提交fd91423，而不是尚不存在的B实现。

本文为新增L1说明，机器清单仅追加这一行，没有改变既有等级、哈希或批准记录。
人工L0保护表新增本文的行留在B的审批范围；其余234项仍保持方案A后的状态。
批准B将授权第五节列出的四个实现/治理路径及必要L1交接更新，**不是永久删除旧回归的授权**。
