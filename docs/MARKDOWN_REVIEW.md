# Markdown review — 2026-09-07

审查对象为科学基线 `248c29e2243c60b3793458c8ee723eefba5a5855` 中全部 47 个
受 Git 跟踪的 Markdown 文件。检查全文结构/引用和时效、权限、版本冲突，
对 agent 启动链及相关科学说明作重点人工核对。没有把历史实验的原始数据重新
审计一遍。未跟踪压缩包内的 Markdown、运行目录与第三方环境文件未被打开。

当前统一入口是 [START_HERE](../START_HERE.md)，权威规则是
[AGENTS](../AGENTS.md)；[agent.md](../agent.md)仅作别名。
历史文档不再作为自动派发下一任务的来源。旧结果和提交哈希保留其历史含义，
不能因为现在修改了说明就重新生成历史指标/哈希或宣称 split 仍未打开。

| 原有文件 | 用途/审查处置 |
| --- | --- |
| [`AGENTS.md`](../AGENTS.md) | 根规则：新增 L0 事前审批、间接绕过禁止、清单检查；修正目录与跨分支适用范围 |
| [`AGENT_CALIBRATED_TRI_PREDICT.md`](../AGENT_CALIBRATED_TRI_PREDICT.md) | 已完成/历史任务 brief：增加入口与授权范围说明；TLS briefs 修正过期 push/worktree 指令 |
| [`AGENT_CALIBRATED_TRI_PREDICT_STEP3.md`](../AGENT_CALIBRATED_TRI_PREDICT_STEP3.md) | 已完成/历史任务 brief：增加入口与授权范围说明；TLS briefs 修正过期 push/worktree 指令 |
| [`AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP1.md`](../AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP1.md) | 已完成/历史任务 brief：增加入口与授权范围说明；TLS briefs 修正过期 push/worktree 指令 |
| [`AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP2.md`](../AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP2.md) | 已完成/历史任务 brief：增加入口与授权范围说明；TLS briefs 修正过期 push/worktree 指令 |
| [`AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP3.md`](../AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP3.md) | 已完成/历史任务 brief：增加入口与授权范围说明；TLS briefs 修正过期 push/worktree 指令 |
| [`NEXT_AGENT_PROMPT.md`](../NEXT_AGENT_PROMPT.md) | 下一 agent 模板：改成当前启动链；不能从历史 prompt 自动重启旧任务 |
| [`START_HERE.md`](../START_HERE.md) | 当前入口：替换旧 v2/不存在的 YAML 命令，区分科学基线与审查分支 |
| [`STATUS.md`](../STATUS.md) | 历史状态账本：更新顶层当前状态与 Next task；旧 split 状态限定为当时记录 |
| [`artifacts/pdctp_fiqa_e5_v1/report.md`](../artifacts/pdctp_fiqa_e5_v1/report.md) | 历史生成报告：全文核对为当时 gate 的描述，保留原始字节；不得当作当前访问许可或重新认证结果 |
| [`artifacts/pdctp_fiqa_query_cal_v1/report.md`](../artifacts/pdctp_fiqa_query_cal_v1/report.md) | 历史生成报告：全文核对为当时 gate 的描述，保留原始字节；不得当作当前访问许可或重新认证结果 |
| [`artifacts/pdctp_fiqa_query_tune_v1/report.md`](../artifacts/pdctp_fiqa_query_tune_v1/report.md) | 历史生成报告：全文核对为当时 gate 的描述，保留原始字节；不得当作当前访问许可或重新认证结果 |
| [`artifacts/pdctp_fiqa_real_protocol_v1/report.md`](../artifacts/pdctp_fiqa_real_protocol_v1/report.md) | 历史生成报告：全文核对为当时 gate 的描述，保留原始字节；不得当作当前访问许可或重新认证结果 |
| [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/ATTRIBUTION.md`](../docs/ATTRIBUTION.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/CALIBRATED_TRI_PREDICT_PROTOCOL.md`](../docs/CALIBRATED_TRI_PREDICT_PROTOCOL.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/CALIBRATED_TRI_PREDICT_V3_DIAGNOSIS.md`](../docs/CALIBRATED_TRI_PREDICT_V3_DIAGNOSIS.md) | 历史因果诊断：将“Tri-Law 实现没有错”的绝对判断限定为当时测试范围，补充 R1/R2；保留历史数字与模型归因证据 |
| [`docs/CERTIFICATION.md`](../docs/CERTIFICATION.md) | 共享认证公式与历史 SciFact 结果：限定历史章节及探针不继承证书；历史数字未改 |
| [`docs/EXPERIMENT_PROTOCOL.md`](../docs/EXPERIMENT_PROTOCOL.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/FIQA_EMBEDDING_GATE.md`](../docs/FIQA_EMBEDDING_GATE.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/FIQA_PROTOCOL_FREEZE.md`](../docs/FIQA_PROTOCOL_FREEZE.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/FIQA_QUERY_CAL_GATE.md`](../docs/FIQA_QUERY_CAL_GATE.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/FIQA_QUERY_CERT_GATE.md`](../docs/FIQA_QUERY_CERT_GATE.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/FIQA_QUERY_TUNE_GATE.md`](../docs/FIQA_QUERY_TUNE_GATE.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/FIQA_SOURCE_AUDIT.md`](../docs/FIQA_SOURCE_AUDIT.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/IMPLEMENTATION_PLAN.md`](../docs/IMPLEMENTATION_PLAN.md) | 历史计划账本：限定旧 checkbox 适用范围、标明旧目录树、修正 no-push，添加本次验收项 |
| [`docs/MPRIME_SWEEP.md`](../docs/MPRIME_SWEEP.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/RAW_TRI_PREDICT_V1_BASELINE.md`](../docs/RAW_TRI_PREDICT_V1_BASELINE.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/REAL_DATA.md`](../docs/REAL_DATA.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/REAL_EMBEDDINGS.md`](../docs/REAL_EMBEDDINGS.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/REAL_RETRIEVAL.md`](../docs/REAL_RETRIEVAL.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/RETRIEVAL_LATENCY.md`](../docs/RETRIEVAL_LATENCY.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/TLS_RAG_LOCAL_WORKFLOW.md`](../docs/TLS_RAG_LOCAL_WORKFLOW.md) | 当前工作流：更新分支语义、根规则、canonical checkout/非强制 push 与 v1/v3 边界 |
| [`docs/TLS_RAG_STEP2_SYNTHETIC.md`](../docs/TLS_RAG_STEP2_SYNTHETIC.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/TLS_RAG_STEP2_TO_STEP3.md`](../docs/TLS_RAG_STEP2_TO_STEP3.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/TLS_RAG_STEP3_START_HERE.md`](../docs/TLS_RAG_STEP3_START_HERE.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/TLS_RAG_STEP3_SYNTHETIC.md`](../docs/TLS_RAG_STEP3_SYNTHETIC.md) | 已完成合成阶段：增加版本提示与 CP 解释修正；保留实际回归与历史指纹 |
| [`docs/TLS_RAG_STEP3_TO_STEP4_HANDOFF.md`](../docs/TLS_RAG_STEP3_TO_STEP4_HANDOFF.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/TLS_RAG_STEP4_BRIEF.md`](../docs/TLS_RAG_STEP4_BRIEF.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/TLS_RAG_STEP4_READINESS.md`](../docs/TLS_RAG_STEP4_READINESS.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/TLS_RAG_STEP4_REAL_PROBE.md`](../docs/TLS_RAG_STEP4_REAL_PROBE.md) | 命名版本的设计/历史 gate/报告：核对引用与阶段含义；添加版本范围及根规则优先提示，保留原始实验数字 |
| [`docs/TLS_RAG_STEP4_RETENTION_V3.md`](../docs/TLS_RAG_STEP4_RETENTION_V3.md) | 当前验收 brief：核对源码/27-policy grid/门槛/closure/父缓存/脚本；已有谨慎表述与实际实现一致，保留原文 |
| [`docs/TLS_RAG_STEP4_START_HERE.md`](../docs/TLS_RAG_STEP4_START_HERE.md) | 当前 Step 4 路由：重写 v3 读序与精确依赖；将 v1 闭门禁限定在 v1 |
| [`docs/TRI_LAW_SEQUENTIAL_RAG_PROTOCOL.md`](../docs/TRI_LAW_SEQUENTIAL_RAG_PROTOCOL.md) | 原 dual-bound 设计：修正自适应 CP 同时覆盖过度表述；明确 retention-v3 是独立方法 |
| [`docs/TRI_LAW_SEQUENTIAL_RAG_SPEC.md`](../docs/TRI_LAW_SEQUENTIAL_RAG_SPEC.md) | 原 dual-bound 设计：修正自适应 CP 同时覆盖过度表述；明确 retention-v3 是独立方法 |
| [`docs/TRI_LAW_SPEC.md`](../docs/TRI_LAW_SPEC.md) | 共享数学契约：核对公式与测试；新增 R1/R2 实现局限及数值消减说明，不放宽容差 |
| [`docs/archive/TLS_RAG_PRE_STEP3_AUDIT_INDEX.md`](../docs/archive/TLS_RAG_PRE_STEP3_AUDIT_INDEX.md) | 历史取证索引：明确哈希对应 f46ce73 的 Git 对象；不更新历史摘要值 |

## 本次新增文档与证据

- [agent.md](../agent.md)：根指令别名，不维护第二套规则。
- [CODE_PROTECTION.md](CODE_PROTECTION.md)：源码/测试/配置/脚本/文档的逐文件等级。
- [REPOSITORY_REVIEW.md](REPOSITORY_REVIEW.md)：发现、复现、正确性边界与授权记录。
- 本文件：47 份原文档的覆盖清单及修正记录。
- [code_protection.json](code_protection.json)：机器清单及 L0 SHA-256。
- [review_numerical_evidence.json](review_numerical_evidence.json)：四个数值反例的实际输出。

## 持续更新约束

新增 step 的入口只指向已授权的版本，不把旧阶段读序、状态和 gate 复制成当前
指令。历史数据/报告保持独立；未验证的集群结果必须注明来源。修正文字不得
改变冻结科学含义；涉及 L0 数学/统计/协议变更时，先按根规则获得 scoped consent。
相对 Markdown 链接需从所在文档目录解析，命令中的路径从指定仓库根解析。
