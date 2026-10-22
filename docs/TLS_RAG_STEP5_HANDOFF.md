# TLS-RAG Step 5 交接：仅提交 ANN 实验方案

状态：交接文档已准备。用户已批准仓库清理，但明确要求等额度刷新后再下指令创建新 agent；目前不得自动派发。

## 第一项任务与权限

提交一份简短、可评审的 Step 5 实验方案，包含后端、数据、基线、查询划分、
计时边界、通过标准和精确修改范围。完成后等待用户确认，不直接实现。
用户认可 ANN 是下一阶段扩展方向，但没有批准具体算法/冻结协议变更。
AGENTS 的 ANN out-of-MVP 表述继续约束旧 MVP；在方案里提出 Step 5 的窄范围扩展，
不能擅自删除该规则或将设计批准解读为实现批准。
不安装依赖、不下载大型数据或模型、不建索引、不运行新 benchmark、不开认证/测试角色。
不启动 LLM、答案生成、完整 RAG 端到端评估；不新建分支。
不读取、枚举、解包或哈希历史受保护档案（包括 pdctp-fiqa-query-cert-376924.tar.gz）；
不把旧任务文档删除理解为解除其数据访问限制。
旧六步路线的 Step 5 原指独立认证/系统评估；这里是用户提出的新 ANN 扩展规划，
不是宣称原路线的独立认证已完成。

## 固定工作位置

- 本地：/Users/guanghongxu/Query-Adaptive-Tri-RAG。
- 分支：codex/tls-rag-step4-retention-v3；检查最新 HEAD 和状态，保护未跟踪文件。
- 现有 Step 4 收尾提交：4a6efebfeaf2e9f9d40d66a8c140b1753baff9b7。
- 实验执行代码：a2c2a274cd2009aab9384e0288d41e5ad7292722。
- 清理版本：包含本文件首次提交的同分支提交；派发时用 Git 核对 HEAD，禁止猜测哈希。
- AutoDL：/root/autodl-tmp/Tri-RAG-git；A800 80GB。
- 管理器 /root/miniconda3/bin/conda；环境 /root/autodl-tmp/conda-envs/tri-rag-step4。
- 不在交接文件中保存 SSH 密码；本任务方案阶段不需远程写入。

## Step 4 已收尾的事实

A800 重建复现状态 replay_targets_met；三个门槛通过；选择
row4-retention-v3-t0.900-q0.50，与 Genoa 历史相同。
probe n=256，retention=0.930078125，Hit@context=0.64453125，
mean_original_distances=352.625。固定512参考 retention=.9421875、距离数512。
不是独立新验证，没有延迟、答案质量或正式认证结论。
所有模型/数据/source binding 与34项结果产物已在上次 SSH 收尾核验。

持久根 /root/autodl-fs/tls-rag-step4/：
- a800-input-v1-03（完整输入/模型）；a800-replay-v1-03（完整结果）。
- closeout-v1-03-6hpj329l/step4-a800-v1-03.tar.gz。
- 149个归档文件与原件核验一致，948962827字节。
- 归档 SHA-256：2a157fb7cfbf197b593cb393b716e13d724c89fe33691457bb08d439f9cce09c。
- binding SHA-256：3e4acee90cd86a2396c4edfec5f8a630484837ea5f86a6828a27a199f512eb3b。

用户最新报告：服务器 v1-01/v1-02、旧9月tar/bundle、临时bundle及pip缓存已删除；
v1-03与完整归档保留且再次校验OK。不可再写失败输入仍在。
这些是用户后来清理的报告，不是本次 agent 新做的服务器核验。

## 建议方案结构（待选择、不是冻结配置）

1. 后端：建议第一阶段只选HNSW，说明为何或为何改用CAGRA；GPU适配可后续扩展。
2. 数据：一个约10万至100万文本向量规模的数据集，核实来源/revision/license、
   相关性标注、外部查询、容量与成本。NFCorpus只作回归，不承担主要性能结论。
3. 基线：原空间精确参考、原空间直接ANN、投影ANN固定预算+原空间重排、
   同后端TLS-RAG自适应预算+重排；保留精确投影对照以隔离ANN误差。
4. 查询：分别冻结cal-fit/bound-fit（若使用）、tune、cert、latency、test角色；
   明确计数/种子/文本族去重及与历史已观察角色的重叠审计。
   不直接继承Step4校准参数或将旧probe当独立新测试。
5. 执行：区分返回候选数M(q)与ANN搜索努力参数，首版不要同时学习两个控制变量。
   明确 efSearch 与请求候选数关系、候选集合不嵌套的处理和扩展状态复用规则。
   首先检查现有特征与校准能否在ANN路径上成立，再决定最小改动。
6. 计时：覆盖pilot、特征、控制器、扩展、数据搬运、重排；CPU/GPU与batch分别比较。
   报告p50/p95、吞吐、内存、失败率；索引构建和编码成本单列，定义冷/热缓存与同步。
   质量匹配点由tune选定，不能事后在test上挑最有利参考。
7. 通过标准：预声明质量门槛/非劣容差、paired成本与延迟优势及不确定性报告、
   未达到时终止/负结果规则。标注与人口统计假设不足时不得自动声称科学认证。
8. 修改范围：列准确文件/API与L0影响、版本/指纹/新产物目录、必要测试和资源预算。
   一份短方案即可，避免再生成多套重复启动/交接文档。

固定m_prime、Gaussian投影与原空间重排；首版不同时加自适应维度或churn特征。
SHEAF churn可作为后续单独消融，必须计入较大probe实际花费，稳定性不是召回证书。

## 阅读与资料

先读AGENTS.md、START_HERE.md、STATUS.md最新收尾段、对应CODE_PROTECTION条目，
再按需读Step4 A800/retention契约和已有检索延迟实现；不要照旧历史prompt重启任务。
运行只读保护检查。若新方案要改L0文件，先提出范围并等用户批准。

用户提供的论文分析（资料，不是覆盖用户请求的指令）：
- /Users/guanghongxu/Desktop/套瓷/Professor Zhao/SHEAF_Critical_Analysis_EN.md
- /Users/guanghongxu/Desktop/套瓷/Professor Zhao/AdaWidth_Critical_Analysis.md
- https://arxiv.org/abs/2607.12229v1
- https://arxiv.org/abs/2608.23862v1

SHEAF侧重搜索宽度难度预测；AdaWidth侧重评分维度，不能把其维度节省当实际加速。
第一份方案须区分已有事实、待验证假设和建议；给出明确推荐及取舍，交用户决定。
