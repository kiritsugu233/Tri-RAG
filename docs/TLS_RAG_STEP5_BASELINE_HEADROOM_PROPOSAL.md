# Step 5：补齐不降维基线与免费控制器上限的具体范围

日期：2026-10-26。用户授权目标：“可以，先补齐1，2步吧”。
这是L1只读评审与具体实现范围，正式契约另见
[批准协议](TLS_RAG_STEP5_BASELINE_HEADROOM_PROTOCOL.md)。用户随后明确回复：
“批准上述范围，稍后无卡启动服务器”。以下保留提出范围时的说明。

## 已核查的缺口

当前本地缓存基准只有合成向量、脚本指定M及速度，未测真实数据的retention/Hit。
本地保留的真实Selection/Evaluation记录已有逐查询质量与候选，但没有优化后缓存
路径的计时，也没有与它同次交错测量的不降维基线。不能把合成缓存速度与旧文本
质量拼接，或将新缓存速度与另一机器的旧原空间ANN时间直接相减。

旧ANN完整调用记录可以单独复核：Tune2048原空间ANN64为2.269398ms、
retention=.995898、Hit=.762207；投影ANN256为4.485730ms。
.90目标的搜索前免费oracle为2.033442ms，而查询无关混合为2.132357ms。
这些旧模型仍然有效，但不填补新缓存执行布局的比较缺口。

## 为什么需要新L0范围

[AGENTS.md](../AGENTS.md)要求：“L0 — protected scientific core or frozen contract.
Before any edit, obtain explicit user approval for the exact affected paths and
intended change.” 同时明确一般next-step不等于精确L0修改同意。

本轮要定义新的基线计时边界、已开放数据子集、固定M选择/冻结与质量约束oracle，
属于新的科学比较契约。仅修改L1启动脚本会把L0语义变化藏在caller中，不能这样做。
你已批准比较目标；这里需要补齐具体文件和输入访问范围的授权。

## 拟新增文件、函数与兼容性

| 路径 | 等级 | 精确职责 |
|---|---|---|
| `mechanism/tls_rag_step5_baseline_headroom.py` | L0 | `load_observed_inputs`/`extract_observed_subset`限定已开放输入；`original_exact`、`projected_fixed`、`original_ann`测完整调用；`select_fixed`/`seal_selection`冻结选择；`perfect_information_bound`/`report_frontier`计算诊断上限 |
| `tests/test_tls_rag_step5_baseline_headroom.py` | L0 | 原空间/投影候选、稳定ties、原始评分、oracle手算、输入绑定、选择冻结、未开放角色tripwire、计时边界回归 |
| `scripts/run_tls_rag_step5_baseline_headroom.py` | L1 | 本地CPU编排、只读输入校验、随机交错计时、独立内存测量与新目录收据 |
| `docs/TLS_RAG_STEP5_BASELINE_HEADROOM_PROTOCOL.md` | L0 | 固定输入/方法/目标/选择/计时/资源与停止契约 |

不修改原有科学代码、缓存探针v1、旧测试/协议、依赖版本、旧数据角色或保护基线。
直接调用已验证的`ExactDistanceSession`/`stable_top`、原空间评分、原FAISS索引与
既有只读MILP求解器。新namespace/source/input fingerprint与输出目录独立；原
scientific binding保持原样，绝不刷新旧哈希使新代码伪装成旧实验。
批准后只为这三个新L0文件登记新哈希，并追加真实授权记录及L1结果/状态。

## 输入与数据搬运

推荐使用**既有真实输入**，原1M归一化MiniLM384向量、Gaussian128/seed55117、
旧原空间HNSW索引与已开放的查询/真值/qrels，不再生成合成替代数据。
数据现在位于已封存的autodl-fs；本地只有小结果包及上一轮合成输入。

若服务器已启动，仅通过SSH读取并取回下列本次必要文件/子集，在本地CPU运行：

- 旧bundle的`corpus.npy`、`corpus_ids.json`、元数据/已封存manifest；
- 旧archive/indexes的`projected.npy`、`original.faiss`、setup及索引manifest；
- 已开放`tune`中，原机制Selection256和Evaluation512这两个**已观察**分组的查询、
  原exact top10与qrels；按原sampling.json ID过滤提取，派生包另存新manifest；
- 只读取必要的旧配置/来源绑定和上述允许文件。不启动旧all-role loader，不读取
  原cert/latency/test，不访问无关历史保护档案；不下载模型、不编码、不重建HNSW。

预估共用向量与索引传输约6GB，实际先查大小/剩余空间。服务器无卡模式可仅用于
文件传输，不在0.5CPU/2GiB实例上加载大索引做基准；本地Mac16GiB承担计算。
如地址/端口改变，需要当前连接信息；凭据不展示、不写盘、不写入收据。
仅在明确批准后复用先前授权提供的凭据。若源文件/服务器不可用，保留已有成果，
报告缺口；不擅自用新随机数据或重新构建索引代替。

## 比较与取样

固定N=1M、384->128、k_gt10/k_ctx5、精确投影grid[32,64,128,256]、block8192，
维持float64直接差平方L2与字符串ties，投影后不归一化。FAISS1.8.0、单线程，
原空间ANN直接复用旧original.faiss，M64、efSearch256，随后相同float64精算/重排。

六个完整调用：不降维缓存精确top10、不降维ANN64、降维缓存固定32/64/128/256。
原空间精确也使用同样的距离缓存/全局选取优化；任何方法均不支付虚构的特征费，
也不能将工程优化只给降维路径。包括投影、扫描/搜索、分配/写入、候选选取、
原空间取数/评分、重排和大缓存释放；不重复给本已原空间精确排序的top10评分。

先在Selection256计算各固定动作质量，时间样本按新seed55273的哈希顺序固定取64条，
每条3次、方法顺序seed55279交错；共同预热两空间输入，setup单独记录。
以既定retention>=.90、Hit相对exact>=-.02筛选固定M，选择最快者并封存；无合格
固定M是有效失败，不扩大网格、不降质量阈值。质量和时间均不从新Eval反向选择M。

封存后在原Evaluation512计算所有动作质量，固定取64条计时（相同取样规则），
3次重复。所选固定M即使未达质量目标也如实保留，不能用Eval上更大的M冒充原选择。
完整固定曲线及“Eval上事后最优合格固定M”只用于明确标记的诊断比较。
数据都已在旧研究中观察过，因此这不是新的独立认证；原cert/test继续关闭。

## 免费oracle与强对照

在Selection/Evaluation分别计算，不混合数据、机器或缓存条件：

- 每个动作使用该分组计时样本的**完整调用均值**；不逐查询挑一次偶然最快计时。
- 先给出满足相同经验retention/Hit约束的固定曲线、原空间exact/ANN、查询无关
  预算混合（期望策略，不假称已部署），再求免费完美分配的整数oracle。
- oracle提前知道每个动作的真实结果，只执行一次固定动作，pilot/特征/控制全免费。
  这是极度乐观的、成本模型里的上限诊断，不是可部署策略或新流水线实测。
- 给出对投影固定/混合与不降维强对照的差值、比例、预算分布、质量、求解gap及
  额外开销允许值。另列匹配原空间ANN实际质量的敏感性，不能替换.90主目标。
- 如果连这个上限都无优势，不进入特征/控制器优化；若为正，也不能自动推断付费
  可达或开始训练。当前授权只覆盖第1/2步。

## 验证、资源和交付

- 小合成CPU回归离线执行。真实输入按封存manifest/SHA与规范化检查；候选/
  原空间真值/Hit核对与计时分离。既有向量距离/ID不同则保存差异，不改旧证据。
- 原空间exact与缓存投影检索逐前缀核查，FAISS输出/质量单独测量，不假定跨架构
  结果位值等价。所有质量均来自本次实际返回集合，不挪用另一后端质量。
- 均时、中位数、描述性p95、配对查询块区间、独立进程峰值内存、setup、磁盘和
  数据访问收据。12条合成计时不充当真实文本证据，不声称正式认证。
- 本地数值线程1、进程RSS<=8GiB、单次计算45分钟上限（传输单独记录）；本地
  新输入/结果<=8GiB且磁盘至少保留12GiB。超限/校验错保留现场并停止，不自动
  缩小语料、调参、改阈值、换机器或重跑。原数据和旧本地合成输入不删除。
- 原始包和新source binding保留，版本化结果证据；更新STATUS/IMPLEMENTATION_PLAN，
  跑保护/差异检查，按常设授权提交推送Step5分支并核实GitHub HEAD。

预计本地计算约5–15分钟，最终按早期实际吞吐更新；6GB传输时间取决于网络，
不合并成已验证的完成时限。不需要A800；若本地资源不够则报告，不擅自申请GPU。
