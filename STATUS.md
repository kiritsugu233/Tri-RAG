# Status

Updated 2026-11-02. Maintained branch: `codex/tls-rag-step5-ann`.
Canonical checkout: `/Users/guanghongxu/Query-Adaptive-Tri-RAG`.

## Benchmark v1：范围获批、实现与本地完整验收（2026-11-02）

- 实际授权：用户对[九路径范围](docs/GAUSSIAN_RAG_BENCHMARK_GAP_AUDIT.md)回复“批准”。新命名空间实现完成，协议见[冻结说明](docs/GAUSSIAN_BENCHMARK_SPACE_PROTOCOL_V1.md)。
- 本地最终完整504/504通过、0跳过、72.571秒；新增23项（含真实HNSW/FastScan、HDF5端到端、真实子进程终止及监控失败回执），旧481项未改。
- 首次502项执行因沙盒禁止ps产生5条错误记录，保留 `runs/gaussian-benchmark-space-v1/local-validation-20261102/full-tests.log`；无沙盒503项先通过，补监控故障测试后最终504项通过，最终日志同目录 `full-tests-final.log`。
- 新本地环境 `/private/tmp/gaussian-benchmark-env-20261102` 通过.pth复用旧固定依赖，仅安装已核验h5py3.11.0；初次未包含旧用户site导致numpy导入失败，补入既有依赖路径后通过，没有升级或修改旧环境/四个native库。
- 完整命令：`TRI_RAG_TEST_PYTHON=/private/tmp/gaussian-benchmark-env-20261102/bin/python STEP5_ROUTE1_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/route1.dylib STEP5_COMPARATORS_V2_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/comparators_v2.dylib STEP5_CONTINUATION_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/continuation_v1.dylib STEP5_CONTINUATION_V2_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/continuation_v2.dylib sh scripts/run_tests.sh`。
- Linux新环境 `/root/autodl-fs/gaussian-benchmark-space-v1/env` 已核验NumPy1.26.4/SciPy1.13.0/FAISS1.8.0/h5py3.11.0，实际FastScan与HNSW seed接口可用；旧环境不变。尚不代表Linux新测试或完整回归通过。
- 官方数据镜像获取进行中，15分钟限额不延长；尚无完整SHA校验或新基准结果。先同步此实现源，再作Linux新测试和数据就绪检查；实验4小时上限不因测试通过自动扩大。
- 保护258项通过，旧科学字节和无关登记保持；治理说明仅追加授权范围，测试入口仅新增模块成员。

## 当前应用研究：真实 benchmark 优先（2026-11-02）

- 用户明确先找真实gap并检验Gaussian质量/成本空间，选择Linux服务器；合成A0暂缓。
- [来源审计与九路径提案](docs/GAUSSIAN_RAG_BENCHMARK_GAP_AUDIT.md)：VIBE HotpotQA完整523万语料、原生recall与严格ID双口径、真实ANN及压缩对照。
- Linux实际14 CPU/120GiB，数据盘约180.9GiB可用；未安装依赖或同步旧checkout。
- 官方下载0bytes失败；诊断为HTTPS连接重置。镜像首页可达但文件未核验。无新检索/索引/测试标签访问。
- Gaussian适配尚未测量。下一步为该提案第5节具体范围批准后实施、完整测试、冻结并运行；不启动旧A0或旧方案4/5。
- 本轮仅L1文档及一条新增L1登记；检查命令为 `python3 scripts/check_code_protection.py`、`git diff --check`、旧登记/文档链接审计；未重复执行历史481项测试。

## 当前交接：方案4已收尾，方案5保留提案、暂不实施（2026-11-02）

- 最新实际答复：“仅保留提案，暂不实施”。方案5状态为deferred/not_run；
  不创建五个科学实现文件、不恢复环境、不读取实验向量或运行R5-A–E。
  保留第41节及既有审计；等待用户以后明确恢复，不持续申请批准或启动后台任务。

- 用户要求检查方案4收尾后开始方案5。已核对当前分支/GitHub HEAD、原科学状态、
  已验证封存索引及八项审查回执；与第39节记录一致。方案4当前方法收尾，不再补跑。
- 收尾保留真实状态：P1为no_candidate；两场景A完整、B决策完整，但额外核查仅5/7，
  原状态仍为resource_stop。未测的20k B-select/剩余seed不变成通过，也不列为自动待办。
- 方案5已完成只读接口审计与[第41节具体R5-A提案](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#41-方案4收尾核对与方案5-r5-a成本预检提案2026-11-02)：
  先量出原空间HNSW完整建图中真正可省的剪枝距离费用，覆盖首次选择和反向重剪，
  再决定是否值得开发Tri比较器。当前没有新建图/投影/检索或新科学结果。
- 第41节列五个新文件、测试入口单项、两份登记的精确范围；该范围本次未批准，已暂缓。
  仅本地CPU、已开放20k语料、无query访问；R5-B–E不随首轮自动启动。
- 环境预检发现旧`/private/tmp/step5-ann-enhancement-env-20261028`无法导入faiss，
  旧头目录缺`faiss/impl/HNSW.h`；不能复用历史467项通过作为本轮验收。
  获批实施时恢复原固定版本/校验过的源码与头，不更新依赖版本或旧原生库。
- 本轮仅四项L1入口/状态/计划/日志更新；科学文件、旧封存、理论框架及治理登记保持。
- 文档复核：旧日志301,304字节完整前缀、65个本地链接/围栏/差异通过。
  全局保护247项通过；其中独立理论任务新增的L1笔记及机器登记保留，不纳入本轮提交。
  本轮没有实现变更，不重复历史467项实现测试。

## 此前独立研究：恢复Gaussian＋Tri-Law的RAG应用，概率桥A0提案（2026-11-02）

- 用户明确已有环构造仍可用于RAG，重点是点对翻转到候选遗漏的差异，并指示“继续”。
  基础构造新颖性审查保留，应用研究恢复；此前停止不再被解释成应用方向的停止令。
- 已只读审查Raw Tri-Predict mean-field与旧Markov桥／实际质量费用问题，形成
  [应用研究说明](docs/GAUSSIAN_RAG_RANK_BRIDGE_RESEARCH.md)。旧科学源、旧日志和封闭数据不改不重开。
- 非冻结符号结果：给定共享目标投影长度的精确边际，配合经典Bernoulli边际尾界后积分，
  无需竞争者独立即可得遗漏区间；理论上嵌套于原无条件边际区间并不松于Markov。
- 手算m2/n4/beta2/正交残差：真实top-4遗漏1/15，无条件独立近似1/81，低估5.4倍；
  条件区间仍为[1/48,1/3]，保留剩余结构不确定性。这不是新MC或已跑测试。
- 第8节已给精确八路径A0：8个单位球几何、16组/48单元、两路各2^16样本、统一224项区间，
  K8192单调积分、600秒/1GiB/64MiB。当前 `implementation_not_approved`，未创建新科学路径或采样。
- 本轮只改L1研究／导航与新增一行L1登记；原481项完整验收保留，不重复实现回归。
  检查命令：`python3 scripts/check_code_protection.py`、`git diff --check`及一次性旧字节／登记／文档审计。
- 实际251项保护、80本地链接、22锚点／围栏通过；旧250条登记和根元数据／科学来源／T1索引保持。
- 当时下一步为A0；现被顶部benchmark优先指示取代，A0暂缓。

## 独立排名尾部研究：后续先行审计，当前候选止于机制笔记（2026-11-02，历史检查点）

- 用户在T1完整验收后指示“继续”；完成进一步一手先行与非冻结符号审查，见
  [独立笔记第12节](docs/JOINT_RANK_THEORY_T0.md#12-后续先行与证明审计保留机制笔记停止当前候选升级2026-11-02)。
- 更直接的先行：Urschel等2017 §3 Lemma2正是本轮缩放后的正负闭合边矩阵对；
  `K=R/2, alpha=r/2`，T1的r=1/4对应alpha=1/8。补读Rising等2015 Thm3.12/3.14。
- 未发现需修L0实现的证明缺陷；高斯排名转换仍需本笔记的卷积、目标长度积分及球提升，
  但目前不足以建立独立发表增量。贡献门判 `covered_or_corollary_only / corollary_only`，
  保留 `novelty_not_established`；没有声称先行逐字包含同一排名下界。
- 补记既有同真子集bit律的容斥推论：不同top-M风险差方向交替，不构成统一安全排序。
  这是符号解释，不改变冻结m=2/M=n核验范围或新增科学测试。
- 保留全部T1证书、失败和481/481完整验收；停止当前JRC-CYCLE-1的T2升级。
  Gaussian类锐区间／同信息经典界增量／低成本摘要条件仍未解，不自动换接口或扩实验找正结果。
- 本轮仅四项L1研究／导航文件；无新MC、数据访问、环境修改或旧Step5工作。
  验证命令：`python3 scripts/check_code_protection.py`、`git diff --check`及一次性文档／旧字节审计；
  实际250项保护、72个本地链接、20锚点／围栏、旧正文／来源／交付索引检查通过。
  无实现变化，不重复481项回归。检查后按常设同步指令提交当前分支并核对远端HEAD。

## 独立排名尾部研究：环境恢复获批，T1完整验收通过（2026-11-02，历史验收）

- 最新实际指示：“补齐完整测试，批准”，批准[T0笔记第10节](docs/JOINT_RANK_THEORY_T0.md#10-为完成全仓验收所需的最小环境恢复范围待批)的专用临时测试环境恢复；此前缺环境失败保留。
- 新环境 `/private/tmp/joint-rank-t1-test-env-20261102`，仅安装faiss-cpu1.8.0和scikit-learn1.6.1；
  官方PyPI arm64 wheel SHA256核对，no-deps/require-hashes/no-index安装，现有NumPy1.26.4/
  SciPy1.13.0/joblib1.4.0/threadpoolctl3.4.0复用；用户全局包、旧不完整环境及仓库依赖文件不改。
- 四原native库哈希保持且均成功加载；新旧FAISS extension逐字节相同，不重建原生库或恢复方案5头文件。
- 从 `/private/tmp` 执行完整测试：**481/481通过、0跳过**；实际收集step5=283、legacy=198，
  含新增14项。测试本体70.081秒，总调用70.949944秒；不是只跑新增或部分子集。
- 重放：设置 `TRI_RAG_TEST_PYTHON=/private/tmp/joint-rank-t1-test-env-20261102/bin/python` 及
  原四个STEP5 native变量后，`sh /Users/guanghongxu/Query-Adaptive-Tri-RAG/scripts/run_tests.sh`；
  精确环境变量/命令/哈希在[T0笔记第11节](docs/JOINT_RANK_THEORY_T0.md#11-环境恢复批准与t1完整验收2026-11-02)及新验收回执。
- 新回执：`runs/joint-rank-theory-v1/t1-20261102-validation`，完整日志SHA256
  `19594051218790c19ee84ea1d2de5ed7200ab9740be10a1031ab932e9d73d4da`。
  原11项T1证书/失败交付索引 `0102a374...092afc4` 保持，不覆盖或改写原失败。
- 最终验收回执9文件/66,529B，交付索引SHA256
  `92e3a3e9c272a31e34df07c67aafabf5a79fe312e2f9cb7f5e912341d327847d`；250项保护、312本地链接/
  19锚点/围栏、旧247项增量及科学/原生来源审计通过，旧全局依赖和证据保持。
- T1确定性实现验收完成；新颖性仍未建立，没有启动T2联合排名采样、真实数据或系统实验。
  按常设指令完成保护/登记/来源审计后，在canonical checkout现有分支提交并同步GitHub。

## 独立排名尾部研究：T1初验通过／全仓环境缺项（2026-11-02，已恢复的历史检查点）

- 实际批准：“批准，你的意思是tri-law已经有人用过了吗”，承接a74e45b的六路径T1提案。
- 三项新L0模块/测试/[协议](docs/JOINT_RANK_CYCLE_T1_PROTOCOL_V1.md)已实现，测试入口仅追加一项；
  治理登记只新增三行/真实批准与获批人工表哈希。旧科学源码/测试/协议/依赖版本及旧日志不改。
- 新14项确定性测试全通过、0跳过，0.038秒；受监督核验及证书生成0.124671秒，
  观测峰值RSS含监督35,553,280B。四固定(n,r)、八组正负几何和单位球Gram已保存。
- 证据：`runs/joint-rank-theory-v1/t1-20261102-bf190bff`，没有MC/数值积分/新查询或系统实验。
- 独立Schur补审计通过八组PSD/秩与重建；证据11文件/105,007B，交付索引SHA256
  `0102a374e530df9cdfb703b1bf21450972ea6172813d342cff476d98d092afc4`。
- 全仓命令为原固定解释器＋四native库下 `sh scripts/run_tests.sh`；实际1.305697秒，
  返回码2，两个旧route4模块缺faiss而收集失败。467是旧检查点，481是新增后的预期，均不写成本轮通过。
- 环境只读核对：旧venv缺pyvenv.cfg及FAISS Python文件；全局sklearn1.4.2而旧代码需1.6.1；
  NumPy1.26.4/SciPy1.13.0可用。未安装环境或更换依赖，未跳过测试/改导入绕开失败。
- 当前 `T1_new_checks_passed / full_regression_blocked_environment`；实现保留本地，必要全仓检查
  未通过前不提交/推送。本轮不标T1完整完成。最小额外环境恢复范围见[T0笔记第10节](docs/JOINT_RANK_THEORY_T0.md#10-为完成全仓验收所需的最小环境恢复范围待批)。
- Tri-Law边际经极化等价于Durrant–Kabán2013 Thm3.2，原文用于分类翻转；
  这不代表已确认其使用Tri-Law名称或已解决本轮联合排名问题。发表新颖性仍未建立。
- 交付前文档/来源审计通过：250项保护，旧247项仅获批人工登记哈希改变；310个本地链接、
  17个锚点/围栏及全部证据哈希一致。测试入口仅多一个成员，额外环境目录尚未创建。

## 独立排名尾部研究：T0首轮文献／符号审计（2026-11-02，历史交付）

- 用户最新指示将研究交给新agent并考虑与Step5隔离；本聊天独立接手，沿用canonical checkout/
  现有分支保存，旧Step5探索日志只读；没有向旧聊天发消息或创建分支/worktree。
- 产物：[独立T0研究笔记](docs/JOINT_RANK_THEORY_T0.md)。复核旧1/3对1/6、条件非中心chi-square
  及球提升线性条件；指出原反向位移反例不能通过统一缩放/平移成为单位球反例。
- 核对框架主要一手材料的指定段落，补入Urschel等2017环/主子式辨识先行；1977论文仅取得
  出版者摘要，完整表达仍待补。不能把完整Gram、条件混合或环符号本身作为新贡献。
- 唯一候选JRC-CYCLE-1：m=2、M=n、n环残差；全部真子集反转联合律相同，
  排名尾部却有正间隔。符号草稿给出正卷积式、显式下界与d=n+2单位球实现；
  n=3/beta2/r=1/4的间隔下界3/16384是解析量，不是实测值或Gaussian全类锐界。
- `novelty_not_established`；候选的排名概率定量转换与单位球实现有清晰可核对对象，
  尚未建立可发表增量。六路径最小确定性T1提案已具体写明，未批准/实现；没有MC或T2实验。
- 只改新L1笔记、启动/状态/里程碑导航及机器清单的一项新L1登记；旧246项/根元数据、
  人工治理登记、科学源码/测试/协议和旧日志保持。保护检查、文档/差异/增量审计后同步。
- 验证命令：`python3 scripts/check_code_protection.py`、`git diff --check`，以及一次性只读
  Markdown/登记/旧字节审计；本轮未重跑467项实现套件，未将旧测试数记作本研究通过。
- 实际文档检查通过：247项保护、63个本地链接/围栏、旧246项/根元数据不变、五路径交付；
  共享HEAD变化导致的两次范围审计失败已记录，最终基于19a22e7复核，保留方案5暂缓记录。
- 下一步：只复核JRC-CYCLE-1及最近先行，按笔记第7节获得精确L0批准后才执行确定性T1；
  若仅是已知工具的无实质差异推论，交付机制笔记并停止，不改研究接口寻找正结果。

## 共同Gaussian投影排名尾部：研究框架交付（2026-11-02，历史交付）

- 用户实际要求：“给出一个该研究问题的研究框架以及可agent执行的md文件”。
- 新增[研究框架与Agent说明](docs/TRI_LAW_JOINT_RANK_RESEARCH_FRAMEWORK.md)，
  用固定k=1、共同Gaussian、rank=1+Z定义唯一概率对象；列明信息层级、可实现性、
  单位球限制、条件可解参考、经典矩/两两边际/树概率界及T0–T4决策门。
- 提供可直接复制的启动任务、待批精确实现路径、建议数值/资源上限、证据与复核清单。
  首项为先行结果与主命题审计；新L0科学实现/协议/模拟尚未批准或执行。
- 当前交付仅为框架及导航；日志第40节记入本次范围。没有自动派发其他聊天、
  新检索/建图、真实数据访问、科学模拟或论文新颖性成立结论。
- 机器清单仅新增一项L1登记，人工L0登记及旧245项登记/等级/哈希与科学源码/依赖/结果保持；
  保护、差异、登记增量、旧日志完整前缀和文档链接/围栏检查后按常设指令同步Step5分支。
  实际检查通过：246项登记、仅一个新增L1条目、74个本地链接；人工治理文件原字节保留。
  文档交付不重跑467项实现套件，不将既有测试数记成本研究通过。
- 下一步：接手者按框架T0先形成文献对照、唯一命题卡及最小L0提案；
  无新颖差异则报告`novelty_not_established`并停止，不自动换接口寻求正结果。

## 两场景实际结果：A无生命周期优势，B无决策增益／核查部分完成（2026-11-01）

- 科学源`1880435`；本地`runs/route4-scenarios-20261101-v1`，详见[第39.3–39.7节](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#393-实际执行范围与资源终止a完整b核查部分完成)。
- A完整：七库203个固定recipe，Gaussian有16个质量达标点，却在Q1/10/100/1000/10000
  全35个生命周期点无赢家。独立曲线求交即使只对原维两对照也无Gaussian胜出区间。
  每库10次query，2k原维17.26–18.72ms对最优达标Gaussian26.09–30.03ms；5k为
  31.52–33.96ms对47.35–48.95ms。20k无合格Gaussian；拟合/选配/交付费用均报告。
- B七库42项顾问决策全部封存。Tri区间13.7956秒，对直接三轮1.2788秒，约10.79倍；
  63动作标出8个高期望风险、0低风险，仍构建21模型/672投影扫描，与直接三轮相同。
  14组Tri/直接三轮部署模型逐字节绑定相同，未改善推荐或减少昂贵工作。
- B实际select完成六库36项，均按各自既定推荐/回退满足门槛；20k这一步未执行。
  额外seed核查5/7完整，第六截断、第七未运行；不得把缺测记成通过或完整B验收。
- 349.469秒后`resource_stop: artifact_bytes_limit`，峰值观测1920.406MiB；科学档案
  2533项/2,176,434,106B≈2.027GiB，2GiB轮询阈值超调约27.61MiB。低估重复明细体积，
  非设备磁盘不足；已停自建进程、保留截断与失败，未扩预算/删旧档或重跑。
- 独立核查448原真值、77,920候选精排、104,320时钟、1015生命周期、4032区间及全部
  2533项文件哈希一致；九项冻结源保持。额外五库FP32/64配对最终/质量差异0，边界
  成员差异4条仍保留。理论对象/部署浮点实现、有限8seed与总体期望分开报告。
- 科学索引SHA`737a7b85d87b082270a443c209799247be020d6f0bdf72fce9930c9c6af24872`；
  审查回执在`runs/route4-scenarios-review-20261101-v1`。未建议自动补跑；原report/cert/test
  关闭，formal_certification=false、fresh_independence=false；当前接法不建议追加投入。

## 两个应用场景v1：已批准、实现验收完成（2026-11-01）

- 实际答复：“批准第38节全部范围，按A→B执行”。八路径范围已实现并登记，旧科学文件不变。
- 完整467/467、零跳过、69.212秒；新22项，Step5=269/legacy198。真实合成worker链、
  区间计数穷举、模型/输入/选择封存、原生PQ ties、共享probe和真实资源停机均通过。
- 首次完整调用缺四个既有native库变量而skip29项，保留日志；最终完整环境已补齐，未改依赖。
- 将冻结/推送源码后运行`runs/route4-scenarios-20261101-v1`；按A→B独立评价，资源45分钟/
  4GiB RSS/2GiB产物。此检查点尚无真实新场景结果；原report/cert/test保持关闭。
- 保护245项通过；详细实现/验收命令见[第39节](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#39-两个场景v1实际批准与执行记录2026-11-01)。

## 两个应用场景：预检完成、新试验范围待批准（2026-11-01）

- 用户要求分别验证临时库总成本、Tri离线风险决策价值；[探索日志第38节](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#38-两个应用场景已有证据预检与独立小试提案2026-11-01)列实际预检及八路径提案。
- 旧P1没有同质量Gaussian或逐库独立建库时钟，不能据阶段耗时宣称生命周期优势。
  新验证必须加入无需训练的原维Flat，并把拟合/编码/选配置/确认/查询费用计算完整。
- 旧36个Gaussian风险上界均无法证明低风险；高上界不是高损失证明。旧配置也全失败，
  无法评价误拒绝安全配置及有效选择覆盖。只读预检未启动新检索/构建或读report。
- 提案为本地CPU七个2k/5k/20k库、历史advice32/select32、全库候选、独立A→B；
  m64/128/192、PCA另96、M128/512/1024，原维/Gaussian/PCA/PQ及真实总费用。
  B拟比较合法上下界三态与1/3轮共享试验、跨方法三probe和总是未决基线；不训练。
- 精确新增五文件、测试入口单项及保护登记窄幅变更列于第38.2节；新科学总上限45分钟、
  4GiB RSS/2GiB产物，无SSH/GPU/1M/新数据下载。尚未获得本节具体路径/协议批准。
- 预检证据`runs/route4-scenarios-preflight-20261101-v1/existing_evidence.json`，SHA256
  `434527ceea6c3df5c3d1640fad97f8e44deaf61aeab7be4a9358c62f1e5ca300`；原科学源/封存不变。

## 方案4 P1原因复核（2026-11-01）

- 按用户要求只读分析已开放结果；见[探索日志第37节](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#37-p1原因复核可修复性与质量成本边界2026-11-01)。没有新检索、变更m/M、训练或打开report。
- 边界点对的相对距离差中位0.9505%，Gaussian192差值噪声标准差中位13.3375%；
  这是点对排序脆弱的解释，47.18%点对反转不等于top-256遗漏概率。
- 八seed的32次遗漏各取一个已保留边界竞争点，SVD精确分解发现29/32的比较在去除
  随机奇异值伸缩项后符号转正；独立Gram特征分解误差<2.45e-15。
  只解释这些已观察比较，不是正交投影的新检索/召回结果，也不保证全部遗漏恢复。
- 旧成本网格中，完整均值不慢于PQ的Gaussian配置20个，质量最好者八seed新增损失
  45.2734%；达到至少10%省时的14个中最好64.4141%。这是已测固定配置的事后描述，
  不是新控制器/连续动作空间的不可能定理。最大维路径修好质量后仍需跨越约2.69倍成本差。
- 聚合界丢失联合尾部信息；收紧界不改变候选集合。扩大M/维度、正交或数据感知表示
  可研究，但尚无Gaussian同时通过质量/净收益证据；PCA/PQ仅在已开放select上通过。
  当前C的93.75%全局保留上限、历史数据限制及认证false保持。
- 本地只读诊断：`runs/route4-cause-review-20261101-v1`，索引SHA256
  `cca873c82551c299a50427c2797d70af5987f207f6b4cbac2991dced415665b4`；原P1交付索引不变。
  只改三项L1文档；不改科学源/配置/测试/保护登记，不重复445项实现回归。

## 方案4 P1完成：无可行Gaussian配置（2026-11-01）

- **P1执行完结、科学门槛未通过：no_candidate；停止本动作空间的P2顾问投入。**
  科学源`b38ad51`，本地`runs/route4-screen-20261101-v1-recovery`；主监督137.777秒，
  峰值观测528.922MiB，原科学封存452项/1,571,700,160B；无超限/运行时错误。
- 20k库；96历史查询仅导出，实际advice32/select32已评估，report32未检索/算真值/计时。
  八设计seed、选择质量661组、157个配置独立完整计时，各query五次。
- 选择集C自身保留93.75%；同C最佳合格PQ M128：0.322162ms/p95 0.397861ms、0新增损失。
  原维FP32完整扫描：1.214284ms；合格PCA96/M256：0.502439ms。
- Gaussian192/M256/FP32：计时seed0.865304ms、额外漏4/320；八seed漏数
  `[4,4,2,5,8,5,2,2]`，平均新增损失1.25%、最差query20%，均不满足0.5%均值/10%尾部。
  免费选M的六维度×两dtype共12个oracle全不可行；不扩大M/维度或追调seed。
- 全竞争者1,518,848对，Tri几何/概率计算2.459秒；advice192/M256计数上界9.560%
  对比八seed实际均值1.758%，界合法但松，不是精确召回预测或独立认证。
- 独立复核64条NumPy原真值、39,584条质量记录、19,973个不同E的精排、25,120个计时
  样本及全部452项科学文件，均一致。FP32/64仅候选内部顺序差异，无成员/最终结果差异。
- 交付索引覆盖465项/1,571,898,774B（原452科学项保留；索引自身另计），SHA256
  `f1f347609a993ba26dcf2ef4c89ca42ce0bc0c6332aa65e37db81d434b42d5dc`；详见日志36.6–36.10。
  `formal_certification=false,fresh_independence=false`，不宣称整体ANN加速。

- 初次105e2f9入口失败于ps权限，科学子进程在读取输入前退出；现场保留，无新检索/真值。
  旧资源测试过度替换Popen，未实际验证ps，这一覆盖缺陷已明确纠正。
- 修复后真实ps/子进程停止、监控异常封存和共享epoch截止时间均验证；完整445/445、
  0跳过、57.439秒（P1共16项；Step5收集247、legacy198）。旧科学参数不变。
- 后续唯一有效科学run使用`runs/route4-screen-20261101-v1-recovery`；确认旧进程为0，
  不复用旧输出，从原45分钟预算保守扣60秒，新监督最多2640秒。最终结果见本节顶部。

- 用户实际指示：“继续P1”，恢复已呈示的第34节八路径范围；旧暂缓决定仍作历史保留。
- 新增五个隔离文件、测试入口单项归类及窄幅登记已完成；不改旧科学源/配置/依赖。
- 初次新增13项及完整442项通过，0跳过，72.162秒；其资源监控覆盖缺陷见上，
  最终16项P1/完整445项验证才包含未替换的真实ps及停机/失败回归。
- 初测保留两项发现：浮点除法再乘回不应要求逐位还原；当前系统拒绝killpg但允许终止
  自建子进程PID。现验证共同Gaussian原始抽样与已追踪PID树停机，未放宽科研阈值。
- 最初计划root的启动失败完整保留；有效恢复run及其负结果见上。科学累计45分钟、
  4GiB RSS/3GiB新产物边界保持，原cert/test关闭；未训练顾问/adapter或执行其他路线。
- 现有四项未提交L1方案5记录保持原字节内容；本轮只将其完整前缀保留并追加第36节。
  保护240项、旧235项等级/旧科学哈希、差异/文档与来源检查在提交前核对。

## 方案5记录：离线建图接口（2026-10-31）

- 用户实际要求：“把这个作为方案5写进去，你检查一下repo中用于存Step5的md文件”。
- 已将讨论追加到[探索日志第35节](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md#35-方案5离线建图时的邻居选择与剪枝2026-10-31)：
  离线HNSW邻居选择/剪枝，区分研究采样与真正建图，拟用Gaussian比较/精算回退，
  最终仍原空间查询。新增参数/比较器、条件风险及图质量桥梁均未验证。
- 记录R5-A成本筛查→固定现场→便宜比较器→真建图→外部查询验证；完整费用、
  已知精确A的强对照、自适应选择与尾部错误、停止条件和后续审批要求均明确。
- 当前状态planned/not_run；只改四项L1文档及导航，不改源码/测试/配置/保护登记，
  不开启新数据或实验。原日志完整保留，方案4“保留P0，暂缓P1实现”决定不变。
- 文档检查：`python3 scripts/check_code_protection.py`、`git diff --check`，
  并核对旧日志完整前缀、四路径L1范围、本地链接/新锚点及代码围栏。
  全部通过：235项保护登记、53个本地链接、新第35节锚点及围栏；旧日志完整字节保留。
  该设计记录保存时只在本地，后于105e2f9原文纳入Step5提交；没有新建图结果。
  方案5记录时未运行429项套件，本轮独立P1的445项结果见顶部。
- 下一步仍需用户决定具体研究/实现范围；本次记录不自动执行方案5或恢复其他路线。

## 方案4后续研究与 Step 5 交接（2026-10-30）

- P0已完成并追加[第33–34节](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md)：
  独立核对2013内积符号翻转律与当前距离反转边际的代换、候选计数界、共同投影反例、
  SRS/PM-LSH及压缩基线差异、可用输入与成本接口。378组表达式核对最大绝对差
  `4.551914400963142e-15`；2880项小集合检查通过。这不是新检索算法验收。
- P0证据目录：`runs/route4-p0-review-20261030-v1`。只校验既有100k向量/ID与768条
  历史已开放tune导出包身份，未抽新子集、检索、建索引、训练或访问封闭角色。
- 该10月30日检查点的实际答复：“保留P0，暂缓P1实现”；11月1日恢复授权及结果见顶部。
- 第34节为未批准/暂缓的P1精确提案：本地CPU、20k/96条历史查询、独立IVF候选池、
  Gaussian/PCA/PQ与完整原空间对照、免费M上限、45分钟科学运行上限。
  包含五个新文件、测试入口单项归类、人工/机器登记的窄幅变更；当时尚未批准/执行。
  不因P0数学成立而声称界足够紧、系统有收益或整篇论文无新意。
- 用户实际指示：“可以，把下一轮研究内容写进方案4的后续里面，然后交付给Step5的agent让它继续按这个路线进行下去”。
- 已追加[探索日志第32节](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md)，
  保留第21节原方案与全部路线1/2历史结果；更新启动入口及里程碑。
- 主线为离线联合`(m,M)`配置＋独立候选池上的Gaussian批量筛选；
  同预算独立视图并集为条件后续。接手聊天：“TLS-RAG Step 5 ANN 实验方案”。
- 首项P0独立复核与新文件/API、角色/输入、参数/资源及验证的L0提案已完成。
  已有研究方向指示无需重复确认；尚未给出的精确受保护实现范围仍按AGENTS审批。
- 当前产物是研究/交接文档。方案4仍无ANN运行、训练、索引构建或新数据访问；
  6000组公式表达核对属于此前只读分析，不是方案4算法验证。
- 文档检查命令：`python3 scripts/check_code_protection.py`、`git diff --check`，
  并核对原日志为新日志的完整前缀、仅五项L1文档变更及本地Markdown链接/围栏。
  本轮全部通过：235项保护登记、56个本地链接；旧登记及科学L0内容未变。
  本次不重跑运行时套件，不将文档校验写成新算法通过。
- 本轮交付至P0结束；P1、后续P2→P3→P4均不自动执行。等待新的用户指示与相应
  精确范围批准。不自动回到旧HNSW逐节点过滤/停止接法，不开启原方案3或旧封闭角色。

## 已执行的测试精简方案 B（2026-10-30）

- [中文审批方案及430方法逐项表](docs/TEST_SUITE_SIMPLIFICATION_PROPOSAL.md)。
- 用户实际批准：“执行”，承接该方案四路径范围；父提交`7b9d44b8c85bbfb964afcaaf8fdbd1c2485ae299`。
- 230项方法保留、2项等价合并为1项、198项退出日常且完整保留。永久删除0项。
- 无参数/`full`完整429；`step5`日常231；`legacy`历史198。实际收集互斥且并集完整。
  部分套件打印未选范围，不能记作全仓通过；所有实现交付前仍必须完整回归。
- M01四场景保留；三个临时错误接受故障均被检出。其余45文件及同文件其他五方法原样。
- 全集收集/导入错误、未知/缺失测试文件、重复ID、未知套件和多余参数验收均会失败；
  仓库外调用及无参数/full等价核对通过。没有新增仓库测试模块或测试方法。
- 固定依赖/四原生库，从`/private/tmp`无参数运行：429/429通过，0跳过，65.411秒。
  显式Step5：231/231通过，0跳过，39.611秒；历史：198/198通过，0跳过，25.973秒。
  缓存测试单独6/6通过；两次部分运行均明确报告完整验证为false。
- 六组科研来源绑定（55/22/13/8/7/10项）保持一致；科学源/配置/依赖及旧结果未变。
  原235项等级/理由和批准历史保留，仅两个获批L0内容哈希及本次实际批准追加。
- 精确重放命令、历史子集结果、日志SHA和兼容限制见[方案B验收](docs/TEST_SUITE_SIMPLIFICATION_PROPOSAL.md#八实际批准与实施验收2026-10-30)。
- 当前产物为现有启动器、等价合并测试及登记/交接文档；用户未跟踪档案保持原样。
  该测试精简检查点原待用户指示；当前研究交接见顶部。进一步永久删减、科研算法/
  部署安全修复仍需独立范围批准。

## 已批准的方案 A：保护检查与登记修复（2026-10-30）

- 实际批准：“批准执行，执行完毕后更新github上的repo，不要另起branch，直接在step5的分支上提交”。
- 父提交 `61b92d7871ca3da703c137abb4d49cdf9eca487e`；继续 Step5 分支。
- 修复 C/C++/头文件和 ignored mechanism 漏检；补齐人工表 44 项，已有等级不变。
- 原 10 个治理测试方法保留，仅新增 1 个参数化方法，覆盖 30 个原生文件/Git 状态组合；
  ignored Python 覆盖扩展为五个目录，并验证被忽略的载荷目录不被扫描。
- 四个获批 L0 文件范围、批准记录和验收见
  [方案 A](docs/REPOSITORY_CLEANUP_REVIEW.md#proposal-a-protected-guard-and-register-repair)。
  原失败证据保留；科学代码/测试/配置/依赖不变。五组 Step5 绑定一致；
  A800 来源清单含保护脚本，其身份随获批编辑改变，旧包按原提交复现，不刷新旧哈希。
- 原检查器执行最终新回归可复现失败；修复后治理测试 11/11 通过。
- 完整固定环境命令：`TRI_RAG_TEST_PYTHON=/private/tmp/step5-ann-enhancement-env-20261028/bin/python sh scripts/run_tests.sh`，
  同时设置审查记录列出的四个 STEP5 原生库环境变量；430/430 通过，无跳过，64.757 秒。
- `python3 scripts/check_code_protection.py` 通过，234 项；人工表完整，
  仅三个获批哈希变化。日志/失败证据及验收记录已保存。
- 该A检查点尚未批准/执行方案B；后续独立批准与执行见本文顶部。A的原验收保持历史含义。

## 上一轮：仓库结构审查与整理


- Review baseline: `50defc35bc9f42f5b24d2dedf5f45a89e51257ad`.
  [Review/approval proposal](docs/REPOSITORY_CLEANUP_REVIEW.md),
  [file/dependency map](docs/REPOSITORY_MAP.md),
  [evidence](docs/repository_cleanup_evidence.json).
- Shortened startup/status/plan navigation; retained detailed history in the fixed
  pre-cleanup Git snapshot. Corrected the outdated next-agent state and retired
  Step 4 branch instruction. Added a repository landing page and lifecycle map.
- Test launcher accepts `TRI_RAG_TEST_PYTHON` and resolves the repository from its
  own location. Suite, assertions and native tests are unchanged.
- No runtime source/config/test/dependency removed or modified; no L0 hash refreshed.
  Old experiments still bind those paths. User archives/report_work/runs preserved.
- Confirmed checker blind spots for new native files/ignored mechanism code;
  human protection register omits40 machine-registered paths. Exact four-path
  protected repair was subsequently approved and implemented as A above.
- Baseline default environment:429 tests,4 missing-FAISS errors,33 skips.
  Existing fixed-version/native environment:429/429 pass,0 skips (66.634s).
  Final launcher from `/private/tmp`:429/429 pass,0 skips (65.151s). Protection234,
  all61 Markdown checks and six unchanged scientific binding maps pass; receipts
  are in the review and evidence file.
- That checkpoint next requested approval for A, now received/executed above. No new
  retrieval, policy fitting, data access, cluster run or scientific claim follows.

## Latest scientific checkpoint — continuation v2

Scientific code `08ed4ce`; exact prior consent:
“批准第26节全部范围，按A→B→C执行”.
[Journal sections27–31](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md)
hold the full process, commands, source/selection/model checks and limitations.

- Historical development:768 queries,4 nested folds,5 timing repeats.
  All768 FAISS/boundary resume checks pass;429 offline tests passed at that milestone.
- Full original ANN=1.168709ms; S=1.248047/G=1.393090/T=1.629399/
  T0=1.657511/selectiveTri=1.415616ms. Current learned/Tri paths fail net gain.
- T retention=.9971354 vs original=.9984375; its delta interval crosses−.002
  tolerance, with one extra5/10-neighbor tail loss. No trained-support/OOD
  abstention fallback; execution is complete, deployment safety acceptance is not.
- Preserve resumable backend and failures; no automatic retry/threshold change,
  adapter/1M/routes3/4, SSH/GPU or closed-role access. No fresh independence or
  formal certification. Route4 was planned at that checkpoint; its later P1
  result is recorded above. Route3 has not run.
- Latest raw evidence: `runs/ann-continuation-v2-20261030-v1`,1658 previously
  verified files/2,149,374,244B. This cleanup did not re-audit or rewrite them.

## Historical records

Detailed status/approvals/commands from all previous stages are retained in
[STATUS at the cleanup baseline](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/STATUS.md), including Step 4 replay,
Step 5 cloud/local storage receipts and original failures. Retrieve offline with:

```bash
git show 50defc35bc9f42f5b24d2dedf5f45a89e51257ad:STATUS.md
```

The unchanged [exploration journal](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md)
and versioned contracts remain local. [Milestone index](docs/IMPLEMENTATION_PLAN.md)
links the historical plan. Historical text describes its original authorization;
read current AGENTS and the user's task before acting.

## Branch-retirement provenance — 2026-10-23

The eight remote references below were previously retired with user approval
and ancestry/open-PR checks. This cleanup performs no additional branch deletion.
Their complete original execution receipt is in the fixed status snapshot above
and `docs/code_protection.json`.

| Retired GitHub branch | Retained tip (full SHA in protection registry) |
|---|---|
| `codex/calibrated-tri-predict-v2` | `26079ce67793844f649d7df16ce818c3b539529f` |
| `codex/calibrated-tri-predict-v3` | `f94c1aa63df459b4d2281e296f400942a8552c00` |
| `codex/tls-rag-step3` | `3913ded9b786d6b6b15c734c750bd13d9c58e12c` |
| `codex/tls-rag-step4-readiness` | `fc38d928e428dc8c10a3ea16e5ec5db5ccfbf72a` |
| `codex/tls-rag-step4-real-probe` | `c08cd3b6800888c8d2a5a5f8c58a77d883a84ede` |
| `codex/tls-rag-step4-retention-v3` | `4f836375c8e45361ef7e3229e66f2c1735abee1b` |
| `codex/tri-law-sequential-rag-step2` | `f46ce73beccf5fddbf05fd87fdb0911318020add` |
| `codex/tri-law-sequential-rag-v1` | `cac654ed73f75db92a1d11c09d10e9cd9973a37f` |
