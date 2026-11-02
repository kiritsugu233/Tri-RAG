# Gaussian＋Tri-Law用于RAG：从点对翻转到候选遗漏

日期：2026-11-02，Asia/Shanghai。入口：`c1b7df4f9842104a62c59ae7b87a7f91c1eab6de`。
等级：L1非冻结研究说明／符号推导／待批实施提案；不是已获批的新科学协议。
状态：`application_research_resumed / analytic_draft_ready / implementation_not_approved`。

## 1. 当前任务与停止范围的纠正

用户在环构造先行审计后明确提出：

> 我觉得这个虽然已经有了同一环符号构造，但是我仍然可以使用这个来应用在RAG上面，以探索gaussian方法+tri-law的可行性。但是前提是解决之前方案存在的问题，单个点对的翻转与最终超出候选范围概率的差异。符号推论的结果代表着什么

解释之后，用户指示“继续”。因此应用可行性研究恢复；不能因为构造已有先行就停止应用研究。
[旧笔记第12节](JOINT_RANK_THEORY_T0.md#12-后续先行与证明审计保留机制笔记停止当前候选升级2026-11-02)
关于独立基础理论贡献的判断保留，只限定JRC-CYCLE-1本身，不作为本应用任务的停止令。
本轮完成只读源码／历史证据审计、符号分析和第8节具体提案；没有执行新科学代码或采样。

唯一目标：在固定Gaussian表示下，明确候选遗漏的正确概率对象、可用信息和误差，
检验能否以足够低的代价支持候选预算决策。已知工具可以使用，应用价值需由效果与费用证明。
保持canonical checkout和既有分支；旧Step5日志只读，旧report/cert/test及未跟踪归档不打开。
不恢复旧方案4/5实验、不改表示、不训练预算器、不调用LLM回答。

## 2. 旧方案到底缺哪一步

只读核对[Raw Tri-Predict源码](../src/tri_rag_harness/tri_predict.py)的
`tri_predict_h_j`、`solve_y_star`、`tri_predict_retention`，以及旧日志§§33.3–33.4、37.4、39.5–39.6。

| 层次 | 当前事实 | 本研究要处理的缺口 |
| --- | --- | --- |
| 精确点对律 | `tri_law_probability`给固定几何、随机Gaussian矩阵的单点对反转率 | 不是候选遗漏率；本轮不修改旧API或边界 |
| Raw Tri-Predict | 源码明确包含LID距离模型、正交／条件独立、结构近似和mean-field门槛 | 条件平均计数的门槛事件不等于随机计数的尾部；不能把该近似叫精确Tri-Law |
| 旧候选池风险界 | 期望线性性＋必要计数条件＋Markov，数学上不要求翻转独立 | 保留均值后丢失联合尾部；合法上界可能无法作预算决策 |
| 旧质量／费用失败 | 已支付的几何顾问费用、候选实际遗漏和同模型选择均有记录 | 更准的风险预测不会改变已算候选；不据此撤销旧no_candidate/resource_stop |

本轮不修Raw公式、不改变它的冻结行为或历史对照；新命名空间直接定义自己的概率对象。
首轮先分离“真实几何下的概率桥”与“线上如何估计这些几何输入”，避免把输入误差、
条件结构假设和尾部近似同时塞进一个校准系数。

## 3. 保留共享目标长度的条件律

首轮k=1。固定 `a=x0-q`、`bj=xj-q`，`A=||a||²>0`、`beta_j=||bj||²/A>1`。
写 `bj=alpha_j*a+rj`，`rj`垂直a，`s_j/A=beta_j-alpha_j²>=0`。
投影行独立N(0,I/m)，投影后不归一化。令

```text
Ij = 1{ ||Pi*bj||² < ||Pi*a||² }, Z=sum Ij,
theta_M = P_Pi(Z>=M), 1<=M<=n,
T = m*||Pi*a||²/A ~ chi_square(m).
```

给定 `Pi*a=w`，`Pi*rj`的边际为 `N(0,(s_j/m)I)`，独立于w。
不要求不同rj相互正交，就有准确条件边际：

```text
pj(t) = F_noncentral_chi_square(
    t/(beta_j-alpha_j²); df=m,
    noncentrality=alpha_j²*t/(beta_j-alpha_j²)).
E_T[pj(T)] = tri_law_probability(beta_j, alpha_j/sqrt(beta_j), m).
```

给定w时的联合反转律只依赖其长度：行空间中的共同正交旋转保留竞争距离，
所以条件于T可以使用同一边际。`s_j=0`时严格较远条件使 `|alpha_j|>1`，该指示几乎处处为0；
单列该分支，不除零。非中心参数与SciPy的`nc`一致，见
[SciPy1.13的ncx2定义](https://docs.scipy.org/doc/scipy-1.13.0/reference/generated/scipy.stats.ncx2.html)。
这种条件Gaussian／非中心chi-square工具已有先行，参见
[Stange等2014 §3.1](https://www.wias-berlin.de/preprint/2005/wias_preprints_2005.pdf)；不主张新概率分布。

如果全部rj两两正交，则给定T的指示确实相互独立，于是
`theta_M=E_T[PoissonBinomialTail(p(T),M)]`。
残差相关时该表达只能作为结构失配诊断，不能当保证。
即使残差正交，**无条件**的指示也通常不独立，因为它们共享T。

## 4. 不假设竞争者独立的条件尾部区间

对任意Bernoulli边际向量p，降序记为 `p_[1]>=...>=p_[n]`，定义

```text
U(p,M) = min(1, min_{s=0,...,M-1} sum_{j=s+1}^n p_[j]/(M-s)),
L(p,M) = 1-U(1-p,n-M+1).
```

这是经典单变量边际尾界，[Padmanabhan–Natarajan作者稿式(7)](https://arxiv.org/pdf/1910.06321)
追溯至Ruger1978／Morgenstern1980；本轮读到该作者稿表达，未取得这两篇原始全文。
其一般Bernoulli类的紧性不等于Gaussian几何类的紧性。
上界的有效性也可直接核对：任意去掉s个指示，`Z>=M`必然使剩余计数至少M−s；
对每个子集用Markov再取最小，即得上式。下界来自补事件。

现在逐T应用，再积分：

```text
lower_cond = E_T[L(p(T),M)] <= theta_M <= E_T[U(p(T),M)] = upper_cond.
```

这不要求残差独立，也不需要竞争者完整Gram。已知真实beta/alpha即可定义这些量。
它仍不能区分同beta/alpha的环构造，区间必须同时容纳两组风险；不是绕过不可辨识反例。

**明确的比较保证。** U是有限个仿射函数的最小值，因而是凹函数；L是凸函数。
令 `pbar=E p(T)`，则Jensen给出

```text
L(pbar,M) <= lower_cond <= theta_M <= upper_cond <= U(pbar,M)
                                             <= min(1,sum(pbar)/M).
```

所以保留共享目标长度后，理论上不会比同边际经典区间和原一阶Markov上界更松；
但可能完全不严格，仍可能不足以支持实际预算。这是已有条件化与概率界的应用推论，
本轮不以它作基础理论新颖性主张。

## 5. 可手算的概率桥夹具及其限制

取m=2、四个竞争者、beta=2、alpha=0。目标投影长度 `Y=||Pi*a||²/A~Exp(1)`，
`p(Y)=1-exp(-Y/2)`，故每个边际都是1/3。
残差互相正交时，条件计数为Binomial(4,p(Y))，而不是无条件Binomial(4,1/3)。

直接积分 `E[p(Y)^s]=s!*Gamma(3)/Gamma(3+s)` 可得
`P(Z=z)=(5-z)/15, z=0,...,4`，从而：

| M | 残差正交时真实遗漏率 | 无条件独立近似 | 无条件边际区间 | 条件边际区间 |
| --- | --- | --- | --- | --- |
| 1 | 2/3 | 65/81 | [1/3,1] | [1/3,37/48] |
| 2 | 2/5 | 11/27 | [1/9,2/3] | [3/16,7/12] |
| 4 | 1/15 | 1/81 | [0,1/3] | [1/48,1/3] |

M=4时盲用独立边际低估5.4倍；M=1时反而高估。说明偏差与门槛有关，
不能靠一个统一倍率解决。这里的分数是符号积分结果，不是新MC或已执行的科学测试。

表中区间由以下同beta、alpha0、m2的闭式得到，n为竞争者数：

```text
upper_cond = n/(M*(beta+1)) * (1-(1-M/n)^(beta+1)),
lower_cond = ((n-M+1)/n)^beta/(beta+1).
```

M=n时上界仍是1/(beta+1)。允许各竞争残差几乎相同，可让所有指示几乎同步；
因此仅凭这些条件边际不能普适地把上界压到正交情形的1/15。
这正是下一步需要量出而不能掩盖的剩余结构不确定性。

## 6. 连接RAG时需要保留的两道边界

**输入边界。** 上面用真实目标及全部beta/alpha，是离线几何参考，不是线上预算器。
第一阶段只判断尾部转换本身是否正确、有用；之后若有效，再研究如何从pilot距离、
重排距离、LID等可部署信息预测风险曲线，并与同pilot输入但不使用Tri的模型比较。
不能把完整Gram、真top-k或已实现召回率当成推理时免费输入。

**概率边界。** 当前theta针对固定几何／固定M而重抽Pi；实际部署常固定Pi，M又由同Pi的pilot选择。
不能直接将固定M的无条件界贴到该自适应动作上。后续必须匹配条件信息或提供对动作统一有效的保证，
并冻结矩阵和策略，在独立query_cert上认证面向新查询的风险。
query_cal只拟合，query_tune只选择，query_latency测费用，query_test最终报告；本提案不打开任何这些真实角色。

对未来top-k、独立于Pi的固定池C，可保留旧分解：`H=G∩C,g=|H|`，
对i∈H，仅数 `C\G`竞争者，遗漏i必需至少 `M-g+1` 个超越。
把第4节的条件上界用于这个计数，得到池内损失上界，再加 `|G\C|/k` 上游损失。
这仍是必要计数事件的上界，不是k>1的精确排名律；首轮不实现该扩展。

## 7. 最小首轮A0：先验证概率桥，不接真实检索

以下是**待批准**的固定范围，不是本轮执行记录。目的为实现／反驳第3–5节的数值对应，
报告区间剩余宽度和输入费用，不承诺真实RAG达标或加速。

**几何。** n=4；base目标a范数1，`bj=alpha*a+sqrt(2-alpha²)*uj`，`alpha∈{0,1/2}`。
uj垂直a，残差Gram固定四类：I4；非对角1/2的等相关；r=1/4正环；只翻闭合边的负环。
四类最小特征值均至少1/2；`G00=1,G0j=alpha,Gjl=alpha²+(2-alpha²)Rjl`，均为有理数。
共8个几何，所有beta=2；投影维度m∈{2,8}，M∈{1,2,4}，共48个概率单元。
只有alpha0的环对继承旧真子集同律性质；alpha1/2的环对仅作为同beta/alpha、不同残差结构的压力测试。

使用既有 `validate_gram_and_lift` 的五列范围，不改旧实现。
对两alpha共同取位移缩放平方1/32，最小6维构造补零到 `ambient_dimension=16`，使m2/8都为实际降维。
可行性手算：`c^T G^-1 c=1/4+((1-alpha/2)^2/(2-alpha²))*1^T R^-1 1<=17/4`，
缩放后至多17/128<1，故可以补正交分量成为单位球。精确点Gram验证与浮点分解分别保存，
不加jitter。所有几何满秩、目标严格最近；外部query与5个corpus身份分离。

**计算对照。** 精确Tri边际；无条件Markov；经典无条件边际区间；条件边际区间；
无条件Poisson-binomial诊断；给定T的Poisson-binomial积分（只在R=I时标为准确结构参考）。
不把新诊断叫Raw Tri-Predict复现，不复用其不同结构／索引约定制造对照。

**积分误差。** 冻结K=8192个等概率T区间，节点 `t_h=chi2.ppf(h/K,m)`。
由于本组 `|alpha|<1`，条件Gaussian球 `B(-alpha*r*e1,r)` 随r增加而嵌套，故pj(t)不减。
L、U和条件Poisson-binomial尾部也按各边际不减。因此在u=F_T(t)上作左右Riemann和，
每个积分的离散化括宽至多1/K；0/1端点用解析极限，不向ncx2传无穷参数。
该论证只覆盖本冻结域；`|alpha|>1`不能沿用单调积分。
保留原始积分括限，并报告它与无条件边际区间的数学交集；空交集按失败处理。
外侧Riemann端点可能比无条件界稍宽，不用它直接反驳精确积分的Jensen关系，
也不靠覆盖原始值或强行单调化来隐藏数值问题。
所有CDF/PPF仍用既有NumPy1.26.4/SciPy1.13.0浮点实现，数值一致性阈值固定1e-10，
不把它当作已证明的舍入误差。报告数学括限、离散误差和浮点核查三个状态，
不把普通SciPy输出称为严格机器区间证书或部署认证。

**两条抽样路径。** 每个(几何,m)各执行原坐标抽Pi、Gram抽Gaussian行两路，
每路固定2^16个独立Pi样本，batch=1024，合计2^21个矩阵样本。
master seeds分别62011/62017；以schema固定的JSON元组 `[namespace,master,geometry_id,m,path]`
（namespace为 `gaussian-rank-bridge-v1`，JSON使用 `separators=(',',':'),ensure_ascii=True`）
的UTF-8 SHA256全部256位（big-endian整数）作为PCG64 seed，不搜索seed、不做同维前缀独立声明。
同一Pi的三个M共用Z，不能计作三个独立样本。
保存逐Pi的bit mask和Z、全几何/点Gram、方法输出、seed来源、来源哈希、全部计时及失败。
bit mask位序固定竞争者1–4，稳定字符串ID。目标浮点并列记 `numerical_tie` 并停止，不改事件epsilon。

**统计检查。** 两条路径各48个尾部率，加32个路径组各4个边际率，共224个二项比例；
Clopper–Pearson双侧区间统一Bonferroni，family alpha=0.01，每项alpha=0.01/224。
既有罕见环间隔不作为MC必分辨目标；不能因区间重叠追加样本。
R=I与解析／条件积分对照；全体边际与精确Tri对照；其余R报告区间覆盖和结构失配。
任何不符先记 `statistical_discrepancy`／`numerical_failure`，不调置信度或删除案例。
同时覆盖是统计事件，不把“落入区间”写成证明；不符也不能直接当数学反例而跳过核查。

**资源与顺序。** 新命名空间 `runs/gaussian-rank-bridge-v1/`，不覆盖旧T1证据。
本地CPU单任务／单BLAS线程，科学run总wall<=600秒、进程树RSS<=1GiB、产物<=64MiB；
批准范围包括只监控／终止本次自建进程。超限保留部分记录和resource_stop，不自动重跑或扩预算。
无数据／模型下载、依赖安装、原生库重建、SSH、GPU或网络科学输入。
先通过新增测试和完整仓库回归，提交／推送／核对科学源码，再运行唯一冻结A0；
最终审计实际结果、保护登记、记录重算和来源，再按常设指令同步结论。
复用T1专用解释器及完整测试所需的既有四库；不借此恢复方案5环境。

**完成／停止。** 必须重现手算分数、方法域和同边际嵌套关系，报告48项绝对误差、区间宽度、
数值离散误差及几何准备／概率计算／两种参考采样的分别费用。
区间仍宽也是有效结果；不以覆盖正确或小n运行快宣称真实RAG可部署。
A0不需要证明显著环差，也不把改善上界当作已经改善候选质量。
后续真实数据、廉价pilot风险预测及独立认证须另列具体协议；本范围不自动进入这些阶段。

## 8. 待批精确八路径：实现并执行上述A0

必要性：本L1笔记足以作符号分析，但新的概率聚合、采样分布、误差和验收断言都属于科学契约。
不能用L1脚本包装旧API来绕开L0；此前的六路径环证书／环境恢复批准不包含本新范围。

| 精确路径 | 等级／拟议修改 |
| --- | --- |
| `mechanism/gaussian_rank_bridge_v1.py` | 新L0：`conditional_inversion_probability(t,beta,alpha,m_prime)`、`bernoulli_tail_bounds(p,M)`、`conditional_tail_reference(...)`、`make_frozen_geometry(...)`、`projection_counts(...)`；条件律／边界、已知尾界、限定域的单调积分、两条参考生成与计数；不修改旧Tri／环模块 |
| `configs/gaussian_rank_bridge_v1.json` | 新L0：冻结第7节8几何／16组／48单元、积分节点、224项区间、seeds及资源和字段schema |
| `tests/test_gaussian_rank_bridge_v1.py` | 新L0：手算分数、条件边际积分回Tri、非中心参数、退化／非法输入、逐状态计数必要条件、预算单调、单位球与PSD、双路径小夹具、seed复现／分批、数值域与真实资源停止链的回归 |
| `scripts/run_gaussian_rank_bridge_v1.py` | 新L1且继承L0审批：来源绑定／manifest、独立run目录、分块记录、资源监控、失败保留和逐记录汇总；不导入旧系统实验的封闭数据 |
| `docs/GAUSSIAN_RANK_BRIDGE_PROTOCOL_V1.md` | 新L0：获批后冻结本提案范围、完整算法/统计/资源语义、可重放命令和诚实的数值精度限定 |
| `scripts/run_tests.sh` | 原等级不变、变更继承L0：只在既有显式清单追加 `test_gaussian_rank_bridge_v1`；不改收集／失败语义或旧测试 |
| `docs/CODE_PROTECTION.md` | L0治理：只追加新路径／实际批准／兼容范围，更新自身获批追加内容的绑定哈希 |
| `docs/code_protection.json` | L0治理：只登记新科学路径／真实批准及上述人工表新哈希；所有旧科学哈希、等级和历史授权保留 |

影响：旧 `tri_law_probability`、`tri_law_conditional_orthogonal`、`validate_gram_and_lift`、
`signed_cycle_gram`只读复用；旧Raw/Calibrated Tri-Predict、缓存版本、数据角色、系统配置及历史产物均不变。
新schema/version/fingerprint与旧namespace分开，不能重标旧结果为新成功。
验证包括新增有意义的科学／资源回归、全部旧回归、保护检查、独立记录重算、几何与来源审计。
冻结A0前的审批必须引用实际用户对本节的同意；当前“继续”只据此开展了研究分析和提案。

## 9. 本轮实际交付／下一步

本轮只有本L1研究说明、独立T0笔记导航、START_HERE、STATUS、IMPLEMENTATION_PLAN及机器清单的新L1行。
未创建第8节新科学路径，未改变现有L0源／测试／协议／治理行；人工保护表保持字节不变。
检查命令：`python3 scripts/check_code_protection.py`、`git diff --check`，另作只读Markdown链接／锚点／围栏、
旧登记／根元数据、旧科学来源和T1交付索引审计。本轮不把手算结果记作执行测试；原481项完整验收保留。
下一步只等待第8节精确实施范围批准；批准后完成A0，不自动访问真实查询或部署预算器。
本轮实际审计通过：保护251项（L0=172/L1=76/L2=3）、80个本地链接、22个锚点／围栏；
全部旧250条登记和根元数据、旧科学／治理来源、旧T0历史正文及两个T1交付索引保持，五项拟议新科学路径未创建。
