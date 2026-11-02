# 共同 Gaussian 投影的排名尾部：独立研究 T0

日期：2026-11-02，Asia/Shanghai。入口提交：`c310eb6c832ebde78d69047ba5340b9f77f7cd48`。
组织：独立理论研究；沿用 canonical checkout 与 `codex/tls-rag-step5-ann` 保存。
等级：L1 工作笔记，包含可修订的符号推导和待批提案，**不是冻结协议或发表优先权声明**。

**最新审查决定。** T1已完成481项完整回归；后续先行／符号审计见[第12节](#12-后续先行与证明审计保留机制笔记停止当前候选升级2026-11-02)。
JRC-CYCLE-1保留为可复核机制笔记，当前贡献门判定为 `covered_or_corollary_only` 中的
`corollary_only`：未建立足够的发表增量，停止该候选的T2数值升级，不撤回既有数学证书。
这不是声称已有论文逐字包含本笔记的排名间隔下界，也不是一般研究问题已被解决。

**首轮结论。** 旧点对律、1/3 对 1/6 反例、完整 Gram 参考、条件正交残差公式均可复核，均不作为新贡献。
本轮保留唯一候选 **JRC-CYCLE-1：单位球上、保持全部低于环长的联合反转分布的排名尾部间隔**。
下面给出一条闭合的符号推导路线，包括有限参数的正下界；没有运行科学代码、积分或 Monte Carlo。
环符号／低阶主子式的辨识机制已有明确先行工作，因此当前发表新颖性仍为
`novelty_not_established`。候选的具体差异是到固定近邻遗漏概率的定量转换及单位球实现，
并非“发现相关环”或“首次用 Gram”。建议只申请最小 T1 核验，不启动 T2 数值搜索。

## 1. 授权、隔离与实际读取

用户最新指示：

> 交付给一个新的agent来做吧，模型就用gpt6.1sol 极高，有必要的话可以先与之前的step5隔离开，这个更像是一个新的研究，你觉得呢

本聊天接手 T0；不向旧 Step5 聊天发送消息。按本次交接的隔离指示，旧探索日志仅作只读历史，
本轮过程进入本文件；原框架末段“追加同一探索日志”的组织建议在此不沿用。
不新建分支/worktree/仓库，不改变旧实验的协议、结果、数据角色或停止决定。

已完整读取 [AGENTS](../AGENTS.md)、[启动入口](../START_HERE.md)、[STATUS](../STATUS.md)、
[研究框架](TRI_LAW_JOINT_RANK_RESEARCH_FRAMEWORK.md)；核对
[保护登记](CODE_PROTECTION.md)和机器清单；只读复核旧日志 §§33.2–33.4、37.4、39.5。
未打开旧原始 report/cert/test 或用户未跟踪归档。进入时 tracked clean，未跟踪档案保留。
进入时保护检查：246 项，L0=169、L1=74、L2=3。

本次授权包括文献检索、符号分析、独立 L1 研究笔记和例行导航/登记。
本文件不创造新 L0 实现、测试、冻结协议或随机数值试验的批准；第7节列出待批精确范围。

## 2. 已知数学对象的独立核对

**2.1 排名事件与并列。** 固定非随机几何，`a=x0-q`，`bj=xj-q`，
`0<||a||<||bj||`；共同矩阵 `Pi` 的元素 iid `N(0,1/m)`。
令 `Ij=1{||Pi*bj||²<||Pi*a||²}`、`Z=sum Ij`、`theta=P(Z>=M)`。
对每个 j，`bj*bj^T-a*a^T` 非零（其迹严格为正），故距离差是非恒零的 Gaussian 多项式，
等于零的概率为零。有限并集仍为零，所以目标排名几乎处处为 `1+Z`。
竞争者之间可以有距离并列；它不改变严格领先目标的身份数。计数单位是竞争者身份。

**2.2 旧边际反例。** `a=e0`，`b1=sqrt(2)e1`，`m=2,M=2`。
令 `Y=||Pi*a||²~Exp(1)`。`b2=-b1` 时两指示相同，遗漏率为 `1/3`；
`b2=sqrt(2)e2` 时给定 Y 两指示独立，条件概率 `1-exp(-Y/2)`，
二者同反转率为 `1-2/(1+1/2)+1/(1+1)=1/6`。两例单点边际均为 `1/3`。
直接乘边际得到 `1/9`，不是正确的共同投影概率。

原反例第一组**不能**通过共同非零缩放和平移变成单位球位移：
若 `q+b1` 和 `q-b1` 与 q 都是单位向量，则分别要求
`2q·b1+||b1||²=0` 和 `-2q·b1+||b1||²=0`，相加矛盾。
逐点归一化也不保持原几何。第二组满列秩，充分缩放后可以提升；不能因此补全第一组。

**2.3 完整 Gram。** `V=[a,b1,...,bn]`、`G=V^T V`。
`Pi*V` 的 m 行独立，行协方差为 `G/m`，故 G 决定全部投影距离联合律。
可实现条件是 `G>=0`、`rank(G)<=d`，并有严格距离对角条件。
奇异 G 合法；不得用 jitter 改变它。Wishart 表征属于已知工具。

**2.4 条件正交残差。** 写 `bj=alpha_j*a+rj`，`rj` 垂直 a；
设 `A=||a||²`、`sj=||rj||²`，`T=m||Pi*a||²/A~chi²_m`。
各非零 rj 两两正交时，`Pi*rj` 相互独立且独立于 `Pi*a`。
给定 `Pi*a=w`，每个 `Pi*bj` 的均值为 `alpha_j*w`，协方差为 `(sj/m)I_m`，于是

```text
pj(t) = F_ncx2(A*t/sj; df=m, nc=alpha_j²*A*t/sj)
theta = integral chi2_m_density(t) * PB_tail(p1(t),...,pn(t); M) dt.
```

非中心参数没有遗漏 m：`m*alpha_j²*||w||²/sj=alpha_j²*A*t/sj`。
给定 w 的概率仅依赖其长度，故条件于 T 仍可用相同乘积/Poisson-binomial。
`sj=0` 时严格最近邻条件强制 `|alpha_j|>1`，该指示几乎处处为0，应单列，不能除零。
只有先计算条件尾部再积分才保留共享目标的依赖；此公式是已知条件混合的应用。

**2.5 单位球提升的精确条件。** 对非零位移列 `v_i`，令 `c_i=-||v_i||²/2`。
要求 `||q||=||q+v_i||=1` 等价于 `V^T q=c` 且 `||q||=1`。
最小范数解存在当且仅当 `c in range(G)`，其范数平方为 `c^T G^+ c`。
若此值不超过1且 `d>rank(V)`，可加正交分量补足单位长度；若 `d=rank(V)`，必须恰等于1。
统一缩放位移为 epsilon*V 后，值域条件不变，最小范数平方乘 `epsilon²`；
所以满列秩构造在增加一个正交维后总能充分缩放提升，秩亏构造则不一定。
所有位移统一缩放保持距离比、角度和每一个投影排序事件。

## 3. 先行文献对照与阅读深度

核对日期均为2026-11-02。以下页码指所链接 PDF 的页序／文内编号；只声称核对所列段落。
“全文可取得”不表示逐行审查全篇；搜索无命中也不证明不存在先行结果。

| 一手来源与定位 | 已核对的内容 | 对本命题的约束／剩余差异 |
| --- | --- | --- |
| [Durrant–Kabán 2013](https://proceedings.mlr.press/v28/durrant13.pdf)，Thm3.2、§4.1、§4.2，PDF3–7页；全文可取得 | 固定向量共同 Gaussian 的精确翻转律及共享矩阵依赖；§4.2还讨论依赖和界 | 极化后就是既有距离反转边际。不能说此前只处理独立翻转；不是本轮贡献 |
| [Khatri–Krishnaiah–Sen 1977](https://www.sciencedirect.com/science/article/pii/0378375877900143)，JSPI1:299–307 | 出版者检索摘要确认相关 Gaussian 二次型的精确／渐近联合分布；直接页面403，未取得全文 | Hermite/Laguerre展开和联合二次型不能主张首次发现；其是否直接覆盖下述尾部差公式仍待全文核对 |
| [Al-Naffouri–Hassibi 2009](https://faculty.kfupm.edu.sa/EE/naffouri/publications/isit09-indef-quad-forms.pdf)，§II–IV，PDF2–4页；全文可取得 | 不定二次型、非中心情形及联合 Hermitian 二次型的积分途径 | 主要显式常数与联合例使用复 Gaussian；不能把其指数−1直接用于任意实单行 Gaussian |
| [Bertsimas–Popescu 2004作者稿](https://flora.insead.edu/fichiersti_wp/inseadwp2004/2004-62.pdf)，§2式(1)–(2)、§4，PDF8–10/18–24页；对应2005论文 | 矩约束概率优化、对偶覆盖多项式和适用条件 | 整数计数矩LP和一般矩界均已有理论；不意味着 Gaussian 可实现类也达到一般分布极值 |
| [Padmanabhan–Natarajan作者稿v1](https://arxiv.org/pdf/1910.06321)，§1–2、Thm2（文内15页）、§4.3 Prop4（20页）；[期刊DOI](https://doi.org/10.1287/ijoo.2019.0038) | 指定一元/树边二元边际时的最优LP界，及顺序统计量应用 | 不要求真实分布在树上条件独立。固定目标比较转成Bernoulli尾部本身不是新意；Hailperin1965状态LP沿本稿追溯，原文未精读 |
| [Stange–Loginova–Dickhaus 2014，WIAS2005号](https://www.wias-berlin.de/preprint/2005/wias_preprints_2005.pdf)，§2、§3.1式(9)–(10)，PDF4–7页；全文可取得 | Wishart对角／多元chi-square，低阶边际概率近似，以及低秩因子下条件非中心gamma乘积 | 正交残差条件参考已有直接工具；一般 product approximation 不自动是界，文中也明确这一限制 |
| [Royen 2015 v1](https://arxiv.org/pdf/1507.00528)，§1式(1.2)–(1.7)、Thm4及式(4.3)–(4.4)，PDF2–3/8–9页；全文可取得 | determinant Laplace变换、signature不变性及主子式展开的概率比较 | 构造依赖这些既有恒等式。Thm4的正相关／无限可分基准条件不直接涵盖所选两个环符号矩阵；不据此声称全新比较方法 |
| [Urschel–Brunel–Moitra–Rigollet 2017](https://proceedings.mlr.press/v70/urschel17a/urschel17a.pdf)，§2.1–2.2式(1)、Prop1，PDF2–3页；全文可取得 | 低阶主子式、signature等价、环符号与cycle sparsity控制辨识 | **最接近的新增先行项**。同低阶信息、长环符号不辨识的矩阵机制已在此明确出现；本轮不能以“环导致不可辨识”为新贡献 |
| [Indyk–Naor 2007作者版](https://web.math.princeton.edu/~naor/homepage%20files/lowdim-journal-new2.pdf)，定义与§4 Thm4.1，PDF1–2/6–7页 | doubling几何中的近邻保持与近似近邻目标 | 不等同于任意固定几何、给定低阶反转摘要的精确遗漏率范围 |
| [Zhang–Li 2023](https://proceedings.mlr.press/v206/zhang23j/zhang23j.pdf)，§3 Lemma1，PDF3页及相邻讨论 | 随机投影前后第k近距离的畸变控制 | 同一身份的投影排名尾部与第k距离保真不同，不能互换 |
| [Sao 2026预印本](https://arxiv.org/pdf/2609.02155)，§7，PDF15–17页及§B.8–B.9 | Gaussian随机数据、共享query排名及信息损失模型；取得41页稿 | 本轮固定几何、只随机Pi；没有从此稿推得单位球同摘要构造，亦不声称其随机数据结论错误 |

额外检索围绕 correlated quadratic forms、Gaussian squares/signature、multivariate gamma/chi-square、
cycle/principal minors、Bernoulli tree bounds、random projection rank identifiability。
最重要的纠偏是补入2017年的环与主子式辨识文献。1977全文缺口及未穷尽的先行覆盖保留。

## 4. 唯一主命题卡：JRC-CYCLE-1

| 项 | 固定内容 |
| --- | --- |
| 几何类 | n>=3，目标 a 与全部竞争残差正交；`||a||=1, ||bj||²=beta>1`，竞争者残差 Gram 是无弦 n 环，边绝对值 r，`0<r<1/2` |
| 两组几何 | `R+` 环边全为+r；`R-` 只把闭合边(1,n)改为−r。`G±=diag(1,beta R±)`；相同 beta/r/n |
| 摘要 | S包含相同S1以及每个真子集 J 的完整联合指示分布 `Law((Ij)_{j in J})`；还可给出该子集的投影距离联合律与目标长度。不包含全环符号 |
| 随机性和量词 | 对每个 n>=3、beta>1、0<r<1/2，几何投影前固定；本命题只取 `m=2,M=n`；单位球实现维数 `d=n+2` 足够 |
| 目标 | 同摘要却有 `W(S)>=Delta_n(beta,r)>0`；给出确定性有限下界，而非仅说明联合分布不同 |
| 候选间隔 | `Delta_n >= 4*n!*beta*r^n / (beta+n/(1-2r))^(n+1)`；第5节给出更精确的正卷积表达 |
| 信息费用 | 摘要允许全部真子集，至多指数规模；这是强信息也可能不充分的存在性命题，不是廉价部署方法。n=3已包含全部两两反转概率 |
| 最接近先行 | Urschel等的环/主子式机制＋多元gamma determinant变换；前者不是Gaussian排名定理，后者是标准联合分布工具 |
| 唯一待核对差异 | 把已知矩阵不辨识转成目标排名事件的严格正间隔，并给共同缩放的单位球构造。差异存在于研究对象／定量陈述，发表新意未建立 |
| 证明／反驳 | 真子集森林的sign switching；环行列式差；m=2的Laplace倒数相减为正卷积；球提升。任一步失败须撤回，不用MC填补 |
| 失败／停止 | 若先行已直接给同一排名间隔/等价直接推论且无实质差异，标`covered_or_corollary_only`，停止实现；若符号/球提升/概率换序不成立，标`claim_failed` |

这不是 Gaussian 全类的最优 W，不给极值匹配，不声称固定绝对常数下界随 n 保持，
不扩到任意 m、M、k>1或有非零目标相关的 rho。n=3是可审查实例，整族是低阶层次的存在性陈述。

## 5. T0符号推导草稿：有限间隔，不靠抽样

以下是本轮推导，待T1以不同路径复核；不能当成已经独立验收的冻结定理。

**5.1 合法性和同摘要。** 每个 R 的对角为1，每行非对角绝对值和为2r，
故最小特征值至少 `ell=1-2r>0`；`G±` 满秩 n+1，是实际向量的 Gram。
取 a=e0、`bj=sqrt(beta)uj`，uj 的 Gram 为 R，且 uj 垂直 e0。
任何真子集 J 的诱导子图都是森林。森林上任意两组边符号可以用对角 `D_J`（元素±1）互换，
即 `R-_J=D_J R+_J D_J`。对应 Gaussian 列只改变符号，其范数向量的联合律完全不变。
目标列独立于所有竞争列；所以包括共同目标长度的子集联合律也相同，继而全部真子集反转律相同。
这不要求存在一个同时作用于全环的 D；全环符号正是不能被此变换消除的信息。

**5.2 m=2的缩放。** 原几何 `||a||=1` 时，记
`U=||Pi*a||²~Exp(1)`、`Xj=||Pi*bj||²/beta`。
每个 Xj 的边际是Exp(1)，其联合Laplace变换为

```text
L±(s)=E exp(-sum sj Xj)=1/D±(s)
D±(s)=det(I+R± diag(s)),  sj>=0.
```

这里指数−1来自两个独立**实** Gaussian 行各贡献−1/2；不借用复数公式的常数。
目标 U 独立于 X。`M=n` 时

```text
theta± = beta * integral_0^infinity exp(-beta*t) F±(t,...,t) dt,
F±(x)=P(Xj<=xj for all j).
```

**5.3 行列式差与尾部差。** 在无弦环上，所有真主子式相同。
全主子式唯一受环符号影响的两个排列项给出

```text
D+(s)-D-(s) = 4*(-1)^(n-1)*r^n*product_j sj.
Laplace(F- - F+)(s) = (L-(s)-L+(s))/product_j sj
                            = 4*(-1)^(n-1)*r^n L-(s)L+(s).
```

令 X-、X+ 为这两个分布的**独立**向量，h 为 `X-+X+` 的密度。
Laplace唯一性和卷积给出

```text
F-(x)-F+(x) = 4*(-1)^(n-1)*r^n h(x),   xj>0.
Delta_n := abs(theta- - theta+)
         = 4*r^n*beta*integral_0^infinity exp(-beta*t) h(t,...,t) dt > 0.
```

此处独立的 X-/X+只用于表示两个分布的卷积，**不是**把同一个投影内的竞争者当独立。
SPD保证正象限内有连续正密度，可将Laplace反演的几乎处处相等提升为逐点相等，
所以在对角线上取值不是把零测集合上未定义的密度当概率。n奇数时theta-大于theta+；偶数方向相反。

**5.4 显式下界。** 对 `Xj=(z1j²+z2j²)/2`，两行协方差均为 R。
在每列使用二维极坐标，独立R=I时密度为 `exp(-sum xj)`。
一般R的密度是归一化角度积分，前因子 `(det R)^(-1)`。
因 `R^(-1)<=I/ell` 且 `det R<=1`（Hadamard不等式），对 `xj>0` 有

```text
f_R(x) >= exp(-sum xj/ell).
h(x) = integral_[0,x] f_-(y) f_+(x-y) dy
     >= (product xj) exp(-sum xj/ell).
Delta_n >= 4*r^n*beta*n! / (beta+n/ell)^(n+1).
```

下界随n衰减，不意味着有宏观统计差异。例如 n=3、beta=2、r=1/4，
两组同全部两两反转分布，但 `theta- - theta+ >= 3/16384`（约0.000183）。
这是解析下界，**不是实测差值**。n=4相同参数的下界仅 `3/400000`。
因此不申请2^16投影便保证分辨该间隔；这种采样设计没有充分依据。

**5.5 单位球实现，不改变事件。** 对 `G=diag(1,beta R)`，

```text
c^T G^(-1)c = (1+beta*1^T R^(-1)1)/4 <= (1+beta*n/ell)/4.
epsilon² = 1/(1+beta*n/ell).
```

两组共用此 epsilon，缩放后的最小解q范数平方都不超过1/4。
取 `q_min=epsilon*V*G^(-1)c`，在 `span(V)` 的正交补中加单位方向分量，补足 `||q||=1`。
设 `x0=q+epsilon*a`、`xj=q+epsilon*bj`，全部为单位向量，且
`0<||x0-q||²=epsilon²<beta*epsilon²=||xj-q||²`。
`d=n+2` 足够；n=3上述实例为d=5、`epsilon²=1/13`。
所有投影距离统一乘epsilon²，故逐投影的 I/Z/遗漏事件及两组摘要保持。

**5.6 已能说明与不能说明的事。** 一个要求覆盖同摘要下所有合法单位球Gaussian几何的区间，
宽度至少为上述Delta；任何仅用该摘要的点估计，在这两组之一上的绝对误差至少Delta/2。
n=3时mu、nu、Var(Z)、全部p_j和p_jl都相同，因此Markov、Cantelli、计数矩LP、
相同边际输入的状态/树LP无法区分这两组；它们的保守区间仍然合法。
没有在本轮实现这些LP或证明Gaussian类上的锐界，也没有声称新界比同信息LP更紧。

已知的正面对照是：当目标相关rho为0、残差非零支持已知是森林，且已知每条边的绝对相关时，
所有符号选择都signature等价，因而这些几何信息足以确定完整范数联合律和排名尾部。
这不是“树边反转概率使指示条件独立”，也没有证明反转概率可以反解全部相关绝对值；
仅作为同一环机制的已知充分条件对照，不另立研究主线。

## 6. 决策门与剩余缺口

状态记录：

- `T0 substantive_review_complete`：完成主要一手材料定位/所列段落核对和符号分析，全文缺口明确。
- `candidate_difference_identified`：完整真子集反转律相同、排名尾部有有限间隔的具体陈述可检验。
- `novelty_not_established`：已知环机制相当接近；尚未证明排名尾部转换构成足够研究增量。
- `T1 proposed_not_approved`；`T2/T3/T4 not_executed`。第5节草稿不标成本轮T3已独立通过。

唯一下一步仍是JRC-CYCLE-1的证明/先行复核。优先补1977全文与该文所引多元gamma资料，
检查有限差的卷积表示是否已有直接结论；再以行列式展开和Gaussian密度两条路径核验。
若只剩标准工具的短推论，可交付机制笔记并停止，不为了得到“正结果”转向ANN接口或扩大试验。
纯符号证明确立与可发表新意是两个不同判断；缺乏后者不等于前者数学错误。

## 7. 最小 T1 受保护范围提案（尚未批准，所有新路径尚未创建）

**必要性与边界。** 当前L1符号笔记已能表达候选，但不能冒充已核验的科学实现。
若要可重放核验，需要新增精确有理证书、几何合法性与回归断言，均定义科学语义，应为L0。
不通过caller、换import或新L1脚本绕开批准。本次只申请确定性T1，不申请MC、LP或T2比较。

| 精确路径 | 等级与拟议修改 |
| --- | --- |
| `mechanism/joint_rank_cycle_v1.py` | 新L0；`signed_cycle_gram(n,r)`、`cycle_determinant_certificate(n,r)`、`validate_gram_and_lift(G)`、`rank_tail_gap_lower_bound(n,beta,r)`、`rank_count_from_squared_norms(...)`。Fraction证书与浮点展示分开；拒绝非法PSD/退化，不加jitter |
| `tests/test_joint_rank_cycle_v1.py` | 新L0；determinant证书、所有真子集signature证书、球提升、rank事件及旧解析例/退化回归；不抽样 |
| `docs/JOINT_RANK_CYCLE_T1_PROTOCOL_V1.md` | 新L0；只冻结获批的确定性核验对象、域、精确断言、数值展示限定及资源停止；引用本笔记草稿，不先把它定为已证结果 |
| `scripts/run_tests.sh` | 原L1，继承L0审批；仅在已有显式模块清单中追加新测试一项，旧收集/失败/套件逻辑不变。共享离线测试入口不表示归入旧Step5系统实验 |
| `docs/CODE_PROTECTION.md` | L0治理；只追加上述新文件行与真实批准记录，不修改旧等级/理由；更新该文件自身已登记内容哈希需同时获批 |
| `docs/code_protection.json` | L0治理；只追加新路径/实际批准，及上述人工登记文件的必要新哈希；不刷新任何旧科学源码哈希 |

**固定核验范围建议。** `n=3,4`，`beta=2`，`r=0,1/4`；正负全环及真子集。
主证书使用精确分数，断言差行列式系数 `4*(-1)^(n-1)*r^n`；
下界精确值分别 `3/16384,3/400000`。r=0必须相同；域外 `r<0,r>=1/2` 拒绝作为本协议输入，
这是该受限族的域限制，不声称所有r>=1/2矩阵都不PSD。
另有固定边界夹具：n=1的严格共线零反转、旧n=2两组1/3和1/6解析积分、
旧反向位移单位球提升失败、非PSD/奇异几何的明确分类，以及仅目标不并列时的rank计数。
有理Gram判定和半正定退化分类分开；球构造的等式以Gram/最小范数证书核验，
浮点坐标残差只作诊断，不用宽容差决定科学事件。

**参数/随机性/资源。** 仅上述人工合成夹具；无数据划分或真实查询，所有随机seed为
`not_applicable`（无随机试验）。使用已有CPU环境，不安装依赖、不下载模型/数据。
新T1科学核验累计至多5分钟、RSS上限1GiB、证据上限32MiB；满额立即停止并保留部分结果，
不自动扩n/m或抽Pi。完整既有离线套件验证另记录费用，不伪称它是理论证据。
新命名空间 `runs/joint-rank-theory-v1/t1-<unique-id>`，不得覆盖已有目录。
本提案不含新runner/config，标准unittest命令即可重放；无须引入本轮用不到的矩LP和采样器。

**验证顺序。** 补先行→独立复核第5节证明→精确分数证书→几何/事件边界回归→
完整旧套件＋新增测试、零跳过→保护/来源/证据审计。若证明或先行门失败，封存失败并停止；
不以“程序通过”宣布新颖性。数值积分/抽样都不在本次范围，今后须另提范围并等待批准。
共享分支最新的方案5交接记录指出旧临时FAISS环境／头文件不完整；本理论任务未恢复它们。
完整回归只能在已可用的原固定环境中执行；若只读检查找不到可用环境，先报告确切恢复需求，
不换版本、不跳过原生测试，也不把当前六路径提案解释为另行安装环境的授权。

**API、历史与指纹后果。** 不修改既有Tri-Law、检索、投影或策略API；新模块不接入系统路径。
新增模块只使用标准库／既有NumPy等已有依赖，不更换固定版本。
新证据绑定上述新科学文件、实际测试入口/保护脚本、git提交和环境身份。
`scripts/run_tests.sh` 已被 route4 P1及场景A/B的SOURCE_PATHS绑定；追加测试成员会改变
**新计算的来源身份**，旧run的历史来源/哈希不能据此刷新，也不重跑旧run。
Step4 repaired acceptance绑定保护脚本，提案不改该脚本；新科学哈希只新增本命名空间。
旧数据角色和历史科学结论无新授权，unit-sphere定理也不等于ANN速度或实际embedding质量结论。

## 8. 本轮验证与交付记录

本轮没有科研测试/模拟结果；467项是既有实现检查点，不记为本研究通过。
实际使用的只读／文档验证命令包括：

```bash
git branch --show-current
git rev-parse HEAD
git status --short
python3 scripts/check_code_protection.py
git diff --check
```

另用一次性只读文档审计检查本轮路径、Markdown本地链接/锚点、围栏、
新登记的唯一性、全部旧246项登记和根元数据相等、全部旧L0内容及旧探索日志字节保持。
机器登记例行只增加本L1文件；人工治理登记未改，未创建第7节科学路径。
研究过程中另一任务推进共享分支，短暂提交状态也发生变化，导致两次固定HEAD/未提交路径范围审计失败。
最终以实际父提交 `19a22e7571cbb8c2e905f058243636d3e2b89153` 重新核对：
五路径理论交付、全部旧246项及根元数据保持，63个本地链接/围栏检查通过；
保护247项（L0=169、L1=75、L2=3）通过，科学路径未创建。
本轮保留另一任务的方案5暂缓决定和日志，不撤销或重写其提交。
下一步批准只可能覆盖实际呈示的第7节范围，不能反写本次为已批准。

## 9. T1实际批准、复核与当前验收缺口（2026-11-02）

用户随后实际回复：

> 批准，你的意思是tri-law已经有人用过了吗

这批准了a74e45b中第7节的六路径确定性核验；第6–8节的“未批准/尚未创建”是该回复之前的T0历史状态。
当前三项新科学文件及测试成员登记已实现，见
[获批T1协议](JOINT_RANK_CYCLE_T1_PROTOCOL_V1.md)。不把批准扩展到安装环境、MC或T2。

**对Tri-Law先行的澄清。** 已再次核对Durrant–Kabán2013 Thm3.2及§4.1：
其对象是同一iid Gaussian矩阵下固定正内积翻负的精确概率，用于分类误差分析。
令 `u=b+a,v=b-a`，则投影内积就是 `||Pi*b||²-||Pi*a||²`，且

```text
c = (beta-1)/sqrt((beta-1)^2+4*beta*(1-rho^2))
p = I_((1-c)/2)(m/2,m/2).
```

这与项目的精确点对距离反转概率代数等价。原文并未因此被认定使用“Tri-Law”这个名称；
这里明确区分同一概率律的先行、该名称/表述以及本轮联合排名问题。
边际本身不能再作首次发现的主张，也不能据此断言整个后续联合尾部研究已经完成。

**补充先行核对。** 1977出版者全文仍403，不能写作已读全文。
[Royen1995](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A5n123.pdf)全文取得并核对§1式(1.2)、
Thm3.1及其条件混合表示；[Royen2007 v1](https://arxiv.org/pdf/0704.0539)核对§1式(1)–(3)和相关说明，
其已明确研究不同协方差的多元gamma卷积。联合分布／卷积表征仍作为已知工具。
[Royen1994树型表示](https://link.springer.com/article/10.1007/BF01720592)只取得出版者摘要。
本轮未在这些已读段落中找到直接陈述同真子集反转律的单位球排名间隔，
但这不证明先行不存在，`novelty_not_established`保持。

**证明复核。** 对照两条有限代数路径：主子式系数与Leibniz排列展开；
全体真子集对角符号证书与全环无此证书；最小范数线性条件与直接点Gram重建。
再次核对两实Gaussian行产生determinant的−1次幂；CDF的Laplace变换多出 `1/product(sj)`，
其恰消去行列式差中的全单项式。正卷积密度在正象限连续，故可以取对角线值。
角度积分前因子为 `1/det(R)`，利用 `R^-1<=I/(1-2r)` 与Hadamard不等式得到下界；
没有把Laplace变换大小直接误当CDF大小。该概率推导是人工符号复核，有限代码证书不取代证明。

**实际验证。** 独立命名空间：
`runs/joint-rank-theory-v1/t1-20261102-bf190bff`。
新增14项确定性测试全部通过、0跳过，unittest本体0.038秒；
含证书生成的受监督核验0.124671秒，观测峰值RSS含监督进程35,553,280字节。
四项固定(n,r)证书、八组正负几何及单位球Gram保存；没有抽Pi或数值积分。
原沙箱不允许ps，受监督核验经工具授权读取自建进程资源后执行；未放宽资源限额。
独立Schur补审计另外用0.005230秒精确复核八组Gram的PSD/秩与位移重建，源哈希一致。
首次审计启动有一项括号语法错误、在任何代码执行前失败，已保留记录并修正审计命令；科学源和证书不变。
证据共11文件/105,007字节；`delivery_index.json` SHA256
`0102a374e530df9cdfb703b1bf21450972ea6172813d342cff476d98d092afc4`。

全仓回归的实际命令为既有固定解释器＋四个既有native库环境变量下
`sh scripts/run_tests.sh`；1.305697秒后收集失败、返回码2，缺 `faiss` 导入。
两个旧route4测试模块未能导入，没有完成全仓测试，不能把历史467项或预期481项记为通过。
失败日志及SHA：`full_regression.log`，
`4c5e88b6cc979e405812f50fe8624b2f5d4394d58ffcd38cdb134b903342d064`。
只读检查还发现全局sklearn为1.4.2，旧continuation_v2明确需要1.6.1；
NumPy1.26.4、SciPy1.13.0已存在。旧临时venv无pyvenv.cfg，FAISS仅余二进制文件。

当前状态：`T1_new_checks_passed / full_regression_blocked_environment`。
不标T1完整验收，不据此提交/推送未通过必要全仓验证的实现；所有变更和失败保留本地，
保护登记只有新三项L0、真实批准和获批人工表哈希变化，旧科学哈希不变。

## 10. 为完成全仓验收所需的最小环境恢复范围（待批）

六路径T1提案明确“不安装依赖”，因此以下是额外恢复范围，不自动执行：

- 新建专用 `/private/tmp/joint-rank-t1-test-env-20261102`，保留旧不完整环境、用户全局包和四个既有native库。
- 只恢复原固定测试版本 `faiss-cpu==1.8.0`、`scikit-learn==1.6.1`，使用官方PyPI相应平台wheel，
  下载时核对其发布SHA256；只装这两个包，不自动解析升级依赖、不修改仓库依赖文件。
- 复用已经存在的NumPy1.26.4、SciPy1.13.0、joblib1.4.0、threadpoolctl3.4.0；
  开始前核对版本/导入与四native库的实际兼容性。不重建native库，不恢复HNSW头文件或执行方案5。
- 使用新解释器和原四库执行完整离线回归；如还有确切缺项则保留失败并报告，不扩大安装范围。
  通过后才完成T1验收、提交/推送既有分支并核对远端HEAD。

这一恢复只支持已批准的离线测试，不新增研究案例、MC、数据访问或新版本科研语义。

## 11. 环境恢复批准与T1完整验收（2026-11-02）

用户对第10节范围实际回复：

> 补齐完整测试，批准

这项补充授权只恢复专用临时测试环境；第7节原先“不安装依赖”的限制及冻结协议的初始授权描述
作为历史保留，以本次明确补充授权执行全仓验证。科学模块、测试断言、冻结数学参数和原生库未因此改变。

**环境与来源。** 新建 `/private/tmp/joint-rank-t1-test-env-20261102`，Python3.9.6，
使用system-site-packages复用原NumPy1.26.4/SciPy1.13.0/joblib1.4.0/threadpoolctl3.4.0。
只从官方PyPI读取发布元数据及下载以下两个arm64 wheel，逐项核对发布SHA256与大小：

| 包与平台wheel | SHA256 |
| --- | --- |
| `faiss_cpu-1.8.0-cp39-cp39-macosx_11_0_arm64.whl` | `8c6fa6b7eaf558307b4ab118a236e8d1da79a8685222928e4dd52e277dba144a` |
| `scikit_learn-1.6.1-cp39-cp39-macosx_12_0_arm64.whl` | `e7be3fa5d2eb9be7d77c3734ff1d599151bb523674be9b834e8da6abe132f44e` |

随后只在新环境执行 `pip install --no-index --no-deps --ignore-installed --require-hashes`，
来源为已验证的本地wheel目录，未自动升级/解析其他依赖。旧不完整环境、全局包和仓库依赖文件保留。
原沙箱DNS不允许访问PyPI，下载命令按本次明确授权经工具审批执行；没有省略哈希检查。
四个既有native库的SHA256与恢复前一致，ctypes加载通过；新旧FAISS extension的共同SHA256为
`dc7ef26c92d6cc247e3ed0f14a9d3143778c0352885ecd4cb34db791c25a86f2`。
未重建native库、恢复HNSW头文件或执行旧Step5实验。

**完整测试。** 实际从 `/private/tmp` 调用以下命令，保持原测试入口逻辑与全部断言：

```bash
export TRI_RAG_TEST_PYTHON=/private/tmp/joint-rank-t1-test-env-20261102/bin/python
export STEP5_ROUTE1_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/route1.dylib
export STEP5_COMPARATORS_V2_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/comparators_v2.dylib
export STEP5_CONTINUATION_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/continuation_v1.dylib
export STEP5_CONTINUATION_V2_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/continuation_v2.dylib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /private/tmp
sh /Users/guanghongxu/Query-Adaptive-Tri-RAG/scripts/run_tests.sh
```

实际收集并执行481项，step5=283/legacy=198，**481通过、0跳过**，
测试本体70.081秒，总调用70.949944秒，`full_repository_validation_passed=true`。
包含全部既有467项和新增14项；本轮没有修改/跳过旧测试或以部分通过替代全仓验证。
首次缺FAISS的失败日志、原11项证书/交付索引和新环境恢复的过程均保留。

验收回执进入新目录 `runs/joint-rank-theory-v1/t1-20261102-validation`，
完整日志SHA256 `19594051218790c19ee84ea1d2de5ed7200ab9740be10a1031ab932e9d73d4da`。
原研究证据目录及其交付索引不变。T1的有限精确证书、独立Schur补审计与全仓回归共同完成本阶段验收；
概率推导的人工复核不变成形式证明，发表新颖性仍为`novelty_not_established`。
不自动启动T2采样、扩大n/m/M、真实数据验证或ANN接口研究。

最终验收回执9文件/66,529字节，`delivery_index.json` SHA256
`92e3a3e9c272a31e34df07c67aafabf5a79fe312e2f9cb7f5e912341d327847d`。
保护250项、旧247项仅获批人工登记哈希改变；312个本地链接、19个锚点/围栏通过。
全部科学/测试入口来源与完整测试时一致，四native库/全局依赖/旧11项证据不变。

## 12. 后续先行与证明审计：保留机制笔记，停止当前候选升级（2026-11-02）

用户在T1完整验收后指示“继续”。本轮延续已授权的一手文献审查与非冻结符号推导，
未改三项L0科学文件、冻结参数、测试入口或任何保护登记；没有新MC、积分、LP或数据访问。
第4–8节及第9–11节保留各自历史状态，本节给出当前贡献判断。

**12.1 最近先行比原对照更具体。** 再读
[Urschel–Brunel–Moitra–Rigollet 2017](https://proceedings.mlr.press/v70/urschel17a/urschel17a.pdf)
§3 Lemma2及式(3)，PDF第5页：其下界构造恰是对角1/2、环边α、仅闭合边符号不同的两矩阵，
并通过容斥把真子集信息相同转成全状态概率差。令 `K±=R±/2, alpha=r/2`，
就得到本轮同一对矩阵；T1固定 `r=1/4` 对应其 `alpha=1/8`，正好在该引理范围内。
这里比对的是构造和辨识机制，不移植其DPP概率、常数或样本复杂度为Gaussian结论。
原第3节只列Prop1，遗漏了这处更加直接的例子；现在明确补全。

补取得[Rising–Kulesza–Taskar 2015作者稿](https://www.alexkulesza.com/pubs/spmap_laa14.pdf)，
核对§3.3 Thm3.12/3.14（PDF第9–10页）：弦自由环乘积控制signature等价，
对应主子式全部相等刻画该等价类。本文只用实对称情形，未审查其算法全篇。
这进一步确认森林符号可消除、长环符号不能消除均属已知矩阵机制。
1977论文仍只读到出版者摘要，未取得全文；不把缺失全文或检索无命中当作新颖性证据。

**12.2 剩余证明的依赖链。** 本笔记的专门化差异是Gaussian固定目标遗漏事件，
不能仅由DPP结论直接代入。逐步审查如下：

| 步骤 | 本轮依据／复核结果 | 贡献判断 |
| --- | --- | --- |
| 同真子集几何、全环不同 | 上述已知环构造与signature不变性；T1有有限有理证书 | 构造本身已有直接先行 |
| 距离范数联合律 | [Royen2015式(1.2)](https://arxiv.org/pdf/1507.00528)的标准determinant Laplace变换；两实Gaussian行给指数−1 | 已知联合分布工具 |
| CDF差到严格正排名间隔 | 第5.3节：两个倒数相减，除以 `product(sj)` 后为两概率向量的卷积密度，再对独立目标长度积分 | 有专门化计算，但目前没有超出这条短推论的独立方法／锐界 |
| 显式下界和单位球 | 第5.4–5.5节：特征值密度下界、gamma积分、满秩线性方程及共同缩放 | 有限定量证书成立；未证明最优、宏观间隔或比同信息经典界更强 |

再次检查了不能省略的两个环节：卷积密度在开正象限连续，因此可从Laplace唯一性的几乎处处等式
取到对角线；球提升共用同一缩放，逐投影保留排序。未发现需要修改当前L0代码／协议的缺陷。
这仍是人工证明审计与有限证书，不是形式证明或独立T4评审。

**12.3 同一命题的容斥推论，说明解释边界。** 以下只分析“全部真子集bit律相同”
这一既有前提；不是新增冻结案例、数值验收或另一主命题。令两组bit向量的全1概率差为
`delta=P_-(I=1)-P_+(I=1)`。若S为恰好取1的位置集合，则容斥中只有全指标项可以不同：

```text
P_-(ones=S)-P_+(ones=S) = (-1)^(n-|S|)*delta.
P_-(Z=z)-P_+(Z=z) = binom(n,z)*(-1)^(n-z)*delta.
P_-(Z>=M)-P_+(Z>=M) = (-1)^(n-M)*binom(n-1,M-1)*delta.
```

最后一式来自相邻二项系数消去。对当前m=2环构造，
`delta=(-1)^(n-1)*Delta_n`，所以风险差为
`(-1)^(M+1)*binom(n-1,M-1)*Delta_n`：不同top-M门槛方向交替，
不存在一组对所有门槛都更安全的随机支配关系。
例如既有n=3的三个系数为 `+1,-2,+1`；这不把M=1/2记成本轮已运行的科学测试。

上述同一容斥机制已在2017式(3)出现；这里是对本笔记符号的推论，不作新贡献。
它还给出 `TV(Law(I-),Law(I+))=2^(n-1)*abs(delta)<=1`，
故全1概率差至多 `2^(1-n)`。这条普适限制不证明当前实际间隔很小；
同样，原 `3/16384` 只是下界，不能据它推断实测效应大小或采样所需次数。

**12.4 决策与未解内容。** 当前贡献门判定为
`covered_or_corollary_only / corollary_only`，发表状态仍为 `novelty_not_established`。
这是基于明确证明依赖链的研究取舍：已知构造加标准概率工具尚不足以支持本轮所要求的独立理论贡献；
不是宣称某篇先行已经逐字给出相同的排名下界，也不是证明今后不可能发表相关工作。

保留T1科学实现、原失败／成功回执和机制解释，停止JRC-CYCLE-1的T2升级。
不为确认已可解析证明的正差追加MC，不通过扩n/m/M、换摘要或转系统接口寻找正结果。
宽泛问题仍未解决：实际规模下同摘要的Gaussian可实现锐风险区间、与同信息经典界的严格增量、
及低成本摘要的可用条件均未获得。未来若恢复研究，先明确其中一个实质缺口及其先行差异，
再提出新命题；当前未把它们列为自动实验任务。

**12.5 本轮交付与验证。** 仅更新本L1笔记、START_HERE、STATUS及IMPLEMENTATION_PLAN；
不改原框架、旧Step5日志、保护清单、科学源、测试、环境或既有证据。
实际命令为 `python3 scripts/check_code_protection.py`、`git diff --check`，
另作一次性只读的变更路径／旧字节／Markdown链接锚点围栏审计。
本轮无实现变化，不重复481项实现回归；481/481是第11节已完成的T1验收结果。
实际审计通过：保护250项（L0=172/L1=75/L2=3）、72个本地链接、20个锚点及围栏；
变更恰为上述四项L1，原研究正文历史字节、科学／治理来源、旧日志及两个交付索引SHA256保持。
