# 第五轮：不对称因子表示与可观测风险（2026-11-06）

后续“可以，继续”已完成[第六轮具体接口筛查](RESEARCH_TRACIN_INTERFACE_SCREEN_20261106.md)。选定Captum最后分类头，原TracIn直接先行、精确缓存与简单符号筛除削弱独立立项依据；实际费用未测，不推进这个小接口为论文主线。累计52PDF/51项工作。以下第五轮一般数学保留，不被本次特例选题决定否定。

用户本轮明确“继续下一步”。完成一个具体表示改变的纸面推导、直接先行核对和确定性自检。此为L1研究索引，完整探索证明未冻结、保存在本地资料包；无科学实现、真实梯度数据、随机采样、模型训练、agent派发或服务器。

## 本轮交付

- [核心研究决定、先行与完整费用](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE5_PRIOR_COST_DECISION_ZH.md>)。
- [完整数学：候选诱导子空间与残差区间](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE5_ASYMMETRIC_FACTOR_MATH_ZH.md>)。
- [下一门可执行brief：只读结构/成本筛查](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE5_NEXT_GATE_ZH.md>)，未派发/未开放新科学执行。
- [确定性核对](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/DETERMINISTIC_CHECKS_STAGE5.json>)、[复核命令](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE5_REPLAY_COMMANDS.md>)和[实际输出](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE5_REPLAY_CHECKS.txt>)。
- [本轮公开来源与读取范围](research_prior_stage5_sources_20261106.json)：新增5PDF，累计51份/50项工作；另1条出版方captcha未下全文的元数据，不计全文总数。固定2026 v6的新题名/范围，未混引v5。

桌面与ignored `runs/research-prior-propositions-20261106/` 同步，PDF不进Git，原文献/失败/四轮笔记保留。数学为本agent自检，无独立同行认证或新颖性保证。

## 得到的具体机制

保留候选的一侧精确因子，只Gaussian压缩另一侧。候选与查询诱导一个已知小子空间，在其上做条件回归；草图残差可以给原分数的有效区间，无需先算原分数、读取原另一因子或额外存自Gram。条件化维度不超过候选因子宽度，不能把查询整体秩误当最低草图维度。

候选因子宽度为1时是标准标量Gaussian回归t区间；一般矩阵响应用条件Gaussian/Wishart独立性和凸性得到保守Student尾，未冒称精确t。固定候选池和预定有限前缀经union可保护淘汰/精确回退。只对独立于银行的固定查询、对银行随机性成立；不是固定银行对所有未来适应性查询的保证。

基础先行包括 [Li等2006](https://hastie.su.domains/Papers/Ping/COLT_rp.pdf) 的条件正态和 [Ahfock等Biometrika2021 §3.1](https://doi.org/10.1093/biomet/asaa062) 的Gaussian草图回归推断；广义风险估计还对照2024/2026版本。**当前有效保证不需要Tri，也未建立新颖性或实际净收益。**

## 决定与费用边界

本候选通过纸面可观测性门，暂保留到一个具体接口的只读结构/成本筛查；不直接进入大模型实验。保留因子有硬存储底线；若同时保存原因子用于回退，银行是额外存储。同预算PCA/量化、低秩表示和精确因子筛除均可利用相同额外预算，必须作为强对照。查询投影、逐点设计/回归、残差、调度和实际回退都要记账。

下一轮只确定一侧因子足够廉价、候选秩和参考变换合法、总存储/回退匹配、完整费用有空间的一个具体接口；不再强行以Tri为必选核心。不满足即关闭此候选；旧PQ关闭不变。无需开启服务器。

## 验证范围

12项精确有理数矩阵分数/方差/正交性检查、4项Student混合尾求积、预算及多重比较尺度核对通过；5份新PDF相关页渲染及全部51PDF字节/SHA/正文验证。初次SciPy df2逆分位数的探索性核对未过，改用解析df2分位数，保留原核对阈值、原命令和失败证据。数值求积不是严格区间认证，无真实分布/过滤/时间/读取字节或发表贡献结论。
