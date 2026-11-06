# 第四轮：独立重复草图的具体机制审查（2026-11-06）

用户本轮明确“继续”。本轮只做纸面机制、先行排重、确定性证明核查；没有科学实现、真实数据、随机采样、训练、agent派发或服务器任务。此为L1进度索引，完整数学保存在未冻结的本地资料包；不改旧科学契约/PQ关闭记录。

## 当前交付

- [核心结论与研究决定](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE4_RESEARCH_DECISION_ZH.md>)。
- [具体机制、有限方差CS保证、费用门](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE4_MECHANISM_AUDIT_ZH.md>)。
- [开放反例族、交换性充分条件、完整资源代数](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE4_VOTING_AND_RESOURCE_PROOFS_ZH.md>)，末尾含确定性复核命令。
- [确定性计算回执](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/DETERMINISTIC_CHECKS_STAGE4.json>)及[实际复核输出](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE4_REPLAY_CHECKS.txt>)。
- [新增公开文献来源/阅读范围/本地SHA](research_prior_stage4_sources_20261106.json)。新增3PDF，累计46份/45项工作；保留旧版本和Bandit首次404下载记录。

桌面资料包与ignored `runs/research-prior-propositions-20261106/` 同步；PDF不进Git。前三轮笔记与manifest完整保留。无独立同行/agent证明认证，也未建立可发表新颖性。

## 实质结果与边界

1. 一般双侧Gaussian的符号多数不认证原分数均值。一个正真实分数差的开放矩阵族，单票错误率超过1/2；其中10001票错误多数约99.958%。有闭式证明及严格不等式，不是随机模拟结果。此反例不否定原dense Gaussian Tri-Law，也不否定原始均值估计。
2. 原始分数+已知方差上界+Catoni重尾置信序列可构造合法的固定查询/固定池top-k淘汰/精确回退基线。对所有点、所有前缀union覆盖，严格处理共享块、适应性阈值/读取和查询独立条件；不能外推为固定银行对任意未来查询的同时保证。
3. 有效保证不需要Tri；[Catoni CS v6 §6](https://arxiv.org/abs/2202.01250v6)和[BanditMIPS ICML2024 Algorithm1](https://proceedings.mlr.press/v235/tiwari24a.html)是直接强基线/先行，不能把组合架构当作已成立的新颖性。
4. 同完整银行字节下独立重复降低方差，却增加查询因子投影；同投影工作时共享网格又可通过更多存储降低方差。因子存储、精确本体/JVP及全部控制开销均需另算，没有完整资源支配结果。
5. 若完整下一阶段费用已不低于现在精确剩余集合，完美筛除也无净收益。当前没有实测梯度读取/JVP费用、gap分布、过滤率或总时间；旧PQ结果不能移作该证据。

## 研究决定

停止通用符号投票版本；均值-CS方案仅保留为有效基线，暂不直接立项为Tri新方法或启动服务器。下一轮须先给出改变完整资源前沿的一个具体表示/结构差异及可否定条件，不能用继续调风险公式、δ或重复次数替代科学差异。此判断不是所有Tri应用不可能的证明。

## 验证

全部46PDF字节/SHA/正文及来源清单核验；本轮3个相关原文页渲染。反例闭式与确定性求积、二项尾、严格有理数比较、二阶矩和固定资源方差/MOM尺度计算复核通过。求积误差仅为估计；MOM表只是特定保守规则的尺度说明，不是最优样本下界。无冻结源/测试/科学契约修改，无需服务器。
