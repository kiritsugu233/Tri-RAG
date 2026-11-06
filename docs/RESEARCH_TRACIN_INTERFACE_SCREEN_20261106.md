# 第六轮：具体TracIn分类头接口的结构/成本筛查（2026-11-06）

用户本轮明确“可以，继续”。完成第五轮唯一下一门要求的一个具体接口审查：公开源码/元数据、精确因子、回退/同预算基线及费用必要条件。此为L1进度索引，探索数学保存在本地未冻结笔记；无新科学实现、真实数据/模型/权重/梯度下载、采样、训练、agent派发或服务器。

## 当前交付

- [核心研究决定](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE6_RESEARCH_DECISION_ZH.md>)。
- [选定接口、官方来源与完整成本筛查](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE6_INTERFACE_COST_SCREEN_ZH.md>)。
- [代数化简、简单筛除与读字节门](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE6_REDUCTION_AND_COST_PROOFS_ZH.md>)。
- [确定性回执](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/DETERMINISTIC_CHECKS_STAGE6.json>)、[可重放命令](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE6_REPLAY_COMMANDS.md>)及[实际输出](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE6_REPLAY_CHECKS.txt>)。
- [公开论文/固定源码来源与SHA](research_prior_stage6_sources_20261106.json)。新增1PDF，累计52份/51项工作；Captum源码仅作为txt保存，不执行。

## 选定接口与结论

选定官方Captum CIFAR-10简单CNN的显式fc3.weight，84→10，单checkpoint CE梯度内积贡献、50000候选。精确因子缓存每点94个数，FP32池载荷18.8MB；不能只与840维展开梯度/重复模型forward比较。一个checkpoint贡献不等于全模型、多checkpoint TracIn或真实因果影响。

原TracIn [AppendixF](https://arxiv.org/abs/2002.08484v3)已直接讨论秩一因子点积/分开投影。该网络的ReLU输入还给便宜符号筛除：精确阈值为正时，输出因子点积非正的点可先排除。第五轮控制在此接口退化为经典Gaussian内积回归，不需要Tri。

在共同简单筛除之后，读m维摘要再回退的理想字节模型，需要额外排除比例f>m/84。m=32为38.10%；加精确h回退时总载荷25.2MB，主银行8.4MB不能当完整存储。同预算强基线可用同额外预算存PCA/量化/索引。没有真实过滤率、时间或物理I/O，实际费用门未通过，不能称实验失败或普遍不可能。

**当前不推进这个小分类头为论文主线，关闭该具体接口的自动实验推进。** 保留一般条件回归数学与该接口作为可能的数值夹具，不自动换更大模型/更多checkpoint投入。旧PQ关闭保持；无需服务器。无独立同行认证、新颖性或净收益结论。

## 来源与验证

固定Captum commit `b89cccbe0add104f734793be69488c078164c532`，保存核心实现/notebook/LICENSE和官方教程/CIFAR页面。源级发现教程的last-child绑定与显式fc3参考有区别，已单列，未运行教程或修改上游，不采用其打印性能数。数据/权重内容SHA和使用条件未核实，未下载它们。

13项精确有理数核恒等式、3项二分类/3项标量回归化简、载荷/额外过滤门槛、源码AST/来源SHA核对通过；TracIn pp3/14/15渲染核对。全部52PDF完整性和本地来源清单核验。分位数/规划尺度是数值说明，不是实际过滤率或有限样本下界。桌面资料包与ignored runs同步，旧笔记/文献保留，PDF全文不进Git。
