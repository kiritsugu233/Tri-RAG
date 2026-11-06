# 第三轮：Tri迁移的定量边界与研究决定（2026-11-06）

用户本轮明确“继续”。完成纸面推导、直接先行排重和确定性核对；没有科学实现、真实数据实验、随机采样、服务器任务或agent派发。下列为L1进度索引，探索证明保存在未冻结的本地资料包，不改旧科学契约。

## 本地交付

- [第三轮核心结论与研究决定](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE3_RESEARCH_DECISION_ZH.md>)。
- [完整定量判据与证明](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/STAGE3_QUANTITATIVE_CRITERIA_ZH.md>)。
- [确定性核对回执](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/DETERMINISTIC_CHECKS_STAGE3.json>)。
- [新增文献来源和阅读范围](research_prior_stage3_sources_20261106.json)。新增6PDF，合计43份／42项工作；保留原版本、失败记录与各轮manifest。

桌面与ignored `runs/research-prior-propositions-20261106/` 同步；PDF全文不进Git。证明为本agent自检，没有独立同行认证或已建立的新颖性。

## 得到的定量结论

1. 两条查询差向量的矩阵子空间若满足一个收缩几何条件，可控制双侧Gaussian输出与dense输出的TV距离，从而给Tri概率加迁移误差。一般界保守且不自动低成本；平坦核心等子类已有更强先行。
2. 比较差可分解为两个独立线性Wishart项和一个混合余项。Berry–Esseen与平滑给简单Gaussian矩代理的绝对风险误差界，未把大误差当作小失败率认证。
3. 真实风险与简单代理逐动作相差≤ε时，按代理最小化风险的损失至多比精确风险最优高2ε。多比较要按margin聚合风险，不能混同总MSE。
4. 对风险目标δ，安全代理规则的选型费用在oracle前沿C*(δ)与C*(δ−2ε)之间（相应可行集非空时）。因此小风险差仍可能跨越大费用台阶；不能由2ε风险界推出小费用收益。
5. 若输入秩小于min(r,s)，完整输出TV等于1；这不否定某个特定排序事件仍可近似正确。两个近似判据都是充分条件，不是穷尽分类。

## 先行与研究决定

[CountGauss，ICML2016，Theorem1](https://proceedings.mlr.press/v48/kapralov16.html)已经通过TV将Gaussian投影下任意应用事件的保证转移到快速替代映射。[Li–Woodruff2021](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.APPROX/RANDOM.2021.35)已有Gaussian矩阵乘积的联合TV逼近。Wiener chaos的相关Gaussian/TV极限与定量正态近似也有强先行。

**当前不建议把“Tri迁移到双侧梯度草图”的宽泛构想继续作为主贡献。** 本轮得到的是适用边界和费用逻辑，尚未找到廉价摘要、独立于简单代理的增量及可利用的实际费用台阶。停止扩展通用迁移证明，不启动在线控制器／系统实验，也不自动另开领域试验。

如后续继续，只核查一个具体机制或输出否定：精确列出摘要的来源与费用、对同信息margin/矩代理/校准的差异，以及能取消哪项实际工作。不能将继续推一般公式或补泛泛文献当作独立里程碑。

## 验证

新增6份PDF类型/页数/字节/SHA/正文提取核验；相关定理页渲染核对。12项精确有理数线性/混合分解及不相关性检查、9项条件协方差矩核对、解析尺度与平滑常数一致性检查完成。无真实梯度结果；数值优化或求积的误差估计不是严格区间证书。当前不需要服务器。
