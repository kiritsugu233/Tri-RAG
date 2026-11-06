# 新研究：先行与命题阶段交付（2026-11-06）

用户目标：优先可发表贡献，方法／领域可调整。实际当前指令：“开始吧，先完成先行与命题阶段，你先收集相关文献，保存到本地文件夹内方便查阅，除了梯度草图的相关课题，也看看逐查询风险控制最新的预印本里面是如何做的。”

本页是 L1 研究进度索引。数学推导留在独立、未冻结的本地探索笔记中；不更改旧 Tri-Law、数据角色、协议、历史结论或审批范围。新方向没有实验／训练／服务器任务已启动。

## 本地交付

桌面查阅包：`/Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题`。

- [阅读入口](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/README.md>)。
- [先行与逐查询风险控制综述](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/PRIOR_AND_RISK_REVIEW_ZH.md>)。
- [探索命题、证明及反例](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/PROPOSITIONS_ZH.md>)。
- [后续 agent brief](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-06_先行与命题/analysis/NEXT_AGENT.md>)，尚未派发新的 agent。

相同工作包保存在 ignored `runs/research-prior-propositions-20261106/`。PDF 全文不进入 Git；[来源回执](research_prior_sources_20261106.json)同步版本、来源、校验和阅读范围，远端可据此重取公开原文。桌面绝对链接仅供本机访问。

## 已完成内容

- 27 份 PDF，26 项工作；LoGra 的 2024 预印本和 2025 正式版分别保留。
- `manifest.json/.csv`、`references.bib`、PDF SHA-256、正文文本与阅读顺序。
- 梯度主线：TRAK、LoGra、Efficient Sketches、GraSS、LoRIF v4、EOGP、TRAK Theory、超参选择及 LM 归因评测设计。
- 风险主线：PQR、ConANN、CTR、ReCIRC、non-monotone CRC、conformal ordering、CopRoute，以及 ADSampling、RaBitQ、SRS、PUFFINN、CRC 背景。
- 数学主线：张量投影、TRP、边际信息、Exact Limits、经典二次型分布。
- 区分相关章节重点阅读与背景定向阅读；未声称逐行复核所有论文或复现作者性能。

一个 OpenReview 早期条目遇到验证，保留失败记录；正式 NeurIPS 2025 LoGra 已下载替代，未将未核实的 ICLR/Scaling Influence Functions 信息用于结论。没有绕过网站验证。

## 先行判断

“草图／量化＋逐查询风险控制＋精化”已有很近先行。[PQR v1](https://arxiv.org/abs/2611.07579v1)覆盖未访问 IVF 分区、精确尾分位数和自适应半径；这是投稿预印本，不是已录用结果。ConANN、CTR、ReCIRC 的输入自适应规则与风险保证对象不同，需要分别核对。

[EOGP](https://arxiv.org/abs/2609.37842v1)已有半白化 PCA／一位存储与匹配字节的排序评测；[LoRIF v4](https://arxiv.org/abs/2601.21929v4)进一步处理低秩存储／I/O。新方向也必须面对强表示基线，不能仅比较 dense Gaussian。

保留的窄问题：结构化梯度草图中，因子几何怎样改变比较风险，以及如何用廉价信息选择两侧／各层预算。探索笔记给出“同输出大小不同风险”“MSE 更低但排序更差”的严格构造和条件分布表示；标准矩／二次型工具明确标为基线，没有认定可发表新颖性或实际收益。

## 验证与下一步

27 个 PDF 的类型、页数、字节、SHA 与正文提取通过；重点公式页面已渲染核对。反例数值用 SciPy F 尾与独立精确 Beta/binomial 有理数恒等式交叉核对，枚举所有 K=64 的正整数因子确认 MSE 分配结论。它们是确定性函数／代数核对，不是真实梯度实验或数学正确性的独立认证。

下一步先扩展数学排重、独立复核证明、核查多 token／预条件与廉价可观测信息。只有这些成立才准备具体的最小数据诊断。若所需信息与直接打分同样昂贵，或贡献只剩标准积分重写，应结束候选。旧 PQ 路线保持关闭，目前不需要服务器。
