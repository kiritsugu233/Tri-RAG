# 晚交互接口：固定源码、简单摘要与最小测量提案

2026-11-07，承接用户“可以开始吧”。本轮完成只读来源审查与未冻结纸面设计，**未实施或运行新的科学实验**。

具体切口是官方 ColBERT/PLAID CPU 的两次质心筛除之后、残差解码/归一化/MaxSim 之前。默认 top-10 配置在此最多留下64篇；可以取消的是常驻内存代码访问与计算，不能称为SSD原向量读取收益。

## 交付与发现

- [固定源码和费用账本](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-07_共享筛除后摘要筛查/source-audit-v1/SOURCE_AUDIT_ZH.md>)：准确函数/行号、H/S₀、尚未支付费用与强先行。
- [两档简单摘要及实数证明](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-07_共享筛除后摘要筛查/source-audit-v1/SIMPLE_SUMMARY_MATH_ZH.md>)：每篇一个半径或每token一个半径；无Gaussian/Tri成分，无新颖性主张。
- [九个确切路径与一次受限诊断提案](</Users/guanghongxu/Desktop/HPC/Tri-law/2026-11-07_共享筛除后摘要筛查/source-audit-v1/MEASUREMENT_SCOPE_PROPOSAL_ZH.md>)：待科学实施的精确范围批准。
- [来源、SHA与确定性核对回执](research_postfilter_source_audit_20261107.json)。

源码改变了两个纸面细节：摘要半径须相对**归一化后的评分向量**；CPU segmented MaxSim有零下限而质心排序无此截断，不能直接用signed质心总分作上界。可用的clipped质心总分还需新增累计/输出，其费用不能记成免费复用。反例、修正公式和原文/源码阅读范围均已保存。

主比较固定未改动官方B0；同锚点B1仅归因。摘要、锚点评分、adapter、合并、全索引新增字节和离线构建均计费。两档摘要预算不同，当前不声称同字节优势。真实几何/净收益未知，未把源码切口存在记为前两门通过。

本轮新增PLAID2022/WARP2025/TACHIOM2026三份固定版本全文，官方源码固定三个commit并仅存txt；跨旧/新资料目录共55PDF/54项工作。WARP融合/直接残差评分、TACHIOM PQ refinement/early exit是后续必须面对的先行，尚未实施或做匹配竞争。

本地工作副本为 ignored runs/research-postfilter-screen-20261107/source-audit-v1；原四个入口文件保留。未下载模型/数据、安装依赖、启动agent/服务器或改旧L0。只完成四个有理数玩具界核对、两个语义反例和固定来源/AST核验；formal_certification=false。

下一步仅为审阅并批准提案中的精确路径与一次受限准备/诊断，然后实现、离线验收、同步GitHub，再在需要时请求Linux节点。当前无需开机，不自动进入Gaussian/Tri或重开已关闭PQ路线。
