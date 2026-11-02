# Gaussian RAG：真实 benchmark 缺口审计与第一步提案

2026-11-02。L1 非冻结研究说明。基线 `f80681b51c2940885f111f7d8faf41cfb27f3bbc`。
状态：`gap_candidate_identified / gaussian_fit_unmeasured / implementation_not_approved`。

## 1. 当前问题与已有证据

用户最新指示：先找真实应用 gap，判断 Gaussian 表示是否有质量与成本空间，再研究预算信息和 Tri 增量。
因此[概率桥研究](GAUSSIAN_RAG_RANK_BRIDGE_RESEARCH.md)第8节合成 A0 暂缓；不是当前前置任务。
旧方案4/5停止、旧 report/cert/test关闭及已记录失败均保留。

[VIBE论文](https://arxiv.org/pdf/2505.17810v3)§5.3、§6.1、Appendix H报告：自然问题等 OOD 查询在匹配 recall 时增加 ANN 搜索费用，训练分布与查询分布的差异影响后端效率。这是可研究的应用缺口。
但论文中通用量化方法仍是强对照，不能把缺口描述成“现有方法无法处理 OOD”。
Gaussian 不依赖语料训练、可降低距离计算维度，是待验证的适配理由；它同时引入排名失真及投影/重排费用，尚无胜出证据。
论文中的非归一化 MIPS/attention 困难不直接适用这里的欧氏 Gaussian 定律，不能改原始度量来制造适用性。

旧20k候选池、固定表示的12个免费最优M oracle全部质量不可行；最佳质量 Gaussian192/M256仍比合格PQ128慢，且有新增遗漏。
这排除了旧动作空间，不能推出所有完整真实基准上的 Gaussian 都不可行；更精确预算预测也不能修复那个旧动作空间。
依据为只读旧[实验日志](TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md)§36.8、§37.6、§39.5，不重新打开旧数据。

## 2. 一手数据与源码核验

[官方VIBE仓库](https://github.com/vector-index-bench/vibe/tree/7ac76c1f8320a411ce7c9cb3791bdd7f2349b30c)，固定提交 `7ac76c1f8320a411ce7c9cb3791bdd7f2349b30c`。
候选为[官方HotpotQA–Harrier文件](https://huggingface.co/datasets/vector-index-bench/vibe/blob/a91cc76a3ab66b0353032e8427b9e34dcee22ded/hotpotqa-harrier-640-normalized.hdf5)。

| 项目 | 已核验来源声明，尚非下载后实测 |
| --- | --- |
| 完整语料 | 5,233,329 × 640，float32，Harrier OSS v1 270m |
| 查询 | 96,845 learn；1,000 official test；自然问题、独立于语料来源 |
| 预处理 | 官方归一化向量；query prompt 为 web_search_query |
| 文件固定版本 | a91cc76a3ab66b0353032e8427b9e34dcee22ded |
| 文件大小 | 13,688,159,632 bytes |
| 文件 SHA256 | 08f1c68cc8bb1f24a29cb64d9ffc20ccc6a91c06d286538e4e659d4afb648114 |

源码审计的实质影响：

- [datasets.py](https://github.com/vector-index-bench/vibe/blob/7ac76c1f8320a411ce7c9cb3791bdd7f2349b30c/vibe/datasets.py)：HotpotQA明确 `ood=True`；若走默认 `ood=False`，部分看似问答的数据集实际从语料抽查询，不能仅凭名称认作真实问题检索。
- [distance.py](https://github.com/vector-index-bench/vibe/blob/7ac76c1f8320a411ce7c9cb3791bdd7f2349b30c/vibe/distance.py)：原生 normalized 距离为 `clip(1-dot,0,2)`。与单位球平方L2排名的等价须审计浮点范数和近同距；不重归一化输入或重写官方真值。
- [export_results.py](https://github.com/vector-index-bench/vibe/blob/7ac76c1f8320a411ce7c9cb3791bdd7f2349b30c/export_results.py)：原生 recall 阈值为第k真值距离加 `1e-3`，不是严格ID重合。原生 recall 为主，严格ID retention另列。
- [runner.py](https://github.com/vector-index-bench/vibe/blob/7ac76c1f8320a411ce7c9cb3791bdd7f2349b30c/vibe/runner.py)：prepare_query 和评测器重算距离不都在查询计时内。Gaussian投影、选候选和实际重排必须进入自有完整查询时钟；另列原生计时口径。
- [FAISS适配器](https://github.com/vector-index-bench/vibe/blob/7ac76c1f8320a411ce7c9cb3791bdd7f2349b30c/vibe/algorithms/faiss/module.py)：已有量化＋精排对照，不能只和全精度HNSW比较。

源码文本仅下载审计、未执行。忽略目录 `runs/benchmark-gap-audit-v1/source-20261102/` 保存8个来源快照，共121,280bytes；`source_audit.json` SHA256 为 `3fa2b7b3d93b819dd2adb36f6311991a64241a0266d5e64ad495923611507818`。
整份数据尚未取得或通过哈希校验；未语义读取官方test数组。

## 3. 已完成 Linux 预检与下载失败

用户选择“现有 Linux 服务器（推荐）”并提供连接，授权本次连接与资源预检。凭据不入库。
既有 known_hosts 中主机密钥与服务器呈现值一致，随后使用严格校验连接；最初新信任文件的未知密钥确认被自动审核拒绝，未绕过。

- 实际 cgroup 配额为14 CPU、120GiB内存；宿主112逻辑CPU/~1TB不作为实验配额。
- `/root/autodl-fs`可用194,225,750,016bytes（约180.9GiB）；`/root/autodl-tmp`约34.6GiB可用。
- 仓库 `/root/autodl-tmp/Tri-RAG-git`，分支正确，HEAD `78a5604e4c86c39252fb400669466eb7cc3aa752`，已跟踪文件干净；未检查未跟踪清单。没有pull/reset。
- 现有 Step5 环境包元数据为 NumPy1.26.4、SciPy1.13.0、FAISS CPU1.8.0；缺h5py。尚未完成FAISS后端符号运行核验，也未安装依赖。
- 首次官方文件获取于2026-11-02 06:52:24 UTC启动，0.269秒后URLError，保留0bytes，未完成哈希校验。远端失败回执：`/root/autodl-fs/gaussian-benchmark-space-v1/source-20261102/source_acquisition.json`。
- 随后只读网络诊断：huggingface.co DNS成功、HTTPS被对端重置（errno104）；hf-mirror.com DNS成功、首页HTTP200。镜像具体对象/整文件尚未核验，不能记作数据已可用。任何替代传输仍必须匹配上节官方SHA与大小。

以上只有资源/来源准备，没有新检索、索引构建、统计或Tri计算结果。

## 4. 第一阶段的可检验设计（待批准，不是冻结协议）

使用完整N，官方预处理与真值，k=100，主目标平均原生recall≥0.95，0.99为敏感性目标；严格ID结果单列。首阶段不研究Tri预算器，pilot和风险计算均为0。

先用learn中固定哈希选取256查询作开发面板（seed63001），不读官方test标签作选择。其余learn保留给未来另行划分；已观察面板不可当作后续独立认证数据。learn若无官方距离，从官方邻居ID和原生向量导出并明确标注，固定32查询做完整原空间真值审计。官方1000 test只在配置及选择结果封存后用于最终报告；首阶段不是正式统计认证。

Gaussian固定m分别为64/128/256/512，seed63013/63017；每个表示内固定矩阵，元素N(0,1/m)，不投影后归一化。候选M为100/200/400/800/1600/3200/6400。先做全语料投影空间精确搜索和原生原空间精排，分块流式，不存全查询×语料距离矩阵。

真实后端提案使用现有固定FAISS1.8.0，记录其与当前VIBE版本的差异，不宣称复现当前排行榜：

- 原空间HNSW：degree16/32、efConstruction200、seed63011，efSearch从100至6400的上述网格。
- Gaussian HNSW：degree32、efConstruction200；开发精确曲线中最小合格m及下一个m，至多两个m、两个seed；若无合格m，m512两seed作负对照。efSearch=M或2M，不混淆搜索努力与候选数。
- 压缩对照：IVF-PQ FastScan＋原空间精排，nlist4096、PQ160×4bit，固定100k语料训练seed63007；nprobe8/16/32/64/128，精排候选100/400/1600。后端不支持时明确阻塞，不静默换算法。
- 开发计时3轮轮换顺序；每方法/质量目标封存一个点后，官方test计时5轮。费用包含投影、搜索、选择、精排，另列建图、预投影、常驻原向量和索引内存/磁盘。

精确投影E与ANN候选A都不是另一者的集合上界。对原生GT身份集合G，严格损失满足：

```text
loss(A) = loss(E) + |G ∩ (E\A)|/k - |G ∩ (A\E)|/k.
```

ANN可能偶然找回投影精确topM外的原始邻居，不能假定其只增加损失。精确oracle失败排除的是对应固定排名/预算族。
每个固定m/seed内评估免费预算oracle，不允许逐query换表示。平均质量目标的可行性用 `mean(max_action recall)`；不能用“每query达95%失败”代替“平均95%不可行”。费用最优oracle用实测逐query动作成本和全局平均质量约束；可报告有证书的LP下界（允许随机化的乐观下界），不能把它称为已实现的确定预算器或用候选数代替延迟。

判定输出必须允许：质量无空间；质量可行但无成本收益；固定Gaussian有收益但无自适应增量；仅开发oracle显示预算空间；资源停止/证据不足。只有最后有预算空间才进入信息缺口研究，随后才比较Tri与同输入经验预测器。

## 5. 精确实施范围与验收提案

新科学协议必须显式保留本基准的原生内积度量与单独的投影L2诊断；不更改旧统一L2协议、旧缓存或旧测试。仅L1编排无法定义这些新的真值、数据角色、oracle和计时语义，因此需要新L0命名空间。

| 精确路径 | 拟定函数/变更与级别 |
| --- | --- |
| `mechanism/gaussian_benchmark_data_v1.py` | 新L0：verify_source、select_roles、native_recall、strict_retention；完整HDF5、稳定字符串ID映射、原生真值 |
| `mechanism/gaussian_benchmark_space_v1.py` | 新L0：project、exact_candidates、run_backend、decompose_loss、oracle_frontier；固定表示与真实费用 |
| `configs/gaussian_benchmark_space_v1.json` | 新L0：第4节固定网格、数据来源、种子和资源限额 |
| `tests/test_gaussian_benchmark_space_v1.py` | 新L0：小型离线HDF5、数值/指标/角色/后端/费用边界测试 |
| `scripts/run_gaussian_benchmark_space_v1.py` | 新L1且继承科学保护：来源校验、分阶段运行、资源终止、回执 |
| `docs/GAUSSIAN_BENCHMARK_SPACE_PROTOCOL_V1.md` | 新L0：冻结第4节协议、真值差异、成本定义及失败条件 |
| `scripts/run_tests.sh` | 只追加新测试模块的明确分类；不改旧测试门槛 |
| `docs/CODE_PROTECTION.md` | 只追加获批路径/实际批准引用及登记本身新hash |
| `docs/code_protection.json` | 只登记新增路径及上述获批治理变化；不更新无关科学基线 |

批准范围还需涵盖：独立临时环境增加h5py3.11.0（不升级旧NumPy/SciPy/FAISS或修改旧环境），官方wheel哈希核对；现有环境无法隔离复用时先报告，不替换旧依赖。
Linux固定源提交通过GitHub同步，先检查未跟踪/分叉，再fetch及ff-only；不reset。先本地完整回归，再远端新模块/后端预检；不能把本地测试冒充Linux完整验收。

h5py已只读取得官方PyPI元数据：Linux cp312 wheel SHA256 `aa6ae84a14103e8dc19266ef4c3e5d7c00b68f21d07f2966f0ca7bdb6c2761fb`；macOS cp39 arm64 wheel SHA256 `6c4b760082626120031d7902cd983d8c1f424cdba2809f1067511ef283629d4b`。尚未下载或安装。

实验总墙钟上限4小时，离线至多8 CPU线程，延迟单线程，RSS96GiB、产物100GiB、临时盘8GiB；不占GPU，不改宿主调度。逐表示处理，不持久化全部8份投影语料；保留实际使用的矩阵/索引及逐query记录。达限保留部分结果和失败，不自动延长。源获取另限15分钟一次，失败保留回执。

验证必须覆盖：来源/hash拒绝、完整N不裁剪、角色隔离、原生阈值和ID计分差异、范数/近tie审计、Gaussian尺度/无二次归一化、精确参考、ANN偶然恢复的带符号分解、平均质量oracle约束及证书、全部付费阶段计时、预算/资源停止、随机种子与指纹变化和重放。完成新测试和完整旧套件，保留失败，不放宽旧容差。

现有API/历史指纹和结果不变；新数据、协议、依赖和实现形成新指纹/目录 `gaussian-benchmark-space-v1`。未来Tri与统计认证不在此次实施范围。
本节尚未获得精确路径批准；此前T1或测试环境批准不移用于这里。依据[AGENTS.md](../AGENTS.md)：L0变更须先批准确切路径和意图。
