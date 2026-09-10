# Step 4 审查修复版有效性验收

2026-09-10。工作目录固定为 `/Users/guanghongxu/Query-Adaptive-Tri-RAG`，
工作分支固定为 `codex/tls-rag-step4-retention-v3`。审查备份分支不改动。
本次用户明确要求：修改和测试由助手完成，**本地提交与 GitHub 推送由用户执行**；
此要求覆盖本次任务的默认自动提交推送规则。

## 批准与版本

用户在收到中文兼容方案后批准：

> 批准，你先修改吧，修干完后给我指令由我本地提交到github上去，然后在slurm的登陆节点git pull同步，等salloc到genoa节点后给我运行测试指令

批准范围：新兼容入口、测试、启动脚本、本文、保护登记，以及 START_HERE、
Step 4 入口、STATUS、实施计划。既有 Tri-Law、v2/v3 源码、测试和协议不修改。
保护规则和等级不降级。本入口属于 L0：显式引入历史数据兼容契约。

运行使用 Tri-Law numerical v2，保留 retention-v3 的 27 候选、拟合、校准和
选择语义。新 binding schema 为 `tls_rag_repaired_retention_binding_v1`；
根结果 schema 为 `tls_rag_repaired_retention_acceptance_v1`。
`evaluation/` 保存原 v3 引擎格式，明确通过 lineage 绑定新的来源文件。
文件夹或结果版本变化不需要创建新 Git 分支。

## 兼容行为与边界

- 仅接受历史提交 `c08cd3b6800888c8d2a5a5f8c58a77d883a84ede` 的完整 v2 源码指纹集。
  预期值由 Git 对象内容计算并固定在新源码中；不接受任意旧版本或新旧混合指纹。
- 核对数据包、模型、协议、角色及旧 evaluation 的文件哈希和身份，必须是
  `no_tune_candidate` 且没有 probe 准备或开启记录。旧文件始终只读。
- 复用已验证的归一化向量、文本和 ID。保持原角色；核对角色互斥和固定 seed 的
  查询族排序。不能证明未记录的人为标签查看不存在，也不保证语义问题族 iid。
- 重新计算三个开发角色的全部前缀特征和保留率监督。旧特征只作哈希审计，
  不复制为当前特征。拟合与校准后在 tune 选择，只有选定策略后才准备 probe。
- 标签读取发生在相应决策文件关闭后。probe 准备前再次核对旧输入和当前源码；
  tune 未通过则不准备 probe，不访问其标签，不自动放宽阈值。
- 旧运行旁的 `.tls-rag-repaired-probe-claims/` 按向量哈希和角色记录原子预约，
  阻止本入口对同一来源改输出名后再次开启 probe。该目录在旧运行根目录之外。
  一旦预约，即使随后中断也保留；不要删除预约并重跑。它是审计保护，不是 OS 安全边界，
  不能发现移走/复制数据、删除记录或其他入口的未记录运行。
- 所有运行文件保存到新的、原先不存在的输出目录。根 manifest 包括嵌套
  evaluation 文件；来源绑定记录旧数据哈希和当前源码哈希。系统环境保存在
  `evaluation/evaluation_environment.json`。预约记录含绝对输出路径；科学结果相同
  不意味着不同路径下的预约元数据逐字节相同。

## 验收目标

原 v3 目标不变：平均 exact top-10 retention ≥ 0.90；Hit@context 相对 exact-original
下降 ≤ 0.02；平均原空间距离计算次数严格 < 512。先要求 tune 合格，再检查
选定策略的 probe。Rows 2–4 用相同 tune 选定阈值和余量分位数作配对比较。
通过这些经验指标不自动证明 Tri-Law 增益、正式认证或服务延迟优势。

状态解释：`heldout_targets_met` 为三个独立留出目标均通过；
`heldout_targets_failed` 为留出验收失败；`no_tune_candidate` 为 tune 无合格候选、
probe 保持关闭。异常为输入/完整性/执行故障。退出码 0 表示流程完成，
**不等于有效性通过**，必须查看 `result.json`。

## 本地提交与 GitHub 推送（用户执行）

从固定目录检查并只暂存本次 10 个文件，避免提交实验归档：

```bash
(
set -eu
cd /Users/guanghongxu/Query-Adaptive-Tri-RAG
test "$(git branch --show-current)" = codex/tls-rag-step4-retention-v3
test -z "$(git diff --cached --name-only)"
python3 scripts/check_code_protection.py
git diff --check
git status --short
git add src/tri_rag_harness/tls_rag_step4_repaired_acceptance.py \
  tests/test_tls_rag_step4_repaired_acceptance.py \
  scripts/run_tls_rag_step4_repaired_acceptance.sh \
  docs/TLS_RAG_STEP4_REPAIRED_ACCEPTANCE.md \
  docs/code_protection.json docs/CODE_PROTECTION.md \
  START_HERE.md docs/TLS_RAG_STEP4_START_HERE.md STATUS.md docs/IMPLEMENTATION_PLAN.md
git diff --cached --stat
git commit -m "Add reviewed-source Step 4 retention acceptance"
git push origin codex/tls-rag-step4-retention-v3
git rev-parse HEAD
)
```

保留最后一行提交 SHA，供集群对照。若已有其他暂存修改，先检查并区分；不自动清空暂存区。

## Slurm 登录节点同步（不运行实验）

使用之前规划的独立 Step 4 checkout，保留 `/home/users/u0001611/Tri-RAG` 中的旧工作。
下面没有新建 Git 分支；目录不存在时仅克隆同一个现有分支：

```bash
(
set -eu
cd /home/users/u0001611
if [ ! -d Tri-RAG-step4-retention-v3 ]; then
  git clone --single-branch --branch codex/tls-rag-step4-retention-v3 \
    https://github.com/kiritsugu233/Tri-RAG.git Tri-RAG-step4-retention-v3
fi
cd Tri-RAG-step4-retention-v3
test -z "$(git status --porcelain --untracked-files=no)"
git switch codex/tls-rag-step4-retention-v3
git pull --ff-only origin codex/tls-rag-step4-retention-v3
git rev-parse HEAD
)
```

确认集群 SHA 与本地推送 SHA 一致。若出现本地修改或分叉，停止同步并检查，不 reset/clean。
登录节点只做 Git 同步和路径检查。

## Genoa 节点：环境、测试和真实验收

不需要 A100，也不需要重新下载模型、安装依赖或计算向量。使用原 `tri-rag`
micromamba 环境。初始资源建议为 4 CPU、16 GB、1 小时（未实测资源上界）；
线程设为 1，以便复现。使用集群实际可用的 account/QOS；如无额外要求：

```bash
salloc --partition=genoa --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:00:00
srun --pty bash
```

必须在获得 allocation 并进入计算节点后执行以下命令。默认旧路径来自此前交接，
尚未直接核实集群文件；命令会检查存在性，若不在该路径，替换为实际完整旧运行根目录。
输出目录固定，禁止覆盖或通过自动日期后缀反复开启 probe。

```bash
(
set -eu
: "${SLURM_JOB_ID:?Enter the allocated Genoa compute node first}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate tri-rag
cd /home/users/u0001611/Tri-RAG-step4-retention-v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
hostname
git rev-parse HEAD
python3 scripts/check_code_protection.py
sh scripts/run_tests.sh

tls_previous=/home/users/u0001611/Tri-RAG-step4-real-probe/step4-nfcorpus-v2-01
tls_output=/home/users/u0001611/Tri-RAG-step4-retention-v3/acceptance-reviewed-v3-01
test -f "$tls_previous/bundle/binding.json"
test -f "$tls_previous/evaluation/artifact_manifest.json"
test ! -e "$tls_output"
sh scripts/run_tls_rag_step4_repaired_acceptance.sh "$tls_previous" "$tls_output"
)
```

新入口运行完整的验证、特征重算、拟合、校准、tune 选择以及条件满足后的 probe。
不用再运行旧 v2 evaluate，也不使用旧 `run_tls_rag_step4_retention_v3.sh` 作为当前入口。
最终查看根目录 `report.md` 和 `result.json`；若中断保留输出与预约记录再分析。

## 本地验证记录

- 新增兼容测试：19/19 通过，5.806 秒。
- 完整 `sh scripts/run_tests.sh`：289 项，288 通过，1 项可选 real-FAISS 跳过，36.368 秒。
- 保护检查：182 个登记文件，L0=122、L1=57、L2=3；原有科学文件与登记项未变化。
- v1 CLI 独立重放两次，六个产物指纹与历史记录完全一致。
- Shell 语法、CLI help、`git diff --check` 和本次 Markdown 链接/围栏检查通过。

实际命令、日志和临时产物位置见 [STATUS](../STATUS.md)。测试仅使用 CPU 合成数据，
不读取本地实验归档，不产生真实 NFCorpus 效果结论。修改未暂存、未提交、未推送。
