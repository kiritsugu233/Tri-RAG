# Step 4 A800 重建复现 v1（2026-10-22）

用户已批准具体 A800 方案，并授权使用现有 Miniconda（“那就用miniconda吧，应该是我记错了”）。
开发目录 `/Users/guanghongxu/Query-Adaptive-Tri-RAG`，分支仍是
`codex/tls-rag-step4-retention-v3`，基础提交 `41697efcff924139a775de11e673caf4a2d68662`。
本次修改仍由用户提交推送。新的代码提交必须单独记录，不能把新代码声称为原基础提交。

## 科学边界

此前 Genoa 的 probe 已开启并达到三个目标（用户提供的日志，非本机原始产物）：
Row4 t=.900 q=.50，retention .9301，Hit@context .6445，平均原空间距离352.62。
AutoDL 目前没有完整旧 v2 输入。本次从固定公开来源重新生成数据和向量，重新拟合并
重放 tune 选择过程，仅作已观察查询上的跨平台复现，不是恢复原模型参数或独立验证。

新配置引用旧数据协议与 retention-v3 协议指纹。固定模型/revision、CPU float32 编码、
float64 归一化、m'=128、投影seed24103、split seed54109、角色256/512/256/256、
预算32/128/512、27个候选和原验收阈值。官方 dev/test qrels 不下载、不开启。
不运行旧 v2 选择、不生成旧 schema binding、不伪造 no_tune_candidate 历史。
不执行 ANN、LLM、答案生成或完整 RAG 延迟测试。

新 schema：`tls_rag_a800_rebuilt_binding_v1`、`tls_rag_a800_replay_result_v1`。
根结果明确 historical_probe_observed=true、independent_validation=false、
certified=false、latency_claim=false。状态为 replay_targets_met/replay_targets_failed/
no_tune_candidate；退出码0不代表目标通过。engine_replay/保留原引擎术语，仅作内部审计，
不能脱离根结果声明新 holdout。选择与历史不一致时原样报告，不据此追调或强制选择Row4。

## 文件与入口

新增 `configs/tls_rag_step4_a800_replay_v1.json`、
`src/tri_rag_harness/tls_rag_step4_a800_replay.py` 和对应测试。
已有科学源文件、旧配置、旧测试保持不变。
唯一主启动入口仍是 `scripts/run_tls_rag_step4_repaired_acceptance.sh`：

```bash
sh scripts/run_tls_rag_step4_repaired_acceptance.sh a800-prepare /root/autodl-fs/tls-rag-step4/a800-input-v1-01
prepare_rc=$?
echo "Preparation exit code: ${prepare_rc}"
```

只有prepare_rc=0、精确HEAD/环境检查通过、保护检查和全套测试通过才单独执行：

```bash
sh scripts/run_tls_rag_step4_repaired_acceptance.sh a800-replay /root/autodl-fs/tls-rag-step4/a800-input-v1-01 /root/autodl-fs/tls-rag-step4/a800-replay-v1-01
replay_rc=$?
echo "Replay exit code: ${replay_rc}"
```

不使用set -e或set -eu。每步检查退出码，失败停止调度，保留交互终端和文件。
两个输出根都必须从未存在。显式根可解析AutoDL挂载链接，内部文件禁止符号链接。
新输入根的a800-replay-attempt为原子尝试预约，失败也保留；不触及旧预约。
该保护不能检测人为复制/删除记录，不是OS访问控制。

## 环境与持久化

远端目标checkout `/root/autodl-tmp/Tri-RAG-git`；持久产物 `/root/autodl-fs/tls-rag-step4/`。
现有Miniconda `/root/miniconda3/bin/conda`；已克隆base到独立环境
`/root/autodl-tmp/conda-envs/tri-rag-step4`，不修改base。
base检查发现Python3.12.3、Torch2.8.0+cu128、NumPy2.3.2，缺少SciPy等。
保留Torch版本，新环境按仓库安装NumPy1.26.4、SciPy1.13.0、ST3.4.1、
Transformers4.48.3、PyArrow19.0.1；安装前保存解析报告，安装后保存版本和pip check。
不通过升级/降级来绕过测试失败。A800只记录硬件，本轮编码和检索均走CPU。

运行前必须核实实际部署提交与本地待交付内容一致、查看状态、检查完整依赖，
再执行保护检查和完整测试；环境变更或失败不允许静默继续。
本地/远端实际完成状态以STATUS记录为准，不把计划命令当作执行结果。
