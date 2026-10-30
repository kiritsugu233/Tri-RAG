# Code protection register — 2026-09-07

权威规则：[AGENTS.md](../AGENTS.md)。机器清单：[code_protection.json](code_protection.json)。
原审查基线：`248c29e2243c60b3793458c8ee723eefba5a5855`。
当前受维护版本：`codex/tls-rag-step5-ann` 的 HEAD，已纳入审查修复提交
`62377d854ddd183925c4b62ca81653e9c14d9c34`。`codex/repo-review-code-protection` 保留在该提交作备份。
用户于 2026-10-23 明确要求新建 Step 5 分支、后续统一在该分支同步并清理冗余旧分支，
替代此前一直沿用 Step 4 分支的安排。仅更新工作流登记，不改变保护等级或审批规则。包含经用户批准的
[Tri-Law 数值 v2 修复](TRI_LAW_NUMERICAL_FIX.md)。机器清单逐项记录批准后的哈希。

| 等级 | 含义 | 变更要求 |
| --- | --- | --- |
| L0 | 科学核心、冻结协议/回归、源码绑定、治理规则 | 编辑前说明必要性、具体范围、兼容影响和测试计划；等待用户明确同意 |
| L1 | 实验编排、配置解析、工具与工作文档 | 当前任务授权内可修改；检查依赖并验证，不能改变 L0 语义或历史结果 |
| L2 | 展示/包标记等低风险内容 | 当前任务授权内处理；科学结论或指标定义变更仍按 L0 |

等级反映改动风险，**不是正确性评级**。Tri-Law 的 R1/R2 已按用户授权修复，原始反例证据保留；
详见[审查报告](REPOSITORY_REVIEW.md)。测试通过仅覆盖已有测试条件。
源码注册项给出直接导入该模块的测试，间接覆盖仍以完整回归为准。
全部模块做了结构/依赖/边界检查；重点模块追加人工契约核对，不宣称逐行形式证明。

保护标注放在登记文件，不插入冻结源码注释，避免更改源文件 SHA-256 使历史
bundle 失效。只读命令：`python3 scripts/check_code_protection.py`。
L0 变更检测到后不得自动重置文件或刷新哈希。批准信息必须来自真实用户消息，
本地 approval 文件不能自行创造授权。登记文件不自哈希，其变化必须对照 Git 审查。
当前工具不是强制 OS/GitHub 权限，也没有自动安装 hook 或 CI。

### Step 5 ANN v1 approved new namespace (2026-10-22)

用户在收到精确路径方案后要求“再仔细检查一遍，看看设计基线有没有问题，没问题就开始执行吧，先在本地改，改好之后同步到github”。
新增实现、复核修正、授权边界及服务器指令见 [ANN v1 契约](TLS_RAG_STEP5_ANN.md)。
旧科学源码、旧测试、依赖固定版本、实验配置及数据访问边界不变；登记仅追加新项。
本文只增加本节及新文件表，不削弱任何既有规则。

| New file | Level | Protected contract |
|---|---|---|
| `src/tri_rag_harness/tls_rag_step5_ann.py` | **L0** | New ANN retrieval, candidate membership, exact references and ties |
| `src/tri_rag_harness/tls_rag_step5_data.py` | **L0** | New source/subcorpus, lexical families, role separation and input identities |
| `src/tri_rag_harness/tls_rag_step5_policy.py` | **L0** | New sampled features, ANN ridge/residual fitting and deployable decisions |
| `src/tri_rag_harness/tls_rag_step5_protocol.py` | **L0** | Role reservations, seals, quality bounds and paired uncertainty |
| `src/tri_rag_harness/tls_rag_step5_benchmark.py` | **L0** | Tune selection, protected-stage orchestration and timing boundaries |
| `configs/tls_rag_step5_ann_v1.json` | **L0** | New-stage frozen parameters, roles, seeds and targets |
| `tests/test_tls_rag_step5_ann.py` | **L0** | New-stage retrieval, leakage, failure and reproduction regressions |
| `docs/TLS_RAG_STEP5_ANN.md` | **L0** | Approved reviewed design, statistical limitations and role boundaries |

保护也沿依赖传播：在 L1 caller、新文件、依赖版本、配置、测试中更换 L0 实现、
阈值/数据角色/比较规则，仍须 L0 事前审批。新增普通文件可在任务范围内登记；
不能顺便删除/降级已有项。整个已有源码兼容性仍受历史冻结规则约束。


## Approved A800 boundary test-fixture repair (2026-09-28)

用户在收到限定的 L0 方案后明确回复“同意，修复完后给我github同步的指令，上传到github上更新repo”。
仅修改 `tests/test_tls_rag_step4_probe.py` 的成功路径夹具：AutoDL A800 的 x86_64 Linux
环境连续两次将 32 条饱和分数确定性分成 1+31，使单条样本箱低于冻结的最少 8 条要求。
测试现对这 32 条独立校准查询注入 16+16 个远离裁剪边界的固定分数；实际生产分箱、
置信界、控制器、选择、角色关闭和复现路径仍全部执行。生产源码、最低样本数、协议、
阈值、真实数据角色、来源绑定和科学结论均未改变。实际失败和验证记录见
[STATUS](../STATUS.md) 的 2026-09-28 当前条目。

## Approved synthetic calibration test repair (2026-09-10)

用户在收到限定测试修复方案后回复“批准”。仅修改
`tests/test_tls_rag_step4_probe.py`：成功路径独立使用 32 条校准查询并检查分箱样本数，
新增固定 4+4 分箱的保守扩展/禁止 probe 回归。共享 fixture 不变；生产源码、
最低样本数、真实协议、阈值和既有测试断言不放宽。
实际失败证据、批准范围及本地验证记录见 [STATUS](../STATUS.md) 的当前条目。
负例只在该测试作用域注入分离的 sufficiency 分数，以确定性重现 4+4 分箱；
实际分箱、置信界、控制器及选择仍执行生产代码。没有运行时补丁或源绑定变化。

## Approved repaired acceptance entry (2026-09-10)

用户明确批准中文兼容方案并要求亲自提交推送。新入口属于 L0 历史数据兼容：
校验固定旧来源、重算特征、保留原策略和未开启 probe；不放宽既有校验器。
实际批准原文、数据影响和运行命令见 [兼容验收记录](TLS_RAG_STEP4_REPAIRED_ACCEPTANCE.md)。
登记新增文件及本文的说明哈希；既有保护等级、科学源码和审批规则均不改变。

| File | Level | Reason |
| --- | --- | --- |
| [`src/tri_rag_harness/tls_rag_step4_repaired_acceptance.py`](../src/tri_rag_harness/tls_rag_step4_repaired_acceptance.py) | **L0** | Approved historical provenance, recomputation and held-out access contract |
| [`tests/test_tls_rag_step4_repaired_acceptance.py`](../tests/test_tls_rag_step4_repaired_acceptance.py) | **L0** | Historical integrity, recomputation, selection/closure and reuse regressions |
| [`scripts/run_tls_rag_step4_repaired_acceptance.sh`](../scripts/run_tls_rag_step4_repaired_acceptance.sh) | **L1** | CPU command wrapper; inherited scientific contracts remain L0 |
| [`docs/TLS_RAG_STEP4_REPAIRED_ACCEPTANCE.md`](TLS_RAG_STEP4_REPAIRED_ACCEPTANCE.md) | **L0** | Approved compatibility, role and acceptance contract |

## Package source

| File | Level | Reason |
| --- | --- | --- |
| [`src/tri_rag_harness/__init__.py`](../src/tri_rag_harness/__init__.py) | **L2** | Package identity only |
| [`src/tri_rag_harness/attribution.py`](../src/tri_rag_harness/attribution.py) | **L1** | Historical oracle attribution runner; never deploy oracle inputs |
| [`src/tri_rag_harness/beir_dataset.py`](../src/tri_rag_harness/beir_dataset.py) | **L0** | Stable source IDs, duplicate-safe external-query split construction |
| [`src/tri_rag_harness/certification.py`](../src/tri_rag_harness/certification.py) | **L0** | Empirical-Bernstein formula, sample-size and certificate identity |
| [`src/tri_rag_harness/config.py`](../src/tri_rag_harness/config.py) | **L1** | Walking-skeleton configuration parsing; numerical contract changes inherit L0 |
| [`src/tri_rag_harness/embeddings.py`](../src/tri_rag_harness/embeddings.py) | **L0** | Input normalization and stable row/ID boundary |
| [`src/tri_rag_harness/indexes.py`](../src/tri_rag_harness/indexes.py) | **L0** | Exact squared-L2 ranking, tie contract and FAISS adapter semantics |
| [`src/tri_rag_harness/lid.py`](../src/tri_rag_harness/lid.py) | **L0** | Euclidean Hill/MLE LID, invalid-distance and fallback contract |
| [`src/tri_rag_harness/manifest.py`](../src/tri_rag_harness/manifest.py) | **L0** | Frozen run identity, split hashes and seeds |
| [`src/tri_rag_harness/mprime_sweep.py`](../src/tri_rag_harness/mprime_sweep.py) | **L1** | Historical tune-only global dimension diagnostic orchestration |
| [`src/tri_rag_harness/pdctp_calibration.py`](../src/tri_rag_harness/pdctp_calibration.py) | **L0** | Frozen pilot-LID/residual fitting and serialization mathematics |
| [`src/tri_rag_harness/pdctp_config.py`](../src/tri_rag_harness/pdctp_config.py) | **L1** | PDCTP configuration boundary; scientific parameter changes inherit L0 |
| [`src/tri_rag_harness/pdctp_dataset_audit.py`](../src/tri_rag_harness/pdctp_dataset_audit.py) | **L0** | Dataset eligibility, duplicate grouping and role witness |
| [`src/tri_rag_harness/pdctp_embedding_audit.py`](../src/tri_rag_harness/pdctp_embedding_audit.py) | **L0** | Frozen embedding/cache integrity and role-opening preconditions |
| [`src/tri_rag_harness/pdctp_features.py`](../src/tri_rag_harness/pdctp_features.py) | **L0** | Deployable feature definition, numerical canonicalization and forbidden inputs |
| [`src/tri_rag_harness/pdctp_fiqa_dataset.py`](../src/tri_rag_harness/pdctp_fiqa_dataset.py) | **L0** | FiQA text-only boundary, stable identity and corpus preparation |
| [`src/tri_rag_harness/pdctp_fiqa_query_cal.py`](../src/tri_rag_harness/pdctp_fiqa_query_cal.py) | **L0** | Cal-only oracle supervision, fitted candidate identities and retention targets |
| [`src/tri_rag_harness/pdctp_fiqa_query_cert.py`](../src/tri_rag_harness/pdctp_fiqa_query_cert.py) | **L0** | Protected cert access, frozen six-hypothesis family and reconstruction |
| [`src/tri_rag_harness/pdctp_fiqa_query_tune.py`](../src/tri_rag_harness/pdctp_fiqa_query_tune.py) | **L0** | Frozen selection, comparator eligibility and downstream policy reconstruction |
| [`src/tri_rag_harness/pdctp_foundation.py`](../src/tri_rag_harness/pdctp_foundation.py) | **L1** | Historical synthetic PDCTP orchestration; scientific behavior remains frozen |
| [`src/tri_rag_harness/pdctp_policies.py`](../src/tri_rag_harness/pdctp_policies.py) | **L0** | Inference input isolation and calibrated budget/fallback decision |
| [`src/tri_rag_harness/pdctp_protocol.py`](../src/tri_rag_harness/pdctp_protocol.py) | **L0** | Five-role leakage guard and terminal transition semantics |
| [`src/tri_rag_harness/pdctp_real_protocol.py`](../src/tri_rag_harness/pdctp_real_protocol.py) | **L0** | Pre-data protocol, role, alpha, power and parameter freeze |
| [`src/tri_rag_harness/pdctp_statistics.py`](../src/tri_rag_harness/pdctp_statistics.py) | **L0** | Paired bounded-mean confidence calculations and Bonferroni allocation |
| [`src/tri_rag_harness/pdctp_v3.py`](../src/tri_rag_harness/pdctp_v3.py) | **L0** | Frozen effective-Tri-LID model/curve repair and policy identities |
| [`src/tri_rag_harness/policies.py`](../src/tri_rag_harness/policies.py) | **L0** | Raw/compiled Tri-Predict and monotone policy scientific contract |
| [`src/tri_rag_harness/projection.py`](../src/tri_rag_harness/projection.py) | **L0** | Dense Gaussian variance, no post-normalization and cache fingerprint |
| [`src/tri_rag_harness/real_dimension_sweep.py`](../src/tri_rag_harness/real_dimension_sweep.py) | **L1** | Historical tune-only dimension selection runner; preserve protocol/results |
| [`src/tri_rag_harness/real_original_baseline.py`](../src/tri_rag_harness/real_original_baseline.py) | **L1** | Historical exact-original reference and evidence metric orchestration |
| [`src/tri_rag_harness/real_policy_certify.py`](../src/tri_rag_harness/real_policy_certify.py) | **L0** | Frozen SciFact cert gate, identities, metric and failure semantics |
| [`src/tri_rag_harness/real_policy_test.py`](../src/tri_rag_harness/real_policy_test.py) | **L0** | Terminal-cert binding and descriptive-only test gate |
| [`src/tri_rag_harness/real_policy_tune.py`](../src/tri_rag_harness/real_policy_tune.py) | **L0** | Tune-only candidate selection and frozen raw/compiled policy identities |
| [`src/tri_rag_harness/real_tune_diagnostics.py`](../src/tri_rag_harness/real_tune_diagnostics.py) | **L1** | Historical tune-only evidence/shuffle diagnostics; no protected outcome tuning |
| [`src/tri_rag_harness/reporting.py`](../src/tri_rag_harness/reporting.py) | **L2** | Render saved statistics; metric definitions and claims inherit L0 |
| [`src/tri_rag_harness/retrieval_benchmark.py`](../src/tri_rag_harness/retrieval_benchmark.py) | **L1** | Systems runner and separated work/timing accounting; backend contract is L0 |
| [`src/tri_rag_harness/run.py`](../src/tri_rag_harness/run.py) | **L1** | Historical synthetic walking skeleton; cert-selected fixed comparisons are diagnostic |
| [`src/tri_rag_harness/synthetic.py`](../src/tri_rag_harness/synthetic.py) | **L1** | Deterministic synthetic fixture distribution; frozen fixture changes inherit L0 |
| [`src/tri_rag_harness/text_embeddings.py`](../src/tri_rag_harness/text_embeddings.py) | **L0** | Pinned model, preprocessing/normalization and content-derived cache validation |
| [`src/tri_rag_harness/tls_rag_step2.py`](../src/tri_rag_harness/tls_rag_step2.py) | **L0** | Frozen exact prefix/retrieval state, two actions, labels and compatibility |
| [`src/tri_rag_harness/tls_rag_step3.py`](../src/tri_rag_harness/tls_rag_step3.py) | **L0** | Frozen observed-pair features, score fitting, CP controller and semantics |
| [`src/tri_rag_harness/tls_rag_step4.py`](../src/tri_rag_harness/tls_rag_step4.py) | **L0** | Frozen v1 readiness protocol and protected-role guards |
| [`src/tri_rag_harness/tls_rag_step4_nfcorpus.py`](../src/tri_rag_harness/tls_rag_step4_nfcorpus.py) | **L0** | Source-bound v2 dataset/model preparation and staged label boundary |
| [`src/tri_rag_harness/tls_rag_step4_probe.py`](../src/tri_rag_harness/tls_rag_step4_probe.py) | **L0** | Source-bound v2 parent pipeline, fixed gate and label-closure semantics |
| [`src/tri_rag_harness/tls_rag_step4_retention_v3.py`](../src/tri_rag_harness/tls_rag_step4_retention_v3.py) | **L0** | Frozen v3 retention targets, margins, 27-policy selection and parent reuse |
| [`src/tri_rag_harness/tri_law.py`](../src/tri_rag_harness/tri_law.py) | **L0** | Exact triplet law; numerical v2 repairs R1/R2 under recorded user approval |
| [`src/tri_rag_harness/tri_predict.py`](../src/tri_rag_harness/tri_predict.py) | **L0** | Frozen Raw Tri-Predict approximation and orthogonal conditional aggregation |
| [`src/tri_rag_harness/utils.py`](../src/tri_rag_harness/utils.py) | **L0** | Canonical JSON/array hashing used in scientific artifact identities |

## Tests

| File | Level | Reason |
| --- | --- | --- |
| [`tests/__init__.py`](../tests/__init__.py) | **L2** | Test package marker |
| [`tests/test_attribution.py`](../tests/test_attribution.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_beir_dataset.py`](../tests/test_beir_dataset.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_certification.py`](../tests/test_certification.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_code_protection.py`](../tests/test_code_protection.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_config_projection.py`](../tests/test_config_projection.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_end_to_end.py`](../tests/test_end_to_end.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_index_retrieval.py`](../tests/test_index_retrieval.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_lid_policy.py`](../tests/test_lid_policy.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_mprime_sweep.py`](../tests/test_mprime_sweep.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_pdctp_calibration.py`](../tests/test_pdctp_calibration.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_pdctp_dataset_audit.py`](../tests/test_pdctp_dataset_audit.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_pdctp_features.py`](../tests/test_pdctp_features.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_pdctp_fiqa_embedding_gate.py`](../tests/test_pdctp_fiqa_embedding_gate.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_pdctp_fiqa_query_cal.py`](../tests/test_pdctp_fiqa_query_cal.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_pdctp_fiqa_query_cert.py`](../tests/test_pdctp_fiqa_query_cert.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_pdctp_fiqa_query_tune.py`](../tests/test_pdctp_fiqa_query_tune.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_pdctp_foundation.py`](../tests/test_pdctp_foundation.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_pdctp_policies.py`](../tests/test_pdctp_policies.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_pdctp_protocol_statistics.py`](../tests/test_pdctp_protocol_statistics.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_pdctp_real_protocol.py`](../tests/test_pdctp_real_protocol.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_pdctp_v3.py`](../tests/test_pdctp_v3.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_real_dimension_sweep.py`](../tests/test_real_dimension_sweep.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_real_original_baseline.py`](../tests/test_real_original_baseline.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_real_policy_certify.py`](../tests/test_real_policy_certify.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_real_policy_test.py`](../tests/test_real_policy_test.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_real_policy_tune.py`](../tests/test_real_policy_tune.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_real_tune_diagnostics.py`](../tests/test_real_tune_diagnostics.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_retrieval_benchmark.py`](../tests/test_retrieval_benchmark.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_text_embeddings.py`](../tests/test_text_embeddings.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_tls_rag_step2.py`](../tests/test_tls_rag_step2.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_tls_rag_step3.py`](../tests/test_tls_rag_step3.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_tls_rag_step4.py`](../tests/test_tls_rag_step4.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_tls_rag_step4_probe.py`](../tests/test_tls_rag_step4_probe.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_tls_rag_step4_retention_v3.py`](../tests/test_tls_rag_step4_retention_v3.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_tri_law.py`](../tests/test_tri_law.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |
| [`tests/test_tri_predict.py`](../tests/test_tri_predict.py) | **L0** | Regression/conformance expectations protect frozen scientific behavior |

## Scripts and dependencies

| File | Level | Reason |
| --- | --- | --- |
| [`pyproject.toml`](../pyproject.toml) | **L0** | Pinned execution/dependency identity affects numerical and model reproducibility |
| [`requirements-embedding-e5.txt`](../requirements-embedding-e5.txt) | **L0** | Pinned execution/dependency identity affects numerical and model reproducibility |
| [`requirements-tls-rag-probe.txt`](../requirements-tls-rag-probe.txt) | **L0** | Pinned execution/dependency identity affects numerical and model reproducibility |
| [`scripts/audit_numerical_contracts.py`](../scripts/audit_numerical_contracts.py) | **L1** | Verification/launcher orchestration; changing scientific gates or expected evidence inherits L0 |
| [`scripts/check_code_protection.py`](../scripts/check_code_protection.py) | **L0** | Read-only governance guard; disabling or weakening requires scoped consent |
| [`scripts/run_tests.sh`](../scripts/run_tests.sh) | **L1** | Verification/launcher orchestration; changing scientific gates or expected evidence inherits L0 |
| [`scripts/run_tls_rag_step4_readiness.sh`](../scripts/run_tls_rag_step4_readiness.sh) | **L1** | Verification/launcher orchestration; changing scientific gates or expected evidence inherits L0 |

## Frozen configurations

| File | Level | Reason |
| --- | --- | --- |
| [`configs/pdctp_fiqa_e5_base_v2_embeddings.json`](../configs/pdctp_fiqa_e5_base_v2_embeddings.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/pdctp_fiqa_query_cal_v1.json`](../configs/pdctp_fiqa_query_cal_v1.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/pdctp_fiqa_query_cert_v1.json`](../configs/pdctp_fiqa_query_cert_v1.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/pdctp_fiqa_query_tune_v1.json`](../configs/pdctp_fiqa_query_tune_v1.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/pdctp_fiqa_real_protocol_freeze_v1.json`](../configs/pdctp_fiqa_real_protocol_freeze_v1.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/pdctp_fiqa_source_audit_v1.json`](../configs/pdctp_fiqa_source_audit_v1.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/pdctp_network_free_foundation_v1.json`](../configs/pdctp_network_free_foundation_v1.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/pdctp_v3_network_free_foundation_v1.json`](../configs/pdctp_v3_network_free_foundation_v1.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/real_scifact_dataset.json`](../configs/real_scifact_dataset.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/real_scifact_e5_base_v2_embeddings.json`](../configs/real_scifact_e5_base_v2_embeddings.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/real_scifact_fixed_dimension_tune.json`](../configs/real_scifact_fixed_dimension_tune.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/real_scifact_original_exact_tune.json`](../configs/real_scifact_original_exact_tune.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/real_scifact_policy_certify.json`](../configs/real_scifact_policy_certify.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/real_scifact_policy_test.json`](../configs/real_scifact_policy_test.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/real_scifact_policy_tune.json`](../configs/real_scifact_policy_tune.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/real_scifact_tune_diagnostics.json`](../configs/real_scifact_tune_diagnostics.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/retrieval_latency_100k_d768.json`](../configs/retrieval_latency_100k_d768.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/retrieval_latency_1m_d1024.json`](../configs/retrieval_latency_1m_d1024.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/retrieval_latency_1m_d1024_faiss_k1984.json`](../configs/retrieval_latency_1m_d1024_faiss_k1984.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/retrieval_latency_smoke.json`](../configs/retrieval_latency_smoke.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/synthetic_attribution_fresh.json`](../configs/synthetic_attribution_fresh.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/synthetic_mprime_sweep_extended_fresh.json`](../configs/synthetic_mprime_sweep_extended_fresh.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/synthetic_mprime_sweep_fresh.json`](../configs/synthetic_mprime_sweep_fresh.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/synthetic_mvp.json`](../configs/synthetic_mvp.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/tls_rag_step2_synthetic_v1.json`](../configs/tls_rag_step2_synthetic_v1.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/tls_rag_step3_synthetic_v1.json`](../configs/tls_rag_step3_synthetic_v1.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/tls_rag_step4_nfcorpus_probe_v2.json`](../configs/tls_rag_step4_nfcorpus_probe_v2.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/tls_rag_step4_protocol_v1.json`](../configs/tls_rag_step4_protocol_v1.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |
| [`configs/tls_rag_step4_retention_v3.json`](../configs/tls_rag_step4_retention_v3.json) | **L0** | Frozen experiment/config identity: roles, seeds, parameters and acceptance targets |

## Documentation and review evidence

| File | Level | Reason |
| --- | --- | --- |
| [`AGENTS.md`](../AGENTS.md) | **L0** | Scientific/protocol or governance contract; change requires prior scoped user consent |
| [`NEXT_AGENT_PROMPT.md`](../NEXT_AGENT_PROMPT.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`START_HERE.md`](../START_HERE.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`STATUS.md`](../STATUS.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`agent.md`](../agent.md) | **L0** | Scientific/protocol or governance contract; change requires prior scoped user consent |
| [`artifacts/pdctp_fiqa_e5_v1/report.md`](../artifacts/pdctp_fiqa_e5_v1/report.md) | **L0** | Historical generated report: immutable evidence, not current task authorization |
| [`artifacts/pdctp_fiqa_query_cal_v1/report.md`](../artifacts/pdctp_fiqa_query_cal_v1/report.md) | **L0** | Historical generated report: immutable evidence, not current task authorization |
| [`artifacts/pdctp_fiqa_query_tune_v1/report.md`](../artifacts/pdctp_fiqa_query_tune_v1/report.md) | **L0** | Historical generated report: immutable evidence, not current task authorization |
| [`artifacts/pdctp_fiqa_real_protocol_v1/report.md`](../artifacts/pdctp_fiqa_real_protocol_v1/report.md) | **L0** | Historical generated report: immutable evidence, not current task authorization |
| [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/ATTRIBUTION.md`](../docs/ATTRIBUTION.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/CALIBRATED_TRI_PREDICT_PROTOCOL.md`](../docs/CALIBRATED_TRI_PREDICT_PROTOCOL.md) | **L0** | Scientific/protocol or governance contract; change requires prior scoped user consent |
| [`docs/CALIBRATED_TRI_PREDICT_V3_DIAGNOSIS.md`](../docs/CALIBRATED_TRI_PREDICT_V3_DIAGNOSIS.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/CERTIFICATION.md`](../docs/CERTIFICATION.md) | **L0** | Scientific/protocol or governance contract; change requires prior scoped user consent |
| [`docs/CODE_PROTECTION.md`](../docs/CODE_PROTECTION.md) | **L0** | Scientific/protocol or governance contract; change requires prior scoped user consent |
| [`docs/EXPERIMENT_PROTOCOL.md`](../docs/EXPERIMENT_PROTOCOL.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/FIQA_EMBEDDING_GATE.md`](../docs/FIQA_EMBEDDING_GATE.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/FIQA_PROTOCOL_FREEZE.md`](../docs/FIQA_PROTOCOL_FREEZE.md) | **L0** | Scientific/protocol or governance contract; change requires prior scoped user consent |
| [`docs/FIQA_QUERY_CAL_GATE.md`](../docs/FIQA_QUERY_CAL_GATE.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/FIQA_QUERY_CERT_GATE.md`](../docs/FIQA_QUERY_CERT_GATE.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/FIQA_QUERY_TUNE_GATE.md`](../docs/FIQA_QUERY_TUNE_GATE.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/FIQA_SOURCE_AUDIT.md`](../docs/FIQA_SOURCE_AUDIT.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/IMPLEMENTATION_PLAN.md`](../docs/IMPLEMENTATION_PLAN.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/MARKDOWN_REVIEW.md`](../docs/MARKDOWN_REVIEW.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/MPRIME_SWEEP.md`](../docs/MPRIME_SWEEP.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/RAW_TRI_PREDICT_V1_BASELINE.md`](../docs/RAW_TRI_PREDICT_V1_BASELINE.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/REAL_DATA.md`](../docs/REAL_DATA.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/REAL_EMBEDDINGS.md`](../docs/REAL_EMBEDDINGS.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/REAL_RETRIEVAL.md`](../docs/REAL_RETRIEVAL.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/REPOSITORY_REVIEW.md`](../docs/REPOSITORY_REVIEW.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/RETRIEVAL_LATENCY.md`](../docs/RETRIEVAL_LATENCY.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/TLS_RAG_LOCAL_WORKFLOW.md`](../docs/TLS_RAG_LOCAL_WORKFLOW.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/TLS_RAG_STEP2_SYNTHETIC.md`](../docs/TLS_RAG_STEP2_SYNTHETIC.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/TLS_RAG_STEP3_SYNTHETIC.md`](../docs/TLS_RAG_STEP3_SYNTHETIC.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/TLS_RAG_STEP4_BRIEF.md`](../docs/TLS_RAG_STEP4_BRIEF.md) | **L0** | Scientific/protocol or governance contract; change requires prior scoped user consent |
| [`docs/TLS_RAG_STEP4_READINESS.md`](../docs/TLS_RAG_STEP4_READINESS.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/TLS_RAG_STEP4_REAL_PROBE.md`](../docs/TLS_RAG_STEP4_REAL_PROBE.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/TLS_RAG_STEP4_RETENTION_V3.md`](../docs/TLS_RAG_STEP4_RETENTION_V3.md) | **L0** | Scientific/protocol or governance contract; change requires prior scoped user consent |
| [`docs/TLS_RAG_STEP4_START_HERE.md`](../docs/TLS_RAG_STEP4_START_HERE.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/TRI_LAW_SEQUENTIAL_RAG_PROTOCOL.md`](../docs/TRI_LAW_SEQUENTIAL_RAG_PROTOCOL.md) | **L0** | Scientific/protocol or governance contract; change requires prior scoped user consent |
| [`docs/TRI_LAW_SEQUENTIAL_RAG_SPEC.md`](../docs/TRI_LAW_SEQUENTIAL_RAG_SPEC.md) | **L0** | Scientific/protocol or governance contract; change requires prior scoped user consent |
| [`docs/TRI_LAW_SPEC.md`](../docs/TRI_LAW_SPEC.md) | **L0** | Scientific/protocol or governance contract; change requires prior scoped user consent |
| [`docs/archive/TLS_RAG_PRE_STEP3_AUDIT_INDEX.md`](../docs/archive/TLS_RAG_PRE_STEP3_AUDIT_INDEX.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/code_protection.json`](../docs/code_protection.json) | **L0** | Scientific/protocol or governance contract; change requires prior scoped user consent |
| [`docs/review_numerical_evidence.json`](../docs/review_numerical_evidence.json) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |

## Approved numerical repair records

| File | Level | Reason |
| --- | --- | --- |
| [docs/TRI_LAW_NUMERICAL_FIX.md](TRI_LAW_NUMERICAL_FIX.md) | **L1** | Repair evidence and handoff; scientific/approval changes inherit L0 |
| [docs/tri_law_numerical_fix_evidence.json](tri_law_numerical_fix_evidence.json) | **L1** | Post-repair audit output; preserve original failure evidence |

## Approved A800 replay (2026-10-22)

User approved the detailed A800 plan, then chose Miniconda. New replay files below are L0;
observed probe is never reclassified as fresh. Existing launcher gains explicit subcommands
and exit handling, with scientific contracts inherited as L0. No existing levels downgraded.
See [contract and paths](TLS_RAG_STEP4_A800_REPLAY.md).

- **L0** `configs/tls_rag_step4_a800_replay_v1.json`: approved reconstructed-data replay contract/regression.
- **L0** `src/tri_rag_harness/tls_rag_step4_a800_replay.py`: approved reconstructed-data replay contract/regression.
- **L0** `tests/test_tls_rag_step4_a800_replay.py`: approved reconstructed-data replay contract/regression.
- **L0** `docs/TLS_RAG_STEP4_A800_REPLAY.md`: approved reconstructed-data replay contract/regression.

### A800 explicit mirror compatibility (2026-10-22)

User approved “实施测试并推送” for the specified mirror scope: new A800 module,
its tests and run document, plus protection hashes. Official logical URLs and
frozen revisions stay unchanged; explicit allowlisted transport and model hashes
are required. Historical adapter and algorithm are untouched.

## Approved retirement and planning handoff — 2026-10-22

用户批准删除11个旧任务/包装路径；活动登记中移除这些文件，删除前哈希与
固定 Git 位置保存在机器登记的批准历史。[审查索引](MARKDOWN_REVIEW.md)说明
约束保留位置。其余等级不变，检查逻辑不变，科学源码/配置/测试不变。

- **L1** `docs/TLS_RAG_STEP5_HANDOFF.md`: planning-only handoff; no ANN implementation authorization.


## Approved Step 5 mechanism study namespace (2026-10-25)

User consent: “我申请了A800节点，你开始工作吧，你连入ssh进去之后执行这次的两阶段测试”. This releases the hardware condition on the
2026-10-24 reviewed small-study plan in docs/IMPLEMENTATION_PLAN.md.
Only the new namespace below and this append-only registration are authorized;
all old scientific files, seals, roles and governance rules remain unchanged.

| Path | Level | Scope |
|---|---|---|
| `mechanism/tls_rag_step5_audit.py` | **L0** | Approved mechanism study; see reviewed plan |
| `mechanism/tls_rag_step5_exact_policy.py` | **L0** | Approved mechanism study; see reviewed plan |
| `configs/tls_rag_step5_mechanism_v1.json` | **L0** | Approved mechanism study; see reviewed plan |
| `tests/test_tls_rag_step5_mechanism.py` | **L0** | Approved mechanism study; see reviewed plan |
| `docs/TLS_RAG_STEP5_MECHANISM_PROTOCOL.md` | **L0** | Approved mechanism study; see reviewed plan |
| `scripts/run_tls_rag_step5_mechanism.sh` | **L1** | Approved mechanism study; see reviewed plan |

## 方案 A：原生源码保护补全与登记同步（2026-10-30）

用户在收到四个精确 L0 路径的中文方案后明确批准：

> 批准执行，执行完毕后更新github上的repo，不要另起branch，直接在step5的分支上提交

批准范围限于 `scripts/check_code_protection.py`、`tests/test_code_protection.py`、
本文件和 `docs/code_protection.json`。必要性、实际反例和验收证据见
[方案 A 记录](REPOSITORY_CLEANUP_REVIEW.md#proposal-a-protected-guard-and-register-repair)。
没有批准测试套件的大规模合并/删除，也没有改变任何科学算法、数据角色或已有等级。

检查器现在通过 Git 清单识别 `.c/.cc/.cpp/.cxx/.h/.hh/.hpp/.hxx`（后缀大小写不敏感），
并对 `src/tests/scripts/configs/mechanism` 补查被忽略的实现文件。原有 Python、shell、
TOML、配置及 requirements 规则保留。被忽略的根目录 runs/data/依赖环境/archives
不被递归遍历；检查器保持只读，不解包实验归档，不创建人类批准。

兼容影响：最终核查确认 Step4 A800 的 22 路径来源清单含本次获批修改的保护脚本，
因此 A800 新计算的来源身份会改变；之前“所有来源绑定不变”的概括不准确。
历史包及其哈希、加载器保持原样，按原精确提交复现，不在当前分支刷新旧包哈希。
当前五组 Step5 来源绑定保持一致；科学算法/配置本体均未修改。

下表补齐原审查遗漏的 40 项及上一轮新增的 4 个 L1 文档/证据项。
等级和理由直接取自既有机器清单；没有文件被降级或免除其继承的 L0 语义。

| File | Level | Reason |
| --- | --- | --- |
| [docs/TLS_RAG_STEP5_COST_AWARE_REVIEW.md](../docs/TLS_RAG_STEP5_COST_AWARE_REVIEW.md) | **L1** | User-authorized planning-only cost-aware strategy review; no executable protocol or protected implementation approval |
| [docs/step5_tri_feature_cost_review_evidence.json](../docs/step5_tri_feature_cost_review_evidence.json) | **L1** | Preserved synthetic profiling receipts and read-only development cost audit; scientific claims and approval changes inherit L0 |
| [scripts/analyze_tls_rag_step5_action_costs.py](../scripts/analyze_tls_rag_step5_action_costs.py) | **L1** | Read-only cost-space diagnostics using observed development records and existing actual calls; non-deployable oracle only, no retrieval or scientific contract changes |
| [docs/TLS_RAG_STEP5_ACTION_COST_SCREEN.md](../docs/TLS_RAG_STEP5_ACTION_COST_SCREEN.md) | **L1** | User-authorized action feasibility screen and follow-up scope proposal; not a frozen protocol or implementation approval |
| [docs/step5_action_cost_screen_evidence.json](../docs/step5_action_cost_screen_evidence.json) | **L1** | Read-only oracle cost-space receipts from already-observed development data; no independent evaluation or measured new-pipeline claim |
| [docs/TLS_RAG_STEP5_REUSE_AUDIT.md](../docs/TLS_RAG_STEP5_REUSE_AUDIT.md) | **L1** | User-requested read-only scan/distance/candidate reuse audit and scope clarification; proposed implementations remain subject to L0 approval |
| [mechanism/tls_rag_step5_reuse_probe.py](../mechanism/tls_rag_step5_reuse_probe.py) | **L0** | Approved isolated exact-distance cache and lazy stable selection; scientific geometry and query-bound cache contract |
| [tests/test_tls_rag_step5_reuse_probe.py](../tests/test_tls_rag_step5_reuse_probe.py) | **L0** | Approved offline exact-equality, ties, distance-count and query-isolation regressions for reuse probe |
| [scripts/profile_tls_rag_step5_reuse_probe.py](../scripts/profile_tls_rag_step5_reuse_probe.py) | **L1** | Approved local synthetic CPU timing/memory orchestration; inherits frozen reuse probe protocol |
| [docs/TLS_RAG_STEP5_REUSE_PROBE_PROTOCOL.md](../docs/TLS_RAG_STEP5_REUSE_PROBE_PROTOCOL.md) | **L0** | Approved local synthetic reuse probe input, fairness, timing and memory contract |
| [docs/TLS_RAG_STEP5_REUSE_PROBE_RESULTS.md](../docs/TLS_RAG_STEP5_REUSE_PROBE_RESULTS.md) | **L1** | Actual authorized local synthetic CPU reuse probe results and measured limitations; no new scientific settings or follow-on run authorization |
| [docs/step5_reuse_probe_evidence.json](../docs/step5_reuse_probe_evidence.json) | **L1** | Preserved complete timing/memory/equality receipts from the approved synthetic CPU probe; earlier artifacts and source bindings remain unchanged |
| [docs/TLS_RAG_STEP5_BASELINE_HEADROOM_PROPOSAL.md](../docs/TLS_RAG_STEP5_BASELINE_HEADROOM_PROPOSAL.md) | **L1** | User-requested read-only scope proposal to complete original-space/fixed-M baselines and perfect-information cost headroom; not implementation or input-access approval |
| [mechanism/tls_rag_step5_baseline_headroom.py](../mechanism/tls_rag_step5_baseline_headroom.py) | **L0** | Approved observed-input access, original/projected baselines, selection sealing and diagnostic perfect-information cost contract |
| [tests/test_tls_rag_step5_baseline_headroom.py](../tests/test_tls_rag_step5_baseline_headroom.py) | **L0** | Approved baseline geometry/ties, observed-input gates, cost-oracle and immutable-output CPU regressions |
| [scripts/run_tls_rag_step5_baseline_headroom.py](../scripts/run_tls_rag_step5_baseline_headroom.py) | **L1** | Approved local CPU/readonly export orchestration inheriting baseline-headroom protocol |
| [docs/TLS_RAG_STEP5_BASELINE_HEADROOM_PROTOCOL.md](../docs/TLS_RAG_STEP5_BASELINE_HEADROOM_PROTOCOL.md) | **L0** | Approved real-input baseline/headroom scientific comparison, data roles, quality and cost protocol |
| [docs/TLS_RAG_STEP5_BASELINE_HEADROOM_RESULTS.md](../docs/TLS_RAG_STEP5_BASELINE_HEADROOM_RESULTS.md) | **L1** | Actual authorized observed-text CPU baseline/free-oracle results and limitations; no policy retuning or new experiments |
| [docs/step5_baseline_headroom_evidence.json](../docs/step5_baseline_headroom_evidence.json) | **L1** | Preserved compact quality, all timing samples, memory, source/input bindings and solver receipts from the approved baseline/headroom comparison |
| [docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md](../docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md) | **L1** | User-requested living research plan and append-only process/result journal for three Tri-assisted ANN feasibility routes; proposed settings are not an executable frozen protocol or scoped L0 implementation consent |
| [mechanism/tls_rag_step5_ann_enhancement.py](../mechanism/tls_rag_step5_ann_enhancement.py) | **L0** | Approved route1 observed-input, source sealing, shadow/quality/cost screening and selection contract; routes2/3 execution deferred |
| [mechanism/tls_rag_step5_ann_enhancement.cpp](../mechanism/tls_rag_step5_ann_enhancement.cpp) | **L0** | Approved isolated native HNSW no-op recording and route1 risk-distance comparison; existing algorithms/libraries unchanged |
| [configs/tls_rag_step5_ann_enhancement_v1.json](../configs/tls_rag_step5_ann_enhancement_v1.json) | **L0** | Approved new route1 settings, data roles, costs, seeds and resource limits; routes2/3 inactive |
| [tests/test_tls_rag_step5_ann_enhancement.py](../tests/test_tls_rag_step5_ann_enhancement.py) | **L0** | Approved route1 geometry, native equality, information-isolation, accounting and failure regressions |
| [scripts/run_tls_rag_step5_ann_enhancement.py](../scripts/run_tls_rag_step5_ann_enhancement.py) | **L1** | Approved bounded route1 CPU runner, independent native build and immutable evidence; routes2/3 not executable |
| [mechanism/tls_rag_step5_ann_comparators_v2.py](../mechanism/tls_rag_step5_ann_comparators_v2.py) | **L0** | User-approved section16 route1 v2 comparator namespace; probability/numeric/cache/role semantics inherit L0; v1 and routes2/3 unchanged |
| [mechanism/tls_rag_step5_ann_comparators_v2.cpp](../mechanism/tls_rag_step5_ann_comparators_v2.cpp) | **L0** | User-approved section16 route1 v2 comparator namespace; probability/numeric/cache/role semantics inherit L0; v1 and routes2/3 unchanged |
| [configs/tls_rag_step5_ann_comparators_v2.json](../configs/tls_rag_step5_ann_comparators_v2.json) | **L0** | User-approved section16 route1 v2 comparator namespace; probability/numeric/cache/role semantics inherit L0; v1 and routes2/3 unchanged |
| [tests/test_tls_rag_step5_ann_comparators_v2.py](../tests/test_tls_rag_step5_ann_comparators_v2.py) | **L0** | User-approved section16 route1 v2 comparator namespace; probability/numeric/cache/role semantics inherit L0; v1 and routes2/3 unchanged |
| [scripts/run_tls_rag_step5_ann_comparators_v2.py](../scripts/run_tls_rag_step5_ann_comparators_v2.py) | **L1** | User-approved section16 route1 v2 comparator namespace; probability/numeric/cache/role semantics inherit L0; v1 and routes2/3 unchanged |
| [mechanism/tls_rag_step5_ann_continuation_v1.py](../mechanism/tls_rag_step5_ann_continuation_v1.py) | **L0** | User-approved section22 route2 first-trial namespace; persistent search/stopping/role/cost semantics inherit scoped L0; all prior sources frozen |
| [mechanism/tls_rag_step5_ann_continuation_v1.cpp](../mechanism/tls_rag_step5_ann_continuation_v1.cpp) | **L0** | User-approved section22 route2 first-trial namespace; persistent search/stopping/role/cost semantics inherit scoped L0; all prior sources frozen |
| [configs/tls_rag_step5_ann_continuation_v1.json](../configs/tls_rag_step5_ann_continuation_v1.json) | **L0** | User-approved section22 route2 first-trial namespace; persistent search/stopping/role/cost semantics inherit scoped L0; all prior sources frozen |
| [tests/test_tls_rag_step5_ann_continuation_v1.py](../tests/test_tls_rag_step5_ann_continuation_v1.py) | **L0** | User-approved section22 route2 first-trial namespace; persistent search/stopping/role/cost semantics inherit scoped L0; all prior sources frozen |
| [scripts/run_tls_rag_step5_ann_continuation_v1.py](../scripts/run_tls_rag_step5_ann_continuation_v1.py) | **L1** | User-approved section22 route2 first-trial namespace; persistent search/stopping/role/cost semantics inherit scoped L0; all prior sources frozen |
| [mechanism/tls_rag_step5_ann_continuation_v2.py](../mechanism/tls_rag_step5_ann_continuation_v2.py) | **L0** | Approved route2 v2 section26 development namespace; old continuation/search/law contracts unchanged; actual exact-scope consent recorded |
| [mechanism/tls_rag_step5_ann_continuation_v2.cpp](../mechanism/tls_rag_step5_ann_continuation_v2.cpp) | **L0** | Approved route2 v2 section26 development namespace; old continuation/search/law contracts unchanged; actual exact-scope consent recorded |
| [configs/tls_rag_step5_ann_continuation_v2.json](../configs/tls_rag_step5_ann_continuation_v2.json) | **L0** | Approved route2 v2 section26 development namespace; old continuation/search/law contracts unchanged; actual exact-scope consent recorded |
| [tests/test_tls_rag_step5_ann_continuation_v2.py](../tests/test_tls_rag_step5_ann_continuation_v2.py) | **L0** | Approved route2 v2 section26 development namespace; old continuation/search/law contracts unchanged; actual exact-scope consent recorded |
| [scripts/run_tls_rag_step5_ann_continuation_v2.py](../scripts/run_tls_rag_step5_ann_continuation_v2.py) | **L1** | Approved route2 v2 section26 development namespace; old continuation/search/law contracts unchanged; actual exact-scope consent recorded |
| [README.md](../README.md) | **L1** | Repository landing page and current navigation; no executable or scientific contract |
| [docs/REPOSITORY_MAP.md](../docs/REPOSITORY_MAP.md) | **L1** | Repository-wide lifecycle/dependency map; authoritative levels remain in protection registry |
| [docs/REPOSITORY_CLEANUP_REVIEW.md](../docs/REPOSITORY_CLEANUP_REVIEW.md) | **L1** | 2026-10-30 structural review, deletion analysis and pending exact L0 proposals; no consent implied |
| [docs/repository_cleanup_evidence.json](../docs/repository_cleanup_evidence.json) | **L1** | Read-only inventory, guard counterexample, preservation and offline-test receipts; no new scientific results |

## 方案 B：获批测试分层与 M01 合并（2026-10-30）

用户收到[中文逐项方案 B](TEST_SUITE_SIMPLIFICATION_PROPOSAL.md)及四路径范围后回复：

> 执行

批准父版本 `7b9d44b8c85bbfb964afcaaf8fdbd1c2485ae299`。范围仅为：
`tests/test_config_projection.py` 的两个缓存身份方法合为一个参数化方法；
`scripts/run_tests.sh` 增加 full/step5/legacy 及完整收集检查；本文记录规则/补提案行；
`docs/code_protection.json` 记录实际批准和本文/该测试的两个内容哈希。
必要 L1 交接文档按方案同步。其余 45 个测试文件和全部科学源码保持原字节；
三个错误身份和同值合法场景都保留，无永久删除、阈值放宽或来源身份迁移。

无参数入口仍为完整套件；当前 full=429、step5=231、legacy=198。
step5 与 legacy 互斥且并集等于 full。日常迭代可显式选 step5；共享核心、
治理/依赖/历史实现变更及任何实现交付提交/推送前仍须 full。
部分通过不得称为全仓通过；未选集合不得记作通过或跳过。全仓收集/导入错误、
未归类新测试文件必须失败。当前计数是实测清单，不是未来新增测试的数量上限。
新增测试继续按其契约登记和审查，不能借分组绕过 L0。

| File | Level | Reason |
| --- | --- | --- |
| [docs/TEST_SUITE_SIMPLIFICATION_PROPOSAL.md](TEST_SUITE_SIMPLIFICATION_PROPOSAL.md) | **L1** | User-requested Chinese proposal B with per-method keep/merge/daily-retirement coverage mapping; planning only, no implementation or weakened-test consent |

该行同步已有机器登记，不降级或修改原有理由。提案文档本身不创造授权；
执行授权来自上面引用的用户回复。实际验收记录见提案末节及 [STATUS](../STATUS.md)。
