# Code protection register — 2026-09-07

权威规则：[AGENTS.md](../AGENTS.md)。机器清单：[code_protection.json](code_protection.json)。
原审查基线：`248c29e2243c60b3793458c8ee723eefba5a5855`。
当前受维护版本：`codex/tls-rag-step4-retention-v3` 的 HEAD，已纳入审查修复提交
`62377d854ddd183925c4b62ca81653e9c14d9c34`。`codex/repo-review-code-protection` 保留在该提交作备份。
用户于 2026-09-07 明确要求此分支安排；仅更新工作流登记，不改变保护等级或审批规则。包含经用户批准的
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

保护也沿依赖传播：在 L1 caller、新文件、依赖版本、配置、测试中更换 L0 实现、
阈值/数据角色/比较规则，仍须 L0 事前审批。新增普通文件可在任务范围内登记；
不能顺便删除/降级已有项。整个已有源码兼容性仍受历史冻结规则约束。


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
| [`scripts/run_tls_rag_step4_probe.sh`](../scripts/run_tls_rag_step4_probe.sh) | **L1** | Verification/launcher orchestration; changing scientific gates or expected evidence inherits L0 |
| [`scripts/run_tls_rag_step4_readiness.sh`](../scripts/run_tls_rag_step4_readiness.sh) | **L1** | Verification/launcher orchestration; changing scientific gates or expected evidence inherits L0 |
| [`scripts/run_tls_rag_step4_retention_v3.sh`](../scripts/run_tls_rag_step4_retention_v3.sh) | **L1** | Verification/launcher orchestration; changing scientific gates or expected evidence inherits L0 |
| [`scripts/slurm_pdctp_foundation.sh`](../scripts/slurm_pdctp_foundation.sh) | **L1** | Verification/launcher orchestration; changing scientific gates or expected evidence inherits L0 |

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
| [`AGENT_CALIBRATED_TRI_PREDICT.md`](../AGENT_CALIBRATED_TRI_PREDICT.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`AGENT_CALIBRATED_TRI_PREDICT_STEP3.md`](../AGENT_CALIBRATED_TRI_PREDICT_STEP3.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP1.md`](../AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP1.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP2.md`](../AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP2.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP3.md`](../AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP3.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
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
| [`docs/TLS_RAG_STEP2_TO_STEP3.md`](../docs/TLS_RAG_STEP2_TO_STEP3.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/TLS_RAG_STEP3_START_HERE.md`](../docs/TLS_RAG_STEP3_START_HERE.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/TLS_RAG_STEP3_SYNTHETIC.md`](../docs/TLS_RAG_STEP3_SYNTHETIC.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
| [`docs/TLS_RAG_STEP3_TO_STEP4_HANDOFF.md`](../docs/TLS_RAG_STEP3_TO_STEP4_HANDOFF.md) | **L1** | Task/reference/review documentation; scientific/approval changes inherit L0 |
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
