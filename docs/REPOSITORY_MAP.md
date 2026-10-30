# Repository map — 2026-10-30

Navigation for current Step 5 and retained experimental families. Baseline:
`50defc35bc9f42f5b24d2dedf5f45a89e51257ad` (242 tracked files).
[START_HERE](../START_HERE.md) and [STATUS](../STATUS.md) describe the current task.
[AGENTS](../AGENTS.md) / [machine register](code_protection.json) remain authoritative
for protection. Lifecycle below does not lower any level or authorize execution.

## Layers and dependency direction

```text
scripts/ + configs/          versioned orchestration and immutable settings
    |                        (runners select a scientific namespace)
    +--> mechanism/          Step5 isolated research implementations/native bridges
    |       |                current continuation_v2 -> continuation_v1
    |       |                continuation_v2.cpp includes continuation_v1.cpp
    |       +--> src/tri_rag_harness/ scientific/geometry/retrieval/data primitives
    +----------> src/tri_rag_harness/ staged harnesses and historical CLIs
                         |
                         +--> tests/         tiny synthetic/offline fixtures
                         +--> runs/, data/   external runtime data, ignored by Git
```

The latest v2 runner imports the v2 implementation, which inherits the v1
Runtime/Session/Native. Both use route1 enhancement helpers for hashing,
exclusive writes, reranking, metrics and accounting. The exact reuse/baseline
probe additionally depends on `tls_rag_step5_audit.ExactScan` and
`tls_rag_step5_exact_policy.PrefixSession`; those are retained dependencies.
`tls_rag_step5_policy` still uses Step3 observed-pair features. Old filenames
therefore cannot be treated as unused modules.

## Lifecycle and entry points

| Family | Role now | Entry / retained record |
| --- | --- | --- |
| Shared law/geometry/search/cache/statistics | Protected primitives used across versions | `src/tri_rag_harness/{tri_law,projection,indexes,lid,utils,...}.py`; [law spec](TRI_LAW_SPEC.md) |
| Raw/Calibrated Tri-Predict, SciFact/FiQA | Historical frozen CLIs, fixtures and protocol/result replay | [Raw baseline](RAW_TRI_PREDICT_V1_BASELINE.md), [PDCTP protocol](CALIBRATED_TRI_PREDICT_PROTOCOL.md) |
| TLS synthetic Steps2/3 | Frozen trajectories/features; some shared code remains imported | [Step2](TLS_RAG_STEP2_SYNTHETIC.md), [Step3](TLS_RAG_STEP3_SYNTHETIC.md) |
| Step4 readiness/probe/retention/repaired/A800 | Historical versioned replay and source-bound guards | [Step4 index](TLS_RAG_STEP4_START_HERE.md) |
| Step5 ANN v1 | Frozen no-tune-candidate experiment; later roles closed | [ANN contract](TLS_RAG_STEP5_ANN.md) |
| Exact mechanism/cache/baseline studies | Completed studies; retain shared backend functions and failures | [Mechanism](TLS_RAG_STEP5_MECHANISM_PROTOCOL.md), [reuse](TLS_RAG_STEP5_REUSE_PROBE_RESULTS.md), [baselines](TLS_RAG_STEP5_BASELINE_HEADROOM_RESULTS.md) |
| Route1 enhancement/comparators | Completed negative speed screens; shared helper imports retained | [Journal §§14/19](TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md) |
| Route2 continuation v1/v2 | Latest completed study; resumable backend retained, learned/Tri integration not accepted | Same journal §§24/27–31; `scripts/run_tls_rag_step5_ann_continuation_v2.py` |
| Routes3/4 | No implementation namespace; plans only | Same journal §§7/21; future exact scope required |

The packaged console command `tri-rag-harness` runs the original walking skeleton
(`tri_rag_harness.run:main`), not the latest Step5 research runner. `mechanism/`
and `scripts/` run from a checkout; pyproject currently packages only `src/`.
This describes existing packaging, not a request to change it.

## Source bindings and deletion decisions

Step5 ANN v1 `source_binding()` hashes its config and all54 package Python files.
Step4 A800 binds22 explicit paths; continuation v2 binds13, comparator v2 binds8,
route1 binds7 and baseline/headroom binds10. These lists were captured before
cleanup and compared after. L1 source files can therefore still be bound by an
L0 historical contract. Reordering imports, moving files or deleting an old CLI
can invalidate an experiment even if an import scanner sees no current caller.

Static unused-import leads include intentional re-exports: v2 runners import
`metrics`, `save`, etc. through their mechanism modules. Do not apply automatic
unused-import deletion or common-helper extraction to these frozen files.
The initial navigation cleanup changed only the unbound L1 test launcher.
The subsequently approved repair A also changes the governance guard/test.
All five captured Step5 bindings stay unchanged; the historical A800 source list
also binds the guard, so that identity changes. Keep old replay artifacts at their
original commit; see [review](REPOSITORY_CLEANUP_REVIEW.md).

## Per-file implementation disposition

All162 tracked paths in `src/`, `mechanism/`, `scripts/`, `tests/`, `configs/` are
listed below. Import counts are AST evidence only; direct CLI/build/hash references
also matter. Protection levels are copied from the review baseline registry.
Tests and configs are retained as conformance/protocol records, not marked dead
because they have no Python importers. No whole source file is approved for deletion.

### Package source

| File | Level | Family | Disposition/evidence |
| --- | --- | --- | --- |
| [src/tri_rag_harness/__init__.py](../src/tri_rag_harness/__init__.py) | L2 | Harness/build/test infrastructure | Retain; 4 code / 4 test importers |
| [src/tri_rag_harness/attribution.py](../src/tri_rag_harness/attribution.py) | L1 | Historical exact retrieval/diagnostics | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/beir_dataset.py](../src/tri_rag_harness/beir_dataset.py) | L0 | Harness/build/test infrastructure | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/certification.py](../src/tri_rag_harness/certification.py) | L0 | Shared scientific primitives | Retain; 10 code / 2 test importers |
| [src/tri_rag_harness/config.py](../src/tri_rag_harness/config.py) | L1 | Harness/build/test infrastructure | Retain; 5 code / 4 test importers |
| [src/tri_rag_harness/embeddings.py](../src/tri_rag_harness/embeddings.py) | L0 | Shared scientific primitives | Retain; 14 code / 2 test importers |
| [src/tri_rag_harness/indexes.py](../src/tri_rag_harness/indexes.py) | L0 | Shared scientific primitives | Retain; 10 code / 4 test importers |
| [src/tri_rag_harness/lid.py](../src/tri_rag_harness/lid.py) | L0 | Shared scientific primitives | Retain; 11 code / 1 test importers |
| [src/tri_rag_harness/manifest.py](../src/tri_rag_harness/manifest.py) | L0 | Harness/build/test infrastructure | Retain; 1 code / 0 test importers |
| [src/tri_rag_harness/mprime_sweep.py](../src/tri_rag_harness/mprime_sweep.py) | L1 | Historical exact retrieval/diagnostics | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/pdctp_calibration.py](../src/tri_rag_harness/pdctp_calibration.py) | L0 | Historical PDCTP | Retain; 4 code / 3 test importers |
| [src/tri_rag_harness/pdctp_config.py](../src/tri_rag_harness/pdctp_config.py) | L1 | Historical PDCTP | Retain; 1 code / 2 test importers |
| [src/tri_rag_harness/pdctp_dataset_audit.py](../src/tri_rag_harness/pdctp_dataset_audit.py) | L0 | Historical PDCTP | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/pdctp_embedding_audit.py](../src/tri_rag_harness/pdctp_embedding_audit.py) | L0 | Historical PDCTP | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/pdctp_features.py](../src/tri_rag_harness/pdctp_features.py) | L0 | Historical PDCTP | Retain; 9 code / 5 test importers |
| [src/tri_rag_harness/pdctp_fiqa_dataset.py](../src/tri_rag_harness/pdctp_fiqa_dataset.py) | L0 | Historical PDCTP | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/pdctp_fiqa_query_cal.py](../src/tri_rag_harness/pdctp_fiqa_query_cal.py) | L0 | Historical PDCTP | Retain; 2 code / 2 test importers; CLI |
| [src/tri_rag_harness/pdctp_fiqa_query_cert.py](../src/tri_rag_harness/pdctp_fiqa_query_cert.py) | L0 | Historical PDCTP | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/pdctp_fiqa_query_tune.py](../src/tri_rag_harness/pdctp_fiqa_query_tune.py) | L0 | Historical PDCTP | Retain; 1 code / 1 test importers; CLI |
| [src/tri_rag_harness/pdctp_foundation.py](../src/tri_rag_harness/pdctp_foundation.py) | L1 | Historical PDCTP | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/pdctp_policies.py](../src/tri_rag_harness/pdctp_policies.py) | L0 | Historical PDCTP | Retain; 4 code / 3 test importers |
| [src/tri_rag_harness/pdctp_protocol.py](../src/tri_rag_harness/pdctp_protocol.py) | L0 | Historical PDCTP | Retain; 6 code / 5 test importers |
| [src/tri_rag_harness/pdctp_real_protocol.py](../src/tri_rag_harness/pdctp_real_protocol.py) | L0 | Historical PDCTP | Retain; 3 code / 1 test importers; CLI |
| [src/tri_rag_harness/pdctp_statistics.py](../src/tri_rag_harness/pdctp_statistics.py) | L0 | Historical PDCTP | Retain; 3 code / 3 test importers |
| [src/tri_rag_harness/pdctp_v3.py](../src/tri_rag_harness/pdctp_v3.py) | L0 | Historical PDCTP | Retain; 0 code / 1 test importers |
| [src/tri_rag_harness/policies.py](../src/tri_rag_harness/policies.py) | L0 | Shared scientific primitives | Retain; 9 code / 5 test importers |
| [src/tri_rag_harness/projection.py](../src/tri_rag_harness/projection.py) | L0 | Shared scientific primitives | Retain; 22 code / 6 test importers |
| [src/tri_rag_harness/real_dimension_sweep.py](../src/tri_rag_harness/real_dimension_sweep.py) | L1 | Historical exact retrieval/diagnostics | Retain; 4 code / 2 test importers; CLI |
| [src/tri_rag_harness/real_original_baseline.py](../src/tri_rag_harness/real_original_baseline.py) | L1 | Historical exact retrieval/diagnostics | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/real_policy_certify.py](../src/tri_rag_harness/real_policy_certify.py) | L0 | Historical exact retrieval/diagnostics | Retain; 2 code / 3 test importers; CLI |
| [src/tri_rag_harness/real_policy_test.py](../src/tri_rag_harness/real_policy_test.py) | L0 | Historical exact retrieval/diagnostics | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/real_policy_tune.py](../src/tri_rag_harness/real_policy_tune.py) | L0 | Historical exact retrieval/diagnostics | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/real_tune_diagnostics.py](../src/tri_rag_harness/real_tune_diagnostics.py) | L1 | Historical exact retrieval/diagnostics | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/reporting.py](../src/tri_rag_harness/reporting.py) | L2 | Harness/build/test infrastructure | Retain; 1 code / 0 test importers |
| [src/tri_rag_harness/retrieval_benchmark.py](../src/tri_rag_harness/retrieval_benchmark.py) | L1 | Historical exact retrieval/diagnostics | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/run.py](../src/tri_rag_harness/run.py) | L1 | Harness/build/test infrastructure | Retain; 2 code / 1 test importers; CLI |
| [src/tri_rag_harness/synthetic.py](../src/tri_rag_harness/synthetic.py) | L1 | Harness/build/test infrastructure | Retain; 3 code / 1 test importers |
| [src/tri_rag_harness/text_embeddings.py](../src/tri_rag_harness/text_embeddings.py) | L0 | Shared scientific primitives | Retain; 10 code / 3 test importers; CLI |
| [src/tri_rag_harness/tls_rag_step2.py](../src/tri_rag_harness/tls_rag_step2.py) | L0 | TLS synthetic Step2 | Retain; 2 code / 2 test importers; CLI |
| [src/tri_rag_harness/tls_rag_step3.py](../src/tri_rag_harness/tls_rag_step3.py) | L0 | TLS synthetic Step3 | Retain; 4 code / 2 test importers; CLI |
| [src/tri_rag_harness/tls_rag_step4.py](../src/tri_rag_harness/tls_rag_step4.py) | L0 | Step4 replay/probe | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/tls_rag_step4_a800_replay.py](../src/tri_rag_harness/tls_rag_step4_a800_replay.py) | L0 | Step4 replay/probe | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/tls_rag_step4_nfcorpus.py](../src/tri_rag_harness/tls_rag_step4_nfcorpus.py) | L0 | Step4 replay/probe | Retain; 1 code / 1 test importers; CLI |
| [src/tri_rag_harness/tls_rag_step4_probe.py](../src/tri_rag_harness/tls_rag_step4_probe.py) | L0 | Step4 replay/probe | Retain; 4 code / 4 test importers; CLI |
| [src/tri_rag_harness/tls_rag_step4_repaired_acceptance.py](../src/tri_rag_harness/tls_rag_step4_repaired_acceptance.py) | L0 | Step4 replay/probe | Retain; 1 code / 1 test importers; CLI |
| [src/tri_rag_harness/tls_rag_step4_retention_v3.py](../src/tri_rag_harness/tls_rag_step4_retention_v3.py) | L0 | Step4 replay/probe | Retain; 2 code / 3 test importers; CLI |
| [src/tri_rag_harness/tls_rag_step5_ann.py](../src/tri_rag_harness/tls_rag_step5_ann.py) | L0 | Step5 ANN/diagnostics | Retain; 4 code / 2 test importers |
| [src/tri_rag_harness/tls_rag_step5_benchmark.py](../src/tri_rag_harness/tls_rag_step5_benchmark.py) | L0 | Step5 ANN/diagnostics | Retain; 0 code / 1 test importers; CLI |
| [src/tri_rag_harness/tls_rag_step5_data.py](../src/tri_rag_harness/tls_rag_step5_data.py) | L0 | Step5 ANN/diagnostics | Retain; 1 code / 1 test importers; CLI |
| [src/tri_rag_harness/tls_rag_step5_policy.py](../src/tri_rag_harness/tls_rag_step5_policy.py) | L0 | Step5 ANN/diagnostics | Retain; 2 code / 2 test importers |
| [src/tri_rag_harness/tls_rag_step5_protocol.py](../src/tri_rag_harness/tls_rag_step5_protocol.py) | L0 | Step5 ANN/diagnostics | Retain; 4 code / 3 test importers |
| [src/tri_rag_harness/tri_law.py](../src/tri_rag_harness/tri_law.py) | L0 | Shared scientific primitives | Retain; 5 code / 6 test importers |
| [src/tri_rag_harness/tri_predict.py](../src/tri_rag_harness/tri_predict.py) | L0 | Shared scientific primitives | Retain; 6 code / 2 test importers |
| [src/tri_rag_harness/utils.py](../src/tri_rag_harness/utils.py) | L0 | Shared scientific primitives | Retain; 45 code / 16 test importers |

### Mechanism and native source

| File | Level | Family | Disposition/evidence |
| --- | --- | --- | --- |
| [mechanism/tls_rag_step5_ann_comparators_v2.cpp](../mechanism/tls_rag_step5_ann_comparators_v2.cpp) | L0 | Route1 comparators v2 | Retain native build/source binding |
| [mechanism/tls_rag_step5_ann_comparators_v2.py](../mechanism/tls_rag_step5_ann_comparators_v2.py) | L0 | Route1 comparators v2 | Retain; 1 code / 1 test importers |
| [mechanism/tls_rag_step5_ann_continuation_v1.cpp](../mechanism/tls_rag_step5_ann_continuation_v1.cpp) | L0 | Route2 continuation v1 | Retain native build/source binding; directly included by v2 |
| [mechanism/tls_rag_step5_ann_continuation_v1.py](../mechanism/tls_rag_step5_ann_continuation_v1.py) | L0 | Route2 continuation v1 | Retain; 3 code / 1 test importers |
| [mechanism/tls_rag_step5_ann_continuation_v2.cpp](../mechanism/tls_rag_step5_ann_continuation_v2.cpp) | L0 | Route2 continuation v2 | Retain native build/source binding |
| [mechanism/tls_rag_step5_ann_continuation_v2.py](../mechanism/tls_rag_step5_ann_continuation_v2.py) | L0 | Route2 continuation v2 | Retain; 1 code / 1 test importers |
| [mechanism/tls_rag_step5_ann_enhancement.cpp](../mechanism/tls_rag_step5_ann_enhancement.cpp) | L0 | Route1 screen v1 | Retain native build/source binding |
| [mechanism/tls_rag_step5_ann_enhancement.py](../mechanism/tls_rag_step5_ann_enhancement.py) | L0 | Route1 screen v1 | Retain; 4 code / 1 test importers |
| [mechanism/tls_rag_step5_audit.py](../mechanism/tls_rag_step5_audit.py) | L0 | Exact mechanism A/B | Retain; 4 code / 3 test importers; CLI |
| [mechanism/tls_rag_step5_baseline_headroom.py](../mechanism/tls_rag_step5_baseline_headroom.py) | L0 | Original/projected baselines | Retain; 1 code / 1 test importers |
| [mechanism/tls_rag_step5_exact_policy.py](../mechanism/tls_rag_step5_exact_policy.py) | L0 | Exact mechanism A/B | Retain; 2 code / 1 test importers |
| [mechanism/tls_rag_step5_reuse_probe.py](../mechanism/tls_rag_step5_reuse_probe.py) | L0 | Exact cache/selection probe | Retain; 2 code / 1 test importers |

### Scripts

| File | Level | Family | Disposition/evidence |
| --- | --- | --- | --- |
| [scripts/analyze_tls_rag_step5_action_costs.py](../scripts/analyze_tls_rag_step5_action_costs.py) | L1 | Step5 ANN/diagnostics | Retain command/build entry |
| [scripts/audit_numerical_contracts.py](../scripts/audit_numerical_contracts.py) | L1 | Harness/build/test infrastructure | Retain command/build entry |
| [scripts/check_code_protection.py](../scripts/check_code_protection.py) | L0 | Governance guard | Retain command/build entry |
| [scripts/profile_tls_rag_step5_reuse_probe.py](../scripts/profile_tls_rag_step5_reuse_probe.py) | L1 | Exact cache/selection probe | Retain command/build entry |
| [scripts/run_tests.sh](../scripts/run_tests.sh) | L1 | Harness/build/test infrastructure | Approved B: default/full429, step5=231, legacy=198; global collection checks |
| [scripts/run_tls_rag_step4_readiness.sh](../scripts/run_tls_rag_step4_readiness.sh) | L1 | Step4 replay/probe | Retain command/build entry |
| [scripts/run_tls_rag_step4_repaired_acceptance.sh](../scripts/run_tls_rag_step4_repaired_acceptance.sh) | L1 | Step4 replay/probe | Retain command/build entry |
| [scripts/run_tls_rag_step5_ann_comparators_v2.py](../scripts/run_tls_rag_step5_ann_comparators_v2.py) | L1 | Route1 comparators v2 | Retain command/build entry |
| [scripts/run_tls_rag_step5_ann_continuation_v1.py](../scripts/run_tls_rag_step5_ann_continuation_v1.py) | L1 | Route2 continuation v1 | Retain command/build entry |
| [scripts/run_tls_rag_step5_ann_continuation_v2.py](../scripts/run_tls_rag_step5_ann_continuation_v2.py) | L1 | Route2 continuation v2 | Retain command/build entry |
| [scripts/run_tls_rag_step5_ann_enhancement.py](../scripts/run_tls_rag_step5_ann_enhancement.py) | L1 | Route1 screen v1 | Retain command/build entry |
| [scripts/run_tls_rag_step5_baseline_headroom.py](../scripts/run_tls_rag_step5_baseline_headroom.py) | L1 | Original/projected baselines | Retain command/build entry |
| [scripts/run_tls_rag_step5_mechanism.sh](../scripts/run_tls_rag_step5_mechanism.sh) | L1 | Exact mechanism A/B | Retain command/build entry |

### Tests

| File | Level | Family | Disposition/evidence |
| --- | --- | --- | --- |
| [tests/__init__.py](../tests/__init__.py) | L2 | Harness/build/test infrastructure | Retain conformance fixture |
| [tests/test_attribution.py](../tests/test_attribution.py) | L0 | Historical exact retrieval/diagnostics | Retain conformance fixture |
| [tests/test_beir_dataset.py](../tests/test_beir_dataset.py) | L0 | Harness/build/test infrastructure | Retain conformance fixture |
| [tests/test_certification.py](../tests/test_certification.py) | L0 | Shared scientific primitives | Retain conformance fixture |
| [tests/test_code_protection.py](../tests/test_code_protection.py) | L0 | Governance guard | Retain conformance fixture |
| [tests/test_config_projection.py](../tests/test_config_projection.py) | L0 | Harness/build/test infrastructure | Approved B M01: two methods→one; all four cache scenarios and other five methods retained |
| [tests/test_end_to_end.py](../tests/test_end_to_end.py) | L0 | Harness/build/test infrastructure | Retain conformance fixture |
| [tests/test_index_retrieval.py](../tests/test_index_retrieval.py) | L0 | Harness/build/test infrastructure | Retain conformance fixture |
| [tests/test_lid_policy.py](../tests/test_lid_policy.py) | L0 | Harness/build/test infrastructure | Retain conformance fixture |
| [tests/test_mprime_sweep.py](../tests/test_mprime_sweep.py) | L0 | Historical exact retrieval/diagnostics | Retain conformance fixture |
| [tests/test_pdctp_calibration.py](../tests/test_pdctp_calibration.py) | L0 | Historical PDCTP | Retain conformance fixture |
| [tests/test_pdctp_dataset_audit.py](../tests/test_pdctp_dataset_audit.py) | L0 | Historical PDCTP | Retain conformance fixture |
| [tests/test_pdctp_features.py](../tests/test_pdctp_features.py) | L0 | Historical PDCTP | Retain conformance fixture |
| [tests/test_pdctp_fiqa_embedding_gate.py](../tests/test_pdctp_fiqa_embedding_gate.py) | L0 | Historical PDCTP | Retain conformance fixture |
| [tests/test_pdctp_fiqa_query_cal.py](../tests/test_pdctp_fiqa_query_cal.py) | L0 | Historical PDCTP | Retain conformance fixture |
| [tests/test_pdctp_fiqa_query_cert.py](../tests/test_pdctp_fiqa_query_cert.py) | L0 | Historical PDCTP | Retain conformance fixture |
| [tests/test_pdctp_fiqa_query_tune.py](../tests/test_pdctp_fiqa_query_tune.py) | L0 | Historical PDCTP | Retain conformance fixture |
| [tests/test_pdctp_foundation.py](../tests/test_pdctp_foundation.py) | L0 | Historical PDCTP | Retain conformance fixture |
| [tests/test_pdctp_policies.py](../tests/test_pdctp_policies.py) | L0 | Historical PDCTP | Retain conformance fixture |
| [tests/test_pdctp_protocol_statistics.py](../tests/test_pdctp_protocol_statistics.py) | L0 | Historical PDCTP | Retain conformance fixture |
| [tests/test_pdctp_real_protocol.py](../tests/test_pdctp_real_protocol.py) | L0 | Historical PDCTP | Retain conformance fixture |
| [tests/test_pdctp_v3.py](../tests/test_pdctp_v3.py) | L0 | Historical PDCTP | Retain conformance fixture |
| [tests/test_real_dimension_sweep.py](../tests/test_real_dimension_sweep.py) | L0 | Historical exact retrieval/diagnostics | Retain conformance fixture |
| [tests/test_real_original_baseline.py](../tests/test_real_original_baseline.py) | L0 | Historical exact retrieval/diagnostics | Retain conformance fixture |
| [tests/test_real_policy_certify.py](../tests/test_real_policy_certify.py) | L0 | Historical exact retrieval/diagnostics | Retain conformance fixture |
| [tests/test_real_policy_test.py](../tests/test_real_policy_test.py) | L0 | Historical exact retrieval/diagnostics | Retain conformance fixture |
| [tests/test_real_policy_tune.py](../tests/test_real_policy_tune.py) | L0 | Historical exact retrieval/diagnostics | Retain conformance fixture |
| [tests/test_real_tune_diagnostics.py](../tests/test_real_tune_diagnostics.py) | L0 | Historical exact retrieval/diagnostics | Retain conformance fixture |
| [tests/test_retrieval_benchmark.py](../tests/test_retrieval_benchmark.py) | L0 | Historical exact retrieval/diagnostics | Retain conformance fixture |
| [tests/test_text_embeddings.py](../tests/test_text_embeddings.py) | L0 | Shared scientific primitives | Retain conformance fixture |
| [tests/test_tls_rag_step2.py](../tests/test_tls_rag_step2.py) | L0 | TLS synthetic Step2 | Retain conformance fixture |
| [tests/test_tls_rag_step3.py](../tests/test_tls_rag_step3.py) | L0 | TLS synthetic Step3 | Retain conformance fixture |
| [tests/test_tls_rag_step4.py](../tests/test_tls_rag_step4.py) | L0 | Step4 replay/probe | Retain conformance fixture |
| [tests/test_tls_rag_step4_a800_replay.py](../tests/test_tls_rag_step4_a800_replay.py) | L0 | Step4 replay/probe | Retain conformance fixture |
| [tests/test_tls_rag_step4_probe.py](../tests/test_tls_rag_step4_probe.py) | L0 | Step4 replay/probe | Retain conformance fixture |
| [tests/test_tls_rag_step4_repaired_acceptance.py](../tests/test_tls_rag_step4_repaired_acceptance.py) | L0 | Step4 replay/probe | Retain conformance fixture |
| [tests/test_tls_rag_step4_retention_v3.py](../tests/test_tls_rag_step4_retention_v3.py) | L0 | Step4 replay/probe | Retain conformance fixture |
| [tests/test_tls_rag_step5_ann.py](../tests/test_tls_rag_step5_ann.py) | L0 | Step5 ANN/diagnostics | Retain conformance fixture |
| [tests/test_tls_rag_step5_ann_comparators_v2.py](../tests/test_tls_rag_step5_ann_comparators_v2.py) | L0 | Route1 comparators v2 | Retain conformance fixture |
| [tests/test_tls_rag_step5_ann_continuation_v1.py](../tests/test_tls_rag_step5_ann_continuation_v1.py) | L0 | Route2 continuation v1 | Retain conformance fixture |
| [tests/test_tls_rag_step5_ann_continuation_v2.py](../tests/test_tls_rag_step5_ann_continuation_v2.py) | L0 | Route2 continuation v2 | Retain conformance fixture |
| [tests/test_tls_rag_step5_ann_enhancement.py](../tests/test_tls_rag_step5_ann_enhancement.py) | L0 | Route1 screen v1 | Retain conformance fixture |
| [tests/test_tls_rag_step5_baseline_headroom.py](../tests/test_tls_rag_step5_baseline_headroom.py) | L0 | Original/projected baselines | Retain conformance fixture |
| [tests/test_tls_rag_step5_mechanism.py](../tests/test_tls_rag_step5_mechanism.py) | L0 | Exact mechanism A/B | Retain conformance fixture |
| [tests/test_tls_rag_step5_reuse_probe.py](../tests/test_tls_rag_step5_reuse_probe.py) | L0 | Exact cache/selection probe | Retain conformance fixture |
| [tests/test_tri_law.py](../tests/test_tri_law.py) | L0 | Shared scientific primitives | Retain conformance fixture |
| [tests/test_tri_predict.py](../tests/test_tri_predict.py) | L0 | Shared scientific primitives | Retain conformance fixture |

### Configurations

| File | Level | Family | Disposition/evidence |
| --- | --- | --- | --- |
| [configs/pdctp_fiqa_e5_base_v2_embeddings.json](../configs/pdctp_fiqa_e5_base_v2_embeddings.json) | L0 | Historical PDCTP | Retain frozen protocol |
| [configs/pdctp_fiqa_query_cal_v1.json](../configs/pdctp_fiqa_query_cal_v1.json) | L0 | Historical PDCTP | Retain frozen protocol |
| [configs/pdctp_fiqa_query_cert_v1.json](../configs/pdctp_fiqa_query_cert_v1.json) | L0 | Historical PDCTP | Retain frozen protocol |
| [configs/pdctp_fiqa_query_tune_v1.json](../configs/pdctp_fiqa_query_tune_v1.json) | L0 | Historical PDCTP | Retain frozen protocol |
| [configs/pdctp_fiqa_real_protocol_freeze_v1.json](../configs/pdctp_fiqa_real_protocol_freeze_v1.json) | L0 | Historical PDCTP | Retain frozen protocol |
| [configs/pdctp_fiqa_source_audit_v1.json](../configs/pdctp_fiqa_source_audit_v1.json) | L0 | Historical PDCTP | Retain frozen protocol |
| [configs/pdctp_network_free_foundation_v1.json](../configs/pdctp_network_free_foundation_v1.json) | L0 | Historical PDCTP | Retain frozen protocol |
| [configs/pdctp_v3_network_free_foundation_v1.json](../configs/pdctp_v3_network_free_foundation_v1.json) | L0 | Historical PDCTP | Retain frozen protocol |
| [configs/real_scifact_dataset.json](../configs/real_scifact_dataset.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/real_scifact_e5_base_v2_embeddings.json](../configs/real_scifact_e5_base_v2_embeddings.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/real_scifact_fixed_dimension_tune.json](../configs/real_scifact_fixed_dimension_tune.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/real_scifact_original_exact_tune.json](../configs/real_scifact_original_exact_tune.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/real_scifact_policy_certify.json](../configs/real_scifact_policy_certify.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/real_scifact_policy_test.json](../configs/real_scifact_policy_test.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/real_scifact_policy_tune.json](../configs/real_scifact_policy_tune.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/real_scifact_tune_diagnostics.json](../configs/real_scifact_tune_diagnostics.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/retrieval_latency_100k_d768.json](../configs/retrieval_latency_100k_d768.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/retrieval_latency_1m_d1024.json](../configs/retrieval_latency_1m_d1024.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/retrieval_latency_1m_d1024_faiss_k1984.json](../configs/retrieval_latency_1m_d1024_faiss_k1984.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/retrieval_latency_smoke.json](../configs/retrieval_latency_smoke.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/synthetic_attribution_fresh.json](../configs/synthetic_attribution_fresh.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/synthetic_mprime_sweep_extended_fresh.json](../configs/synthetic_mprime_sweep_extended_fresh.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/synthetic_mprime_sweep_fresh.json](../configs/synthetic_mprime_sweep_fresh.json) | L0 | Historical exact retrieval/diagnostics | Retain frozen protocol |
| [configs/synthetic_mvp.json](../configs/synthetic_mvp.json) | L0 | Harness/build/test infrastructure | Retain frozen protocol |
| [configs/tls_rag_step2_synthetic_v1.json](../configs/tls_rag_step2_synthetic_v1.json) | L0 | TLS synthetic Step2 | Retain frozen protocol |
| [configs/tls_rag_step3_synthetic_v1.json](../configs/tls_rag_step3_synthetic_v1.json) | L0 | TLS synthetic Step3 | Retain frozen protocol |
| [configs/tls_rag_step4_a800_replay_v1.json](../configs/tls_rag_step4_a800_replay_v1.json) | L0 | Step4 replay/probe | Retain frozen protocol |
| [configs/tls_rag_step4_nfcorpus_probe_v2.json](../configs/tls_rag_step4_nfcorpus_probe_v2.json) | L0 | Step4 replay/probe | Retain frozen protocol |
| [configs/tls_rag_step4_protocol_v1.json](../configs/tls_rag_step4_protocol_v1.json) | L0 | Step4 replay/probe | Retain frozen protocol |
| [configs/tls_rag_step4_retention_v3.json](../configs/tls_rag_step4_retention_v3.json) | L0 | Step4 replay/probe | Retain frozen protocol |
| [configs/tls_rag_step5_ann_comparators_v2.json](../configs/tls_rag_step5_ann_comparators_v2.json) | L0 | Route1 comparators v2 | Retain frozen protocol |
| [configs/tls_rag_step5_ann_continuation_v1.json](../configs/tls_rag_step5_ann_continuation_v1.json) | L0 | Route2 continuation v1 | Retain frozen protocol |
| [configs/tls_rag_step5_ann_continuation_v2.json](../configs/tls_rag_step5_ann_continuation_v2.json) | L0 | Route2 continuation v2 | Retain frozen protocol |
| [configs/tls_rag_step5_ann_enhancement_v1.json](../configs/tls_rag_step5_ann_enhancement_v1.json) | L0 | Route1 screen v1 | Retain frozen protocol |
| [configs/tls_rag_step5_ann_v1.json](../configs/tls_rag_step5_ann_v1.json) | L0 | Step5 ANN/diagnostics | Retain frozen protocol |
| [configs/tls_rag_step5_mechanism_v1.json](../configs/tls_rag_step5_mechanism_v1.json) | L0 | Exact mechanism A/B | Retain frozen protocol |

## Test execution after approved proposal B

[Proposal B](TEST_SUITE_SIMPLIFICATION_PROPOSAL.md) retains the complete430-method
baseline disposition/coverage table. Its only method merge yields429 current
methods across the same46 files. Daily `step5` has231 methods in21 files; historical
`legacy` has198 in25 files, with no overlap or missing methods. No assertion or
test file was permanently deleted; the other45 test files remain byte-identical.
The explicit manifests live in the existing launcher; new files fail until
classified and new methods in classified files are automatically discovered.
No argument/`full` still covers the whole repository and remains required before
implementation delivery. Import/collection errors anywhere fail all modes.
The merge and launcher do not change any of the six captured scientific source maps.

## Documentation and payloads

- Root README/START_HERE/NEXT_AGENT_PROMPT/STATUS are short current navigation.
  [Implementation milestone index](IMPLEMENTATION_PLAN.md) links the complete
  historical plan at its fixed Git commit. Results/consent are preserved there.
- [Living journal](TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md) remains unchanged;
  read relevant sections rather than making every task preload it.
- At the review baseline, L0 `docs/CODE_PROTECTION.md` omits40 machine-registered files.
  Approved repair A synchronizes these plus the four new L1 entries, preserving
  all machine levels. Details are in the [cleanup review](REPOSITORY_CLEANUP_REVIEW.md).
- `artifacts/` has13 tracked historical result/witness files; preserved. JSON
  evidence under `docs/` is also preserved. Old results are not current run orders.
- `runs/` and `data/` contain ignored runtime payloads (tracked `.gitkeep` only).
  Untracked experiment archives, bundle and `report_work/` are user assets;
  they were not opened, hashed or deleted for this cleanup.
- `.gitignore`, `pyproject.toml`, and both requirements files retain existing
  dependency/build behavior. No dependency or package namespace was changed.
