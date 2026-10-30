# Implementation milestones

Updated 2026-10-30. Current stage: TLS-RAG Step 5 on `codex/tls-rag-step5-ann`.
Read [START_HERE](../START_HERE.md) and [STATUS](../STATUS.md) before work.
Unchecked historical items are not execution authorization.

## Repository review and cleanup

- [x] Check canonical branch/HEAD/status and protection before editing.
- [x] Inventory all242 tracked baseline files; parse118 Python files, inspect
  native include/dependency paths, configs, test coverage and source bindings.
- [x] Trace current Step 5 and retained historical families; publish a per-file
  [repository map](REPOSITORY_MAP.md) and [cleanup findings](REPOSITORY_CLEANUP_REVIEW.md).
- [x] Remove repeated historical narration from startup/status/plan while retaining
  its exact Git snapshot; correct obsolete workflow/next-agent instructions.
- [x] Keep all scientific sources/tests/configs/source bindings unchanged.
  No whole source file qualifies for automatic deletion in this review.
- [x] Run429 tests in the existing fixed-version/native environment without skips;
  document default-environment missing-FAISS failure rather than hiding it.
- [x] Make the L1 test launcher independent of cwd with explicit interpreter choice;
  final full rerun from `/private/tmp`:429/429,0 skips. Protection/link/binding checks pass.
- [ ] Obtain exact four-path L0 approval before checker/register repair (review A).

## Scientific milestone index

| Family/milestone | Recorded state | Detailed record |
| --- | --- | --- |
| Raw Tri-Predict v1 | Terminal negative; frozen | [Baseline](RAW_TRI_PREDICT_V1_BASELINE.md) |
| Calibrated Tri-Predict v2/v3 | Historical; frozen | [Protocol](CALIBRATED_TRI_PREDICT_PROTOCOL.md), [diagnosis](CALIBRATED_TRI_PREDICT_V3_DIAGNOSIS.md) |
| TLS-RAG Steps2/3 | Synthetic implementation complete | [Step2](TLS_RAG_STEP2_SYNTHETIC.md), [Step3](TLS_RAG_STEP3_SYNTHETIC.md) |
| Step4 repaired/A800 replay | Replay gates passed; no independent effectiveness/latency claim | [A800 contract](TLS_RAG_STEP4_A800_REPLAY.md), [historical status](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/STATUS.md) |
| Step5 ANN v1 | No tune candidate; protected later roles closed | [Contract](TLS_RAG_STEP5_ANN.md), [historical status](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/STATUS.md) |
| Step5 exact mechanism/reuse/baselines | Studies complete; conclusions version scoped | [Repository map](REPOSITORY_MAP.md) |
| Route1 v1/v2 | Negative net-speed results | [Journal §§14/19](TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md) |
| Route2 continuation v1/v2 | A→B→C complete; current learned/Tri paths fail net gain | [Journal §§24/27–31](TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md) |
| Route3 independent-view rescue | Not run | Same journal §7; separate scope needed |
| Route4 offline geometry/index configuration | Plan only | Same journal §21; separate scope needed |

## Pending decisions and safety limits

- [ ] User decision on the [guard/register repair](REPOSITORY_CLEANUP_REVIEW.md#proposal-a-protected-guard-and-register-repair).
- [ ] Trained-support/OOD fallback remains unimplemented for continuation v2.
  A future repair needs its own L0 scope/version and evaluation; do not relabel
  the historical study as safe deployment or rerun protected data automatically.
- [ ] Fresh independent confirmation, new corpus/1M, adapter/routes3/4 and formal
  certification are separate future work, not assigned by this checklist.

## Full historical plan

The complete prior1521-line ledger, original approvals, unchecked milestones,
commands and compatibility constraints remain available at
[the cleanup baseline](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/docs/IMPLEMENTATION_PLAN.md). Offline retrieval:

```bash
git show 50defc35bc9f42f5b24d2dedf5f45a89e51257ad:docs/IMPLEMENTATION_PLAN.md
```

The source-bound scientific contracts and append-only research journal remain
at their existing local paths. This index does not replace those contracts.
