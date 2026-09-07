# TLS-RAG Step 3 minimal entry

> **Version-scoped reference.** This document describes its named method and
> milestone, including historical results and commands. It is not a current
> assignment or permission to reopen data. Start at [root AGENTS](../AGENTS.md)
> and [the current entry](../START_HERE.md). Current L0 protection and standing
> checkout/push instructions apply; later versioned briefs define successor work.


Status: authorized by the user after successful local and Genoa Step 2
verification. This was the entry for the now-completed Step 3 task.
The current entry is [START_HERE.md](../START_HERE.md).

## Baseline

Start from the handoff commit that contains this file and verify that
`f46ce73beccf5fddbf05fd87fdb0911318020add` is an ancestor. Step 2 is frozen:
its exact retrieval, two-action interface, decision/supervision separation,
config, tests, and portable identities must remain compatible.

Step 3 is still CPU-only, network-free, synthetic-only, and LLM-free. It adds:

1. the exact observed-pair Tri-Law risk feature profile from the frozen spec;
2. transparent remaining-gain and current-sufficiency score models;
3. disjoint synthetic model-fit and bound-fit calibration partitions;
4. candidate/stage/bin/reachability-specific one-sided confidence tables;
5. conservative dual-bound STOP logic; and
6. Rows 1--4 of the frozen one-factor synthetic ablation ladder.

It does not authorize real data, protected roles, source/download work,
`query_tune`, `query_cert`, `query_latency`, `query_test`, approximate indexes,
GPU benchmarking, serving, LLM calls, answer generation, certification, or a
scientific performance claim. Those remain later gates.

## Mandatory reading order and allowlist

Read these files completely and in this order before editing:

1. `AGENTS.md`;
2. `AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP3.md`;
3. `docs/TLS_RAG_STEP3_START_HERE.md`;
4. `docs/TLS_RAG_STEP2_TO_STEP3.md`;
5. `docs/TRI_LAW_SEQUENTIAL_RAG_SPEC.md`;
6. `docs/TRI_LAW_SEQUENTIAL_RAG_PROTOCOL.md`;
7. `docs/TRI_LAW_SPEC.md`;
8. `src/tri_rag_harness/tls_rag_step2.py`;
9. `configs/tls_rag_step2_synthetic_v1.json`;
10. `tests/test_tls_rag_step2.py`; and
11. `src/tri_rag_harness/tri_law.py`.

After that, only the exact definitions imported by those files may be opened
from `projection.py`, `indexes.py`, `embeddings.py`, and `utils.py`. Repository
search may be used to locate names, but do not open unrelated documents,
historical implementations, artifacts, runs, real-data adapters, protected
roles, or archives. If the allowlist is insufficient, stop and report the
specific missing file and reason before reading it.

In particular, do not preload `STATUS.md`, `docs/IMPLEMENTATION_PLAN.md`, the
Step 1/2 briefs, old v1/v2/v3 agent prompts or design documents, real-data
documents, reports under `artifacts/`, or `runs/`. Their audit chain is retained
at the Step 2 commit and indexed separately for humans.

## Non-negotiable boundary

- Normalize before projection; use Gaussian standard deviation
  `1/sqrt(m_prime)`; never renormalize after projection.
- Use squared L2, stable string-ID ties, one projected scan, prefix reuse,
  newly exposed original distances only, accumulated exact reranking, and the
  unchanged exact top-`k_ctx` context builder.
- Tri-Law accepts only strictly ordered, fully exposed original displacement
  pairs. Equal-distance, zero, nonfinite, or collinear cases are counted and
  handled exactly as frozen. No unseen geometry or scalar-LID reconstruction.
- Tri-Law outputs and aggregates are ex-ante geometric features—not posterior
  missing-evidence probabilities, evidence probabilities, or guarantees.
- Labels may fit synthetic score/calibration artifacts only in explicit
  calibration code. Inference state, controller, and Phase A never receive a
  supervision-store handle or forbidden field.
- `STOP` requires both a valid `U_gain <= delta_gain` and a valid
  `L_suff >= tau_sufficient`. Every failure, empty/underpowered cell, invalid
  state, or missing required profile forces next-grid expansion when possible.
- Do not modify exact Tri-Law formulas/tolerances/tests or historical v1/v2/v3
  behavior and identities.

Detailed implementation and acceptance requirements are in the Step 3 brief;
the exact Step 2 interfaces and handoff risks are in the transition document.
