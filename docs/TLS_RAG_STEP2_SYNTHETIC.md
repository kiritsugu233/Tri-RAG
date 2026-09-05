# TLS-RAG Step 2 synthetic skeleton

## Scope and result

Step 2 implements only the CPU/network-free walking skeleton authorized by
`AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP2.md`. It uses seven external synthetic
queries, twelve normalized float64 corpus vectors, one fixed dense Gaussian
projection, the grid `[3, 6, 12]`, `k_gt=2`, `k_ctx=2`, and a fixed maximum of
two expansions. The controller is a frozen per-query action schedule used to
exercise the interface. It is not learned, fitted, calibrated, selected, or
certified.

Each query is projected once. One exact projected squared-L2 full scan is
stable-sorted by string ID, only the pilot prefix is initially exposed, and
each expansion reveals exactly the next grid prefix. Original squared-L2
distances are evaluated once for newly exposed candidates, cached, and used to
stable-rerank the entire accumulated prefix. Context is always the exact
original-space top-`k_ctx` within that prefix. Projected vectors are never
renormalized.

The only controller actions are `STOP` and
`EXPAND_TO_NEXT_GRID_VALUE`. Its immutable `tls_rag_decision_input_v1` state
contains deterministic query/plan features, the exposed prefix, cached
original measurements, reranking/context, structural summaries, and validity
fields. Recursive guards reject roles, qrels, evidence/support labels and IDs,
gain/sufficiency labels, answer fields, oracle/effective LID, exact full-corpus
top-k identities, retention, future outcomes, and protected-role outcomes.

## Two-phase evidence boundary

Phase A constructs, serializes, fingerprints, and closes the complete
label-free trajectory. The separate `EvidenceLabelStore` is constructed only
after `phase_a_decisions.jsonl` exists. Phase B joins those labels without
altering the Phase A fingerprint and reconstructs, per query and stage:

- marginal candidate evidence gain;
- marginal final-context evidence gain;
- remaining useful evidence over later frozen budgets;
- current final-context sufficiency;
- candidate and context facet coverage; and
- exact top-`k_gt` retention as a separate diagnostic.

The fixture includes candidate gain without context gain, an empty immediate
shell followed by later context gain, duplicate/equal distances, a zero
displacement, an empty plan, invalid deterministic features, pilot stop,
multiple expansion, maximum expansion, full-corpus exhaustion, and terminal
evidence nonattainment. The same-distance-curve/different-angle fixture is
used only in tests to construct observed-pair `beta` and `rho` directly and to
call the unchanged exact `tri_law_probability` API. Production Tri-Law risk
aggregation is absent.

## Run and verify

The runner refuses to overwrite a nonempty output directory. A fresh local run
uses:

```bash
cd /Users/guanghongxu/Query-Adaptive-Tri-RAG
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m tri_rag_harness.tls_rag_step2 \
  --config configs/tls_rag_step2_synthetic_v1.json \
  --output runs/tls_rag_step2_synthetic_v1
```

The complete CPU/network-free regression uses:

```bash
cd /Users/guanghongxu/Query-Adaptive-Tri-RAG
./scripts/run_tests.sh
```

The Step 2 focused suite contains 13 tests. The full local suite completed 180
tests: 179 passed, one optional real-FAISS CPU conformance test was skipped
because FAISS is absent, and zero failed. The Step 2 test runs the fixture in
two fresh temporary directories and compares every portable artifact byte for
byte.

## Artifacts and accounting

The runner writes these portable artifacts:

- `manifest.json`: frozen config, fixture, projection, ID-map, plan/label-store,
  and Phase A/B fingerprints plus scope declarations;
- `projection.json`: normalization and dense-Gaussian projection identity;
- `id_maps.json`: ordered stable corpus and external-query IDs;
- `evidence_plan_schema.json`: deterministic plans and annotation schema, not
  passage evidence labels;
- `evidence_label_store.json`: the separate Phase B passage annotations;
- `phase_a_decisions.jsonl`: label-free state/action records for every visited
  stage;
- `phase_b_supervision.jsonl`: separately joined supervision and diagnostics;
- `work_counters.json`: per-stage and total deterministic work counts;
- `aggregates.json`: values reconstructed from query-level records; and
- `report.md`: scope, fingerprints, structural result, and disclaimers.

`timings.json` is deliberately nonportable and excluded from scientific
fingerprints. It separates setup/fixture generation from query projection,
pilot scan/ranking, prefix reuse, new original distances, reranking, plan,
controller, and context timings. Deterministic work counters separate the same
operations.

The frozen run has 7 queries and 18 visited stages: 11 expansions and 7 stops.
It performs 7 projected full scans and 84 projected distance evaluations, with
69 unique original-distance evaluations. Candidate gain occurs at 2 stages,
final-context gain at 1 stage, and 3 terminal queries record evidence
nonattainment. These are engineered fixture diagnostics, not estimates for a
real query distribution.

## Boundary and remaining risks

No network, download, new dependency, real dataset/model, protected role,
returned archive, approximate index, GPU, LLM, answer generation, or answer
evaluation is used. Raw Tri-Predict v1, PDCTP v2/v3, exact Tri-Law, their
historical schemas, and their existing tests are unchanged.

This exact NumPy full scan is correctness-oriented, not a latency or serving
result. The evidence annotations and fixed schedule are deliberately synthetic
and cannot establish target identifiability, calibration, stopping quality,
certification, or generalization. Timings are local measurements and are not
portable. Step 3 remains unauthorized; its risk profile, learned controller,
and calibration must not be inferred from this fixture.
