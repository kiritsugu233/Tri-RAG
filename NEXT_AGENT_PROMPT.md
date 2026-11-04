# Step5 compact brief: C0 retained, C1 deferred

Updated 2026-11-04. This replaces obsolete active instructions in this file.
Actual user consent is quoted at the top of the [current plan](docs/PQ_REFINEMENT_COST_GEOMETRY_PLAN_V0.md).
The user approved that MD, asked the EXISTING Step5 agent to finish gate 1 first,
retired schemes 1-4 from working context, and requested notice when server power-on is needed.

## Current decision

The user's latest actual replies are “暂缓C1，保留C0结果” and “暂不开机”.
Keep the completed C0 outputs and C1 proposal. Do not create C1 code, run new
measurements, access SSH, ask for power-on again or create background follow-ups.
Resume only after a new explicit instruction covering the deferred scope.
Gate 1 remains unresolved: aggregate rerank fraction does not establish actual
cancellable read/compute costs and cannot count as a completed/pass result.
Do not enter gate 2 residual/sketch experiments or gate 3 Tri implementation.
Do not expand representations, action grids or query roles.

Latest checkpoint: C0 and local C1 preparation are complete; read plan sections10–11.
Do not repeat the C0 run unless investigating a discrepancy. C1's new exact eight
paths are proposed, not approved, and explicitly deferred. No C1 code has been created.
Cost gate is unresolved; actual gather/read-vs-compute costs remain unknown.

## Minimal read set

1. START_HERE.md latest current entry, AGENTS.md, relevant protection entries, this brief.
2. docs/PQ_REFINEMENT_COST_GEOMETRY_PLAN_V0.md execution note and sections 2-4/6.
3. runs/route6-S0-manual-r2-recovery-20261104/analysis/analysis_zh.md.
4. Its latency_components.csv and ../priority/ivf/opq64/{64,256}/timings.json;
   ../priority/representations/pq64/cpu14/timings.json. Read other provenance only as needed.

Canonical checkout: /Users/guanghongxu/Query-Adaptive-Tri-RAG.
Continuing branch: codex/tls-rag-step5-ann. Pre-handoff HEAD: 15712d15db85bdee034cd8b67b5ff65c8496aa4b.
Inspect actual HEAD/status, preserve untracked archives, do not create another checkout/branch.
Protection checks before edits/handoff; exact L0 approval and data-role guards remain binding.
Routine authorized L1 analysis/docs need no renewed generic approval.
Classify new files by actual impact; do not treat every new analysis file as L0 automatically.
Commit/push/verify the continuing branch after authorized work and required checks.

## Facts to retain

- Full S0 completed normally. Local recovery verified 798 selected products plus index;
  17 large caches and the full original source remain on the server. It may be off.
- Corpus: 5,233,329 x 640 FP32. k=100. Native clip(1-dot,0,2), original ties unchanged.
- 128 observed development queries: 64 discovery +64 check; no independent certification.
- CPU timed cohort: 16 queries per role, 3 calls each; 48 calls are not 48 independent queries.
- Primary OPQ64/IVF64/M6400: check total15.149721ms, rerank4.771083ms, fraction31.493%.
- Quality control IVF256/M6400: total36.366667ms, rerank5.029427ms, fraction13.830%.
- Low-headroom PQ64/Flat/M3200: total137.272857ms, rerank2.862618ms, fraction2.085%.
- Rerank includes gathering/copying, distance, sorting: not a physical-read counter.
- Proposed engineering target is10% total reduction; halving primary rerank leaves
  about0.870569ms for new summary access, query projection and control.
- C0 independently checks288 candidate hashes/outputs and96 query groups/72 cost
  assumptions. Primary check F/T query-cluster95% interval30.660%–32.375%; this
  describes observed development queries, not certification or hardware drift.
- Free removal of all PQ Flat rerank gives only2.085%; stop that refinement interface.
- M6400 logical gathers total16,640,000B including100 output rereads; virtual page
  coverage/NumPy requests are not physical DRAM/SSD traffic. C1 manifest is ready;
  CPU-only server with >=32GiB RAM can supply original vectors/index already there.
- Current S0 Gaussian/Hybrid main representations had no competitive Tri entry point.
  Auxiliary residual summaries are a different, still unverified hypothesis.

## Context to retire

- Schemes1-4 are rejected/retired by the user. Remove their pending tasks and long
  implementation narratives from active summaries. Do not resume or rerun them.
- Route5 remains an unimplemented historical proposal, not current work.
- Old Step4 execution details, monitoring incidents, advisor/LID tuning and old
  controller plans are historical. Do not reread the whole exploration journal.
- Gaussian cycle/T0/T1/M0/B0 work is reference only; retain necessary conclusions
  and evidence pointers, not independent theory work as pending tasks.
- Preserve historical files, scientific contracts, failures and consent records.
  Context cleanup does not delete evidence or weaken governance.
- Recover only a targeted historical dependency if a concrete current question needs it.
- This is working-summary cleanup, not a claim that model history was physically erased.
  If no supported compaction tool is available, continue directly using this active brief.

## Gate1 deliverables

Query-level cost audit; coverage/provenance; cancellable-work decomposition;
read-vs-compute evidence or the exact minimal server replay still needed;
10% target with5%/20% sensitivity; same-hardware paired uncertainty;
pass/insufficient/unresolved verdict with limitations. Do not declare the gate
complete while required physical read/compute decomposition remains unmeasured.

When a new L0 contract or file is necessary, prepare exact paths/impact/validation
for scoped consent while continuing unaffected local analysis. Do not ask whether
starting gate1 is allowed: the user already instructed it.
Report in the Step5 chat when the user resumes work. No other chat messaging or automatic gate2 start is authorized.

Current evidence: docs/pq_refinement_cost_c0_evidence_v0.json;
runs/pq-refinement-cost-C0-20261104-v2/analysis_zh.md and C1_input_manifest.json.
