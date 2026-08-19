# Prompt for the Next Implementation Agent

Work inside this directory and implement the Query-Adaptive Tri-RAG experimental harness described here.

Before editing anything, read these files completely in order:

1. `AGENTS.md`
2. `START_HERE.md`
3. `docs/ARCHITECTURE.md`
4. `docs/TRI_LAW_SPEC.md`
5. `docs/EXPERIMENT_PROTOCOL.md`
6. `docs/CERTIFICATION.md`
7. `docs/IMPLEMENTATION_PLAN.md`

## Task for this first implementation pass

Complete Milestones 0 through 3 from `docs/IMPLEMENTATION_PLAN.md`:

- repository/package/config/manifest skeleton;
- CPU-only synthetic external-query walking skeleton;
- normalized original embeddings and fixed dense Gaussian projection without post-projection normalization;
- exact original and projected squared-L2 retrieval;
- paper-conformant exact Tri-Law, orthogonal conditional law, and Monte Carlo conformance tests;
- pilot retrieval, original-distance pilot reranking, and deployable query-LID estimation;
- diagnostic oracle LID;
- fixed-budget and monotone binned adaptive policies;
- per-query artifact logging;
- independent empirical-Bernstein certification;
- unit and integration tests;
- `STATUS.md` with exact run/test commands and current results.

Do not implement a real text embedding model, download a real dataset, call an LLM, or add an approximate index in this pass. The synthetic harness must be correct before external dependencies are introduced.

## Required demonstration

Provide one tiny checked-in or deterministically generated synthetic configuration for which a single command:

1. creates disjoint tune/cert/test external queries;
2. builds the original and projected indexes;
3. fits the monotone binned policy on tune queries;
4. freezes and fingerprints the policy;
5. certifies it on certification queries;
6. evaluates it on test queries;
7. writes the manifest, policy, per-query records, certification artifact, aggregates, timings, and Markdown report.

The run may legitimately fail the configured statistical target. Report failure faithfully; do not manipulate the split or budget after observing certification.

## Completion criteria

- All required tests in `AGENTS.md` that apply to Milestones 0-3 pass.
- The exact Tri-Law conformance suite passes before any Tri-Predict implementation begins.
- Tune/cert/test IDs are demonstrably disjoint.
- No policy inference method can access labels, exact ground-truth IDs, or realized recall.
- The query-level output is sufficient to recompute every aggregate and confidence bound.
- Pilot, expansion, and original reranking costs are logged separately.
- Repeating the same run with the same seeds yields identical policy and metric artifacts, except explicitly documented timestamps/runtime noise.
- `docs/IMPLEMENTATION_PLAN.md` and `STATUS.md` accurately reflect completed and remaining work.

When a design detail is ambiguous, choose the smallest testable implementation consistent with `AGENTS.md`, document the choice in `STATUS.md`, and continue without expanding scope.
