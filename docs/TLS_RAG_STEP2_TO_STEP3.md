# TLS-RAG Step 2 to Step 3 handoff

> **Version-scoped reference.** This document describes its named method and
> milestone, including historical results and commands. It is not a current
> assignment or permission to reopen data. Start at [root AGENTS](../AGENTS.md)
> and [the current entry](../START_HERE.md). Current L0 protection and standing
> checkout/push instructions apply; later versioned briefs define successor work.


## Accepted Step 2 baseline

Step 2 is frozen at
`f46ce73beccf5fddbf05fd87fdb0911318020add`. Local and Genoa CPU replays both
completed all 180 tests with 179 passes, one expected optional real-FAISS skip,
and zero failures. Genoa resolved the same full commit and completed in
315.264 seconds.

The Step 2 runner uses 12 normalized corpus vectors, 7 external queries, one
fixed dense Gaussian projection, `m_prime=2`, `M_grid=[3,6,12]`, `k_gt=2`,
`k_ctx=2`, and at most two expansions. Across 18 visited stages it performs 7
projected scans, 84 projected distance evaluations, and 69 unique original
distance evaluations. It emits 11 expansions and 7 stops.

The portable manifest, Phase A, and Phase B fingerprints are respectively:

- `7d7f2329d56ea608117f8956eeb37926550b9beb5a176fe5f427c598b5d03eee`;
- `78c4e4869ffca61a7a82ab014b9a3bd9c1513824c6d7d83ad6c2180f4428c2f3`; and
- `a3d3620538c76bcc8a64b17c8dac619ac4b279be13abd43308758b43efda56e4`.

Step 3 must import or wrap the Step 2 retrieval/state machinery rather than
silently reimplementing different normalization, projection, search, action,
context, supervision, fingerprint, work, or timing semantics. Compatibility
tests must prove the Step 2 fixed-schedule run remains identical.

## Reusable Step 2 interfaces

The primary definitions in `tri_rag_harness.tls_rag_step2` are:

- `Step2Config`, `load_step2_config`, `build_step2_environment`;
- `EvidencePlan`, `FacetSlot`, `EvidenceLabelStore`, `PassageEvidence`;
- `DecisionInput`, `StateValidity`, `assert_deployable_only`;
- `Action`, `ControllerDecision`, `FixedScheduleController`;
- `run_phase_a`, `join_phase_b`, and `run_step2`; and
- `same_distance_different_angle_fixture`, portable artifact names, and
  separate work/timing schemas.

Step 2 deliberately exposes no production Tri-Law feature, learned score,
confidence table, calibrated controller, or risk aggregation. Its Phase A
stores the full projected ranking internally for later prefix exposure, but
each decision object contains only its current prefix. Phase B alone contains
evidence IDs, gains, coverage, sufficiency, retention, and full exact top-k
diagnostics.

## Step 3 implementation target

Use new `tls_rag_step3` schemas, versions, fingerprints, config, run namespace,
module, tests, artifacts, and documentation. Keep the implementation a
network-free synthetic research harness, not a production controller framework.

The observed-pair profile must implement the exact frozen orientation and
summaries:

- `u=x_i-q`, `v=x_j-q`, with `||u|| < ||v||`;
- `beta=||v||^2/||u||^2` and
  `rho=<u,v>/(||u||||v||)`;
- context-boundary pairs from the farthest context item to every farther
  exposed non-context item;
- core-frontier pairs from every context item to the frozen most-recent shell;
- count, invalid/tied/collinear counts, mean, maximum, and quantiles 0.50,
  0.90, 0.99 for valid exact single-pair probabilities;
- frozen core/shell projected-to-original log-distortion quantiles; and
- previous-state deltas with explicit first-state behavior.

Do not call dependent pair probabilities independent, sum them into a theorem,
or label them posterior risk. Do not use unexposed original distances or infer
them from LID. The same-distance/different-angle and realized-projection
counterexamples must remain explicit.

Fit transparent remaining-useful-evidence and current-context-sufficiency
scores on a deterministic synthetic model-fit partition only. Fit score bins
and exact one-sided Clopper-Pearson reachable-bin event-rate limits on a
disjoint synthetic bound-fit partition. Query IDs—not stages—are the
independent units. Freeze minimum cell size and simultaneous alpha allocation
across candidates, stages, bins, and both outcomes. Empty, underpowered, or
invalid cells return `[0,1]` and prohibit STOP.

For every preregistered candidate, build calibration tables sequentially from
only calibration queries that would reach each stage under that candidate's
already frozen earlier actions. The inference controller receives only the
versioned deployable state, frozen model artifacts, and frozen lookup tables.
It never receives labels, roles, exact full-corpus neighbors, realized
retention, or future outcomes.

Implement exactly the frozen synthetic ablations:

1. fixed-grid exact reference;
2. sequential distance/distortion/redundancy/validity features, no Tri-Law;
3. Row 2 plus only the observed-pair Tri-Law profile; and
4. Row 3 plus only deterministic plan/facet-prediction features.

Rows 2--4 must share all other data, fit algorithm, partitions, bins, alpha,
candidate registry, thresholds, fallback, retrieval, and context components.
Synthetic comparisons are code-path diagnostics only; no policy may be called
selected, certified, or superior for real use.

## Required fresh files and artifacts

Prefer these isolated additions:

- `src/tri_rag_harness/tls_rag_step3.py`;
- `configs/tls_rag_step3_synthetic_v1.json`;
- `tests/test_tls_rag_step3.py`; and
- `docs/TLS_RAG_STEP3_SYNTHETIC.md`.

The runner must emit fingerprinted feature/model/calibration/candidate specs,
synthetic partition IDs, separate pre-supervision decisions and post-decision
labels, query-stage records, one-factor ablation records, work counters,
separate nonportable timings, reconstructable aggregates, and a report with
explicit synthetic/no-certificate/no-latency/no-answer disclaimers. Run twice
in fresh directories and compare every portable artifact.

## Acceptance and stop boundary

Focused tests must cover exact observed-pair geometry, ties/zero/collinear
exclusions, profile quantiles and deltas, no unseen access, angle
counterexamples, row-by-row feature isolation, deterministic disjoint fit/bound
partitions, hand-computed Clopper-Pearson limits and alpha allocation,
candidate-specific reachability, vacuous cells, dual-bound STOP, conservative
fallback, label isolation, Phase A fingerprint preservation, Step 2 exact
compatibility, accounting, artifact reconstruction, and two-run determinism.

Run the complete CPU/network-free suite. Update the Step 3 document,
`docs/IMPLEMENTATION_PLAN.md`, and `STATUS.md`, commit on the fresh Step 3 task
branch, and stop for user review. Do not begin Step 4, inspect any real or
protected role/archive, download anything, run an approximate index/GPU/LLM,
or make calibration, certification, latency, evidence-quality, or answer claims
beyond the explicitly synthetic fixture.
