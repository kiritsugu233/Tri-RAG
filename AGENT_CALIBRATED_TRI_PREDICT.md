# Agent brief: Pilot-Distance Calibrated Tri-Predict v2

You are the implementation agent for the successor to a completed negative
baseline. Work only on branch `codex/calibrated-tri-predict-v2` unless the user
explicitly directs otherwise.

## Mission

Implement and rigorously validate **Pilot-Distance Calibrated Tri-Predict
(PDCTP)** on top of Raw Tri-Predict v1. The scientific hypothesis is that two
separate calibration layers can correct:

- systematic pilot-LID bias; and
- analytic LID-to-budget miscalibration.

The desired positive result is not guaranteed. Your job is to build a protocol
that can honestly pass or fail, not to obtain a predetermined conclusion.

## Read before editing

Read these files completely in order:

1. `AGENTS.md`
2. `docs/RAW_TRI_PREDICT_V1_BASELINE.md`
3. `docs/CALIBRATED_TRI_PREDICT_PROTOCOL.md`
4. `STATUS.md`
5. `docs/REAL_RETRIEVAL.md`
6. `docs/CERTIFICATION.md`
7. `docs/TRI_LAW_SPEC.md`
8. `docs/IMPLEMENTATION_PLAN.md`

The root `AGENTS.md` correctness rules remain binding. This brief adds v2
boundaries; it does not relax normalization, projection, exact-search,
split-isolation, certification, artifact, or reporting requirements.

## Immutable baseline

Verify before work:

```bash
git rev-parse raw-tri-predict-v1-terminal-negative^{commit}
git branch --show-current
```

The tag must resolve to `fb09c00`, and the branch must be
`codex/calibrated-tri-predict-v2`.

Do not modify v1 behavior in place. Keep exact Tri-Law, Raw Tri-Predict policy
loading, and accepted v1 schemas backward compatible. New calibrated policies
must have distinct modules/classes, schema names, fingerprints, configs, and
run directories. Never rerun or retune the observed SciFact cert/test splits.

## Required method

Implement the two stages specified in
`docs/CALIBRATED_TRI_PREDICT_PROTOCOL.md`:

1. a deterministic pilot-distance feature extractor and deployable log-linear
   LID calibrator fit on `query_cal`; and
2. a quantile budget-residual calibrator for
   `log(M_required/M_raw)`, retaining Raw Tri-Predict as the explicit theory
   anchor.

At inference, the method may use only pilot original/projected distances,
pilot-LID validity, frozen metadata, and Raw Tri predictions. Oracle LID,
exact top-k, qrels, retention, answers, and split roles are forbidden.

Implement explicit ablations for LID calibration only, budget calibration only,
and full PDCTP. Do not substitute an opaque end-to-end regressor.

## New-data protocol

SciFact is exhausted. Begin with a source audit for the provisional BEIR FiQA
candidate. Pin archive/revision/license identities, validate qrels, group
normalized duplicate query text, and prove disjoint `query_cal`, `query_tune`,
`query_cert`, `query_latency`, and `query_test` roles. If the data cannot support
the predeclared power plan, stop before method evaluation and report the issue.

Reuse the pinned E5 model revision for continuity, but use a new dataset,
embedding cache, projection seed, and artifact namespace. Freeze `m_prime=192`
for the first v2 protocol; do not run another dimension sweep.

Only `query_cal` may fit calibration parameters. Only `query_tune` may select a
candidate/operating point. `query_cert` is one-time certification,
`query_latency` is a label-free systems benchmark, and `query_test` remains
untouched until every earlier gate is terminal.

## First implementation pass

Complete these tasks before requesting a real cluster run:

1. Add versioned config schemas for feature extraction, calibration candidates,
   five data roles, selection, certification, and latency.
2. Implement pilot-distance feature extraction with hand-computed tests,
   squared-L2-to-Euclidean conversion, deterministic invalid handling, and a
   serialized feature fingerprint.
3. Implement the constrained log-linear LID calibrator and round-trip/tamper
   tests.
4. Implement the quantile residual calibrator, grid ceiling, clipping, fallback,
   and round-trip/tamper tests.
5. Implement a synthetic five-role walking skeleton proving that fit, selection,
   certification, latency, and test boundaries reject leakage.
6. Implement fixed, monotone, Raw Tri, both ablations, and PDCTP behind one
   decision interface.
7. Add a sample-size/power-planning artifact and paired-bound fixtures before
   preparing real protected splits.
8. Update `docs/IMPLEMENTATION_PLAN.md` and `STATUS.md` with exact commands,
   tests, artifacts, risks, and the next stop/go gate.

Do not download real data, embed FiQA, open `query_cert`, run an LLM, or add an
approximate index in this first pass. CPU tests must be network-free.

## Required tests

In addition to existing tests, cover at least:

- exact feature values for a hand-computed distance profile;
- invariance to a common positive scaling for ratio-only features and explicit
  non-invariance of the radius feature;
- deterministic invalid-state handling and fallback for zero, duplicate,
  nonfinite, insufficient, or unsorted pilot data;
- no feature or decision access to qrels/exact top-k/split role;
- LID-calibrator coefficient/domain/schema/fingerprint validation;
- residual quantile objective and a hand-computed optimum fixture;
- `M_cal` grid safety, lower bound, terminal fallback, and deterministic ties;
- positive and negative residual corrections on low/high synthetic difficulty;
- Raw v1 decisions remain unchanged under the new installation;
- cal/tune/cert/latency/test identity disjointness and protected-access refusal;
- no refit after tune selection and no mutation after certification;
- complete per-query reconstruction of all certification bounds;
- deterministic artifacts under repeated synthetic runs; and
- one projected scan shared by pilot and expansion.

## Statistical and claim discipline

Freeze targets, evidence tolerances, alpha allocation, candidate grids, seeds,
hardware protocol, and sample-size plan before protected access. Use paired
query-level statistics for evidence, budget, and latency comparisons. Apply the
predeclared family-wise correction.

The strong claim requires full PDCTP to meet fresh retention and evidence
constraints and reduce both mean candidate budget and measured steady-state
latency against every eligible fixed, monotone, and Raw Tri baseline. If a
comparator is infeasible, distinguish feasibility dominance from matched-
quality cost superiority. If any gate fails, preserve the failure and stop;
never alter a split, feature, grid, tolerance, or threshold after the fact.

LLM answer generation remains out of scope until the new retrieval/evidence and
latency gates are terminal.

## User/cluster workflow

The user runs Slurm manually. After each coherent local change:

1. run all CPU tests;
2. update status/plan documents;
3. commit on the v2 branch;
4. give the user exact local `git push`, cluster `git pull`, and Slurm commands;
5. ask for complete logs/artifacts when independent audit is required.

Never claim a cluster result that the user has not returned and that has not
been independently reduced.

## Completion of your first turn

Lead with what was implemented and tested. State which stop/go gate is next.
Provide the commit ID and exact push/pull/Slurm commands. Do not proceed to real
certification merely because synthetic tests pass; advance one gate at a time.
