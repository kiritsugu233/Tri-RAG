# Pilot-Distance Calibrated Tri-Predict v2 protocol

## 1. Research objective

Raw Tri-Predict v1 is a terminal negative baseline. The v2 hypothesis is that a
small, deployable calibration layer can preserve the theory-derived ordering
signal while correcting both observed failure mechanisms:

1. bias in scalar pilot LID relative to the local geometry exposed by the full
   pilot distance profile; and
2. miscalibration between analytic Tri-Predict budget and the budget actually
   required for target retention.

The proposed method is **Pilot-Distance Calibrated Tri-Predict** (PDCTP). A
positive claim is a target, not an assumption. The experiment succeeds only if
one frozen PDCTP policy passes fresh retention and evidence constraints and then
shows statistically supported budget and latency reductions against eligible
fixed, monotone-binned, and Raw Tri-Predict baselines. A failed gate remains a
valid terminal result.

## 2. Version and code boundary

- Raw baseline tag: `raw-tri-predict-v1-terminal-negative` at `fb09c00`.
- Successor development branch: `codex/calibrated-tri-predict-v2`.
- Keep `tri_law.py`, the Raw Tri-Predict scientific policy, and all v1 artifact
  loaders backward compatible.
- New method code must use distinct classes, schemas, fingerprints, run
  directories, and report labels. Recommended names are
  `PilotDistanceFeatureExtractor`, `PilotLIDCalibrator`,
  `TriBudgetResidualCalibrator`, and `CalibratedTriPredictPolicy`.
- Never label PDCTP output as a v1 result or change a v1 serialized policy to
  produce calibrated decisions.

## 3. Inference contract

PDCTP may use only information available after the normal single pilot pass:

- the sorted original-space distances of the `M_pilot` projected candidates;
- the corresponding projected-space distances;
- scalar pilot LID, validity, and failure reason;
- fixed corpus/projection/search metadata; and
- the frozen Raw Tri-Predict curve evaluated from a calibrated deployable LID.

It may not use oracle LID, exact original top-k identities, realized retention,
qrels, answer labels, query split names, or corpus-wide original distances at
inference. Calibration artifacts are loaded once and must not adapt online.

The pilot and expansion phases must continue to reuse one projected search.
Feature extraction, calibrated policy evaluation, and artifact lookup latency
must be logged separately.

## 4. Frozen pilot-distance feature specification

The first v2 implementation must use a small explicit feature vector, not raw
text or a learned neural encoder. Let `r_i` be stable-sorted positive Euclidean
original-space pilot distances and `s_i` the corresponding Euclidean projected
distances. Squared-L2 outputs must be square-rooted before the ratios below.
Freeze the epsilon, index positions, rounding, and invalid-value behavior in a
versioned artifact.

Required features are:

1. `log_pilot_lid`;
2. `log_radius = log(r_k + epsilon)` at the frozen LID boundary `k`;
3. mean and standard deviation of `log((r_k + epsilon)/(r_i + epsilon))` for
   the valid interior distances;
4. inner-half and outer-half slopes of that log-ratio profile and their
   difference (curvature);
5. frozen quantiles of consecutive normalized original-distance gaps;
6. mean and standard deviation of
   `log((s_i + epsilon)/(r_i + epsilon))`, exposing projection distortion; and
7. explicit validity/count indicators.

All feature normalization statistics are fit on `query_cal` only. The feature
extractor must be deterministic, stable under the existing ID tie contract, and
covered by hand-computed fixtures. Feature additions after certification require
a new protocol version and fresh data.

## 5. Two-stage calibrated method

### 5.1 Pilot-LID calibration

Fit a regularized log-linear model on `query_cal`:

```text
log(lambda_cal) = intercept + theta^T standardized_pilot_features
```

The target may be oracle LID only on `query_cal`; oracle values never enter an
inference record. Constrain the coefficient of `log_pilot_lid` to be
nonnegative, clip the output to a predeclared domain, and serialize coefficients,
normalization statistics, fit IDs, objective, regularization, and fingerprint.

Predeclare a small regularization grid. Fit every candidate on `query_cal` and
select one on `query_tune`; do not refit after selection in the first protocol.
Report pilot-versus-oracle and calibrated-versus-oracle errors as diagnostics,
but select the final policy by the complete retention/evidence/cost rule below,
not by LID error alone.

### 5.2 Analytic budget-residual calibration

For each `query_cal` query, each predeclared retention-training level, and the
frozen budget grid, compute:

- the Raw Tri-Predict budget `M_raw` using `lambda_cal`;
- the realized smallest grid budget `M_required` reaching that training-level
  top-`k_gt` retention; and
- the residual target `log(M_required / M_raw)`.

Fit a small regularized linear quantile model
`delta_phi(pilot_features)` to that residual. The deployable decision is:

```text
M_cal = grid_ceiling(M_raw * exp(delta_phi + safety_offset))
```

The output is always clipped to the frozen grid and never below
`max(k_gt, M_pilot)`. This residual form keeps Raw Tri-Predict as an explicit
theory anchor while allowing low-budget corrections to be positive and severe
high-LID overallocations to be negative.

Predeclare candidate Raw Tri thresholds, residual training levels, quantiles,
regularization values, and safety offsets before inspecting `query_tune`.
Candidates are fit on `query_cal`; one complete tuple is selected on
`query_tune` by the frozen lexicographic rule:

1. retain only candidates meeting the tune retention lower-bound target;
2. retain only candidates meeting candidate- and final-evidence
   noninferiority tolerances against the tune-selected fixed reference;
3. minimize the common coordinate-work objective;
4. break exact ties by lower mean budget and then canonical fingerprint.

No candidate may be revised after `query_cert` is opened.

### 5.3 Required ablations

Freeze and evaluate, without post-cert selection:

- Raw Tri-Predict v1 algorithm on the new data;
- pilot-LID calibration only;
- budget-residual calibration only, using raw pilot LID;
- full PDCTP; and
- a shuffled-pilot-profile diagnostic on `query_tune` only.

These ablations are required to show whether each independent v1 failure was
actually addressed. Only full PDCTP is eligible for the primary v2 claim.

## 6. Fresh data protocol

SciFact is exhausted for method development. Do not access its cert/test
records from v2 selection code.

The recommended first candidate is a pinned BEIR FiQA release because it offers
a larger external-query pool and a medium corpus compatible with exact
correctness checks. This is provisional until the source archive, license,
official split counts, qrel integrity, and normalized-text duplicates are
audited. If it cannot support the power plan, stop and document the reason
before choosing a replacement dataset or inspecting method outcomes.

Discovery metadata only, not an accepted v2 identity: the BEIR project lists
FiQA as public with train/dev/test qrels, approximately 57K corpus documents,
648 test queries, and archive MD5 `17918ed23cd04fb15047f73e6c3bd9d9` at
`https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip`.
The adapter must independently verify the downloaded bytes, record SHA-256 and
member identities, and count unique eligible queries before GO.

Reuse the already pinned E5 model revision for continuity, but create a new
dataset fingerprint, embedding cache, projection seed, and run namespace.
Fix `m_prime=192` before label access for the first protocol; do not rerun a
dimension sweep. Derive one geometric candidate-budget grid from corpus size
using a checked-in deterministic rule, then freeze the exact values before
calibration.

After normalized-text duplicate grouping and exclusion against the untouched
official test set, assign roles as follows:

- `query_cal`: fit feature normalization, LID calibration, and budget residuals;
- `query_tune`: select all policy hyperparameters and operating points;
- `query_cert`: one-time independent scientific certification;
- `query_latency`: labels ignored; frozen systems comparison only;
- `query_test`: one-time descriptive final report after all other gates are
  terminal.

Prefer source-native partitions when suitable. Any subdivision must use a
seeded, label-free stable group hash so duplicate normalized query text cannot
cross roles. Serialize ordered ID lists and hashes. The five roles, corpus, E5
revision, projection, `m_prime`, pilot contract, budget grid, feature version,
candidate grids, tolerances, alpha allocation, hardware protocol, and seeds must
all be frozen before the corresponding protected split is opened.

## 7. Baselines and fair operating points

All policies use the same corpus, embeddings, projection, projected ranking,
pilot contract, budget grid, exact original reranker, `k_gt`, and `k_ctx`.
Freeze on `query_tune`:

1. the smallest fixed projected-space `M` meeting the tune constraints;
2. a monotone-binned policy fit from the new calibration/tune data;
3. Raw Tri-Predict with its predeclared threshold candidates; and
4. one full PDCTP policy.

Original-space exact retrieval remains the quality reference, not a candidate-
budget peer. Report its latency separately.

A comparator that fails the fresh quality constraints is infeasible and cannot
support a matched-quality cost claim. Report feasibility dominance separately.
To claim that PDCTP reduces cost relative to a comparator, both policies must
meet the same retention/evidence constraints and PDCTP must pass the paired cost
test against that comparator.

## 8. Independent certification

Before any real run, write a sample-size/power artifact and freeze numerical
targets. Do not copy tolerances merely because they make the observed result
pass. The primary certification family contains:

1. an absolute empirical-Bernstein lower bound for PDCTP mean embedding
   retention;
2. paired noninferiority lower bounds for candidate evidence recall and final
   evidence recall@`k_ctx` versus the fixed reference;
3. paired superiority upper bounds for normalized candidate budget versus each
   eligible fixed, monotone, and Raw Tri comparator; and
4. an explicit family-wise alpha allocation, initially Bonferroni unless a
   different procedure is implemented and tested before data access.

Save every per-query value needed to recompute the bounds. Certification is
terminal for the frozen policy. If PDCTP fails any primary gate, report failure
without changing a feature, threshold, residual model, grid, tolerance, or
split.

The strong primary statement is allowed only if all required gates pass:

> On fresh external queries, frozen Pilot-Distance Calibrated Tri-Predict meets
> the predeclared retention and evidence constraints and reduces mean candidate
> budget relative to every eligible fixed, monotone-binned, and Raw Tri-Predict
> comparator.

Otherwise use the exact weaker statement supported by the terminal artifacts.

## 9. Latency protocol

Latency is a separate systems gate after scientific certification. Use the
frozen `query_latency` IDs, exact FAISS CPU first and one frozen NVIDIA GPU
backend second. Freeze hardware class, device count, package versions, threads,
batching mode, warmup count, repetitions, method-order randomization seed,
boundary guard, and cache state.

Construct randomized paired blocks so each query/method is measured under the
same repetition. Report stage times, p50/p95/p99 total latency, distance counts,
bytes, CPU RSS, GPU memory, and index-build/setup time. The primary latency claim
uses a predeclared one-sided paired interval for mean steady-state latency with
family-wise correction across fixed, monotone, and Raw Tri comparisons. Tail
latencies remain descriptive unless a separate tested procedure is frozen.

Compiled/loaded calibration evaluation must remain outside index construction
but inside per-query retrieval timing. Offline fit/compile time is reported
separately. A latency win may not be inferred from candidate count alone.

## 10. Artifact contract

At minimum, produce fingerprinted artifacts for:

- dataset, split groups, embeddings, projection, and budget grid;
- pilot-distance feature specification and normalization;
- every fit candidate and ordered `query_cal` ID hash;
- frozen LID calibrator and residual calibrator;
- Raw, monotone, fixed, ablation, and PDCTP policies;
- tune selection with rejected/eligible candidates;
- per-query cal/tune/cert/latency/test records in separate directories;
- certification hypotheses, alpha allocation, bounds, and terminal decision;
- paired latency blocks and environment/memory metadata; and
- a generated report distinguishing retention, candidate evidence, final
  evidence, budget, coordinate work, and measured latency.

Timings and timestamps must remain outside portable scientific identities.
Every protected runner validates all frozen upstream artifacts before reading
its split.

## 11. Stop/go gates

1. Do not touch real data until synthetic feature/calibrator tests pass.
2. Do not inspect `query_tune` until all candidates are fit on `query_cal`.
3. Do not inspect `query_cert` until one complete policy and all hypotheses are
   frozen.
4. Do not run latency until the scientific certificate is terminal.
5. Do not inspect `query_test` until certification and latency protocols are
   terminal and no selection remains.
6. Never use SciFact cert/test outcomes to change v2.
7. Never promise a positive result; preserve every failure.
