# TLS-RAG Step 3 implementation brief

## 1. Mission

Complete only TLS-RAG Step 3: the CPU/network-free synthetic observed-pair
Tri-Law feature profile, transparent gain/sufficiency scoring, disjoint
synthetic fit/bound calibration, conservative dual-bound controller, and Rows
1--4 of the frozen one-factor ablation ladder. Commit the result on the fresh
Step 3 task branch and stop for user review.

Step 3 is not a real-data experiment, policy selection, certification, latency
benchmark, production system, or answer experiment. Do not begin Step 4.

## 2. Minimal reading contract

Before editing, follow the complete ordered allowlist in
`docs/TLS_RAG_STEP3_START_HERE.md`. Do not preload the historical status/plan,
Step 1/2 prompts, Raw Tri-Predict or PDCTP documents/code, real-data documents,
artifacts, runs, protected roles, or archives. The only optional supporting
reads are exact imported symbol definitions in `projection.py`, `indexes.py`,
`embeddings.py`, and `utils.py`.

If a missing file is genuinely required, stop and identify it before widening
the read set. Repository filename/symbol searches are allowed; opening an
unrelated match is not.

At final documentation time only, locate the TLS-RAG section headings in
`docs/IMPLEMENTATION_PLAN.md` and `STATUS.md` with `rg`, then read and update
only those bounded sections. Do not ingest their historical bodies.

## 3. Starting point and protection boundary

The task prompt supplies the exact documentation-handoff commit. Verify it is
`HEAD` or an ancestor and verify that Step 2 commit
`f46ce73beccf5fddbf05fd87fdb0911318020add` is an ancestor. Create or use an
isolated `codex/` Step 3 task branch. Do not rewrite, squash, force-push, reset,
or amend the frozen lineage.

Do not modify:

- `src/tri_rag_harness/tri_law.py` or `tests/test_tri_law.py`;
- Step 2 behavior, schemas, config, portable artifacts, or expected tests;
- Raw Tri-Predict v1 or PDCTP v2/v3 code, schemas, artifacts, and tests; or
- any real/protected role state, returned archive, or historical result.

No network, dataset/model/dependency download, real-data adapter, approximate
index, FAISS work, GPU run, LLM call, answer generation, or answer evaluation
is authorized. Use only existing NumPy/SciPy and the deterministic synthetic
fixture.

## 4. Required isolated implementation

Prefer adding only:

- `src/tri_rag_harness/tls_rag_step3.py`;
- `configs/tls_rag_step3_synthetic_v1.json`;
- `tests/test_tls_rag_step3.py`; and
- `docs/TLS_RAG_STEP3_SYNTHETIC.md`.

Import or wrap Step 2 exact retrieval and evidence definitions. New objects,
schemas, versions, fingerprints, namespaces, and artifacts must use
`tls_rag_step3` or `tri_law_sequential_rag_step3`. Do not create a broad generic
controller framework.

Freeze every stochastic seed and all feature names, shell size, quantiles,
invalid behavior, score model parameters, regularization choices, partition
rules, bin rules, minimum cell size, alpha allocation, candidate registry,
thresholds, and fallbacks in the checked-in config/manifest.

## 5. Exact observed-pair Tri-Law profile

Use only candidates in the current exposed prefix whose original distances and
embeddings are already available. For observed displacements `u=x_i-q` and
`v=x_j-q`, orient a valid pair so `||u|| < ||v||`, then compute:

```text
beta = ||v||^2 / ||u||^2
rho  = <u,v> / (||u|| ||v||)
```

Call the unchanged `tri_law_probability(beta, rho, m_prime)` for each valid
pair. Implement exactly:

- `context_boundary_risk`: farthest context item versus each farther exposed
  non-context item;
- `core_frontier_risk`: every context item versus the frozen newest shell;
- valid, invalid, tied, zero, nonfinite, and collinear counts;
- valid probability mean, maximum, and quantiles 0.50/0.90/0.99;
- frozen core/shell projected-to-original log-distortion quantiles; and
- previous-state deltas with explicit first-state values and validity.

Stable retrieval ties remain. Equal original distances do not define
`beta>1`; exclude and count them. Zero/nonfinite displacements are invalid.
Collinear pairs retain the exact API's boundary behavior but must be counted as
specified and may invalidate a required profile. Duplicate projected distances
are counted and never treated as independent evidence.

Never inspect or infer unexposed original distances/angles. Never reconstruct
them from scalar LID. Pair probabilities are dependent geometric features and
must not be summed or described as a theorem, posterior missing-evidence
probability, relevance probability, sufficiency probability, or guarantee.

## 6. Synthetic scoring and calibration

Create enough deterministic synthetic external queries to exercise valid and
invalid cells without changing the Step 2 fixture. Partition query IDs by a
frozen label-free stable hash into disjoint synthetic model-fit, bound-fit, and
evaluation sets. These names must be explicitly synthetic and must not open or
impersonate real `query_cal/tune/cert/latency/test` roles.

Fit transparent versioned score models for:

- remaining useful evidence; and
- current final-context sufficiency.

Model fitting may use Phase B labels only from the synthetic model-fit
partition. Feature normalization and all learned parameters are frozen and
fingerprinted. Missing/nonfinite input has a deterministic invalid result, not
imputation from future outcomes.

On the disjoint synthetic bound-fit partition, freeze score-bin edges from
label-free scores and compute exact one-sided Clopper-Pearson reachable-bin
event-rate limits. Allocate family-wise alpha explicitly across every
candidate, stage, bin, and both outcomes. Treat query ID—not multiple stages
from one query—as the independent unit. Empty or underpowered cells must return
the vacuous interval `[0,1]`.

Build candidate-specific stage tables sequentially: a stage receives only
bound-fit queries whose earlier frozen decisions under that same candidate
reach the stage. Save the reachable query IDs or a lossless reconstructable
reference and test the reconstruction.

## 7. Controller and ablation contract

The calibrated controller accepts only a versioned immutable deployable state
and frozen model/calibration artifacts. It has no supervision-store handle.
Reuse the exact Step 2 action enum; no third action and no grid jump.

STOP is permitted only when all mandatory state/profile/model/cell validity
checks pass and both conditions hold:

```text
U_gain <= delta_gain
L_suff >= tau_sufficient
```

Otherwise expand exactly to the next grid value when possible. Invalid or
nonfinite features, schema mismatch, prediction failure, missing required
Tri-Law support, invalid/empty/underpowered cells, zero query displacement,
empty evidence plan, or either failed inequality forces expansion. At the
terminal budget, stop with explicit exhaustion, maximum-expansion, invalid, or
evidence-nonattainment reasons; never enlarge the grid or relax thresholds.

Implement only these synthetic rows:

1. fixed-grid exact reference;
2. sequential distance/distortion/redundancy/validity features;
3. Row 2 plus only the observed-pair Tri-Law profile; and
4. Row 3 plus only deterministic evidence-plan/facet-prediction features.

Rows 2--4 share every component except the named feature addition. Save and
test their exact feature-name differences. Do not select a winning policy or
claim superiority; report diagnostic outcomes for all rows.

## 8. Phase and artifact contract

Every synthetic evaluation runner has two explicit phases:

- Phase A builds and serializes the full inference trajectory without opening
  evaluation supervision, fingerprints it, and closes it.
- Phase B joins only the permitted synthetic evaluation labels, reconstructs
  targets/diagnostics, and proves the Phase A fingerprint is unchanged.

Keep complete query-stage records. Aggregates alone are invalid. Separately
record deterministic work and nonportable timing for query projection,
projected full scan/pilot, prefix reuse, new original distances, exact rerank,
risk features, deterministic plan/facet features, score inference, calibration
lookup, controller, and context. Offline fixture generation, feature/model fit,
and calibration setup are separate from query work/latency.

The runner must emit at least:

- manifest, config/fixture/projection/ID-map identities;
- risk-feature specification and feature registry;
- synthetic partition proof and candidate registry;
- score-model artifacts and candidate-specific calibration tables;
- Phase A decision JSONL and separate Phase B supervision JSONL;
- one-factor ablation query/stage records;
- work counters, nonportable timings, reconstructable aggregates; and
- a report explicitly denying real-data, selection, certification, latency,
  posterior, evidence-guarantee, and answer claims.

Run twice in fresh directories and compare every portable artifact byte for
byte. Timings and runtime timestamps are nonportable and excluded from portable
fingerprints.

## 9. Required focused tests

Cover at least:

1. Step 2 fixed-schedule output and fingerprints remain identical;
2. exact hand-computed observed `beta/rho` and unchanged Tri-Law API values;
3. current-prefix-only pair construction and refusal of unseen candidates;
4. context-boundary/core-frontier orientation, summaries, quantiles, and
   previous-state deltas;
5. tied/equal, duplicate, zero, nonfinite, collinear, and empty-profile counts;
6. same distance curve/different angles produces different profiles;
7. realized projection outcome does not change the ex-ante interpretation;
8. Rows 2--4 differ by exactly the frozen one-factor feature sets;
9. deterministic disjoint synthetic fit/bound/evaluation partitions;
10. model-fit labels cannot reach bound fitting or inference objects;
11. hand-computed one-sided Clopper-Pearson limits and family-wise alpha;
12. candidate/stage/bin reachability uses independent query IDs;
13. empty/underpowered cells return `[0,1]` and prohibit STOP;
14. both bound inequalities and every validity flag are required for STOP;
15. all invalid states expand next-grid-only and terminate explicitly;
16. recursive forbidden-field rejection and no label-store reachability;
17. Phase A serialization precedes the label join and its fingerprint survives;
18. complete separate work/timing accounting and aggregate reconstruction;
19. two-run byte-identical portable artifacts; and
20. no network, dependency, real/protected role/archive, approximate search,
   GPU, LLM, answer, Step 4 runner, or scientific claim.

Every pre-existing CPU test must remain unchanged and pass.

## 10. Verification, documentation, and Git

The local full regression command is:

```bash
cd /Users/guanghongxu/Query-Adaptive-Tri-RAG
./scripts/run_tests.sh
```

Document the exact Step 3 runner command using the checked-in module, config,
and a fresh output directory. The final handoff must include the actual branch,
full commit hash, focused/full test counts, pass/fail/skip and runtime, portable
fingerprints, exact push and cluster-sync commands, and this manual Slurm shape:

```bash
cd /home/users/u0001611/Tri-RAG
salloc \
  --job-name=tls-rag-step3 \
  --cpus-per-task=1 \
  --mem=8G \
  --time=00:30:00
```

After allocation:

```bash
cd /home/users/u0001611/Tri-RAG
srun --ntasks=1 --cpus-per-task=1 bash -lc '
cd /home/users/u0001611/Tri-RAG
eval "$(micromamba shell hook --shell bash)"
micromamba activate tri-rag
./scripts/run_tests.sh
'
```

At the end, update `docs/TLS_RAG_STEP3_SYNTHETIC.md` plus only the bounded
TLS-RAG sections of `docs/IMPLEMENTATION_PLAN.md` and `STATUS.md`. Commit every
Step 3 change on the isolated task branch. Do not push automatically. Stop and
wait for user review; passing Step 3 tests does not authorize Step 4.
