# TLS-RAG Step 4 retention v3: next effectiveness acceptance

> **After the approved Tri-Law repair:** this is the historical v3 experiment
> brief and replay command for the old source-bound v2 bundle. Current development
> uses `codex/tls-rag-step4-retention-v3` HEAD, which includes the reviewed
> baseline `62377d8`; `codex/repo-review-code-protection` stays as its backup. Its repaired `tri_law.py` has a new
> hash, so the unchanged loader correctly refuses the old bundle. Do not use the
> clone/run instructions below as the current development startup, or rewrite
> old hashes. Use the original reviewed commit for historical replay; new runs
> need reviewed bindings and appropriate independent roles from the corrected
> code. See [the repair record](TRI_LAW_NUMERICAL_FIX.md).

Canonical local checkout: `/Users/guanghongxu/Query-Adaptive-Tri-RAG`.
Branch: `codex/tls-rag-step4-retention-v3`, successor of `c08cd3b6800888c8d2a5a5f8c58a77d883a84ede`.
The user requested implementing the retention-aligned successor and providing the
next real-data acceptance command. Step 2/3, exact Tri-Law and Step 4 v1/v2 stay frozen.

## Problem observed in v2

The user reported `no_tune_candidate` on 256 NFCorpus tune queries. All three
adaptive rows had mean budget 32, retention 0.6453 and context hit rate 0.6211.
Exact-original hit rate was 0.6328; fixed-512 retention was 0.9551. Row 3 added
41.83984375 risk-pair evaluations per query without changing retrieval results.
These are user-reported aggregates, not an independently inspected cluster bundle.

V2 fitted a binary judged-context-hit proxy, while acceptance also required 90%
mean exact top-10 retention. Its inherited gain/sufficiency thresholds did not
constrain retention. V3 replaces that stopping mechanism and keeps the observed
v2 failure intact. Merely running v3 successfully does not establish effectiveness.

## Frozen successor experiment

Protocol: `configs/tls_rag_step4_retention_v3.json`.
Fingerprint: `1d730efc412851c45b5ae2a600387ed4aca6c749aceaff87c30f40eebf55e026`.

Use the same pinned model, normalized embeddings, m'=128 projection, seed 24103,
exact squared-L2/string-tie retrieval, k_gt=10, k_ctx=5 and [32,128,512] grid as v2.
No new data downloads, GPU embedding or dependencies are required.

The explicitly authorized reuse is:

- V2 `query_cal_model_fit` (256): fit a separate standardized ridge model for
  each stage and feature family (Rows 2–4), with target **continuous exact top-10
  retention**, regularization 0.01. Exact top-k is offline supervision only.
- V2 `query_cal_bound_fit` (512): compute prediction minus actual retention.
  Safety margins are max(0, empirical residual quantile), using `method=higher`
  at q=0.5, 0.8 or 0.9. Fewer than eight valid queries leaves the model/cell invalid.
- V2 `query_tune` (256): development/selection data, already observed in v2.
  Select among the predeclared 3 feature families × 3 stop thresholds
  [0.90,0.925,0.95] × 3 margin quantiles = 27 candidates.
- Preserve the same **unopened** V2 `query_probe` (256) as the held-out sample.
  It is prepared/opened only after the selected v3 policy has been serialized.
  No cal/tune query is moved into this role. Official dev/test remain closed.

At inference, STOP if a valid current-prefix prediction minus its frozen stage
margin (clipped to [0,1]) reaches the policy threshold. Otherwise expand to the
next grid budget. The final budget forces an explicit terminal STOP without
asserting retention success. The controller receives only the registered
`Step3DeployableState`, never qrels, realized retention, exact top-k or future
prefix outcomes. Invalid states/calibration cause expansion, not optimistic stops.

These residual margins are empirical score adjustments, **not confidence bounds**
or per-query retention guarantees. Stagewise calibration is not a proof about
adaptive stopping. Semantic question-family independence remains unaudited.
This milestone is descriptive held-out acceptance, not formal certification.

Acceptance targets are unchanged, on tune and then on the selected policy's probe:

1. Mean exact top-10 retention >=0.90.
2. Mean context-hit rate no more than 0.02 below exact-original.
3. Mean original query-to-document distance evaluations strictly below 512.

Select the qualifying tune policy with the smallest mean original distance work;
tie by ascending policy ID. If none qualifies, stop with `no_tune_candidate` and
keep probe closed. No threshold changes or automatic retries follow that failure.
After selection, probe also reports Rows 2–4 at the **same selected threshold and
margin quantile**, alongside fixed 32/128/512 and exact-original. These are matched
feature comparisons; probe outcomes never choose a replacement policy.
A passing selected controller does not automatically demonstrate a Tri-Law benefit.

## Existing paths and safety of reuse

The documented parent run is:
`/home/users/u0001611/Tri-RAG-step4-real-probe/step4-nfcorpus-v2-01`.
Only its `bundle/` and exact v2 `evaluation/` files needed for this successor are read.
Do not read unrelated archives, previous project datasets or other run namespaces.

The loader checks the old bundle hashes, embedding metadata and frozen source code.
It verifies the v2 `result.json`, `selection.json`, `input_binding.json`, `roles.json`
and `protocol.json` against the old artifact manifest, requires `no_tune_candidate`,
and rejects any existing v2 probe features, decisions, metrics, summary, label-opening
receipt or opened-label file. This cannot detect a person's unrecorded inspection
of held-out labels; the probe must actually have remained unseen.

The exact cached feature files reused are:
`evaluation/query_cal_model_fit.features.jsonl`,
`evaluation/query_cal_bound_fit.features.jsonl`, and
`evaluation/query_tune.features.jsonl`.
Their hashes are checked before copying. Complete query/stage identities and feature
registries are validated. Existing labels enter only the permitted development or
held-out metric joins. Cal roles use newly reconstructed retention targets, with
label-free feature closure receipts written first. Probe decisions are serialized
and closed before its labels and exact-reference metrics are reconstructed.

The old run is never modified. A new absent output directory stores the v3 protocol,
lineage hashes, environment/Slurm identity, retention targets, stage models,
calibration residuals/margins, decisions, closure receipts, tune selection and
query-level metrics. `query_tune.stopping_audit.json` and, if opened,
`query_probe.stopping_audit.json` list stop budgets and reasons for each policy.
The final `artifact_manifest.json` hashes all files in the new result directory.

All-prefix preparation and cached replay are offline audit work. Work counters
separate pilot/expansion original distances, reranking, feature-pair and controller
work; they are not measured serving latency. Reusing vectors/caches makes this
acceptance CPU-only, even when launched within the user's A100 allocation.

## GitHub synchronization and manual acceptance

Use a separate checkout so the existing v2 run and code remain available:

```bash
cd /home/users/u0001611
git clone --single-branch --branch codex/tls-rag-step4-retention-v3 \
  https://github.com/kiritsugu233/Tri-RAG.git Tri-RAG-step4-retention-v3
cd /home/users/u0001611/Tri-RAG-step4-retention-v3
git rev-parse HEAD
```

Compare HEAD to the commit delivered with this milestone, then use that exact
commit for the run. Within the existing compute allocation, activate micromamba:

```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate tri-rag
cd /home/users/u0001611/Tri-RAG-step4-retention-v3
```

New focused test:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover \
  -s tests -p test_tls_rag_step4_retention_v3.py -v
```

Full regression and the **next actual effectiveness acceptance**, in one sequence:

```bash
(
set -eu
cd /home/users/u0001611/Tri-RAG-step4-retention-v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
sh scripts/run_tests.sh
sh scripts/run_tls_rag_step4_retention_v3.sh \
  /home/users/u0001611/Tri-RAG-step4-real-probe/step4-nfcorpus-v2-01 \
  /home/users/u0001611/Tri-RAG-step4-retention-v3/acceptance-v3-01
)
```

The final command fits, calibrates, selects and, if tune passes, evaluates the
held-out probe. It prints `result.json`'s compact status and `report.md`.
There is no intervening approval, second run command, network or GPU setup step.

Interpret the status:

- `heldout_targets_met`: all three empirical held-out targets passed for the
  tune-selected policy; inspect matched feature deltas before attributing benefit
  to Tri-Law. No formal certification or latency claim follows.
- `heldout_targets_failed`: the selected policy failed at least one held-out target.
  Preserve this negative result; any redesign needs new unseen evaluation queries.
- `no_tune_candidate`: no preregistered policy qualified on tune; probe stayed closed.
- An exception is an input/integrity/implementation failure, not a passed acceptance.

Do not delete an opened-probe result or change thresholds and reuse it as fresh
validation. If the old v2 run used a different output directory, substitute its
actual root only; do not mix files from different runs.

## Validation

- Focused new v3 suite: 12/12 passed (3.254 s before the environment-record addition;
  all 12 also passed in both final full regressions).
- Complete command: `sh scripts/run_tls_rag_step4_readiness.sh /private/tmp/tls-rag-retention-v3-regression-01`.
  Two rounds each ran 255 tests: 254 passed, one optional real-FAISS skip;
  durations 31.693 s and 31.435 s. Original focused v1 tests: 26/26 each.
- Six frozen v1 artifact files remain byte-identical with their previous fingerprints.
  Outputs: `/private/tmp/tls-rag-retention-v3-regression-01/{a,b}`;
  log: `/private/tmp/tls-rag-retention-v3-regression-01.log`.
- New complete-pipeline fixtures produce equal artifact manifests in two runs.
  Tests cover continuous targets (including fractional values), low-score expansion,
  margin effects, invalid inputs, terminal nonattainment, unchanged gates, label
  closure, cache reuse/tampering, unopened-probe guards and the actual CLI path.
- Existing Step 2/3, exact Tri-Law and v1/v2 code/config/tests are unchanged.

No local real data were accessed during implementation. The actual v3 cluster
result remains to be produced by the command above.
