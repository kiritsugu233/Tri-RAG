# TLS-RAG v1 fresh-data and evaluation protocol

Status: Step 1 design freeze. This protocol defines future gates. It authorizes
no real-data access, download, embedding run, protected-role run, or LLM call.

## 1. Scientific boundary

Tri-Law Guided Sequential RAG Controller (TLS-RAG) is a new successor family.
Raw Tri-Predict v1, Pilot-Distance Calibrated Tri-Predict v2, and the v3 causal
repair are frozen historical references. Their fitted parameters, FiQA
cal/tune outcomes, protected identities, and SciFact outcomes must not fit,
select, threshold, or support a TLS-RAG claim.

The primary hypothesis is that a sequential controller using deployable
retrieval/evidence state can preserve evidence quality while reducing measured
work and latency relative to a properly matched fixed budget. The added-value
hypothesis for Tri-Law is narrower: an observed-pair Tri-Law risk profile adds
useful allocation information beyond the otherwise identical sequential
controller. A negative result at either hypothesis is valid and terminal.

## 2. Fresh-data requirements

A source audit must select a dataset not used to develop v1, v2, or v3. It must
pin source URL/revision, byte hashes, license and use restrictions, archive
members, corpus/query counts, query-corpus disjointness, stable IDs, qrel and
evidence-reference integrity, missing/empty texts, and duplicate normalized
query text.

Before accepting the source, the audit must establish that its annotations can
identify the targets in the TLS-RAG specification:

- atomic evidence facets tied to the deterministic evidence plan;
- passage-to-facet support labels;
- independent-source groups when a facet requires multiple supports;
- contradiction or invalid-evidence labels when the frozen plan uses them; and
- a completeness contract sufficient to label current context sufficiency and
  remaining useful evidence across the frozen budget grid.

Ordinary relevance qrels identify candidate relevance gain but do not, by
themselves, prove facet completeness or context sufficiency. If the required
annotation is missing, the audit must either obtain a separately pinned,
independently produced annotation before role assignment or stop. It may define
a differently named candidate-gain-only study, but it may not weaken TLS-RAG's
primary evidence claim after viewing outcomes.

All corpus and query embeddings must be normalized before one frozen dense
Gaussian projection with variance `1/m_prime`; projected vectors are not
renormalized. Search and reranking use squared L2 and exact stable string-ID
ties. The first backend preserves one projected ranking and reuses prefixes.

## 3. Duplicate-safe five-role assignment

After normalizing query text with a frozen NFKC/casefold/whitespace procedure,
group identical text and keep every group within exactly one role. Stable IDs
and normalized texts must have empty intersections across:

- `query_cal`
- `query_tune`
- `query_cert`
- `query_latency`
- `query_test`

Use native partitions where they preserve these rules. Otherwise assign whole
groups through a seeded, label-free stable hash. Serialize ordered ID lists,
normalized-text hashes, group assignments, counts, and pairwise-disjointness
proofs. Power and calibration-cell requirements must be met before outcomes
are inspected. If they are not met, the source gate fails.

## 4. Role permissions and supervision

| Role | Allowed supervision and purpose | Forbidden use |
|---|---|---|
| `query_cal` | Fit evidence-gain, remaining-gain, and sufficiency score models; fit deterministic feature normalization; calibrate score mappings and one-sided uncertainty limits. Facet/support/contradiction labels, exact full-grid candidate/context outcomes, qrels, and exact top-k/retention may create training labels or diagnostics. Internally split by stable hash into `query_cal_fit` and `query_cal_bounds`. | No policy/threshold selection. Answer labels do not fit the retrieval controller. No cal outcome enters a decision object. |
| `query_tune` | Evaluate all preregistered fixed and adaptive candidates; select exactly one candidate per eligible ablation and the single primary controller; choose `delta_gain`, `tau_sufficient`, and other preregistered operating points. Evidence/retention labels may evaluate quality only. | No refit, feature invention, candidate addition, uncertainty recalibration, or use of cert/latency/test. |
| `query_cert` | One-time independent evaluation of frozen evidence, sufficiency, retention, and work hypotheses. Decisions and pre-supervision records are finalized before labels are joined. | No selection, fitting, retuning, fallback change, repeated certificate, or latency/answer claim. |
| `query_latency` | Label-free randomized paired systems measurement of the frozen eligible methods after certification is terminal. Only query inputs, deployable state, work counters, timings, memory, and environment data are used. | No qrels, evidence/answer labels, policy changes, clipping, or quality selection. |
| `query_test` | One-time descriptive evaluation of the final frozen methods after certification and latency are terminal. Retrieval/evidence labels may be joined after decisions. Step 6 may later use frozen answer labels only under its separate protocol. | No fit, selection, new certificate, threshold change, or replacement of a failed gate. |

Oracle LID, exact full-corpus top-k identities, realized retention, qrels,
evidence labels, answer labels, split role, and future expansion outcomes are
never fields in the inference decision object, regardless of role.

### Enforced label separation

Each runner has two phases. Phase A receives only a typed allowlisted inference
object, computes the complete trajectory, hashes it, and closes it. Phase B may
load the role-specific supervision store and append label/outcome fields.
Policy code has no supervision-store handle. Schema tests reject forbidden
names recursively; role guards reject wrong IDs before parsing outcome columns;
decision fingerprints are recomputed before and after the join and must match.
Query-level records keep the two namespaces visibly separate.

## 5. Calibration and controller candidate freeze

Before `query_tune` access, freeze:

1. a label-free stable-hash split of `query_cal` into model-fit and bound-fit
   queries;
2. the transparent score-model class, regularization grid, feature ablations,
   missing-value behavior, stage handling, and score output domain;
3. label-free score-bin construction, minimum cell size, underpowered-cell
   fallback, one-sided Clopper-Pearson calculation, and simultaneous alpha
   allocation over gain/sufficiency, stages, bins, and candidates;
4. candidate values of `delta_gain`, `tau_sufficient`, maximum expansions,
   and every other operating parameter;
5. the full fixed-budget grid and Rows 1--4 of the one-factor ablation ladder;
6. quality targets, noninferiority margins, work definition, selection order,
   deterministic ties, and sample-size plan.

Correlated states from one query are never counted as independent calibration
samples. Limits are computed separately at each stage over independent query
IDs and separately for every preregistered candidate. Candidate-specific tables
are built sequentially: only calibration queries whose earlier frozen actions
reach stage `t` enter its cells. The family-wise allocation covers candidates,
stages, bins, and both outcomes. Empty or underpowered cells return `[0, 1]`,
which prevents STOP. The artifact must call these values reachable-bin event-
rate confidence limits, not per-query posterior probabilities or exact Tri-Law
bounds.

## 6. Fixed reference and tune selection

All methods use identical corpus/query embeddings, projection, projected
ranking, pilot, budget grid, exact original reranker, context builder, evidence
plan, `k_gt`, `k_ctx`, terminal budget, and work accounting.

### 6.1 Matched fixed reference

For every fixed grid budget, compute tune query-level embedding retention,
candidate evidence coverage, final-context evidence coverage, context
sufficiency, and work. A fixed candidate is quality-eligible only if all
predeclared absolute lower bounds pass. Select `F*` as the eligible fixed budget
with least mean common work, breaking ties by lower budget then canonical
fingerprint. If none is eligible, the protocol stops before adaptive selection.

This fixed reference is the primary matched comparator. An adaptive candidate
must meet the same absolute constraints and must pass paired noninferiority
against `F*` for both candidate and final-context evidence. A method with weaker
quality is not allowed to claim a cost win.

### 6.2 Cost-superiority eligibility

For query `i`, let `W_i(A)` be the frozen common-work measure for method `A`,
including query projection, projected search/prefix handling, all original
distance evaluations and reranks, controller/evidence-plan computation, and
context construction with preregistered coordinate/byte weights. Offline fit
and setup are reported separately. Candidate count is a component, not the
objective by itself.

An adaptive candidate is tune-eligible only when the predeclared one-sided
paired upper confidence bound satisfies:

```text
U_tune(mean_i[(W_i(A) - W_i(F*)) / W_i(F*)]) < 0.
```

Requiring this strict tune-side superiority prevents the v2 failure in which a
minimum-cost member could be selected inside an entirely inefficient family.
If no member of a family passes quality and cost eligibility, that family has
no selected policy; no threshold or fallback is revised.

Among eligible candidates, select lexicographically:

1. lowest mean common work;
2. lowest mean candidate budget;
3. canonical candidate fingerprint.

The primary method is Row 4 only if it is eligible. Rows 2 and 3 remain required
one-factor ablations. Row 3 supports a Tri-Law feature benefit only if its
predeclared paired quality/work comparison against Row 2 passes; otherwise the
result falsifies or fails to support that benefit. Historical v1/v2/v3 methods
are descriptive references and do not share fitted components.

## 7. Independent certification hypotheses

Every confidence procedure, direction, target, margin, family, and alpha share
is frozen before `query_cert`. Query is the independent unit; all bounds are
reconstructable from saved paired records. The primary evidence/retention
family for the frozen TLS-RAG policy contains at least:

1. **Context sufficiency:** a one-sided lower confidence bound for mean stopped-
   context sufficiency is at least the preregistered `sufficiency_target`.
2. **Final-context evidence coverage:** an absolute lower bound meets
   `context_coverage_target`, and a paired lower bound versus `F*` is no less
   than `-epsilon_context`.
3. **Candidate evidence coverage:** a paired lower bound versus `F*` is no less
   than `-epsilon_candidate`.
4. **Embedding retention diagnostic:** a separate absolute lower bound for
   mean top-`k_gt` retention meets `retention_target`. This does not substitute
   for either evidence hypothesis.
5. **Work superiority:** a paired upper bound for normalized common-work
   difference versus `F*` is strictly below zero. Equivalent comparisons are
   required against every other quality-eligible frozen comparator named in
   the primary claim.

Use the preregistered empirical-Bernstein implementation for bounded means and
paired differences, with a tested family-wise correction (Bonferroni by
default). Report each component even if an earlier component fails. The
retrieval/evidence certificate passes only if every primary component passes.
Failure is terminal and preserves the frozen policy.

Certification must also report, separately: immediate candidate gain, immediate
context gain, remaining-gain calibration, sufficiency calibration, coverage by
facet/stage, risk-profile validity, terminal reasons, candidate budgets, and
embedding retention. None is silently converted into an answer-quality claim.

## 8. Paired systems gate

Only after the certificate is terminal may `query_latency` be opened. Freeze
the exact backend, hardware class/device count, package versions, CPU threads,
batching, cache state, prefix-reuse implementation, warmups, repetitions,
method-order randomization seed, boundary guard, memory measurement, and
failure policy.

Record separate latency and work for:

- query projection;
- pilot projected search;
- every expansion search or prefix-reuse operation;
- original-space distance evaluation and reranking at every stage;
- deterministic evidence planning and controller evaluation;
- final-context construction;
- setup/index build and offline fitting outside query latency; and
- optional LLM planning/feedback/generation only in Step 6.

The primary latency hypothesis is, for each required eligible comparator `B`:

```text
H0: E[latency_TLS - latency_B] >= 0
H1: E[latency_TLS - latency_B] < 0
```

The preregistered one-sided paired upper confidence bound for mean steady-state
latency must be below zero under family-wise correction. Work superiority is
tested analogously and remains a separate hypothesis. Report p50/p95/p99,
bytes, distance counts, expansion counts, controller time, CPU RSS, device
memory, setup time, and failures, but do not infer latency from budget or
candidate count. Backend incompatibility is a terminal systems failure; never
clip the selected policy to make it run.

## 9. Later LLM gate and answer hypothesis

No LLM call is permitted until both conditions hold:

1. the independent retrieval/evidence certificate passes; and
2. paired measured latency superiority over the matched fixed baseline passes.

Step 6 then requires a separate instruction and frozen protocol. It may expose
only a structured signal such as `supported_claims`, `missing_facets`,
`cited_context_ids`, `contradiction_detected`, and
`request_one_more_expansion`. The LLM cannot emit a budget, alter the grid, or
supply a calibrated probability. Freeze model/revision, prompt, schema,
contexts, decoding, seeds, failure handling, and generator for every comparator.

The minimum answer hypothesis is paired noninferiority of frozen TLS-RAG plus
the common generator versus `F*` plus the same generator:

```text
LCB(E[answer_score_TLS - answer_score_F*]) >= -epsilon_answer.
```

If frozen LLM feedback is claimed as beneficial, its one-factor ablation must
also pass a separately preregistered paired superiority hypothesis against the
same TLS-RAG controller without feedback. Answer labels never enter retrieval
decisions. Answer failure cannot replace a retrieval policy or reopen earlier
roles.

## 10. Seeds and fingerprints

Every stochastic component has an explicit seed, including role assignment,
projection, synthetic generation, model initialization or solver randomness,
cross-fitting, candidate order, bootstrap or permutation procedures if later
authorized, and latency method order.

Content-derived fingerprints bind at least:

- source bytes/revision/license record, corpus IDs/text hash, query IDs,
  normalized query-text groups, role order, and evidence annotation schema/data;
- embedding provider/model revision, text formatting, dimensions, dtype,
  normalization flag, array hashes, and software contract;
- projection family, variance, seed, matrix hash, `m_prime`, no-renormalization
  flag, exact backend, squared-L2 semantics, batching, and tie rules;
- `M_pilot`, ordered budget grid, terminal budget, maximum expansions, `k_gt`,
  `k_ctx`, context builder, evidence-plan generator, and facet matcher;
- every state/risk-profile feature name and version, shell size, quantiles,
  invalid behavior, model parameters, calibration partitions/bins/limits,
  alpha allocation, thresholds, and fallback;
- baselines, candidate registry, selection rule, quality targets, margins,
  hypotheses, work weights, latency protocol, code commit, and all seeds; and
- in Step 6, model/prompt/context/schema/decoding/generation identities.

Cache reuse is refused unless all relevant metadata matches. Runtime timestamps
and timings are excluded from portable scientific fingerprints but stored with
environment and run identities. Calibration and controller artifacts use new
TLS-RAG names, schemas, namespaces, and directories.

## 11. Query-level record contract

Aggregate-only output is invalid. For every query and visited stage, retain:

- query ID, frozen protocol and policy fingerprints, step, budget, remaining
  steps, action, stop/fallback reason, and validity flags;
- exposed candidate IDs, projected ranks/distances, original query distances,
  stable reranked IDs, and context IDs;
- complete deployable feature values, observed-pair counts, `beta`/`rho`
  summary inputs or a lossless reconstructable reference, Tri-Law profile,
  distortion, redundancy, plan/facet predictions, scores, bins, and bounds;
- stage work counts and timings for projection, pilot, prefix reuse/expansion,
  original distance/rerank, planning/controller, and context;
- in a separately joined supervision namespace, exact top-k/retention,
  candidate gain, context gain, remaining gain, facet coverage, sufficiency,
  evidence IDs, and qrels; and
- latency block/repetition/method order and environment records on
  `query_latency`, with no supervision namespace.

Records must reconstruct every fit target, calibration cell, candidate
decision, bound, aggregate, terminal result, and comparison. Stable string IDs
are used at data boundaries; row numbers appear only through explicit ID maps.

## 12. Gate order

1. Step 2: CPU/network-free exact retrieval/evidence skeleton and fixed
   sequential interface.
2. Step 3: unchanged exact Tri-Law profile, calibrated controller, and
   one-factor synthetic ablations.
3. Step 4: fresh source audit, annotation sufficiency audit, power plan, role
   freeze, `query_cal` fit/calibration, then one-time `query_tune` selection.
4. Step 5: one-time `query_cert`, followed only after terminal audit by
   label-free paired `query_latency`; then one descriptive `query_test` report.
5. Step 6: frozen LLM feedback and answer evaluation only after both prior
   positive gates and separate user authorization.

No later gate can repair an earlier failure. No `query_test` outcome can become
selection or certification evidence.

## 13. Terminal failure and no-retuning contract

The following are valid terminal outcomes: insufficient fresh queries,
unidentifiable evidence sufficiency, annotation-integrity failure, no eligible
fixed reference, no adaptive candidate with strict tune cost superiority,
underpowered calibration cells, certification failure, backend incompatibility,
latency failure, context/answer noninferiority failure, or protocol tampering.

After `query_cert` access, any change to the following invalidates the
certificate and requires a new dataset or fresh independent role identities:

- corpus/query/evidence labels or role assignments;
- embedding model/revision/format/normalization or cached arrays;
- projection seed/matrix/family/dimension or distance/search/tie semantics;
- pilot, budget grid, maximum expansions, action/transition/fallback rule;
- evidence plan, facet matcher, context builder, `k_gt`, or `k_ctx`;
- any feature, Tri-Law aggregation, score/calibration model, parameter, bin,
  confidence procedure, alpha, threshold, or validity rule;
- policy candidate registry, selection rule, quality target, margin, metric,
  hypothesis, work definition, comparator, or latency protocol; or
- any later LLM model, prompt, schema, decoding, context format, or failure rule
  used by the claimed system.

Software changes that could affect numerical results, decisions, work, or
latency are mutations unless exact equivalence is proved under the frozen
artifact contract. A failed target is reported; budgets, thresholds, margins,
and splits are never silently enlarged or relaxed.

## 14. Step 1 authorization statement

This Step 1 protocol is documentation only. No real-data run is authorized.
No dataset/model/dependency may be downloaded, no query-cal/tune/cert/latency/
test runner may be executed, no returned protected archive may be inspected,
and no LLM or answer generation may be called. Step 2 begins only after the
user reviews this design and supplies a separate authorization.
