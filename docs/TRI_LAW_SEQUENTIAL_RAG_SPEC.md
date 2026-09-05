# Tri-Law Guided Sequential RAG Controller (TLS-RAG) v1 specification

Status: Step 1 design freeze. This document defines a successor algorithm
family. It is not a Tri-Predict, PDCTP, or Calibrated Tri-Predict version, and
it does not authorize Step 2 implementation or any real-data run.

## 1. Problem statement

Let a corpus contain stable string document IDs and normalized embeddings
`x_i`, and let an external query have normalized embedding `q`. One dense
Gaussian matrix `Pi` with entries `N(0, 1/m_prime)` is drawn once and frozen.
Projected vectors are not renormalized. Retrieval uses squared L2 both in the
normalized original space and in projected space.

TLS-RAG must choose how far to expose one frozen projected ranking. Starting at
`M_0 = M_pilot`, it may stop or reveal exactly the next value of a frozen
budget grid `G = (M_0, ..., M_T)`. At every visited budget, all exposed
candidates are reranked by exact original-space query distance. The first
context builder returns the stable exact original-space top-`k_ctx` candidates.
The objective is to stop only when further retrieval is unlikely to add useful
evidence and the current context is likely to satisfy a frozen evidence plan,
while using less measured work and latency than a quality-matched fixed budget.

The controller estimates evidence outcomes; it does not receive evidence
labels at inference. Exact Tri-Law values are allowed only as geometric risk
features on candidates whose original embeddings have already been evaluated.

### Non-goals

- No per-query projection dimension, sparse projection, approximate index,
  learned embedding model, arbitrary budget jump, or distributed serving.
- No direct scalar-LID reconstruction of unseen rank distances and no larger
  residual model for top-k retention.
- No learned or diversity-aware context builder in the first controller.
- No answer generation or LLM call in Steps 1 through 5.
- No claim that embedding retention, Tri-Law, or a calibrated evidence score
  guarantees answer correctness.
- No formal Tri-Law theorem for the sequential RAG system.

## 2. Frozen objects and notation

The protocol freezes the corpus and query source, embedding model and revision,
normalization, projection matrix and seed, `m_prime`, projected ranking tie
rule, `M_pilot`, budget grid, maximum expansions, `k_gt`, `k_ctx`, evidence-plan
generator, context builder, feature schemas, action rules, fallback rules,
calibration procedure, controller candidates, thresholds, metrics, and seeds.

For query `q` and step `t`:

- `P(q)` is a deterministic, structured evidence plan.
- `R` is the single stable projected full-corpus ranking. Only prefix
  `C_t = R[:M_t]` is exposed to the controller.
- `D_t` contains exact original-space query distances for every ID in `C_t`.
- `H_t = top_original(C_t, k_ctx)` is the fixed final context, with stable ID
  ties.
- `a_t` is either `STOP` or `EXPAND_TO_NEXT_GRID_VALUE`.

An evidence plan contains required atomic facets. A facet may specify entities,
relations, comparison sides, temporal constraints, a required number of
independent supporting passages, and a contradiction rule. In Steps 1 through
5, the plan is produced by deterministic code or a frozen network-free fixture.
An empty or invalid plan is not interpreted as automatically sufficient.

## 3. Decision process

### 3.1 State

The deployable state is:

```text
S_t = {
  deterministic query/evidence-plan features,
  M_t, t, remaining grid steps,
  exposed projected-prefix IDs and projected squared distances,
  exact original squared distances and exact-reranked order within C_t,
  exact-reranked distance/gap/frontier summaries,
  projected/original distortion summaries,
  Tri-Law risk profile over valid observed candidate pairs,
  observed-candidate redundancy/diversity summaries,
  deterministic facet-match and predicted facet-coverage features,
  previous-state deltas,
  explicit validity/failure indicators
}
```

Raw text may be used only by the frozen deterministic evidence-plan and facet
feature code. A future learned scorer must be fit on `query_cal`, serialized,
and accept the same deployable inputs. Split role is runner metadata and is not
part of `S_t` or the policy decision object.

### 3.2 Actions

The only nonterminal actions are:

```text
STOP
EXPAND_TO_NEXT_GRID_VALUE
```

`EXPAND_TO_NEXT_GRID_VALUE` changes `M_t` to `M_{t+1}`. It cannot skip a grid
value or change `m_prime`. At `M_T`, expansion is unavailable and the controller
must emit a terminal stop reason. Step 2 uses `M_T = N` so full-corpus exhaustion
is directly testable. A later real protocol may freeze `M_T < N`, but must label
that condition `maximum_budget_reached`, not `corpus_exhausted`.

### 3.3 Transition

One projected ranking is computed once. An expansion exposes only the next
prefix, computes original-space query distances for newly exposed candidates,
reranks the accumulated prefix exactly, rebuilds `H_t`, and recomputes state
features. No prior candidate is discarded from `C_t`; `H_t` may change because
new candidates can displace old context items. Pilot, every expansion, each
rerank, controller evaluation, and context construction have separate work and
latency counters.

### 3.4 Evidence targets

For supervised evaluation only, each corpus passage can be annotated with the
plan facets it supports, an independent-source group, and contradiction or
invalid-evidence flags. Let `slots(P)` be the required facet/support slots and
let `covered(A, P)` be the slots satisfied by passage set `A` under the frozen
independence and contradiction rules. Define:

```text
coverage(A, P) = |covered(A, P)| / |slots(P)|
sufficient(A, P) = 1 iff every required slot is covered and no frozen
                    blocking contradiction rule is triggered
```

The following labels are distinct:

- Marginal candidate evidence gain at `t` is one iff the new projected-prefix
  candidates at `M_{t+1}` fill at least one slot not covered by `C_t`.
- Marginal final-context evidence gain at `t` is one iff
  `coverage(H_{t+1}, P) > coverage(H_t, P)`. Candidate gain does not imply
  context gain.
- Current evidence sufficiency is `sufficient(H_t, P)`.
- Remaining useful evidence is one iff some later frozen grid state has higher
  final-context coverage than `H_t`, or converts an insufficient `H_t` into a
  sufficient context. This horizon label prevents a zero-gain immediate shell
  from being mistaken for proof that no later shell is useful.

The primary gain target for stopping is remaining useful evidence. Immediate
candidate and final-context gains remain separate training diagnostics and may
support an expansion-utility score. If a real dataset provides only document
qrels, candidate relevance gain can be identified, but facet coverage,
independent support, contradictions, final-context sufficiency, and the horizon
target generally cannot. Such a dataset requires additional facet-level
annotations before TLS-RAG can make its primary evidence claim.

### 3.5 Calibrated stopping rule

The provisional formula is revised to avoid presenting point predictions as
posterior probabilities. The frozen calibration procedure produces:

- `U_gain(S_t)`: a simultaneous one-sided upper confidence limit for the
  remaining-useful-evidence event rate among independent calibration queries
  at the same stage and frozen calibrated-score bin; and
- `L_suff(S_t)`: a simultaneous one-sided lower confidence limit for the
  current-sufficiency event rate in the corresponding stage/bin.

Score models are fit on a stable-hash `query_cal_fit` subset. Score mappings and
one-sided binomial limits are fit on a disjoint stable-hash `query_cal_bounds`
subset. Score-bin edges are frozen from label-free score quantiles. Tables are
built stagewise for every preregistered controller candidate: a stage-`t` cell
contains only calibration queries that would reach `t` under that candidate's
already frozen earlier actions. Each candidate/stage/bin/outcome cell uses an
exact one-sided Clopper-Pearson limit with the predeclared family-wise alpha
allocation. An empty or underpowered cell returns the vacuous interval
`[0, 1]`. Query IDs, rather than correlated states from the same query, are the
independent units. These are reachable-bin event-rate bounds under the frozen
exchangeability assumption; they are not exact individual posteriors.

`delta_gain`, `tau_sufficient`, the candidate score model, and all controller
hyperparameters form a preregistered candidate grid. Models and candidate-
specific bound tables use only `query_cal`; one complete controller is selected
once on `query_tune`. The action rule is:

```text
STOP only if
  U_gain(S_t) <= delta_gain
  and L_suff(S_t) >= tau_sufficient
  and all mandatory state and calibration validity flags are true;
otherwise EXPAND_TO_NEXT_GRID_VALUE.
```

Thus an uncalibrated point score never justifies STOP. If the fresh-data audit
cannot supply the labels above or enough independent calibration queries per
stage/bin, the primary rule is not identifiable. The smallest testable
alternative is a candidate-relevance-gain controller with no evidence-
sufficiency claim; it must use a different method/version and cannot be called
the primary TLS-RAG controller.

### 3.6 Terminal and fallback behavior

- Invalid/nonfinite features, invalid calibration cells, feature schema
  mismatch, zero query displacement, no valid Tri-Law pairs when that ablation
  requires them, and prediction failure force expansion.
- Equal original distances do not define `beta > 1`. Stable string-ID ties are
  retained for retrieval, the tied pairs are excluded from the Tri-Law profile,
  and exclusion counts are logged. If required profile support is lost, expand.
- Duplicate projected distances use stable string-ID ties and are not treated
  as independent evidence.
- An empty/invalid evidence plan forces expansion through the frozen terminal
  budget and emits `invalid_evidence_plan`; it never yields automatic
  sufficiency.
- If both bounds are valid but either stop inequality fails, expand.
- At `M_T`, emit `corpus_exhausted`, `maximum_budget_reached`,
  `maximum_expansions_reached`, or `evidence_nonattainment` as applicable,
  construct the frozen context, and stop without enlarging the grid.
- A terminal context may remain insufficient. That is a recorded failure, not
  a reason to retune, repeat certification, or silently retrieve more.

## 4. Inference observables and forbidden fields

| Field or quantity | Inference status | Source and rule |
|---|---|---|
| Stable query/document IDs | Allowed | Data boundary and deterministic ties; IDs carry no split label. |
| Normalized query and observed candidate embeddings | Allowed | Frozen embedding cache; only candidates in `C_t`. |
| Fixed `Pi`, `m_prime`, and projection metadata | Allowed | Frozen globally; projection is never changed per query. |
| Projected prefix IDs/distances through `M_t` | Allowed | One frozen ranking prefix only. |
| Original query distances for candidates in `C_t` | Allowed | Computed by pilot/expansion reranking. |
| Exact reranked order and distance gaps within `C_t` | Allowed | Does not reveal full-corpus exact neighbors. |
| Deterministic evidence plan and plan features | Allowed | Frozen code/fixture, no LLM in Steps 1--5. |
| Deterministic lexical/facet-match features on `H_t` | Allowed | Predictions/features only, never annotation labels. |
| Observed candidate pairwise distances/redundancy | Allowed | Computed only within `C_t`; cost is logged. |
| Observed-pair `beta`, `rho`, and Tri-Law profile | Allowed | Both original displacement vectors have been evaluated. |
| Projection distortion and state-to-state deltas | Allowed | Derived only from exposed candidates and prior state. |
| Frozen calibrated score/bin/interval lookup | Allowed | Learned only on `query_cal`; immutable at inference. |
| Current budget, expansion count, validity reasons | Allowed | Controller bookkeeping. |
| Query split/role | Forbidden | Runner metadata must not enter the decision object. |
| Qrels, facet/support labels, evidence IDs | Forbidden | Supervision store is opened only after decisions are fixed. |
| Candidate/context gain and sufficiency labels | Forbidden | Training/evaluation targets only. |
| Answer labels, generated answers, answer correctness | Forbidden | Outside Steps 1--5 and never a retrieval input. |
| Oracle LID or effective Tri-LID targets | Forbidden | Historical/fit diagnostics only. |
| Full-corpus original distances or exact top-k IDs | Forbidden | Supervision/retention diagnostic only. |
| Realized retention or future expansion outcomes | Forbidden | Post-decision labels only. |
| Protected cert/latency/test outcomes | Forbidden | Never fit or select a controller. |
| Unseen-candidate original distances, angles, relevance | Forbidden | They cannot be imputed with the failed scalar LID law. |

After the pilot, the controller knows only the first state's allowed rows. Each
expansion adds the new prefix candidates and their allowed measurements. Only
after the complete action trajectory is serialized may the evaluation layer
join evidence, retention, or answer supervision.

## 5. Exact Tri-Law boundary and risk profile

For observed displacements `u = x_i - q` and `v = x_j - q` with
`||u|| < ||v||`:

```text
beta = ||v||^2 / ||u||^2
rho  = <u, v> / (||u|| ||v||)
```

`tri_law_probability(beta, rho, m_prime)` is exactly the probability, over a
new random dense Gaussian projection, that this single pair's projected order
is inverted. The function and its precision/tolerances remain unchanged and
separate from TLS-RAG.

Once the one deployment projection has been drawn and its outcome observed,
the marginal value is not automatically a posterior probability that an unseen
document is missing. Pair risks are dependent because candidates share the
query and projection. Tri-Law does not assign relevance, evidence utility,
context utility, sufficiency, answer correctness, or stability. Unknown
distances and angles for unseen documents are never replaced by a scalar LID
power law. Any aggregate below is a feature or calibrated risk score, not an
exact theorem or bound.

The frozen first risk profile uses only valid observed pairs:

1. `context_boundary_risk`: exact pair probabilities between the farthest item
   in `H_t` and each farther observed non-context candidate;
2. `core_frontier_risk`: probabilities between every item in `H_t` and a
   frozen-size shell of the most recently exposed projected-prefix candidates,
   retaining only strictly ordered original-distance pairs;
3. for both sets, count, invalid/tied/collinear counts, mean, maximum, and
   quantiles `0.50`, `0.90`, and `0.99`;
4. observed projected/original log-distortion quantiles for core and shell; and
5. changes in all valid summaries since the previous visited budget.

At the pilot, the shell is the last `min(shell_size, M_pilot)` projected ranks;
later it is the newly exposed prefix slice. `shell_size`, quantiles, empty-set
behavior, and pair orientation are frozen. These components describe observed
geometric fragility without reconstructing an unseen rank-distance curve.

## 6. Exact, deterministic, approximate, and calibrated components

| Component | Scientific status |
|---|---|
| Normalization, fixed Gaussian projection, squared-L2 search, prefix reuse, exact original rerank | Exact/deterministic under the frozen numerical contract. |
| `tri_law_probability` on one valid observed pair | Exact ex-ante single-triplet marginal law. |
| Deterministic evidence plan and lexical/facet features | Deterministic heuristic; not an evidence label. |
| Risk-profile aggregation and redundancy/diversity summaries | Engineered features; no independence or posterior claim. |
| Evidence-gain/sufficiency score models | Approximate supervised models fit on `query_cal`. |
| Score-bin Clopper-Pearson limits | Exact binomial limits for frozen bin event rates under independent-query/exchangeability assumptions; not per-query posterior bounds. |
| Sequential STOP decision | Calibrated policy selected on `query_tune`; it is not a Tri-Law theorem. |
| Embedding retention, evidence quality, work, latency, answer quality | Separate empirical claims with separate hypotheses and gates. |

## 7. Fixed context-builder contract

The first context builder is exactly the first `k_ctx` candidates after stable
original-space reranking of `C_t`. It uses squared L2 and stable string-ID ties,
does not use evidence labels, plan features, Tri-Law, diversity, or controller
scores, and has its work/latency logged separately. If `|C_t| < k_ctx`, it
returns all candidates and records the shortfall. Its identity is frozen before
selection. A learned, evidence-aware, or diversity builder requires a later
one-factor ablation and a fresh certificate if changed.

## 8. Frozen one-factor ablation ladder

| Row | Method | Only change from prior row |
|---|---|---|
| 1 | Fixed-budget exact projected retrieval + exact rerank | Reference budgets from the same grid; no sequential controller. |
| 2 | Sequential controller without Tri-Law features | Adds sequential STOP/EXPAND using distance, distortion, redundancy, and validity features. |
| 3 | Same controller plus Tri-Law risk profile | Adds only the observed-pair risk-profile fields in Section 5. |
| 4 | Same controller plus deterministic evidence-plan/facet features | Adds only frozen plan and deterministic facet-match/coverage-prediction features. |
| 5 | Context-builder ablation, only if separately authorized | Replaces only exact top-`k_ctx`; all retrieval/controller objects stay frozen. |
| 6 | Frozen LLM feedback, only after the retrieval/evidence and latency gates pass | Adds only the restricted structured next-action request. |

Rows 2--4 use the same training algorithm, calibration partition, candidate
grid, uncertainty procedure, budget grid, fallback, and context builder. Raw
Tri-Predict v1 and terminal PDCTP v2/v3 remain historical references with no
shared successor fit or selection parameters. Tri-Law feature benefit is
falsified if Row 3 does not pass its predeclared paired tune and independent
certification improvement criterion over Row 2, or if it raises work/latency
without quality benefit. It remains in the report as a negative ablation and
cannot be rescued by post-cert retuning.

## 9. Tiny counterexamples the design must handle

1. **Same distance curve, different angles.** Two toy corpora have identical
   query-distance ranks (and therefore the same scalar LID) but different
   candidate angles. Their exact observed-pair Tri-Law profiles differ. TLS-RAG
   must preserve that difference and must not reconstruct either profile from
   scalar LID.
2. **Candidate gain without context gain.** An expansion retrieves a relevant
   passage, but exact top-`k_ctx` reranking excludes it. Candidate gain is one,
   final-context gain is zero, and sufficiency is unchanged. The controller and
   records must not conflate these labels.
3. **Empty next shell, useful later shell.** The next grid prefix adds no new
   facet, while the following prefix adds the missing required facet. A model
   using only immediate gain could stop incorrectly; remaining-useful-evidence
   is one until the later evidence is reached.
4. **Marginal law after a realized projection.** Two observed pairs can have
   the same exact ex-ante Tri-Law probability even though the frozen projection
   has already inverted one and preserved the other. The probability remains a
   profile feature and is never relabeled as a posterior missing-document risk.
5. **Degenerate inputs.** Equal original distances, a zero displacement, or an
   empty evidence plan produce explicit invalid counts and conservative
   expansion, followed by a recorded terminal failure if no valid stop occurs.

## 10. One-query pseudocode

```text
function TLS_RAG_QUERY(q, frozen):
    assert q is external and normalized
    plan = deterministic_evidence_plan(q.text)
    q_proj = Pi @ q.embedding
    projected_ranking = exact_projected_rank_once(q_proj, projected_corpus,
                                                   stable_id_ties=True)
    log_cost("query_projection")
    log_cost("pilot_projected_search")

    previous_state = NONE
    previous_M = 0
    for t, M_t in enumerate(frozen.budget_grid):
        new_ids = projected_ranking[previous_M:M_t]
        evaluate_exact_original_query_distances(q, new_ids)
        C_t = projected_ranking[:M_t]
        reranked = stable_original_rerank(C_t)
        H_t = reranked[:k_ctx]
        log_cost("expansion_prefix_reuse" if t > 0 else "pilot_prefix")
        log_cost("original_rerank")
        log_cost("final_context")

        state = build_deployable_state(q, plan, C_t, reranked, H_t,
                                       previous_state)
        assert state contains no role, qrels, evidence labels, exact full-corpus
               top-k, realized retention, oracle LID, answers, or future outcome

        U_gain = calibrated_remaining_gain_upper(state)
        L_suff = calibrated_sufficiency_lower(state)
        valid = mandatory_validity(state, U_gain, L_suff)
        can_expand = (t + 1 < len(frozen.budget_grid)
                      and t < frozen.maximum_expansions)

        if valid and U_gain <= delta_gain and L_suff >= tau_sufficient:
            action = STOP
            reason = "calibrated_gain_low_and_sufficiency_high"
        else if can_expand:
            action = EXPAND_TO_NEXT_GRID_VALUE
            reason = conservative_reason(state, U_gain, L_suff)
        else:
            action = STOP
            reason = frozen_terminal_reason(M_t, plan, state)

        serialize_pre_supervision_state_and_decision(state, action, reason)
        log_cost("controller")
        if action == STOP:
            return H_t, complete_cost_record(), reason
        previous_state = state
        previous_M = M_t
```

Supervision is joined only after the returned trajectory is immutable.

## 11. Failure modes and refusal behavior

TLS-RAG refuses to fit, select, certify, or reuse artifacts when fingerprints,
roles, query-text disjointness, label schemas, projection metadata, normalized
embedding hashes, action grids, or upstream files do not match. It refuses
calibration with insufficient independent queries, real data without
identifiable evidence targets, a policy input containing forbidden fields, and
any post-cert mutation. Numerical or feature failures cause conservative
expansion, not a cheaper guessed budget. Exhaustion and target nonattainment are
valid terminal outputs. Failure never triggers silent grid growth or threshold
relaxation.

## 12. Step 2 CPU/network-free acceptance tests

Step 2 remains unapproved until the user accepts this design. Its smallest
walking skeleton must test:

- normalized external query/corpus vectors, Gaussian scale
  `1/sqrt(m_prime)`, no projected renormalization, and squared-L2 search;
- one deterministic projected ranking reused by pilot and every prefix;
- exact original reranking and fixed top-`k_ctx` context with stable string-ID
  ties;
- `STOP`/next-grid-only actions, maximum-expansion behavior, and full-corpus
  exhaustion;
- immutable pre-supervision state/decision records whose schema rejects every
  forbidden field in Section 4;
- deterministic fixture evidence plans and facet/support labels stored outside
  the inference object;
- exact reconstruction of candidate gain, context gain, remaining gain,
  sufficiency, and coverage from labels after the trajectory is fixed;
- hand-computed observable `beta`/`rho` pairs and risk-profile summaries using
  the unchanged exact Tri-Law API;
- tied, duplicate, collinear, zero-distance, empty-plan, invalid-feature, and
  nonattainment fallbacks;
- all five counterexamples in Section 9;
- separate query projection, pilot scan, expansion prefix reuse, reranking,
  evidence-plan/controller, and context counters/timings;
- query-level byte/value reproducibility under identical seeds; and
- a hard network/LLM prohibition and no new dataset, model, dependency, or
  approximate index.

## 13. Resolution ledger for Step 1 questions

| Question | Resolution |
|---|---|
| Pilot/expansion/supervision knowledge | Resolved by Sections 3 and 4; labels join only after the trajectory is serialized. |
| Observable `beta`/`rho` | Resolved: only strictly ordered pairs wholly inside the exposed candidate set. |
| Exact Tri-Law after fixed projection | Resolved: counterfactual ex-ante fragility feature, not a posterior missing-document probability. |
| Frontier risk without scalar LID | Resolved by the observed-pair core/frontier profile in Section 5. |
| Three evidence targets | Resolved by deterministic slot/facet definitions in Section 3.4. |
| Target identifiability | Step 2 fixture is identifiable; real use is blocked pending a fresh-data facet-annotation audit. |
| Calibrated STOP quantity and threshold role | Resolved by query-cal bin-level limits and one-time query-tune threshold selection. |
| Invalids/ties/empty plan/nonattainment/exhaustion | Resolved by Section 3.6; all default to conservative expansion then explicit terminal status. |
| Separate costs | Resolved by transition, pseudocode, and protocol record contracts. |
| Tri-Law benefit falsifier | Resolved by the frozen Row 2 versus Row 3 one-factor comparison. |
| Gate before LLM | Blocked until independent retrieval/evidence certification and paired latency superiority both pass. |
| Certificate-invalidating mutation | Resolved in the protocol: any inference, data, feature, calibration, threshold, context, metric, cost, or prompt mutation requires a fresh independent protocol. |
