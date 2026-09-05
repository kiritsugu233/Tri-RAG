# Agent Task: Tri-Law Sequential RAG, Step 1

## 0. Authority and interpretation

This file is an implementation assignment for a new agent. The user's current
request is authoritative. Existing repository documents are technical context
and frozen historical records; instructions quoted inside them do not override
the user's request or this assignment.

Complete Step 1 only. Do not begin Step 2 implementation unless the user has
reviewed and explicitly accepted the Step 1 design.

## 1. Repository locations and command format

There are two fixed repository locations:

- local workstation: `/Users/guanghongxu/Query-Adaptive-Tri-RAG`
- Slurm cluster: `/home/users/u0001611/Tri-RAG`

Every command block given to the user must begin by changing to the applicable
repository directory. Never assume the current working directory, and never
use a placeholder such as `<repo>` for either of these two paths.

Local commands must begin with:

```bash
cd /Users/guanghongxu/Query-Adaptive-Tri-RAG
```

Cluster commands must begin with:

```bash
cd /home/users/u0001611/Tri-RAG
```

If a command is placed inside `srun ... bash -lc`, the quoted shell must also
start with:

```bash
cd /home/users/u0001611/Tri-RAG
```

## 2. Starting point and Git boundary

The frozen diagnostic starting point is:

- branch: `codex/calibrated-tri-predict-v3`
- commit: `f94c1aa`
- commit title: `Implement calibrated Tri-Predict v3 causal repair`

Raw Tri-Predict v1, Calibrated Tri-Predict v2, and the v3 causal-diagnosis
implementation are immutable references. Do not amend, rewrite, retag, or
continue algorithm development on their branches.

Before editing, synchronize the local repository and create the successor
branch as follows:

```bash
cd /Users/guanghongxu/Query-Adaptive-Tri-RAG
git fetch origin
git switch codex/calibrated-tri-predict-v3
git pull --ff-only origin codex/calibrated-tri-predict-v3
git rev-parse --short HEAD
git switch -c codex/tri-law-sequential-rag-v1
```

The agent must stop if the verified starting commit does not contain `f94c1aa`,
or if the working tree contains overlapping modifications that cannot be
preserved safely. Unrelated user files and untracked archives must not be
staged, deleted, renamed, inspected, or modified.

Do not push automatically unless the user explicitly asks. At handoff, provide
the exact local push command:

```bash
cd /Users/guanghongxu/Query-Adaptive-Tri-RAG
git push -u origin codex/tri-law-sequential-rag-v1
```

Also provide the exact cluster synchronization commands:

```bash
cd /home/users/u0001611/Tri-RAG
git fetch origin
git switch codex/tri-law-sequential-rag-v1
git pull --ff-only origin codex/tri-law-sequential-rag-v1
git rev-parse --short HEAD
```

If the branch does not yet exist locally on the cluster, give this alternative
instead of using a force/reset command:

```bash
cd /home/users/u0001611/Tri-RAG
git fetch origin
git switch --track origin/codex/tri-law-sequential-rag-v1
git rev-parse --short HEAD
```

## 3. Successor-program roadmap

The successor is a new algorithm family, not Calibrated Tri-Predict v4. Treat
Tri-Predict as a reference baseline and its failures as design evidence.

The work is divided into six steps:

1. **Step 1 — problem definition and design freeze.** Define the sequential
   RAG decision problem, the mathematically valid role of Tri-Law, deployable
   state/action/label boundaries, ablations, protocol roles, and acceptance
   gates. This is the only step authorized by this file.
2. **Step 2 — network-free synthetic retrieval/evidence skeleton.** Implement
   a tiny exact projected-search, exact-rerank, evidence-labeled environment
   and a fixed sequential stop/expand interface without running an LLM.
3. **Step 3 — Tri-Law risk profile and sequential controller.** Implement the
   exact pairwise risk feature layer, evidence-gain/sufficiency calibration,
   uncertainty-aware stopping, and one-factor synthetic ablations.
4. **Step 4 — fresh real-data source and five-role gates.** Audit a new dataset
   with evidence/answer annotations, freeze disjoint cal/tune/cert/latency/test
   identities, then fit on `query_cal` and select once on `query_tune`.
5. **Step 5 — retrieval/evidence certification and latency.** Independently
   certify evidence quality and retrieval behavior, measure paired latency,
   and perform one descriptive retrieval/evidence test evaluation.
6. **Step 6 — frozen LLM reasoning feedback.** Only after Step 5 passes, add a
   frozen structured LLM evidence-gap interface, certify its trigger behavior,
   and evaluate answer quality without changing the retrieval controller.

Every future step requires a separate agent instruction file and explicit user
authorization. A failure at any gate is a valid terminal result.

## 4. Mission of the new algorithm

Design a query-adaptive retrieval controller around the actual RAG process.
The controller must predict whether further retrieval can add useful evidence
to the final context, rather than predicting only embedding top-k retention.

The provisional method name is:

**Tri-Law Guided Sequential RAG Controller (TLS-RAG)**

The name is provisional during Step 1 and may be changed in the design freeze.
It must not use `Tri-Predict`, `PDCTP`, or `Calibrated Tri-Predict` in its method
identity.

The core decision is sequential:

```text
query
  -> projected pilot retrieval
  -> exact original-space reranking
  -> observed retrieval/evidence state
  -> STOP or EXPAND to the next frozen budget
  -> repeat within a frozen maximum number of expansions
  -> construct final context
  -> optional LLM stage only after the retrieval/evidence gate passes
```

The first version should use the conservative action space
`{STOP, EXPAND_TO_NEXT_GRID_VALUE}`. Do not introduce arbitrary jumps, dynamic
projection dimensions, approximate indexes, or a learned embedding model in
the first implementation.

## 5. Why this is a new method rather than another Tri-Predict repair

Step 1 must explicitly carry forward the following causal findings:

- the exact dense-Gaussian Tri-Law implementation passed conformance;
- finite-rank quadrature was not materially inaccurate;
- the scalar LID rank-distance power law was the earliest failing layer;
- real shared-projection geometry violated the orthogonal/independent
  mean-field approximation in a budget-dependent way;
- pilot candidate construction strongly biased pilot LID;
- oracle geometric LID was the wrong curve-calibration target;
- one scalar effective dimension could not fit low and high budget regimes;
- the v2 residual model compensated for prediction error rather than repairing
  its cause;
- terminal fallback dominated much of the observed cost failure;
- the frozen v2 selection rule could accept an inefficient adaptive method by
  design;
- embedding retention and answer correctness are different scientific claims.

Therefore Step 1 must not propose another scalar-LID transform, a larger
residual model, or a more complicated direct prediction of top-k retention as
the primary algorithm.

## 6. Mathematically valid role of Tri-Law

Keep `tri_law_probability(beta, rho, m_prime)` separate from every aggregation,
calibration, evidence, and answer model.

For two observed query-to-document displacement vectors `u` and `v`, with
`||u|| < ||v||`, define:

```text
beta = ||v||^2 / ||u||^2
rho  = <u, v> / (||u|| ||v||)
```

The exact dense-Gaussian Tri-Law gives the probability, over a new random dense
Gaussian projection of dimension `m_prime`, that their projected ordering is
inverted. It is an exact single-triplet, ex-ante probability.

Step 1 must state these limitations explicitly:

1. After the one fixed deployment projection has been drawn and its projected
   distances observed, the marginal Tri-Law probability is not automatically
   a posterior probability that an unseen document is missing.
2. Pairwise Tri-Law probabilities are not independent across documents because
   every document shares the same query and projection matrix.
3. Tri-Law alone does not assign relevance, evidence utility, context utility,
   answer correctness, or answer stability.
4. Unknown original-space distances or angles for unseen candidates cannot be
   replaced by the failed scalar LID rank-distance power law.
5. Any aggregation of Tri-Law quantities must be named as a feature,
   approximation, calibrated risk score, or bound—not an exact Tri-Law theorem.

The proposed valid use is a **Tri-Law risk profile** computed from pairs whose
original embeddings have already been evaluated during pilot/expansion
reranking. Candidate profile components may include pairwise inversion-risk
quantiles, top-k boundary risk, frontier-shell risk, observed projected/original
distortion, and how these quantities change after each expansion. Step 1 must
decide which components are identifiable at inference without labels.

## 7. Provisional sequential decision model

Step 1 must formalize, revise, or reject the following provisional state:

```text
S_t = {
  deployable query/evidence-plan features,
  current budget M_t and expansion count,
  exact-reranked candidate distance/gap profile,
  projected/original distortion profile,
  Tri-Law pairwise risk profile on observed candidates,
  candidate redundancy/diversity features,
  evidence-facet coverage predictions,
  explicit validity/failure indicators
}
```

The state at inference must not contain qrels, evidence labels, answer labels,
oracle LID, exact full-corpus top-k identities, realized recall, or protected
split outcomes.

The retrieval-only controller should estimate two distinct events for the next
frozen expansion:

- **marginal evidence gain:** whether the expansion adds new relevant evidence
  to the candidate set or fixed final-context builder;
- **current evidence sufficiency:** whether the current final context contains
  the evidence required by the query's frozen evidence plan.

A provisional stopping rule is:

```text
STOP only when
  upper_bound(P(useful evidence remains beyond M_t | S_t)) <= delta_gain
and
  lower_bound(P(current evidence is sufficient | S_t)) >= tau_sufficient;
otherwise EXPAND_TO_NEXT_GRID_VALUE.
```

Step 1 must specify how these uncertainty bounds would be calibrated using only
allowed roles. Do not silently treat a point prediction as a bound. If this
stopping rule is not identifiable under available supervision, say so and
propose the smallest testable alternative.

The final context builder must initially remain fixed—preferably exact
original-space top-`k_ctx`—so retrieval-budget effects are not confounded with
an evidence-aware reranker. A learned/diversity context builder would require a
later isolated ablation.

## 8. RAG and LLM reasoning boundary

Step 1 must define an **evidence plan** as a structured representation of what
the query requires, for example entities, relations, comparison facets,
temporal constraints, and the number/type of independent supporting passages.

For Steps 1 through 5, this plan must be produced by deterministic code or a
frozen, network-free fixture. Do not call an LLM.

Step 6 may add a frozen LLM interface with outputs such as:

```text
supported_claims
missing_facets
cited_context_ids
contradiction_detected
request_one_more_expansion
```

The LLM must never emit an unrestricted budget. Its structured signal may only
request the controller's next frozen action. Self-reported confidence must not
be treated as calibrated probability. Model, revision, prompt, context format,
decoding, output schema, seeds, and failure handling must all be frozen before
evaluation.

Answer generation remains prohibited until the retrieval/evidence harness
passes its acceptance gate.

## 9. Fresh protocol and data separation

TLS-RAG requires a new dataset and fresh identities. Do not reuse v2 FiQA
query-tune, query-cert, query-latency, or query-test outcomes for fitting,
selection, thresholds, ablations, or claims.

The future protocol must contain at least:

- `query_cal`: fit evidence-gain/sufficiency models and calibrate uncertainty;
- `query_tune`: choose controller thresholds and one candidate per method;
- `query_cert`: independent retention/evidence certification;
- `query_latency`: label-free paired systems measurement;
- `query_test`: one-time descriptive final evaluation after prior gates close.

All IDs and normalized query texts must be disjoint across roles. Policy
selection and certification must not use the same queries. Any post-cert change
to a policy, feature, threshold, context builder, or inference prompt invalidates
the certificate and requires a new independent protocol.

Step 1 must define which labels are permitted for fitting on `query_cal` and
selection on `query_tune`, while proving that none of those labels appear in
the inference decision object.

## 10. Cost and evaluation contract

The primary systems objective is measured end-to-end work and latency, not
candidate count alone. Preserve separate accounting for:

- query projection;
- pilot projected search;
- every expansion search or prefix reuse;
- original-space reranking at each stage;
- evidence-plan and controller computation;
- final context construction;
- optional LLM planning/feedback and final generation in Step 6 only.

The first exact backend should preserve one projected ranking and reuse its
prefixes. Report pilot, expansion, reranking, controller, and final-context
costs separately.

Future positive claims must require all of the following:

1. independent evidence-quality certification;
2. embedding-retention reporting as a separate diagnostic;
3. paired latency superiority, not only candidate/work reduction;
4. noninferior answer quality once the LLM stage is authorized;
5. superiority over a properly matched fixed baseline.

## 11. Required one-factor ablations

Step 1 must freeze an ablation ladder in which each row adds only one component:

1. fixed-budget exact retrieval;
2. sequential controller without Tri-Law features;
3. the same controller plus Tri-Law risk-profile features;
4. the same controller plus deterministic evidence-plan/facet features;
5. a separate context-builder ablation only if later authorized;
6. frozen LLM feedback only after the retrieval/evidence gate passes.

Also retain Raw Tri-Predict v1 and the terminal v2/v3 results as historical
reference baselines. They must not share fitted successor parameters.

The selection gate must require actual tune-side cost superiority over the
matched fixed reference in addition to quality eligibility. It must not repeat
the v2 rule that selected a minimum-cost member independently within every
family even when the entire family was more expensive than fixed.

## 12. Read before editing

Read the following files completely and in this exact order:

1. `AGENTS.md`
2. `AGENT_CALIBRATED_TRI_PREDICT.md`
3. `AGENT_CALIBRATED_TRI_PREDICT_STEP3.md`
4. `docs/TRI_LAW_SPEC.md`
5. `docs/CALIBRATED_TRI_PREDICT_V3_DIAGNOSIS.md`
6. `docs/CALIBRATED_TRI_PREDICT_PROTOCOL.md`
7. `docs/IMPLEMENTATION_PLAN.md`
8. `STATUS.md`
9. `src/tri_rag_harness/tri_law.py`
10. `src/tri_rag_harness/tri_predict.py`
11. `src/tri_rag_harness/pdctp_features.py`
12. `src/tri_rag_harness/pdctp_v3.py`
13. `tests/test_tri_law.py`
14. `tests/test_tri_predict.py`
15. `tests/test_pdctp_v3.py`

Do not begin edits until all files above have been read. Do not delegate this
reading to another agent.

## 13. Step 1 authorized work

Step 1 is a design-and-freeze milestone. It authorizes:

- read-only inspection of repository code, tests, and documented negative
  results;
- CPU/network-free mathematical checks on tiny synthetic vectors;
- writing the successor algorithm specification and protocol design;
- writing explicit state/action/label schemas as documentation or pseudocode;
- updating `docs/IMPLEMENTATION_PLAN.md` and `STATUS.md`;
- adding a Step 2 agent-instruction draft only if clearly marked unapproved.

Step 1 does not authorize:

- implementation of the controller or a large new framework;
- downloading any dataset, model, or dependency;
- running an LLM or answer generation;
- reading any returned query-cert archive;
- accessing query-cert, query-latency, or query-test identities/outcomes;
- fitting or selecting a successor policy on v2 query-tune outcomes;
- modifying Raw Tri-Predict v1, frozen v2, or v3 behavior/schemas/artifacts;
- changing Tri-Law formulas, precision, tolerances, or conformance tests;
- claiming that Tri-Law guarantees evidence or answer correctness.

If existing local untracked files reveal a protected archive name, do not open,
hash, list, extract, inspect, stage, or otherwise access that archive.

## 14. Step 1 required deliverables

Create these new files with new method names and no v1/v2/v3 artifact reuse:

1. `docs/TRI_LAW_SEQUENTIAL_RAG_SPEC.md`
2. `docs/TRI_LAW_SEQUENTIAL_RAG_PROTOCOL.md`

Update:

3. `docs/IMPLEMENTATION_PLAN.md`
4. `STATUS.md`

`TRI_LAW_SEQUENTIAL_RAG_SPEC.md` must contain:

- exact problem statement and non-goals;
- state, action, transition, stopping, and fallback definitions;
- a table of every inference-time observable and forbidden field;
- exact versus approximate/calibrated components;
- the valid role and limitations of Tri-Law;
- evidence-gain and evidence-sufficiency target definitions;
- candidate context-builder contract;
- uncertainty/calibration proposal;
- complete one-factor ablation ladder;
- at least three tiny synthetic counterexamples that the design must handle;
- failure modes and refusal behavior;
- pseudocode for one full sequential query;
- Step 2 implementation acceptance tests.

`TRI_LAW_SEQUENTIAL_RAG_PROTOCOL.md` must contain:

- fresh-data requirements and leakage boundaries;
- cal/tune/cert/latency/test roles and allowed supervision;
- exact policy-selection and matched-fixed comparison rules;
- independent evidence and retention certification hypotheses;
- paired work and latency hypotheses;
- later LLM-stage gate and answer-quality hypothesis;
- deterministic seeds and fingerprint requirements;
- query-level record requirements;
- terminal failure and no-retuning rules;
- explicit statement that no real-data run is authorized by Step 1.

## 15. Questions Step 1 must resolve

The written design must give concrete answers to all of these questions:

1. What is known after pilot retrieval, after each expansion, and only after
   supervision is revealed?
2. Which `beta` and `rho` values are genuinely observable at inference?
3. What can exact Tri-Law validly say once the fixed projection outcome has
   already been observed?
4. How will the method represent frontier risk without reconstructing the
   failed scalar LID rank-distance model?
5. What precisely counts as marginal candidate evidence gain, final-context
   evidence gain, and evidence sufficiency?
6. Can these targets be identified from the proposed annotations? If not, what
   additional annotation is required?
7. What calibrated quantity justifies STOP, and on which independent role is
   its threshold chosen?
8. What happens on invalid features, ties, duplicate distances, empty evidence
   plans, nonattainment, and full-corpus exhaustion?
9. How are pilot, expansion, reranking, controller, context, and later LLM costs
   separately measured?
10. What evidence would falsify the proposed benefit of Tri-Law features?
11. What gate must pass before any LLM call is permitted?
12. What exact mutation would invalidate a future certificate?

Do not hide unresolved issues behind future implementation. Mark each item as
resolved, empirically testable in Step 2, or blocked pending a fresh-data audit.

## 16. Step 1 acceptance criteria

Step 1 is complete only when:

- the starting branch/commit and frozen baselines are verified;
- the required files have been read in order;
- both successor design documents exist and are internally consistent;
- exact Tri-Law claims are separated from calibrated RAG claims;
- the sequential state contains deployable inputs only;
- evidence labels and answer labels are absent from inference;
- the stopping rule has an explicit uncertainty-calibration path;
- the matched-fixed cost-superiority gate is explicit;
- the one-factor ablation ladder is frozen;
- Step 2 has a small CPU/network-free test plan rather than a large framework;
- existing CPU tests pass without changing frozen behavior;
- `STATUS.md` and `docs/IMPLEMENTATION_PLAN.md` are updated;
- all changes are committed on `codex/tri-law-sequential-rag-v1`;
- the handoff reports the exact commit and changed files;
- the handoff provides local push, cluster pull, and manual `salloc`/`srun`
  commands, with the correct `cd` line first in every command block;
- no protected role or returned certification archive was accessed.

## 17. Test and Slurm handoff commands

The local regression command given to the user must be:

```bash
cd /Users/guanghongxu/Query-Adaptive-Tri-RAG
./scripts/run_tests.sh
```

The manual Slurm allocation command must be presented as:

```bash
cd /home/users/u0001611/Tri-RAG
salloc \
  --job-name=tls-rag-step1 \
  --cpus-per-task=1 \
  --mem=8G \
  --time=00:15:00
```

After allocation succeeds, give this exact shape, retaining the repository
`cd` inside the allocated shell:

```bash
cd /home/users/u0001611/Tri-RAG
srun --ntasks=1 --cpus-per-task=1 bash -lc '
cd /home/users/u0001611/Tri-RAG
eval "$(micromamba shell hook --shell bash)"
micromamba activate tri-rag
./scripts/run_tests.sh
'
```

Do not place downloads, protected-role runners, real-data evaluation, or LLM
calls in the Step 1 Slurm command.

## 18. Required final response

Lead with the design outcome, not a chronology of edits. The final response to
the user must include:

- the proposed algorithm identity and one-paragraph decision rule;
- the most important distinction between exact Tri-Law risk and calibrated
  evidence risk;
- unresolved or blocked scientific questions;
- exact tests passed/failed/skipped;
- branch name and commit hash;
- clickable absolute paths to the two design documents;
- the local push commands;
- the cluster pull commands;
- the manual `salloc` and `srun` commands;
- an explicit statement that no protected role, download, or LLM was used.

Stop after Step 1 and wait for user review.
