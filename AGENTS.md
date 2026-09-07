# Agent Instructions

## Repository-wide code protection and approval (2026-09-07)

These rules apply to this entire repository, including source, tests, configs,
scripts and documentation. `AGENTS.md` is canonical; `agent.md` is only an alias.
The user explicitly requested this protection policy. Start with
`START_HERE.md`, then the relevant entries in `docs/CODE_PROTECTION.md` and the
current task brief. Historical briefs do not override these rules, the standing
checkout/push instruction, or the user's explicit current authorization.
Old read allowlists constrain their original implementation tasks; they do not
block a user-authorized repository-wide source/Markdown review.

Every implementation file has an explicit level in `docs/code_protection.json`;
the human-readable per-file register is `docs/CODE_PROTECTION.md`.

- **L0 — protected scientific core or frozen contract. Before any edit, obtain
  explicit user approval for the exact affected paths and intended change.**
  This includes formulas, numeric tolerances, conformance tests, retrieval
  geometry/ties, calibration/stopping semantics, split/certification guards,
  frozen protocol settings, cache/fingerprint behavior and bound source files.
- **L1 — controlled implementation.** Changes within the user's task may proceed
  with impact analysis and relevant tests, provided all L0 contracts and frozen
  results remain unchanged. A semantic change to an L0 contract through a caller,
  wrapper, new implementation, dependency, or config is itself an L0 change.
- **L2 — presentation or maintenance.** Routine authorized changes may proceed;
  do not change scientific claims, recorded results or authorization boundaries
  under the guise of wording or formatting.

Protection level is change risk, **not a correctness grade**. Known defects,
review evidence and limitations are in `docs/REPOSITORY_REVIEW.md`. In particular,
Tri-Law has reproducible floating-point boundary defects at the reviewed baseline;
neither passing existing tests nor L0 registration means it is defect-free.

For an L0 request, first do read-only analysis and prepare a concrete proposal:

1. Give exact file/function names and the reproducible defect or requirement.
2. Explain why modification is necessary and why an L1-only solution cannot
   preserve the intended contract. Describe the proposed change precisely.
3. List affected APIs, tests, fingerprints, historical artifacts, data roles,
   scientific claims and compatibility consequences; give the validation plan.
4. Ask the user to approve that scope and **wait before editing protected files**.
   A general next-step/fix/refactor instruction is not scoped L0 consent unless
   it explicitly authorizes that protected change. Do not ask again when such
   consent already exists in the current task; record its scope at handoff.
5. After approval, make only the approved changes, retain failure evidence,
   use a new scientific version/namespace when needed, run required checks and
   update the protection baseline with the approval reference. Changed policy
   after cert/probe inspection requires fresh independent evaluation as applicable.

Do not evade L0 by copying/replacing a protected implementation, monkeypatching,
changing imports, loosening/skipping tests, changing dependency versions, changing
hashes/expected results, deleting/renaming files, or downgrading the registry.
Approval/protection rules in this file, `agent.md`, `docs/CODE_PROTECTION.md`,
`docs/code_protection.json`, and `scripts/check_code_protection.py` are themselves
L0 governance. Weakening these controls requires the same prior approval.
Adding a new file's L1/L2 registry row within an authorized task is routine;
it cannot alter existing entries or exempt inherited L0 semantics.

Run `python3 scripts/check_code_protection.py` before editing and before handoff.
It checks inventory and L0 SHA-256 values without modifying files; it cannot
verify that a human approved a change. A mismatch is evidence to investigate,
not permission to reset user changes or automatically refresh hashes. Unlisted
code must be classified before editing; use L0 for scientific core/contracts,
L1 for other implementation, L2 only for presentation/maintenance.
The approval record must quote or reference actual user consent; never fabricate
consent in a local file. Repo tooling is a review guard, not OS access control.

## Canonical local checkout and GitHub synchronization

User instruction effective after the Step 4 readiness milestone:

- All subsequent local TLS-RAG steps must use
  `/Users/guanghongxu/Query-Adaptive-Tri-RAG` as the working directory.
  Inspect its branch, HEAD and tracked/untracked status before changing it.
- Do not create or continue local step implementation in a Codex-managed
  worktree unless the user explicitly authorizes an exception. If a task starts
  in another directory, execute repository work in the canonical checkout;
  never assume the main checkout contains another worktree's commits.
- Preserve user modifications and untracked files. Do not reset, clean, delete
  archives, or overwrite a divergent branch to synchronize the checkout.
- After an authorized step passes its required checks, commit from the canonical
  checkout, push its explicit step branch to GitHub, and verify the remote HEAD.
  This is the user's standing synchronization instruction; it supersedes older
  TLS-RAG handoffs that said not to push automatically. Never force-push or merge
  the GitHub default branch without separate authorization.
- Cluster work uses a separate Linux checkout synchronized through GitHub at
  the reviewed commit. Provide exact replay commands and record actual results;
  do not infer a cluster result from a local run.

These instructions apply to the entire canonical repository above.

## Mission

Build a rigorous experimental harness for fixed-`m_prime`, query-adaptive `M(q)` retrieval. Optimize for a trustworthy, runnable MVP rather than breadth.

## Scope boundaries

In scope:

- external queries that are disjoint from corpus items;
- normalized dense text embeddings;
- one fixed dense Gaussian projection and one fixed projected dimension;
- a standalone implementation and Monte Carlo validation of the paper's exact dense-Gaussian Tri-Law;
- exact projected-space search for the first correctness-oriented backend;
- exact original-space reranking of retrieved candidates;
- pilot-based query LID estimation;
- binned and analytic adaptive-budget policies;
- independent statistical certification;
- embedding retention, evidence recall, cost, latency, and optional answer evaluation.

Out of scope for the MVP:

- choosing a projection dimension per query;
- training a new embedding model;
- HNSW/PQ/IVF approximation effects;
- sparse projection families;
- distributed serving;
- claiming a formal Tri-Law theorem for the full adaptive RAG system;
- claiming that embedding retention guarantees answer correctness.

## Non-negotiable correctness rules

1. Normalize corpus and query embeddings before projection when using cosine-equivalent retrieval.
2. Use a dense Gaussian projection with entries `N(0, 1/m_prime)`, where `1/m_prime` is the variance. NumPy's `normal(..., scale=...)` takes a standard deviation, so code must use `scale=1/sqrt(m_prime)`, never `1/m_prime`.
3. Do not renormalize vectors after projection. The paper's dense-Gaussian distance law is for projected Euclidean norms, not cosine similarity after projected renormalization.
4. Search with squared L2 distance in both the normalized original space and projected space.
5. Freeze the embedding model, corpus, projection seed, `m_prime`, budget grid, and data splits before certification.
6. Never use evidence labels, answer labels, exact top-k identities, or realized recall to choose `M(q)` at inference time.
7. `oracle_exact` LID is diagnostic only. Main deployment claims must use `pilot_rerank` LID.
8. Do not select a policy and certify it on the same queries. Policy selection uses `query_tune`; certification uses `query_cert`; final reporting uses `query_test`.
9. If any policy hyperparameter is changed after inspecting `query_cert`, certification is invalid and must be rerun on a fresh independent split.
10. Every stochastic component must have an explicit seed in the run manifest.
11. Keep query-level records. Aggregate-only CSV files are insufficient for auditing or recomputing bounds.
12. Report the pilot pass, expansion pass, and original-space reranking costs separately.
13. Keep `tri_law_probability(beta, rho, m_prime)` separate from Tri-Predict. The former is the exact single-triplet law; the latter aggregates the orthogonal conditional specialization through additional LID, structural, independence, and mean-field approximations.

## Calibrated Tri-Predict v2 addendum

These additional rules apply to the Calibrated Tri-Predict family wherever its
code is checked out; branch names do not disable them. They do not alter the
tagged Raw Tri-Predict v1 baseline. TLS-RAG uses its separately versioned protocol;
never confuse Calibrated Tri-Predict v3 with TLS-RAG Step 4 retention v3:

1. Add a separate `query_cal` role for fitting calibration parameters. Policy
   candidate selection still uses `query_tune`; scientific certification uses
   `query_cert`; label-free systems measurement uses `query_latency`; final
   reporting uses `query_test`.
2. `oracle_exact` LID may supervise the pilot-LID calibrator on `query_cal`
   only. It remains forbidden at inference and in tune/cert/latency/test policy
   decisions. Main deployment claims must use the frozen pilot-distance
   calibrator and deployable pilot inputs.
3. Realized retention and exact top-k identities may label `query_cal` budget-
   residual fitting records, but may never enter a policy decision at inference.
4. Preserve Raw Tri-Predict behavior and schemas. Calibrated methods require
   new names, versions, fingerprints, and run namespaces.
5. Do not use the observed SciFact cert/test records to fit or select v2. Use a
   new dataset and fresh cal/tune/cert/latency/test identities.
6. A positive v2 claim requires independent retention/evidence certification
   and measured paired latency superiority. Candidate-count reduction alone is
   not a latency result.

## Engineering rules

- Start with NumPy/SciPy and an exact vector-search backend. Add FAISS only behind a small adapter.
- Prefer memory-mapped arrays and batched matrix operations; do not materialize a full query-by-corpus distance matrix for large datasets.
- IDs must be stable strings at data boundaries. Array row numbers belong only in explicit ID-map artifacts.
- Cache artifacts using content-derived fingerprints that include the embedding model, normalization flag, corpus hash, projection seed, and dimension.
- Refuse to reuse a cache when its metadata does not match the current run.
- All metrics and confidence bounds must be reproducible from saved per-query records.
- A failed target is a valid output. Do not silently enlarge budgets after certification.
- Tests must run on CPU with a tiny synthetic dataset and no network access.

## Required implementation order

1. Build a tiny synthetic end-to-end walking skeleton.
2. Add exact original and projected retrieval.
3. Implement and validate the exact Tri-Law and its orthogonal conditional specialization according to `docs/TRI_LAW_SPEC.md`.
4. Add pilot-based and oracle LID estimators.
5. Add fixed-budget and monotone binned policies.
6. Add query-level logging and empirical-Bernstein certification.
7. Add analytic Tri-Predict policy only after Tri-Law conformance tests pass.
8. Add one real external-query dataset adapter.
9. Add evidence metrics.
10. Add answer generation only after the retrieval harness passes acceptance tests.

Do not begin with LLM answer generation. It is the most expensive and least diagnostic component.

## Required tests

- projection entries have the expected scale within statistical tolerance;
- projected vectors are not renormalized;
- exact Tri-Law matches Monte Carlo inversion rates within a predeclared binomial-error tolerance;
- exact Tri-Law returns zero for the collinear boundary and its orthogonal specialization reduces to the `F(m_prime, m_prime)` tail at threshold `beta`;
- marginalizing the orthogonal conditional chi-square law numerically agrees with the orthogonal marginal Tri-Law;
- exact original top-k matches a brute-force reference on a toy dataset;
- candidate retention equals the overlap after exact original reranking;
- LID estimator rejects zero/duplicate/insufficient distances cleanly;
- budget policy only emits values from the configured grid and never below `max(k_gt, M_pilot)`;
- monotone policy never reduces budget for a higher-LID bin;
- fixed-budget recall is nondecreasing with budget on exact search;
- empirical-Bernstein radius matches a hand-computed fixture;
- tune/cert/test IDs are disjoint;
- changing projection metadata invalidates cached projected embeddings;
- the same manifest and seeds reproduce identical per-query results.

## Handoff behavior

At the end of each milestone, update `docs/IMPLEMENTATION_PLAN.md` with checked boxes and add a short `STATUS.md` containing:

- what runs;
- exact command used;
- tests passed/failed;
- current artifacts;
- next task;
- known deviations or risks.
