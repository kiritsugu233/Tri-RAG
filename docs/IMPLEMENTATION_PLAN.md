# Implementation Plan

This checklist is the working plan for the implementation agent. Check items only after tests and artifacts exist.

## Milestone 0: repository skeleton

- [x] Add `pyproject.toml` with a minimal supported Python version and pinned/locked dependencies.
- [x] Create `src/tri_rag_harness/`, `tests/`, `configs/`, `scripts/`, and `runs/` conventions.
- [x] Implement structured configuration loading and validation.
- [x] Implement deterministic seed setup and run-manifest creation.
- [x] Add CPU-only CI/local test command.
- [x] Add `STATUS.md`.

Acceptance:

- one command validates a config and writes a manifest;
- tests run without network access.

## Milestone 1: synthetic walking skeleton

- [x] Generate a tiny clustered corpus and disjoint external queries with known relevant clusters.
- [x] Implement embedding-array ingestion without any text model dependency.
- [x] L2-normalize original vectors.
- [x] Implement exact batched squared-L2 search with deterministic ties.
- [x] Compute exact original top-`k_gt` ground truth.
- [x] Generate and apply dense Gaussian projection using variance `1/m_prime` (`scale=1/sqrt(m_prime)` in NumPy).
- [x] Implement exact projected retrieval and original exact reranking.
- [x] Implement `tri_law_probability(beta, rho, m_prime)` in `tri_law.py`.
- [x] Implement the orthogonal conditional chi-square law.
- [x] Add deterministic Monte Carlo conformance experiments over several `(beta, rho, m_prime)` cases.
- [x] Test collinear, orthogonal, monotonicity, and marginalization identities from `docs/TRI_LAW_SPEC.md`.
- [x] Save query-level records and aggregate metrics.

Acceptance:

- fixed-`M` retention is nondecreasing over the configured budget grid;
- a run is byte-for-byte or value-for-value reproducible under the same seeds;
- projected vectors are demonstrably not renormalized.
- Tri-Law conformance tests pass under predeclared statistical/numerical tolerances.

## Milestone 2: LID and adaptive baseline

- [x] Implement robust Hill/MLE LID estimator.
- [x] Implement `oracle_exact` LID.
- [x] Implement two-stage `pilot_rerank` LID with cached pilot distances.
- [x] Quantify pilot-LID error versus oracle-LID.
- [x] Implement fixed-budget policies.
- [x] Implement monotone binned empirical policy fit on tune queries.
- [x] Implement fallback behavior for invalid LID estimates.

Acceptance:

- policy emits only allowed budgets;
- no labels or exact-neighbor identities are accessed by `choose`;
- tune/cert/test leakage tests pass;
- per-query logs expose pilot overhead and LID failures.

## Milestone 3: certification

- [x] Implement empirical-Bernstein radius and lower bound.
- [x] Add hand-calculated numeric fixtures.
- [x] Implement overall adaptive-policy certificate.
- [x] Implement optional Bonferroni-corrected per-bin certificates.
- [x] Implement sample-size planning or an explicit insufficient-sample warning.
- [x] Serialize `certification.json` and generate a report section from it.

Acceptance:

- a frozen synthetic policy can pass/fail certification deterministically;
- policy or split fingerprint changes invalidate old certification artifacts;
- the report cannot claim a pass when the stored artifact says failure.

## Milestone 4: analytic query-adaptive Tri-Predict

- [x] Refuse to start this milestone unless the exact Tri-Law conformance tests pass.
- [x] Implement exact finite-rank summation for `h_j(y)` on small `N`.
- [x] Implement stable chi-square CDF evaluation.
- [x] Implement bounded root finding for `y_j_star`.
- [x] Implement `y_j_star = infinity` and unit retention when `M - j >= N - k_gt - 1`.
- [x] Implement predicted retention across `M_grid`.
- [x] Implement deterministic rank integration/geometric sampling for larger `N`.
- [x] Test approximation against exact summation on small problems.
- [x] Implement `M(q)` selection and saturation logging.
- [x] Add optional scalar safety correction fitted only on tune queries.
- [x] Add a tune-only global `m_prime`/threshold sweep with a serialized selection rule.
- [x] Freeze the selected dimension before an independent fresh certification run.
- [x] Reject cert/test records at the dimension-selection boundary.

Acceptance:

- predicted recall is nondecreasing in `M` within numerical tolerance;
- increasing LID does not accidentally yield a lower configured budget after monotonic enforcement;
- analytic and empirical adaptive policies can be compared through the same interface.

## Milestone 4.5: retrieval-only systems benchmark

- [x] Add a memmap-compatible streaming exact squared-L2 backend.
- [x] Eliminate repeated pilot/expansion projected scans in the main harness.
- [x] Retain a legacy double-scan control in the latency benchmark.
- [x] Add realistic `100k x 768` and `1M x 1024` configurations.
- [x] Record query projection, search, LID, Tri-Predict, expansion, rerank, and total latency.
- [x] Record p50/p95/p99, distance counts, bytes scanned, cache size, and process RSS.
- [x] Verify that reuse and double scan make identical decisions under exact search.
- [x] Reproduce the 100k baseline on Genoa with one BLAS thread.
- [x] Run the 1M scale-up after the 100k gate passes.
- [x] Compile a frozen analytic policy into adjacent-float64 LID decision intervals.
- [x] Add fingerprinted artifact loading and dense/reference boundary equivalence tests.
- [x] Reproduce compiled-policy lookup latency and exact decision equivalence on Genoa.
- [x] Add an optional exact FAISS `IndexFlatL2` CPU/GPU adapter after archiving the exact baseline.
- [x] Add offline adapter, boundary-tie refusal, and benchmark integration tests.
- [x] Share one FAISS GPU resource pool across original/projected indexes.
- [x] Resolve bounded FAISS boundary ties with one-scan overfetch and recorded deterministic refinement.
- [x] Pass conformance against a real FAISS CPU build on the cluster.
- [x] Pass the 100k FAISS GPU smoke/latency gate before running the 1M comparison.
- [x] Reject FAISS GPU `k + boundary guard > 2048` before fixture generation.
- [x] Complete the 1M FAISS CPU/GPU comparison with a separately frozen `M_max=1984` grid.

Acceptance:

- no query-by-corpus distance matrix is materialized;
- reuse performs one projected scan and the legacy control performs two;
- both Tri-Predict paths choose identical budgets and retention;
- a non-NumPy backend must match NumPy candidate sets at `k_gt`, `M_pilot`, and
  every budget cutoff, row-aligned squared distances, compiled-policy
  decisions, reranked top-k rows, and retention before artifacts are accepted;
- FAISS boundary refinement must report its requested neighbors, host distance
  evaluations, and latency without counting an unreported second scan;
- run artifacts contain stage latency, work, memory, environment, and query-level records.

## Milestone 5: one real external-query retrieval dataset

- [x] Implement a dataset adapter producing corpus, external queries, splits, qrels, and optional answers.
- [x] Pin dataset revision/hash and store license/source metadata.
- [x] Implement pluggable text embedding with caching and explicit model revision.
- [x] Run a fingerprint-bound exact original-space baseline on `query_tune` only.
- [x] Run exact original/projected retrieval in memory-safe batches.
- [x] Freeze one `m_prime` using tune queries only.
- [x] Run required policies on tune, then freeze them.
- [x] Freeze cross-platform LID precision and separate compiled deployment identity.
- [x] Reproduce and independently audit the frozen scientific policy result on Genoa.
- [x] Implement and freeze a certification-only runner using synthetic fixtures only.
- [x] Certify on untouched certification queries.
- [x] Implement and freeze a descriptive test-only runner using synthetic fixtures only.
- [x] Evaluate the three frozen policies once on test queries without selection or recertification.

Acceptance:

- all required manifest fields and per-query records exist;
- at least one fixed policy and both adaptive policies complete end to end;
- failures and policy saturation are visible in the report.

## Milestone 6: evidence evaluation

- [x] Compute candidate and final-context evidence hit/recall.
- [ ] Compute separate evidence empirical-Bernstein bounds when requested.
- [x] Add matched fixed-cost and fixed-quality comparisons.
- [x] Add LID/retention/evidence relationship tables or plots.
- [x] Add shuffled-LID control.

The frozen test run reports auditable final-context evidence metrics at cutoffs
1/5/10. The posthoc tune-only run additionally records candidate-set evidence
separately from the exact original reranked context for all 403 tune queries and
all 16 frozen budgets. Slurm job `374032` reproduced the diagnostic twice, and
an independent reduction replayed the full fixed grid, relationship strata,
matched comparisons, and all 1,000 shuffled-LID controls. Separate evidence
confidence bounds remain optional and were not requested; no new certificate
was created from these posthoc outcomes.

Acceptance:

- qrel-free queries are handled explicitly;
- evidence results can be regenerated from saved query-level artifacts;
- the report distinguishes embedding retention from evidence recall.

## Milestone 7: optional answer generation

- [ ] Add a generator adapter only after retrieval milestones pass.
- [ ] Freeze prompt, model/version, decoding parameters, and context formatting.
- [ ] Cache prompts, contexts, raw outputs, and scores.
- [ ] Compute deterministic answer metrics where available.
- [ ] Analyze answer quality by evidence and embedding-retention strata.

Acceptance:

- retrieval can still run independently of the generator;
- generator failures do not remove query records;
- no text claims that embedding certification guarantees answer correctness.

## Final report requirements

- [x] State corpus/query sizes and prove query IDs are external/disjoint.
- [x] State `d`, `m_prime`, projection seed, `M_pilot`, `M_grid`, `k_gt`, and `k_ctx`.
- [x] Report policy budget distributions.
- [x] Identify the smallest certified fixed-`M` baseline.
- [x] Report adaptive saving against that baseline.
- [x] Report overall and optional per-bin certificates.
- [x] Report pilot-versus-oracle LID gap.
- [x] Report embedding and evidence metrics separately and state that answer
  generation was not run.
- [x] Document all negative results, deviations, and remaining threats to validity.

These report requirements are closed for Raw Tri-Predict v1. v2 must produce a
new report from its fresh data identities rather than amending the v1 result.

## Raw Tri-Predict v1 closure

- [x] Complete tune-only selection, independent certification, and one-time test.
- [x] Preserve Raw Tri-Predict's terminal FAIL without post-cert retuning.
- [x] Complete pilot-versus-model attribution and tune-only evidence diagnostics.
- [x] Independently audit returned query-level artifacts and shuffled controls.
- [x] Tag the closed implementation as `raw-tri-predict-v1-terminal-negative`.
- [x] Record the immutable result in `docs/RAW_TRI_PREDICT_V1_BASELINE.md`.

## Calibrated Tri-Predict v2: network-free foundation

- [x] Add a versioned pilot-distance feature specification and extractor.
- [x] Add a constrained log-linear pilot-LID calibrator.
- [x] Add a quantile analytic-budget residual calibrator.
- [x] Add LID-only, residual-only, and full PDCTP policy variants.
- [x] Preserve Raw Tri-Predict v1 decisions and artifact loading unchanged.
- [x] Add five-role cal/tune/cert/latency/test guards and synthetic fixtures.
- [x] Add paired evidence/budget bounds and family-wise-alpha fixtures.
- [x] Add a deterministic sample-size/power artifact.
- [x] Pass all existing and new CPU tests without network access.
- [x] Reproduce and independently audit all 20 artifacts byte for byte on Genoa.

The network-free gate is implemented by
`configs/pdctp_network_free_foundation_v1.json` and
`tri_rag_harness.pdctp_foundation`. It writes separate feature, candidate-fit,
selected-calibrator, policy-suite, split, hypothesis, paired-bound, label-free
latency-dry-run, protocol-state, and power-plan artifacts. The checked-in
worst-case empirical-Bernstein power plan requires 1,567 fresh certification
queries for the full paired family. The 16-query synthetic certification
fixture therefore fails all primary bounds as expected; it validates terminal
failure and reconstruction behavior and is not a scientific claim.

The 125-test local suite reports 124 passes and one optional real-FAISS
conformance skip because FAISS is absent from the offline Mac environment. The
end-to-end v2 test runs the complete five-role skeleton twice and compares all
20 output artifacts byte for byte. Every one of its 312 base/decision records
uses one cached projected scan, and all `query_latency` records are label-free.
The first Genoa artifact audit preserved every selected tuple, budget, metric,
and terminal decision but exposed cross-platform float/fingerprint drift. The
feature/LID and residual lattices are now explicit and tested. The first
canonical rerun improved exact agreement from 8/20 to 16/20; the remaining four
files traced only to three unselected-candidate coefficients. The final
five-decimal rerun at commit `64348b0` passed 124 of 125 tests with one optional
FAISS skip and matched all 20 Mac artifacts byte for byte. Its returned archive
SHA-256 is
`5d2fb7adf248819c4adfb5493328ed3db8fb64ad6deb39d9a78d32b36c045012`.

Acceptance:

- calibration inference reads only deployable pilot quantities;
- fit records contain `query_cal` only and selection records contain
  `query_tune` only;
- no protected runner can read its split before validating every upstream
  fingerprint;
- all policies share one exact projected ranking and original reranker;
- v1 tests and frozen decisions remain unchanged.

## Calibrated Tri-Predict v2: fresh real-data gates

- [x] Audit and pin a new external-query dataset; FiQA passes the source gate.
- [x] Freeze duplicate-safe five-role IDs and a power-supported protocol.
- [x] Implement canonical qrel-free FiQA text preparation and an independent
  E5 cache audit runner without opening a protocol role.
- [x] Build and independently audit a new E5 embedding cache.
- [x] Freeze `m_prime=192`, a fresh projection seed, and one budget grid.
- [x] Implement and test the fingerprint-gated `query_cal` runner and compact,
  reconstructable all-candidate artifact format.
- [x] Execute and independently audit all calibration fits on `query_cal` only.
- [x] Implement and test the fingerprint-gated all-family `query_tune` runner.
- [x] Select one complete PDCTP policy on `query_tune` only.
- [x] Freeze fixed, monotone, Raw Tri, ablation, and PDCTP comparators.
- [x] Independently replay all 2,086 tune candidates, selected policies, and
  hypotheses from returned query-level artifacts.
- [x] Implement and freeze the fingerprint-gated `query_cert` runner without
  opening certification outcomes.
- [ ] Certify once on untouched `query_cert` with family-wise correction.
- [ ] Run a frozen paired CPU/GPU latency protocol on `query_latency`.
- [ ] Evaluate once on `query_test` only after all prior gates are terminal.

Acceptance:

- the strong claim is emitted only if PDCTP meets retention/evidence
  constraints and passes paired budget and latency superiority against every
  eligible comparator;
- feasibility dominance is not mislabeled as matched-quality cost superiority;
- a failed gate is terminal and does not trigger retuning.

The FiQA source gate is documented in `docs/FIQA_SOURCE_AUDIT.md`. The pinned
17,948,027-byte archive matches official MD5
`17918ed23cd04fb15047f73e6c3bd9d9` and independently measured SHA-256
`32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02`.
All qrel references pass. The eligible native train/dev/test counts are
5,500/500/648, normalized query text has no duplicates, and the constructive
five-role witness assigns cal/tune/cert/latency/test counts of
1,966/1,967/1,567/500/648. Its `GO_TO_PROTOCOL_FREEZE` decision is a capacity
result only and does not authorize embeddings, method evaluation, or protected
outcome access. The audit also exposes 38 empty corpus items referenced by
positive qrels; the subsequent protocol gate freezes their `[EMPTY_DOCUMENT]`
representation rather than silently removing them. The complete offline regression reports 129
passes, one optional real-FAISS skip, and zero failures across 130 tests.
Slurm allocation `374320` independently reproduced this gate on `genoa06` at
commit `d5c31a3`: the same 130-test result passed in 30.641 seconds, and both
audit artifacts matched byte for byte. The returned archive passed local
SHA-256 verification at
`098910e034dfb1790913699a3c3ad9e4c13106852722821dde8218539e31f46e`.

The complete real-data preregistration is documented in
`docs/FIQA_PROTOCOL_FREEZE.md` and emitted by
`tri_rag_harness.pdctp_real_protocol`. Config/protocol fingerprints are
`47c602c777e9e4589597ae996a7d1459407ae916b376854699569c115ebdfc41`
and `cb3ef70f3ffc801c248f3269e0807480f0ee5a51cde41a52573a03f228a42368`.
It freezes the exact 6,648 ordered role IDs, the 38-item empty-document marker
contract, pinned E5 revision, fresh projection seed `83047`, `m_prime=192`, a
21-value full-corpus-terminal budget grid, 1,620 PDCTP tuples, tune/certification
targets, and paired CPU/GPU latency rules. All roles remain closed and the GO
decision permits only dataset preparation and independent embedding-cache
audit. The 135-test offline suite reports 134 passes, one optional FAISS skip,
and zero failures. Slurm allocation `374320` independently reproduced the gate
on `genoa06` at commit `2a5b44f`: all four artifacts matched byte for byte and
the same test result completed in 30.706 seconds. The returned archive passed
local SHA-256 verification at
`36e8698f01777abc2f6dde5ee5e69385f1f9ca8298ca59e3fff47c6a421d165e`.

The next gate implementation is documented in `docs/FIQA_EMBEDDING_GATE.md`.
The qrel-free `pdctp_fiqa_text_only_v1` preparation opens only corpus/query ZIP
members, retains all 38 frozen `[EMPTY_DOCUMENT]` rows, orders all 6,648 query
texts by the exact closed five-role assignments, and fingerprints both IDs and
formatted E5 text. Two real local preparations matched all six artifacts byte
for byte; the dataset-manifest fingerprint is
`bfc25daad8d2d382390a0a42c3aa03b96e965965ba17c2065aaf8bef00903240`.
The E5 request config fingerprint is
`dce9c5f590c0348672dc3ab6f90a8e07e5b170c2174a5c2aab5b9eaeabc8bc78`.
An independent audit runner rehashes formatted inputs, row IDs, arrays,
normalization, model snapshot, runtime, token lengths, and complete cache
identities. Slurm allocation `375414` on `a100-0` at commit `1671b2b` built the
cache once, safely reused it once, and emitted two byte-identical audits. The
embedding and audit fingerprints are
`079545ef7c6af8ab27a5c8382dbd8174905f1bb537df59e94d572b6c2f2b04c1`
and
`54af315d5b94b43a81be71ea29ab860635f0748a97108e0cda120a510947dd71`.
The returned archive SHA-256 is
`87288fd7e913930474c9f764017780b729ab00404e93d88e2fb9ffc0359c1133`.
Local transfer verification, 6/6 cross-platform preparation comparison, full
array rehash/norm audit, and byte comparison against both cluster audits all
passed. The 141-test local suite reports 140 passes, one expected optional
FAISS skip, and zero failures. The embedding gate is closed; `query_cal` is
ready but has not yet been opened.

The `query_cal` implementation and accepted run are documented in
`docs/FIQA_QUERY_CAL_GATE.md`. Its config fingerprint
`7ff0bdf656ebc22026702622e933975ffe56b3814bb384b1db99effde51df36b`
binds every accepted upstream identity and the complete frozen calibration ID
order. Synthetic tests cover exact one-scan retrieval, deterministic records,
fit-only role scope, compact exact-pinball equivalence, complete 1,620/405
operating-point enumeration, reconstruction, and tamper refusal. The new
compact solver is opt-in and preserves the network-free foundation's legacy
solver serialization and behavior.

Slurm allocation `375414` executed all 1,966 `query_cal` records and returned
archive SHA-256
`686781b787c5a64a00b81996547594abfd5ffcd60107927844a40a552872089c`.
Independent reconstruction and a complete local refit reproduced the frozen
LID and residual bundles exactly. Their fingerprints are
`4526c8b752325e3cae040d8b450c76cd0df77571b9e6d6080bd1a53ba4a56a1e`
and
`ed6cf0f7056fc1b7345b5303a7ad71815a2b8df6677c6c1c2c0e231bbf9c9f31`.
The audit preserves, rather than hides, 33 duplicate-distance feature
failures, 15 pilot-LID failures, and seven oracle-LID failures. It also records
cross-platform last-decimal drift in 60 diagnostic oracle-LID values and one
saved projected distance, all bounded by approximately `1.01e-10`; every
deployable feature, ranking, target, candidate, and operating point remains
identical. Audit fingerprint
`e3cd09d125a868b685df02b23f0706926fa5786752d863f17bf13d6293de7884`
accepted only the fit gate and authorized the subsequent query_tune runner.
That authorization has now been consumed by the accepted tune run documented
below; cert/latency/test remain closed. The post-query-cal audit suite reported
146 passes, one optional real-FAISS skip, and zero failures across 147 tests.
This corrected audit identity supersedes `1e2e09d9...`, whose sole error was a
manually transcribed 62-character query-record file hash; the returned file
already matched the original run manifest's 64-character hash.

The next protected runner is documented in `docs/FIQA_QUERY_TUNE_GATE.md`.
Config fingerprint
`06c647625bb01192b54ae0698e9e4150fe4fec0d2b4407858de74c763573d7d0`
binds the corrected accepted query_cal audit and every prior source, role,
embedding, calibration, and power-plan identity. A real-returned-artifact dry
validation reconstructed all 2,025 residual operating points and exactly
replayed the closed post-calibration state without reading tune outcomes.

After validation, the runner may open only all 1,967 query_tune IDs. It parses
document/relevance fields from the combined native train qrel file only after
the first-column query ID passes the tune-role filter. One frozen projected
scan provides pilot and expansion order; exact original reranking produces
retention plus candidate/final evidence curves for all 21 budgets. The full
`2086 x 1967` per-query candidate budget matrix prevents aggregate-only
selection records.

Six method families are optimized independently under the same quality
constraints, then stored as a deduplicated, fully reconstructable component
registry. A success freezes selection and the already preregistered hypotheses
without opening query_cert; a missing eligible family is terminal. Eight
query-tune tests cover the accepted audit, profile reuse, filtering,
deterministic retrieval/evidence, complete family enumeration, terminal
failure, selection/suite reconstruction, Raw v1 immutability, and tamper
refusal. The post-audit 155-test offline suite reports 154 passes, one expected
optional FAISS skip, and zero failures. The real
query_tune run on allocation `376924` passed all 153 then-current cluster tests
and froze all six families. Its returned archive SHA-256 is
`3cfbeb6abd65b4e01991bc79065c2c244c8bc356f86deddae88a9ae4b7084969`.

Independent audit fingerprint
`f9a375115ed2c7f461bbd16add72735a6b9da44c309dd91f4782ecb01f0e5924`
and file SHA-256
`2fcb974179bd6d0f7299c9cb410655c5989bdb99eb58d73455b2a45f47f686a0`
reloaded 4,958 tune-positive qrel rows, recomputed every query curve and all
2,086 candidate evaluations, reproduced 11,802 selected-policy records, and
exactly replayed the complete candidate budget matrix, selection, component
registry, policy suite, hypotheses, projection, and post-tune state. The
selected PDCTP is quality-eligible but uses 7.318% more common coordinate work
than fixed on tune and exceeds the GPU stable-selection limit on 262 queries.
These are frozen negative/feasibility signals, not grounds for retuning.

The observed multi-hour silent interval came from threshold-duplicated
Tri-Predict computation. Exact profile reuse plus progress logging now reduces
the profile-input path from 49,175 threshold requests to 9,835 shared inputs.
A full real-record replay proves value-identical candidate outputs and does not
change Raw Tri-Predict v1.

The one-time certification runner is now implemented and documented in
`docs/FIQA_QUERY_CERT_GATE.md`. Config fingerprint
`c6357a748f7f3262f481c1f597d3f25acb76892a9dbfc621735effe6b0bd8143`
binds the accepted tune audit and all 16 returned tune files, selected suite,
hypotheses, power plan, projection, embedding cache, source archive, and full
1,567-ID cert order. Six synthetic tests cover role-scoped qrel parsing,
deployable-only decisions made before supervision reduction, deterministic
one-scan records, complete paired-bound reconstruction, and terminal PASS/FAIL
behavior. The 161-test offline suite reports 160 passes, one expected optional
FAISS skip, and zero failures. A metadata-only replay reconstructed all six
real frozen policies, the hypotheses, and the exact post-tune guard state while
leaving query_cert closed. The remaining unchecked cert item is the one-time
real execution; no cert outcome has been read locally.

## Calibrated Tri-Predict v3: Step 3 causal repair

- [x] Independently validate every allowed returned-archive SHA, run-file hash,
  record/candidate fingerprint, upstream binding, query-tune selection, and
  protocol-state replay in fresh temporary directories.
- [x] Keep query-cert, query-latency, and query-test identities and outcomes
  closed; perform no download, LLM call, answer generation, or v2 mutation.
- [x] Reconfirm exact dense-Gaussian Tri-Law conformance and projection scale.
- [x] Compare deterministic geometric-rank quadrature with exact finite-rank
  aggregation over the prescribed layer grid.
- [x] Diagnose rank-distance, mean-field geometry, LID input, calibration
  target, budget residual, fallback, and selection layers separately.
- [x] Write the nine-part causal conclusion and identify the scalar
  rank-distance power law as the earliest failing layer.
- [x] Add an isolated v3 effective-curve-shape calibrator fitted on query-cal
  only, with scalar and low/high effective-Tri-LID modes.
- [x] Preserve a one-factor ablation: both modes share features, fit records,
  numerical problem, Raw Tri-Predict curve implementation, and decision input.
- [x] Add an exact in-memory float64 prediction-grid cache whose complete key
  includes the numerical implementation version and whose state is never
  serialized into scientific artifacts.
- [x] Prove on a complete tiny candidate suite that cached and uncached budgets,
  selection objects, artifact values, and fingerprints are identical.
- [x] Pass the full CPU/network-free suite without changing Raw Tri-Predict v1,
  frozen v2, exact Tri-Law, or numerical tolerances.
- [ ] Freeze and execute a new real-data five-role v3 protocol on a new dataset.

Acceptance state:

- Step 3 diagnosis and network-free implementation are complete on
  `codex/calibrated-tri-predict-v3`;
- the exact test command is `./scripts/run_tests.sh` and reports 166 passes,
  one expected optional real-FAISS skip, and zero failures across 167 tests;
- `docs/CALIBRATED_TRI_PREDICT_V3_DIAGNOSIS.md` contains the layer matrix,
  directions and magnitudes, deterministic derivations, and causal answers;
- the checked-in v3 configuration is
  `configs/pdctp_v3_network_free_foundation_v1.json` with a new run namespace;
- a real v3 policy remains intentionally unevaluated until a fresh dataset and
  independent cal/tune/cert/latency/test identities are frozen.

## TLS-RAG successor: Step 1 design freeze

- [x] Verify the frozen v3 starting point at commit `f94c1aa` and create the
  separate `codex/tri-law-sequential-rag-v1` branch.
- [x] Preserve Raw Tri-Predict v1, PDCTP v2/v3, exact Tri-Law, protected roles,
  and returned certification archives without modification or access.
- [x] Define the TLS-RAG sequential problem, state, next-grid-only actions,
  transitions, stopping rule, terminal states, and conservative fallback.
- [x] Restrict inference state to exposed pilot/expansion observations and
  explicitly forbid roles, qrels, evidence/answer labels, oracle LID, exact
  full-corpus top-k identities, and realized outcomes.
- [x] Restrict exact Tri-Law to an ex-ante single-observed-pair law and define
  all frontier aggregation as a feature or calibrated risk score.
- [x] Define distinct candidate gain, final-context gain, remaining gain, and
  current evidence-sufficiency targets under a frozen evidence plan.
- [x] Freeze a query-cal fit/bounds split and bin-level one-sided uncertainty
  calibration path; do not treat a point score as a posterior or bound.
- [x] Freeze the fixed top-`k_ctx` context builder and the complete one-factor
  ablation ladder through the later conditional LLM row.
- [x] Define fresh-data cal/tune/cert/latency/test roles, label permissions,
  leakage guards, matched-fixed selection, independent hypotheses, and
  terminal no-retuning rules.
- [x] Require strict tune-side paired common-work superiority over the matched
  fixed reference before an adaptive candidate is eligible.
- [x] Define a small CPU/network-free Step 2 acceptance plan with synthetic
  counterexamples and no controller framework, real data, model download,
  approximate index, or LLM.
- [x] Pass the complete existing CPU/network-free regression suite without
  changing frozen behavior.
- [x] Commit all Step 1 documentation on the successor branch and stop for user
  review before Step 2.

Acceptance state:

- the algorithm specification is `docs/TRI_LAW_SEQUENTIAL_RAG_SPEC.md`;
- the fresh-data/evaluation protocol is
  `docs/TRI_LAW_SEQUENTIAL_RAG_PROTOCOL.md`;
- Step 2 is not authorized until the user explicitly accepts this design;
- real-data evidence sufficiency remains blocked pending a fresh source with
  facet/support/completeness annotations adequate to identify the frozen
  targets.
- the local regression reports 166 passes, one expected optional real-FAISS
  skip, and zero failures across 167 tests.

## Suggested future code tree

```text
query-adaptive-tri-rag-harness/
  AGENTS.md
  START_HERE.md
  pyproject.toml
  configs/
    mvp_scifact.yaml
  src/tri_rag_harness/
    config.py
    manifest.py
    data/
    embeddings/
    projection.py
    indexes/
    lid.py
    tri_law.py
    tri_predict.py
    policies/
    retrieval.py
    metrics.py
    certification.py
    reporting.py
    run.py
  tests/
  docs/
  runs/
```

## Stop/go gates

1. Do not implement Tri-Predict until exact Tri-Law conformance tests pass.
2. Do not add a real embedding model until the synthetic walking skeleton passes.
3. Do not implement answer generation until retrieval certification works.
4. Do not optimize approximate indexes until the exact-search results establish a benefit.
5. Do not claim a positive result unless the adaptive policy beats the properly matched certified fixed baseline.
