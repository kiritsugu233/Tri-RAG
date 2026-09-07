# TLS-RAG Step 3 to Step 4 handoff

> **Version-scoped reference.** This document describes its named method and
> milestone, including historical results and commands. It is not a current
> assignment or permission to reopen data. Start at [root AGENTS](../AGENTS.md)
> and [the current entry](../START_HERE.md). Current L0 protection and standing
> checkout/push instructions apply; later versioned briefs define successor work.


Status: Step 3 is closed. This file records the original v1 readiness handoff; the current
entry is [TLS_RAG_STEP4_START_HERE.md](TLS_RAG_STEP4_START_HERE.md).

## Required provenance

Begin from a commit that contains the Step 3 closeout commit
`c32eefb2902173c30ea136e9df04be5abe081f09`. Before reading implementation
material, verify that all of the following are ancestors of `HEAD`:

- frozen Step 2: `f46ce73beccf5fddbf05fd87fdb0911318020add`;
- Step 2 archive and Step 3 handoff: `c5815f7e9b7f32360bd692552758f156dbc3d173`;
- final Step 3 implementation and portability fix:
  `3913ded9b786d6b6b15c734c750bd13d9c58e12c`; and
- Step 3 closeout: `c32eefb2902173c30ea136e9df04be5abe081f09`.

Refuse to proceed if the worktree is dirty in files needed for Step 4 or if any
required ancestor is absent. Do not reset, discard, or overwrite user changes.

## Strict initial reading order and allowlist

Read each allowed file completely and in this exact order:

1. repository-root `AGENTS.md`;
2. this file, `docs/TLS_RAG_STEP3_TO_STEP4_HANDOFF.md`;
3. `docs/TLS_RAG_STEP3_SYNTHETIC.md`;
4. `configs/tls_rag_step3_synthetic_v1.json`;
5. `src/tri_rag_harness/tls_rag_step3.py`;
6. `tests/test_tls_rag_step3.py`;
7. `pyproject.toml` and `scripts/run_tests.sh`, only as needed to run the
   established offline tests.

The frozen Step 2 source/config/tests may be inspected only if a precise Step 3
interface question cannot be answered from the files above. In that case, read
only these three files and no Step 2 prose or prompt:

- `configs/tls_rag_step2_synthetic_v1.json`;
- `src/tri_rag_harness/tls_rag_step2.py`;
- `tests/test_tls_rag_step2.py`.

If this list is insufficient, stop before reading anything else and report the
exact missing path, the exact unanswered question, and why that file is the
minimum necessary source. Do not broaden the list by analogy.

## Explicit reading prohibitions

At startup, do not read `STATUS.md` or `docs/IMPLEMENTATION_PLAN.md`. At the end
of an implemented milestone, locate only the TLS-RAG Step 4 insertion point and
edit a bounded local section; never read either file from beginning to end.

Do not read any of the following unless the user separately authorizes an exact
file after a reported allowlist gap:

- `docs/TLS_RAG_STEP3_START_HERE.md` or any earlier start/handoff document;
- `AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP3.md`, Step 1/2 prompts, or earlier agent
  briefs;
- historical v1/v2/v3 plans, reports, gates, postmortems, or status material;
- unrelated project Markdown, old milestone notes, or task-residue Markdown;
- any archive, artifact directory, run directory, returned bundle, or log;
- any real-data, protected-role, certification, test, latency, or answer-quality
  document or data file.

Filename discovery is allowed only to determine whether a specifically named
Step 4 path already exists. Discovery does not authorize reading its contents.

## Frozen Step 3 result

Step 3 advances TLS-RAG from a fixed exact-retrieval schedule to a synthetic,
auditable, query-adaptive two-action controller. It provides observed-pair
Tri-Law risk profiles, transparent remaining-gain and current-sufficiency
scores, mutually exclusive synthetic fit/bound/evaluation partitions,
candidate/stage/bin/reachability-specific Clopper--Pearson calibration,
conservative dual-bound stopping, and Rows 1--4 one-factor diagnostics.

The focused local suite passed 23 of 23 tests. The full local CPU suite ran 203
tests: 202 passed and one optional real-FAISS test skipped. The user subsequently
reported that the final cluster replay passed completely. The repository does
not contain the cluster job ID, node, environment versions, timings, or log, so
do not invent them.

Frozen local audit identities are:

- manifest: `864031f5a3a160db09b1b637b68529677d56f12b7ebbb04bf0f3e42d2c1c671e`;
- Step 3 config: `3901dd0c742944ebd8bb23893488a8ab4950e28e4c48ac7c4fa143873f8d0101`;
- Step 3 Phase A: `5272b52bc008e52d8d6cc132fcb19b91166258a3dd27b064f219297052e94e52`;
- Step 3 Phase B: `1754cf5e325fa8c0c950ab42ec3db26b560d96c6e52f2eb229d4cd3567a6d0a2`;
- Step 2 Phase A semantic trajectory:
  `119e46b2670c8830a4de7da57a94cc990e6316d990f6c8a8bd8417385be8b5ec`;
- Step 2 Phase B semantic supervision:
  `ef498b9ed0a10f76c6af753471b9935f3586eabd0cc1992dadb4c492472bf97b`.

The manifest intentionally retains host-observed raw Step 2 identities. A raw
manifest fingerprint is therefore a same-host audit identity, not a cross-host
acceptance gate. Cross-host compatibility uses both complete frozen semantic
identities.

Step 3 made no real-data, policy-selection, certification, latency, evidence-
guarantee, answer-quality, or production claim.

## Step 4 initial milestone

Step 4 begins with a real-transfer protocol freeze, not with a real-data run.
Its first objective is to turn the frozen Step 3 mechanism into a pre-registered
and reviewable protocol that could later be evaluated on external queries
without leaking outcomes into inference or reusing calibration queries for
selection, certification, or final reporting.

Complete the following in order:

1. Write a concise Step 4 brief and a minimal Step 4 start-here file. Freeze the
   scientific question, allowed data roles, role-opening order, independent
   units, candidate set, score/bin/reachability semantics, family-wise error
   allocation, success/failure gates, artifact identities, and no-retuning rule.
2. Specify how the frozen Step 3 two-action controller and Rows 1--4 diagnostics
   transfer without modifying Step 2, Step 3, or exact Tri-Law behavior.
3. Implement only CPU, network-free, synthetic fixtures and input guards needed
   to test that protocol. Tests must prove role disjointness, label isolation,
   pre-data fingerprint gating, terminal failure behavior, and deterministic
   artifacts.
4. Run the focused Step 4 tests and the complete CPU regression twice where
   artifact reproducibility is relevant. Update only new Step 4 documentation
   and bounded TLS-RAG sections of the shared status/plan files.
5. Commit the protocol/readiness milestone and stop at the real-data gate.

Do not open, download, parse, enumerate, summarize, or fingerprint real or
protected data during this initial milestone. If a real adapter or role manifest
is later required, first report the exact minimum file/data paths and obtain a
separate authorization that names them.

## Non-negotiable preservation and scope boundaries

- Keep frozen Step 2 exact retrieval, normalization, projection, stable string
  ties, prefix reuse, exact reranking, context construction, two actions, label
  isolation, artifacts, and compatibility semantics unchanged.
- Keep the Step 3 controller, schemas, candidates, partitions, scores,
  calibration semantics, stopping rule, Rows 1--4, artifacts, and fingerprints
  unchanged. Add new Step 4 names, schemas, namespaces, and fingerprints.
- Do not change `tri_law_probability` or any exact Tri-Law implementation or
  conformance test.
- Use CPU only and no network or new dependency. Do not run approximate indexes,
  GPU code, LLMs, answer generation, or answer evaluation.
- Do not access archives, prior runs, real/protected roles, certification/test
  outcomes, or historical v1/v2/v3 materials.
- Do not make selection, certification, latency, quality, posterior,
  evidence-guarantee, or production claims.
- Do not start any later Step 5 work.

## Stop conditions

Stop and report rather than guessing if the Step 4 scientific question cannot
be made precise from this handoff and the frozen Step 3 surface, if any required
role or statistical rule is ambiguous, if a prohibited file appears necessary,
or if preservation requires changing a frozen Step 2/3 or exact Tri-Law file.

At handoff, report the exact tests, branch, commit, new artifact fingerprints,
same-host reproducibility result, untouched frozen-file diff, remaining risks,
and any push/cluster/Slurm commands. For current work, root AGENTS supersedes the original no-push instruction:
commit and push the explicit reviewed task branch after checks, without force.
