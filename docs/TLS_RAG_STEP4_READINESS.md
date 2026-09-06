# TLS-RAG Step 4 readiness closeout

Status: protocol and synthetic readiness complete; real-data gate closed.
Branch: `codex/tls-rag-step4-readiness`. Base/handoff:
`0e4df85df3cd6bc9427476702a3b90339576636e`.

## Provenance and reading boundary

The initial worktree was clean and detached at the handoff commit. Read-only
`git merge-base --is-ancestor` checks returned 0 for all five required commits:
`f46ce73beccf5fddbf05fd87fdb0911318020add`,
`c5815f7e9b7f32360bd692552758f156dbc3d173`,
`3913ded9b786d6b6b15c734c750bd13d9c58e12c`,
`c32eefb2902173c30ea136e9df04be5abe081f09`, and the handoff above.

The initial reading order was root AGENTS, the Step 3-to-Step 4 handoff, then
Step 3 synthetic document, config, full source and tests. The established
pyproject/test launcher were read for test execution. The conditionally allowed
Step 2 config was subsequently read to resolve the exact retrieval constants
`k_gt`, `k_ctx`, projection seed and grid. No other Step 2 source/prose was
actively inspected. Historical documents and prior run bundles were not used.
Shared STATUS/plan had no Step 4 heading; new bounded Step 4 sections were
appended without reading their existing bodies. Existing CPU tests were run as
authorized; they were not modified or used as a route to inspect other material.

The user resolved the initial statistical ambiguity: preserve internal Step 3
calibration behavior without asserting joint coverage for adaptive cells; use
independent certification for future scientific acceptance. No real/protected
data was opened by this new runner. No real-data or protected-role paths were
discovered, enumerated, downloaded or fingerprinted for Step 4.

## What runs

`tri_rag_harness.tls_rag_step4` is a standalone readiness entry point with no
data/config override or real-enable switch. It validates the pinned protocol
and upstream config before generating a 24-query, six-role synthetic identity
fixture. The role state machine checks complete Phase A receipts, label-opening
order, prior receipts, immutable snapshots and terminal failures. Its successful
path is explicitly a simulation; `simulated_gate_pass` does not evaluate labels
or implement selection/certification.

The controller probe set contains 63 hand-constructed states: Rows 2--4, seven
validity/bound scenarios and three stages. Calls go directly to the unchanged
Step 3 controller. Input guards reject unknown/nested fields, wrong IDs/types,
unregistered features, nonfinite values, wrong grid states and fingerprint
drift. Typed invalid states expand through 3/6/12 then stop with an explicit
failure. Probe tables/models are manual branch fixtures, not fitted estimates.

The future real protocol fixes the candidate set, roles/counts, opening order,
independent question-family unit, targets, error allocation and no-retuning
rule. Internal alpha 0.2 and future certification alpha 0.05 are distinct.
Real model/corpus/role/annotation identities are unbound and gate-blocking.

## Exact verification

From the repository root, in the existing environment:

```bash
sh scripts/run_tls_rag_step4_readiness.sh /private/tmp/tls-rag-step4-readiness-3608-audit-01
```

The script ran this focused command twice:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_tls_rag_step4.py' -v
```

Each focused run: **26 passed, 0 failed, 0 skipped**. It ran the complete
unchanged CPU launcher twice with `sh scripts/run_tests.sh`. Each full run:
**229 tests, 228 passed, 0 failed, 1 skipped**, with the optional real-FAISS
conformance test skipped because FAISS is not installed. The final full run
took 25.883 seconds; this is test execution time, not retrieval latency.
The full suite includes frozen Step 2/3 compatibility and exact Tri-Law
conformance. Step 3 still verifies both complete Step 2 semantic identities and
same-host portable artifact equality in its own synthetic tests.

An earlier focused development run found four errors (the same nonfinite
contract issue in three tests, and a macOS temporary-directory symlink alias in
one test). These were resolved entirely in new Step 4 code/tests: nonfinite
inputs are rejected as upstream requires; test output parents use physical
temporary paths. The two complete final replay rounds passed after those fixes.

Environment: Python 3.9.6 at
`/Library/Developer/CommandLineTools/usr/bin/python3`, NumPy 1.26.4,
SciPy 1.13.0, macOS 26.5.1 arm64. No new dependency was installed. The focused
reproducibility test blocks socket creation and connection attempts.

## New synthetic artifacts

The two newly created directories are
`/private/tmp/tls-rag-step4-readiness-3608-audit-01/a` and
`/private/tmp/tls-rag-step4-readiness-3608-audit-01/b`. All six files compared
byte-for-byte equal using `diff -r`. This is a same-host reproducibility result.
No timing or environment-dependent fields enter these artifacts. Temporary
outputs are untracked; the runner, protocol and identities below reproduce them.

| Identity | SHA-256 |
| --- | --- |
| Protocol canonical JSON | `35a3ae863249d1f1d7aec7f0b2fbaae877871093216523177b67e5c40b0a802e` |
| Synthetic fixture | `bbb87661a7511c4b03e63d7543d4fa5c6f039185fb295740a5e29def72ad56f6` |
| Step 4 source bytes | `199ab95f9ee5f04456837b628d7c9e5f06b511191e8e9d781738c5b336340f33` |
| manifest.json | `e70dc296e46443efcff2f7885540ea9c54238a246a8f9fcba84a956551daab62` |
| protocol_identity.json | `b6b2272b9a9673ad4af7aaf29ce89a8b0a0099b9b3c7e3b4472fd575543b29a3` |
| synthetic_role_proof.json | `303917418f57f99f2233614431840d6b7755dc6f1f0fe6cbcc6898ea7da9177b` |
| synthetic_transitions.json | `40ac78c133ecc547834c273f41d61051f2b8a19ff1aac1c7226181d568f39446` |
| synthetic_controller_probes.json | `0d237c98547d13c4b604bfe4c12c90d3ca6ad0ba39cf95ae468f2dd6302db9fa` |
| denial_report.json | `a50cc9a59231e2e43c338cbcd74221700f22df3227c192fb533c1d9e85b54179` |

Artifact fingerprints hash canonical JSON excluding the artifact's own
`fingerprint` field. The manifest also records SHA-256 of each other complete
file, including that field and its newline; tests reconstruct both layers.

## Preservation

The following command returned exit 0 and no output:

```bash
git diff --exit-code 0e4df85df3cd6bc9427476702a3b90339576636e -- \
  src/tri_rag_harness/tls_rag_step2.py src/tri_rag_harness/tls_rag_step3.py \
  src/tri_rag_harness/tri_law.py \
  configs/tls_rag_step2_synthetic_v1.json configs/tls_rag_step3_synthetic_v1.json \
  tests/test_tls_rag_step2.py tests/test_tls_rag_step3.py tests/test_tri_law.py
```

No preexisting implementation/config/test file changed. Shared documentation
changes are append-only, explicitly marked TLS-RAG Step 4 sections. All other
changes use new Step 4 paths. `git diff --check` passed.

## Manual push and cluster replay

No push, remote fetch, cluster login or Slurm submission was executed. On the
local checkout, inspect the final commit using `git rev-parse HEAD`. To push
manually after reviewing it:

```bash
git push -u origin codex/tls-rag-step4-readiness
```

On an existing clean cluster checkout after that push, fetch and select the
reviewed commit. Substitute the full hash reported in the task closeout for
`REVIEWED_COMMIT`; it is intentionally not an invented cluster revision:

```bash
git fetch origin codex/tls-rag-step4-readiness
git switch --detach REVIEWED_COMMIT
git merge-base --is-ancestor 0e4df85df3cd6bc9427476702a3b90339576636e HEAD
sh scripts/run_tls_rag_step4_readiness.sh /tmp/tls-rag-step4-cluster-replay-01
```

For a Slurm site, the following is an unexecuted template using an already
available CPU NumPy/SciPy environment and the reviewed checkout as working
directory. Supply any site-required account/partition flags yourself; no
cluster host, allocation or environment module has been inspected or assumed.

```bash
sbatch --job-name=tls-rag-step4-readiness --cpus-per-task=1 --mem=4G --time=00:10:00 \
  --wrap='sh scripts/run_tls_rag_step4_readiness.sh "${SLURM_TMPDIR:-/tmp}/tls-rag-step4-${SLURM_JOB_ID}"'
```

Record the actual job ID, exact commit/environment and new synthetic output
identities after replay. There is no Step 4 cluster result to report yet.

## Remaining risks and next task

The first milestone is complete. Stop at the real-data gate. The next task is
separate exact-path authorization and pre-data binding review; the current
runner cannot open real inputs even if such authorization is later supplied.
It would require a separately implemented adapter and guard integration.

Synthetic IDs and hash receipts do not establish real iid sampling, semantic
deduplication, annotation validity or deployable feature provenance. Manual
controller probes do not demonstrate real transfer performance. The literal
small grid and prespecified target/sample counts may fail or be underpowered;
that is a valid result. Internal adaptive-cell joint coverage remains unproved.
The future certifier, selector and paired latency runner are specified only to
the extent needed for role/gate separation and are not implemented here. No
retention/evidence guarantee, answer-quality, latency or production claim is
made. Step 5 has not started.
