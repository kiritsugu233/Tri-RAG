# Repository review and cleanup — 2026-10-30

Baseline: `50defc35bc9f42f5b24d2dedf5f45a89e51257ad` on
`codex/tls-rag-step5-ann`, canonical checkout
`/Users/guanghongxu/Query-Adaptive-Tri-RAG`.
[Repository map](REPOSITORY_MAP.md) lists every implementation file and its role;
[evidence](repository_cleanup_evidence.json) records counts, hashes and tests.

The authorized L1 cleanup is complete. **No whole source file was established
as safely deletable.** Important changes below remain proposals for the user.
No scientific source/config/test/dependency, native implementation, existing
protection level/hash or historical experiment was changed.

## Review coverage

The baseline contains242 tracked files:54 package Python modules,8 mechanism
Python files,4 C++ files,13 scripts,47 Python test files,36 configs,58 Markdown
files and retained build/dependency/evidence/artifact files. The register has230
entries (161 L0,66 L1,3 L2). Python sources total52,435 lines including tests.

- Every tracked implementation was inventoried and Python-parsed; direct imports,
  entry points, native includes, config/path references and scientific bindings
  were checked. All58 tracked Markdown files passed relative-link/fence checks.
- Current Step5 data/role/ledger/feature/decision/timing paths and continuation
  v1/v2 Python/native code received focused manual inspection, including the
  current runner's nested selection and label-free workers. Historical retained
  families were checked against their entry points/dependencies and the prior
  review; unchanged old code was not re-proved line by line.
- Duplicate helper bodies and unused-import leads were inspected conservatively.
  Text/name matching does not prove reachability or deletion safety. Existing
  code tests provide executable evidence within their fixture scope.
- No user archive, report_work content, large runtime dataset or closed-role
  label was opened/hashed. No retrieval experiment, model fit on real data,
  network/SSH/GPU job, branch deletion or dependency installation occurred.

This is a repository structure/contract review with focused current-code review,
not a formal proof of every scientific implementation or a fresh validation of
historical results.

## Findings and disposition

| ID | Priority | Finding | Disposition |
| --- | --- | --- | --- |
| C1 | P1 | Guard excludes new C++/header paths and ignored mechanism Python | Reproduced; protected repair A pending |
| C2 | P2 | Human protection register omits40 machine-registered paths | Exact paths in evidence; protected repair A pending |
| C3 | P2 | Next-agent template says real Step5 preparation/evaluation awaits instruction; workflow names retired Step4 branch | Corrected L1 navigation to actual current state/standing AGENTS |
| C4 | P2 | START_HERE/STATUS/plan repeat historical states as current, increasing stale-work risk | Short current summaries, fixed-Git historical links and lifecycle map |
| C5 | P2 | Plain NumPy/SciPy environment cannot complete advertised Step5 suite |4 missing-FAISS errors retained; fixed existing env passes429; explicit interpreter/cwd support and documentation added |
| C6 | P2, known safety limit | Continuation v2 lacks trained-support/OOD abstention | Current source confirms known journal limitation; not changed or relabeled accepted |

### C1: concrete guard counterexample

`scripts/check_code_protection.py:code_path` accepts `.py/.sh/.toml`, configs and
requirements files. It returns false for `mechanism/new_native.cpp` and `.hpp`.
`inventory` recursively inspects only src/tests/scripts/configs, omitting
mechanism. In a disposable Git repository with unregistered
`mechanism/new_native.cpp` and Git-ignored `mechanism/ignored.py`:

```text
code_path_cpp: false
code_path_hpp: false
inventory: []
check(schema_version=1, files=[]): []
```

This was produced by the unchanged actual guard's functions. Existing registered
C++ files still have L0 hashes and are checked; the failure is discovering new
unregistered implementations. The Python test fixture was outside this checkout;
no protected file or working experiment was modified. Receipt is in the evidence.

### C2: protection table drift

At the review baseline, the machine registry contains all existing code paths,
but the L0 human register stops after the early mechanism namespace.40 entries (later reuse,
baseline, route1/comparator/continuation files and documents) are absent from that
table. The machine levels are valid and unchanged. The new map shows those levels;
it does not silently replace or downgrade the authoritative register. The four
new L1 navigation/review entries also await human-table synchronization under A.

### C5: tests and environment

Plain `sh scripts/run_tests.sh` at the baseline ran429 tests:392 passed,
4 errored (`ModuleNotFoundError: faiss`),33 skipped. The four errors are in
`tests/test_tls_rag_step5_baseline_headroom.py`; other integration tests explicitly
skip without the optional/native environment. This exposes a dependency/readiness
mismatch rather than an observed algorithm regression. No test was weakened.

The existing environment has NumPy1.26.4, SciPy1.13.0, FAISS1.8.0 and
scikit-learn1.6.1 plus the four previously compiled native libraries. It passed
all429 tests with no skips before cleanup. The updated L1 launcher selects this
interpreter explicitly and resolves its repo from its own path, retaining the
same discovery command, tests and assertions. It still reports missing packages
when a caller selects an incomplete environment; it does not auto-install or
silently skip failing tests.

### C6: retain current safety failure

`mechanism/tls_rag_step5_ann_continuation_v2.py:Predictor.__call__` checks shape/
finite values; `Runtime.call` and `decide` have no learned-support abstention.
This agrees with the existing journal §§30–31. Numerical input checks are not
an OOD guard. No deployment/safety acceptance can be inferred from passing tests.
Fixing this would change stopping decisions and needs a separately scoped L0
scientific version/evaluation task; it is not part of maintenance cleanup A.

## Deletion and framework decisions

No source/config/test was deleted, moved or consolidated. Reasons are concrete:

| Candidate family | Evidence against automatic deletion |
| --- | --- |
| `tls_rag_step5_ann_continuation_v1.py/.cpp` | v2 subclasses/imports v1; v2.cpp directly includes v1.cpp |
| `tls_rag_step5_ann_enhancement.py` | Helpers re-exported through comparator/continuation modules to runners; current binding includes it |
| `tls_rag_step5_audit.py`, `tls_rag_step5_exact_policy.py`, `tls_rag_step5_reuse_probe.py` | Shared ExactScan/PrefixSession/cache implementations remain used by baseline/reuse code |
| `tls_rag_step2.py`, `tls_rag_step3.py` | Feature/state compatibility and current Step5 policy imports remain |
| `pdctp_*`, `real_*`, `attribution.py`, `mprime_sweep.py`, `retrieval_benchmark.py`, `run.py` | Retained CLI/tests/protocol history; all54 package files are covered by the Step5 ANN source binding, even L1 files |
| Versioned configs, tests and launchers | Scientific protocol/conformance and version-specific source seals; completed/negative study does not mean useless file |
| Tracked artifacts/JSON evidence | Reproducibility and original failure/approval history; cannot be inferred disposable |
| Untracked archives/bundle/report_work/runs/data | User-owned/experimental payloads; not proven temporary, untouched |

Common hashing/JSON helpers do repeat across historical modules. Removing them
would change source seals and can conflate different strictness/overwrite rules.
Unused local imports also include deliberate public re-exports. A broad formatter,
linter auto-fix or moving old code into an archive directory would create protected
changes for little current benefit. Future shared-library extraction should be
scoped against a new experimental namespace, with explicit compatibility/equality
checks, rather than refreshing old artifact hashes.

Implemented structure cleanup:

- Added a short README, lifecycle/dependency map and per-file disposition for all162
  implementation/config/test paths. Existing package/module/import paths stay valid.
- Replaced accumulated startup/status/plan narration with bounded current summaries.
  The complete original text/approvals/commands remain in Git at the full baseline
  SHA; evidence records original hashes and exact offline `git show` commands.
  No duplicate archive tree was added. Branch-retirement tips remain in STATUS.
- Corrected current workflow and next-agent routing. The long append-only scientific
  journal, frozen briefs and numerical failure evidence retain their original bytes.
- Changed only the L1 test launcher among executable files. Four new L1 navigation/
  review/evidence rows were appended; all230 prior rows and all other registry root
  values remain equal to baseline. No approval history was fabricated.

## Proposal A: protected guard and register repair

**Pending user decision; not implemented.** Exact protected paths:

1. `scripts/check_code_protection.py` — extend `code_path` to native C/C++ source/
   header suffixes and include `mechanism/` in ignored-file discovery. Keep
   read-only behavior and preserve current inventory exclusions for user data,
   environments, archives and runtime outputs.
2. `tests/test_code_protection.py` — add disposable-fixture regressions for new
   untracked/tracked native sources/headers and ignored mechanism code. Keep all
   existing tests/levels/hash checks unchanged; add negative and exclusion cases.
3. `docs/CODE_PROTECTION.md` — synchronize the40 omitted existing entries and new
   review L1 entries with the machine inventory; document the corrected checker
   coverage. No existing grade, consent requirement or scientific rule changes.
4. `docs/code_protection.json` — record actual scoped user approval and only the
   resulting approved checker/test/human-register hashes. Preserve every existing
   classification and scientific hash; no automatic blanket baseline refresh.

Why necessary: C1 is an actual bypass of discovery for the repository's native
implementation language, and C2 can mislead an agent reading the human table.
An L1 wrapper/map cannot repair the canonical guard or L0 documentation. Their
protected content must be edited directly; general cleanup authorization does
not satisfy the exact-path L0 rule the user requested.

Impact: guard coverage and governance tests/documentation only; no retrieval,
Tri-Law formula/tolerance, candidate ordering, calibration, stopping, model,
config, data-role, cache/bundle schema, historical result or scientific claim
changes. Existing scientific binding maps must remain byte-identical. The guard
is still a review tool, not a human-approval verifier or OS permission system.

Validation after approval: reproduce C1 as a detected error in disposable fixtures;
run existing/new guard tests and the full fixed-version429-plus-added suite; verify
native no-op/resume tests, unchanged source-binding maps/all unrelated L0 hashes,
no scanned data/archive content, Markdown/table coverage and `git diff --check`.
Then commit/push this same branch and independently verify remote HEAD.

No other important-code deletion/refactor is requested by proposal A. Current
recommendation is to retain historical scientific namespaces; a future retirement
would need its own exact paths, preserved contracts and migration/replay plan.

## Validation

Baseline default log: `/private/tmp/tri-rag-cleanup-baseline-20261030.log`.
Complete environment baseline log: `/private/tmp/tri-rag-cleanup-full-20261030.log`
(429/429,0 skips,66.634s). Final launcher validation runs the same suite from
`/private/tmp`, verifying independent cwd resolution and interpreter selection:

```bash
export TRI_RAG_TEST_PYTHON=/private/tmp/step5-ann-enhancement-env-20261028/bin/python
export STEP5_ROUTE1_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/route1.dylib
export STEP5_COMPARATORS_V2_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/comparators_v2.dylib
export STEP5_CONTINUATION_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/continuation_v1.dylib
export STEP5_CONTINUATION_V2_NATIVE=/private/tmp/step5-ann-enhancement-native-20261028/continuation_v2.dylib
cd /private/tmp
sh /Users/guanghongxu/Query-Adaptive-Tri-RAG/scripts/run_tests.sh
```

These paths identify the existing local Mac environment/libraries; they are not
portable cluster setup instructions. No new experiment/model/data access follows.
Final protection/link/binding/diff/launcher receipts are recorded in the evidence
file and the final verification note below.

### Final verification

- Final launcher from `/private/tmp`:429/429 pass,0 skips,65.151s, exit0.
- Protection check:234 registered files (161 L0,70 L1,3 L2); no L0 hash changes.
- All61 current Markdown files have valid relative targets and balanced fences.
- The original230 registry rows/root metadata are unchanged; only four new L1 rows.
- All165 existing implementation/build/dependency paths were byte-compared with
  baseline; only `scripts/run_tests.sh` changed. All six captured scientific
  namespace binding maps remain identical; no historical hash was refreshed.
- Existing four-case Tri-Law audit passes; old numerical evidence remains unchanged.
- Navigation line totals and full test-log SHA-256 receipts are in the evidence
  JSON. This cleanup makes no new real-data, cluster, latency or certification claim.
