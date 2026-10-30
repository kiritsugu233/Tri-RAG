# Repository review and cleanup — 2026-10-30

Baseline: `50defc35bc9f42f5b24d2dedf5f45a89e51257ad` on
`codex/tls-rag-step5-ann`, canonical checkout
`/Users/guanghongxu/Query-Adaptive-Tri-RAG`.
[Repository map](REPOSITORY_MAP.md) lists every implementation file and its role;
[evidence](repository_cleanup_evidence.json) records counts, hashes and tests.

The authorized L1 cleanup is complete. **No whole source file was established
as safely deletable.** 方案 A 随后获得用户明确批准并实施；测试套件精简方案 B 未获批准。
原结构清理未修改科学代码；后续方案 A 仅修改获批治理检查器、治理测试和保护登记哈希，
科学源码、科学测试、配置、依赖、原生算法和历史实验保持原样。

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
| C1 | P1 | Guard excludes new C++/header paths and ignored mechanism Python | 原反例保留；已按获批方案 A 修复 |
| C2 | P2 | Human protection register omits40 machine-registered paths | 已按方案 A 补齐原 40 项与新增 4 项 |
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
new L1 navigation/review entries were subsequently synchronized under the approved A repair.

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

**方案 A：已获批准的保护检查与登记修复（2026-10-30）**

实际批准原文：

> 批准执行，执行完毕后更新github上的repo，不要另起branch，直接在step5的分支上提交

该批准承接上一条中文方案 A，仅涉及以下四个 L0 路径；不包含测试套件精简方案 B。
提交父版本为 `61b92d7871ca3da703c137abb4d49cdf9eca487e`，继续使用
`codex/tls-rag-step5-ann`，不创建新分支。

| 路径 | 获批修改与实际实现 |
| --- | --- |
| `scripts/check_code_protection.py` | 增加 C/C++ 源码及头文件后缀识别，并补查被 Git 忽略的 mechanism 实现；保留原有只读和隔离范围 |
| `tests/test_code_protection.py` | 保留原 10 个测试方法，扩展既有 ignored-source 场景，仅新增 1 个参数化方法覆盖真实 Git 状态和排除范围 |
| `docs/CODE_PROTECTION.md` | 从既有机器清单补齐 40 个漏列项目及上一轮新增的 4 个 L1 项，说明实际检查范围 |
| `docs/code_protection.json` | 记录真实批准，仅更新检查器、治理测试、人工表的 3 个获批哈希；自哈希例外保持原样 |

必要性：原检查器对新 `.cpp/.hpp` 等实现可能返回成功，人工保护表也不完整。
L1 说明文档无法修复权威检查器，必须在上述 L0 范围内修复。原始失败反例和
历史清理证据保留在 [evidence](repository_cleanup_evidence.json)，修复结果另记，
不把原失败覆盖成通过。

实现与测试范围：

- 原生后缀 `.c/.cc/.cpp/.cxx/.h/.hh/.hpp/.hxx`，大小写不敏感。
- Git 清单负责已跟踪及未忽略的未跟踪代码；五个实现目录的递归检查补查 ignored 代码。
- 临时 Git 仓库测试覆盖 10 个后缀示例 × 3 种 Git 状态，共 30 个组合；
  既有测试扩展覆盖五个目录的 ignored Python 源码。
- 核对被忽略的 runs/data/.venv/archives 不进入清单，且清单发现阶段不读取载荷字节。
- 所有旧治理测试方法保留；治理测试 10→11，全量方法数 429→430。
  没有为了缩减数量而删除旧断言，也没有为每种后缀独立增加方法。

影响边界：不修改 Tri-Law 公式/容差、检索排序、候选集合、校准、停止决策、模型、
配置、数据角色、科学缓存内容、历史结果及现有等级。当前五组 Step5 来源绑定
与修复前一致；没有改写旧实验哈希。测试大规模整理仍需独立方案 B。

**兼容性更正：** 原方案“所有来源绑定不变”的表述遗漏了 A800 来源清单中的保护脚本。
最终逐路径比较确认，Step4 A800 的 22 项中只有 `scripts/check_code_protection.py`
因本次明确批准的编辑改变 SHA；旧回放包在当前代码下可能被严格校验拒绝。
历史复现继续使用其原精确提交，旧包、旧哈希和验收器不变；本次不迁移任何实验包。
这一来源身份变化不表示科学算法改变，也不构成新的实验/数据访问授权。

验证：使用最终新测试在隔离临时副本中运行原检查器，确认旧缺陷导致失败；
修复后治理测试通过。随后运行包含现有原生库的完整离线套件，核对科学绑定、
其余 L0 哈希、人工/机器清单一致性、Markdown 链接和 diff；结果追加于下文。
检查器仍是审查辅助工具，不能证明人类批准或充当 OS 权限控制。

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

### 原结构整理验收（方案 A 之前）

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

### 方案 A 最终验收（2026-10-30）

- 用最终新测试在临时副本执行原 `61b92d7` 检查器：11 个测试方法中 2 个失败，
  参数化场景记录 32 条失败；旧缺陷可复现。原审查反例证据未被覆盖或删除。
- 修复后治理测试 11/11 通过；固定版本/四个原生库的完整套件 430/430 通过，
  无跳过、64.757 秒。命令同上节固定环境的 `sh scripts/run_tests.sh`。
- 当前 234 项登记（161 L0/70 L1/3 L2）保护检查通过，人工表覆盖全部 234 项。
  仅三个获批内容哈希和当前批准记录更新，原有等级及批准历史保留。
- 比较全部 165 个实现/构建/依赖路径，仅保护检查器及治理测试变化。
  五组 Step5 绑定完全一致；A800 仅获批保护脚本的来源哈希变化，兼容影响如上。
- 完整失败/成功日志 SHA-256、覆盖组合和来源差异保存在原 evidence 的
  `approved_guard_repair` 字段，原清理证据保持其历史含义。
- 大规模测试精简方案 B 未执行；无新实验、集群、模型下载或历史角色访问。

本次交付补充检查：全部 61 份 Markdown 相对链接/围栏及 `git diff --check` 通过。

## 测试精简方案 B（独立提案，未批准）

用户要求先列出逐项保留/合并/退役及覆盖依据，再批准。
[中文方案 B](TEST_SUITE_SIMPLIFICATION_PROPOSAL.md)包含430方法完整处置表、
实际断言/调用、M01精确四场景映射、来源绑定影响及四个实施/治理路径。
拟将198项退出日常但保留完整回归，2个缓存检查合为1个参数化方法；
日常231/完整429，永久删除0项。当前仍是430项，尚未修改测试或执行入口。
