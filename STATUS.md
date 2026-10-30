# Status

Updated 2026-10-30. Maintained branch: `codex/tls-rag-step5-ann`.
Canonical checkout: `/Users/guanghongxu/Query-Adaptive-Tri-RAG`.

## 已批准的方案 A：保护检查与登记修复（2026-10-30）

- 实际批准：“批准执行，执行完毕后更新github上的repo，不要另起branch，直接在step5的分支上提交”。
- 父提交 `61b92d7871ca3da703c137abb4d49cdf9eca487e`；继续 Step5 分支。
- 修复 C/C++/头文件和 ignored mechanism 漏检；补齐人工表 44 项，已有等级不变。
- 原 10 个治理测试方法保留，仅新增 1 个参数化方法，覆盖 30 个原生文件/Git 状态组合；
  ignored Python 覆盖扩展为五个目录，并验证被忽略的载荷目录不被扫描。
- 四个获批 L0 文件范围、批准记录和验收见
  [方案 A](docs/REPOSITORY_CLEANUP_REVIEW.md#proposal-a-protected-guard-and-register-repair)。
  原失败证据保留；科学代码/测试/配置/依赖不变。五组 Step5 绑定一致；
  A800 来源清单含保护脚本，其身份随获批编辑改变，旧包按原提交复现，不刷新旧哈希。
- 原检查器执行最终新回归可复现失败；修复后治理测试 11/11 通过。
- 完整固定环境命令：`TRI_RAG_TEST_PYTHON=/private/tmp/step5-ann-enhancement-env-20261028/bin/python sh scripts/run_tests.sh`，
  同时设置审查记录列出的四个 STEP5 原生库环境变量；430/430 通过，无跳过，64.757 秒。
- `python3 scripts/check_code_protection.py` 通过，234 项；人工表完整，
  仅三个获批哈希变化。日志/失败证据及验收记录已保存。
- 大规模测试精简方案 B 未获批准，未执行。下一任务等待用户指示；按既定要求同步本次 Step5 提交。

## 上一轮：仓库结构审查与整理


- Review baseline: `50defc35bc9f42f5b24d2dedf5f45a89e51257ad`.
  [Review/approval proposal](docs/REPOSITORY_CLEANUP_REVIEW.md),
  [file/dependency map](docs/REPOSITORY_MAP.md),
  [evidence](docs/repository_cleanup_evidence.json).
- Shortened startup/status/plan navigation; retained detailed history in the fixed
  pre-cleanup Git snapshot. Corrected the outdated next-agent state and retired
  Step 4 branch instruction. Added a repository landing page and lifecycle map.
- Test launcher accepts `TRI_RAG_TEST_PYTHON` and resolves the repository from its
  own location. Suite, assertions and native tests are unchanged.
- No runtime source/config/test/dependency removed or modified; no L0 hash refreshed.
  Old experiments still bind those paths. User archives/report_work/runs preserved.
- Confirmed checker blind spots for new native files/ignored mechanism code;
  human protection register omits40 machine-registered paths. Exact four-path
  protected repair was subsequently approved and implemented as A above.
- Baseline default environment:429 tests,4 missing-FAISS errors,33 skips.
  Existing fixed-version/native environment:429/429 pass,0 skips (66.634s).
  Final launcher from `/private/tmp`:429/429 pass,0 skips (65.151s). Protection234,
  all61 Markdown checks and six unchanged scientific binding maps pass; receipts
  are in the review and evidence file.
- That checkpoint next requested approval for A, now received/executed above. No new
  retrieval, policy fitting, data access, cluster run or scientific claim follows.

## Latest scientific checkpoint — continuation v2

Scientific code `08ed4ce`; exact prior consent:
“批准第26节全部范围，按A→B→C执行”.
[Journal sections27–31](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md)
hold the full process, commands, source/selection/model checks and limitations.

- Historical development:768 queries,4 nested folds,5 timing repeats.
  All768 FAISS/boundary resume checks pass;429 offline tests passed at that milestone.
- Full original ANN=1.168709ms; S=1.248047/G=1.393090/T=1.629399/
  T0=1.657511/selectiveTri=1.415616ms. Current learned/Tri paths fail net gain.
- T retention=.9971354 vs original=.9984375; its delta interval crosses−.002
  tolerance, with one extra5/10-neighbor tail loss. No trained-support/OOD
  abstention fallback; execution is complete, deployment safety acceptance is not.
- Preserve resumable backend and failures; no automatic retry/threshold change,
  adapter/1M/routes3/4, SSH/GPU or closed-role access. No fresh independence or
  formal certification. Route4 remains planned; route3 has not run.
- Latest raw evidence: `runs/ann-continuation-v2-20261030-v1`,1658 previously
  verified files/2,149,374,244B. This cleanup did not re-audit or rewrite them.

## Historical records

Detailed status/approvals/commands from all previous stages are retained in
[STATUS at the cleanup baseline](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/STATUS.md), including Step 4 replay,
Step 5 cloud/local storage receipts and original failures. Retrieve offline with:

```bash
git show 50defc35bc9f42f5b24d2dedf5f45a89e51257ad:STATUS.md
```

The unchanged [exploration journal](docs/TLS_RAG_STEP5_ANN_ENHANCEMENT_EXPLORATION.md)
and versioned contracts remain local. [Milestone index](docs/IMPLEMENTATION_PLAN.md)
links the historical plan. Historical text describes its original authorization;
read current AGENTS and the user's task before acting.

## Branch-retirement provenance — 2026-10-23

The eight remote references below were previously retired with user approval
and ancestry/open-PR checks. This cleanup performs no additional branch deletion.
Their complete original execution receipt is in the fixed status snapshot above
and `docs/code_protection.json`.

| Retired GitHub branch | Retained tip (full SHA in protection registry) |
|---|---|
| `codex/calibrated-tri-predict-v2` | `26079ce67793844f649d7df16ce818c3b539529f` |
| `codex/calibrated-tri-predict-v3` | `f94c1aa63df459b4d2281e296f400942a8552c00` |
| `codex/tls-rag-step3` | `3913ded9b786d6b6b15c734c750bd13d9c58e12c` |
| `codex/tls-rag-step4-readiness` | `fc38d928e428dc8c10a3ea16e5ec5db5ccfbf72a` |
| `codex/tls-rag-step4-real-probe` | `c08cd3b6800888c8d2a5a5f8c58a77d883a84ede` |
| `codex/tls-rag-step4-retention-v3` | `4f836375c8e45361ef7e3229e66f2c1735abee1b` |
| `codex/tri-law-sequential-rag-step2` | `f46ce73beccf5fddbf05fd87fdb0911318020add` |
| `codex/tri-law-sequential-rag-v1` | `cac654ed73f75db92a1d11c09d10e9cd9973a37f` |
