# Status

Updated 2026-10-30. Maintained branch: `codex/tls-rag-step5-ann`.
Canonical checkout: `/Users/guanghongxu/Query-Adaptive-Tri-RAG`.

## Repository review and cleanup

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
  protected repair is proposed in the review; pending, not implemented.
- Baseline default environment:429 tests,4 missing-FAISS errors,33 skips.
  Existing fixed-version/native environment:429/429 pass,0 skips (66.634s).
  Final launcher from `/private/tmp`:429/429 pass,0 skips (65.151s). Protection234,
  all61 Markdown checks and six unchanged scientific binding maps pass; receipts
  are in the review and evidence file.
- Next task: user decision on the protected guard/register repair. No new
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
