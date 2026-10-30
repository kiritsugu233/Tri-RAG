# TLS-RAG local and cluster workflow

Updated 2026-10-30. Authoritative rules: [AGENTS](../AGENTS.md).
Use `/Users/guanghongxu/Query-Adaptive-Tri-RAG` and the continuing
`codex/tls-rag-step5-ann` branch. Inspect branch, HEAD and tracked/untracked
status before editing; preserve user changes and existing archives.
The review backup `codex/repo-review-code-protection` remains at
`62377d854ddd183925c4b62ca81653e9c14d9c34`.

After an authorized milestone passes required checks, commit/push the continuing
branch and verify its remote HEAD, per the standing synchronization instruction.
Remote: `https://github.com/kiritsugu233/Tri-RAG.git`. Read current
[START_HERE](../START_HERE.md) and [STATUS](../STATUS.md) for the actual stage.

## Local verification

```bash
python3 scripts/check_code_protection.py
sh scripts/run_tests.sh
```

The launcher resolves its repository and accepts `TRI_RAG_TEST_PYTHON` for an
existing interpreter. Complete Step 5 verification needs fixed FAISS/sklearn and
native libraries; see [validation](REPOSITORY_CLEANUP_REVIEW.md#validation).
No dependency installation or model/data download occurs in the test launcher.

## Cluster replay

Cluster execution uses a separate Linux checkout synchronized through GitHub
at the reviewed full commit. Preserve cluster modifications; verify exact SHA,
environment, command, exit code and actual result. The present route1/2 native
builders are macOS scoped, not validated Linux launchers.
A new remote experiment follows the user's specific task and versioned contract.

Historical Step 4/Genoa/A800 directory/branch/commit-and-push arrangements are
retained at [the pre-cleanup workflow](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/docs/TLS_RAG_LOCAL_WORKFLOW.md), with exact execution
records in [historical STATUS](https://github.com/kiritsugu233/Tri-RAG/blob/50defc35bc9f42f5b24d2dedf5f45a89e51257ad/STATUS.md). Those retired branch instructions
and milestone-only commit arrangements are superseded by current AGENTS.
