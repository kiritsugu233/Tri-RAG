# TLS-RAG canonical local and cluster workflow

The user fixed the local working directory for every subsequent step to:

```text
/Users/guanghongxu/Query-Adaptive-Tri-RAG
```

Run local implementation, tests, Git commits and GitHub synchronization there.
Do not use `.codex/worktrees/...` for new local steps without a new explicit
exception. Preserve existing untracked files and user changes. The single continuing
Step 4 branch is `codex/tls-rag-step4-retention-v3`, fast-forwarded to include
reviewed baseline `62377d854ddd183925c4b62ca81653e9c14d9c34` and the approved Tri-Law
numerical v2 repair. Inspect actual HEAD. Preserve `codex/repo-review-code-protection`
at that baseline as a backup. Do not push ongoing work to it or create another
branch for each Step 4 update; the continuing branch name stays the same across
versions. `248c29e` is only the historical pre-review experiment commit.
Historical initial readiness commit:
`155070d8df96a773d83a650db84e6ec69d12c51e`. The remote is
`https://github.com/kiritsugu233/Tri-RAG.git`.

For every authorized milestone, validate the exact checkout, complete tests,
commit and push the continuing branch without force, then compare the local
commit with the GitHub branch HEAD. The standing user instruction authorizes
this GitHub synchronization. The default branch is not implicitly changed.

The Step 4 replay script refuses other local macOS working directories. From
the canonical directory run it with a new, absent output parent:

```bash
sh scripts/run_tls_rag_step4_readiness.sh /private/tmp/tls-rag-step4-local-NEW
```

The script still allows a Linux cluster checkout. GitHub is the synchronization
source: fetch the named step branch, check out the reviewed full commit in a
clean/dedicated cluster checkout, and verify that the fetched branch contains
that commit. Avoid reset/clean or overwriting local cluster changes.

Use the existing CPU environment with NumPy 1.26.4 and SciPy 1.13.0; do not
install dependencies or download data inside the test job. A Genoa CPU node is
appropriate. The initial replay request is one CPU, 4 GB and ten minutes with
no GPU. Set BLAS/OpenMP thread counts to one. Use the site's actual partition
and account. No cluster path, partition or account is inferred from this file.

The test job runs focused/full CPU suites twice and compares its two fresh
synthetic output directories. Keep the reviewed commit, environment, Slurm job
ID, output and exit code. Cluster success requires the actual job result.
The v1 readiness real-data gate remains closed. Separately authorized v2/v3
probe work follows its versioned brief and exact paths; this workflow does not
authorize additional data access or Step 5. Read root AGENTS and run
`python3 scripts/check_code_protection.py` before edits and at handoff.
