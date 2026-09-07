# Next agent startup

Use `/Users/guanghongxu/Query-Adaptive-Tri-RAG` for local repository work.
Inspect branch, HEAD and tracked/untracked changes; preserve user files.

Read in order:

1. [AGENTS.md](AGENTS.md), including the L0 approval rule.
2. [START_HERE.md](START_HERE.md), for current status and task routing.
3. [Code protection registry](docs/CODE_PROTECTION.md), for the exact files
   relevant to the user's task; run `python3 scripts/check_code_protection.py`.
4. The current task's explicitly named brief. For TLS-RAG Step 4 retention v3,
   use [the Step 4 entry](docs/TLS_RAG_STEP4_START_HERE.md).

Continue on `codex/repo-review-code-protection` at its latest reviewed-and-repaired
HEAD, including Tri-Law numerical v2. Do not create a new branch for every step.
`248c29e` and `c40b7b0` are historical experiment/review-only snapshots.
Read `docs/TRI_LAW_NUMERICAL_FIX.md` before reusing old source-bound bundles.
No real acceptance run or Step 5 is authorized by this numerical repair. Do not restart calibrated v2,
TLS-RAG Step 2, or Step 3 from old prompts. Historical reading allowlists apply
to their original tasks; they do not override root governance or the user's
explicit current task, including a repository-wide review.

Original numerical defects and review scope are recorded in
[the repository review](docs/REPOSITORY_REVIEW.md); their approved repair is in
[the numerical repair record](docs/TRI_LAW_NUMERICAL_FIX.md). L0 protection means
change control, not a claim that the code is defect-free. Before proposing
an L0 fix, explain its necessity, exact affected paths, compatibility and
validation plan, then wait for explicit user approval before editing.
