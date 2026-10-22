# Next agent startup

Use `/Users/guanghongxu/Query-Adaptive-Tri-RAG` and the existing
`codex/tls-rag-step4-retention-v3` branch. Preserve user files and frozen history.

Read [AGENTS.md](AGENTS.md), [START_HERE.md](START_HERE.md), then the
[Step 5 handoff](docs/TLS_RAG_STEP5_HANDOFF.md) and relevant protection entries.
Run `python3 scripts/check_code_protection.py` before proposing edits.

Agent creation/dispatch is deferred until the user's next explicit instruction.
When dispatched, submit only the short Step 5 ANN experiment proposal and wait
for the user's approval before implementation. Do not execute old task prompts.
