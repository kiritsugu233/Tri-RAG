# Query-Adaptive Tri-RAG

Research harness for fixed-dimension Gaussian projected retrieval, query-adaptive
candidate budgets, exact reranking, and versioned ANN mechanism studies.

Current stage: **TLS-RAG Step 5**, branch `codex/tls-rag-step5-ann`.
The latest continuation v2 study is complete. The tested learned/Tri stopping
paths did not achieve net latency gain; deployment safety is incomplete.

- [Start here](START_HERE.md): current state and reading order.
- [Repository map](docs/REPOSITORY_MAP.md): code families, dependencies and file roles.
- [Status](STATUS.md): latest results, validation and links to historical records.
- [Agent rules](AGENTS.md) and [protection register](docs/CODE_PROTECTION.md).
- [2026-10-30 cleanup review](docs/REPOSITORY_CLEANUP_REVIEW.md): findings and the approved guard repair.

Offline tests, from an environment with the required dependencies:

```bash
python3 scripts/check_code_protection.py
sh scripts/run_tests.sh
```

The Step 5 suite includes FAISS-dependent tests. The complete native suite also
requires the fixed-version native libraries described in
[the validation instructions](docs/REPOSITORY_CLEANUP_REVIEW.md#validation).
Set `TRI_RAG_TEST_PYTHON=/absolute/path/to/python` to select an existing environment.
The launcher installs nothing and runs the same unittest suite.

Historical experiments, frozen contracts and failed results are retained for
replay. Opening old roles or running another experiment requires its task scope;
this landing page is navigation, not an execution request.
