# TLS-RAG Step 4 start here

Entry provenance: `0e4df85df3cd6bc9427476702a3b90339576636e` must be an ancestor.
The original Step 3-to-Step 4 handoff remains authoritative for startup access.

For this new milestone, read the following new files in order:

1. `docs/TLS_RAG_STEP4_BRIEF.md`
2. `configs/tls_rag_step4_protocol_v1.json`
3. `src/tri_rag_harness/tls_rag_step4.py`
4. `tests/test_tls_rag_step4.py`
5. `scripts/run_tls_rag_step4_readiness.sh`
6. `docs/TLS_RAG_STEP4_READINESS.md`

Use only the already allowed Step 3 files and the established offline test
launcher for upstream interfaces/regressions. Do not read historical documents,
data, logs, archives or prior artifacts. Shared STATUS/plan access is limited
to the marked TLS-RAG Step 4 local sections.

Run `sh scripts/run_tls_rag_step4_readiness.sh /tmp/<new-empty-parent>` from the
repository root using the existing NumPy/SciPy environment. The script performs
focused and full CPU tests twice and compares two freshly generated synthetic
artifact sets. It never downloads dependencies, submits jobs or pushes.

Stop after protocol/readiness verification and commit. The real-data gate is
closed. A future task must obtain exact-path authorization and complete a
pre-data binding before implementing any real adapter. Step 5 is not authorized.
