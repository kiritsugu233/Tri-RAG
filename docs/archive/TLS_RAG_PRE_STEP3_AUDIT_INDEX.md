# TLS-RAG pre-Step 3 audit index

Status: logically archived and frozen on 2026-09-05.

This index compresses the Step 1 design and Step 2 implementation history
without moving, deleting, rewriting, or duplicating the original files. Git is
the canonical archive. Keeping the original paths stable avoids breaking
tests, links, hashes, and earlier handoffs while allowing the Step 3 agent to
use a much smaller active reading set.

## Frozen lineage

| Commit | Purpose |
|---|---|
| `cac654ed73f75db92a1d11c09d10e9cd9973a37f` | TLS-RAG Step 1 design freeze |
| `d9be55f39f53fe05e23fc78a45ac303d0a7758dc` | Step 2 assignment brief |
| `a11d983afc660f1cf4ea034525a551f1a7b94f1c` | Restored Step 1 agent brief |
| `f46ce73beccf5fddbf05fd87fdb0911318020add` | Completed Step 2 synthetic skeleton |

All four commits are in the Step 2 ancestry. The Step 2 branch is
`codex/tri-law-sequential-rag-step2`. The complete tree selection covering the
TLS-RAG docs, Step 1/2 briefs, Step 2 module/config/test, and all files below
`docs/` at `f46ce73` has SHA-256
`d028b5e34afa6887c71451aec8e274fbc0fa341fbcf9079bc9143f9c012d0648`.

## Frozen file integrity at Step 2

| File | SHA-256 |
|---|---|
| `AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP1.md` | `f86b5542db7dbd4b13db467fda0991cc7828b8f1711f6e7a33329fbeefa828d7` |
| `AGENT_TRI_LAW_SEQUENTIAL_RAG_STEP2.md` | `8fafa3bd108197f6378698d0af60fbd30a9d4c389f11ae997a804412ac9fb6b0` |
| `docs/TRI_LAW_SEQUENTIAL_RAG_SPEC.md` | `672973a2f00f365064796551e668e63fa50620e18399f8c324b6e83d7914ed2d` |
| `docs/TRI_LAW_SEQUENTIAL_RAG_PROTOCOL.md` | `33a8a052187789ae350c08316863b2102a3a10873a0633e3fa43905a57febd94` |
| `docs/TRI_LAW_SPEC.md` | `f5cfe996dbc58b6ccfc5b0f08dad8372451079b5988a4da1164aadeea852f79d` |
| `docs/TLS_RAG_STEP2_SYNTHETIC.md` | `b379bbd72cfe7330ad245fb6e2293871ca5547b2e4d0ef92a2bfca1c19b1c550` |
| `src/tri_rag_harness/tls_rag_step2.py` | `a6be67c944237ea8f7649f18eb6c6e84f71c94b1546f62cdbb037da7540ed3dd` |
| `configs/tls_rag_step2_synthetic_v1.json` | `257c032be82e945ab757b3c1342d1d87f6cd399667b786008122c358f62ac5aa` |
| `tests/test_tls_rag_step2.py` | `c97033e59789b698e126bdd69094bb93e4493466d41742aa7fe8138ac24f13ee` |
| `docs/IMPLEMENTATION_PLAN.md` | `a8f8d95bd73d4de35c4749dd05e28bb260b26195eaf84e3b01dc37c43f1f3633` |
| `STATUS.md` | `0597e7bc5ff9b839191e34d65b56033ed06bce4b82f49f625f4e3dddcc3ff213` |

The plan and status hashes describe their Step 2 versions. Later append-only
handoff updates are expected to change those two working documents; the
`f46ce73` Git objects remain the immutable originals.

## Step 2 scientific/audit identity

- Config fingerprint:
  `d0506bbfe06c5a1bbb737cbfca8ccc2daacf7eef4a1ba7779df148e8713803af`.
- Fixture fingerprint:
  `60739ff51de1027a21a56cdf734a6d47cc55a320edc90c8e645dcc9e92b317d8`.
- Manifest fingerprint:
  `7d7f2329d56ea608117f8956eeb37926550b9beb5a176fe5f427c598b5d03eee`.
- Phase A decision fingerprint:
  `78c4e4869ffca61a7a82ab014b9a3bd9c1513824c6d7d83ad6c2180f4428c2f3`.
- Phase B supervision fingerprint:
  `a3d3620538c76bcc8a64b17c8dac619ac4b279be13abd43308758b43efda56e4`.
- Local full regression: 180 tests, 179 passed, one optional real-FAISS skip,
  zero failures, 24.002 seconds.
- Genoa replay at the exact Step 2 commit: 180 tests, 179 passed, the same
  optional real-FAISS skip, zero failures, 315.264 seconds.
- Documentation-handoff regression after creating the logical archive and
  minimal read set: 180 tests, 179 passed, the same optional real-FAISS skip,
  zero failures, 24.204 seconds. Step 2 and exact Tri-Law files had zero diff.

## Archive boundary

The following remain available through their paths and Git history but are
inactive for Step 3 unless a concrete compatibility failure requires one
specific file:

- the Step 1 and Step 2 agent briefs;
- the large historical `STATUS.md` and `docs/IMPLEMENTATION_PLAN.md` narratives;
- Raw Tri-Predict v1, PDCTP v2/v3, SciFact, FiQA, certification, real-data,
  embedding, and latency documentation;
- `NEXT_AGENT_PROMPT.md`, old calibrated-agent prompts, reports under
  `artifacts/`, and any `runs/` output; and
- every returned or protected archive and all protected role data.

Step 3 starts from `docs/TLS_RAG_STEP3_START_HERE.md` and
`docs/TLS_RAG_STEP2_TO_STEP3.md`. This audit index is for human provenance and
need not be loaded by the Step 3 agent.
