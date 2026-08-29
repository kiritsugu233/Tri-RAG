# FiQA PDCTP real-data protocol freeze

## Scope and decision

This gate converts the audited FiQA role-capacity witness into the actual
Pilot-Distance Calibrated Tri-Predict v2 preregistration. It freezes dataset
handling, all five ordered role identities, E5 continuity, projection and
retrieval parameters, candidate grids, selection/certification rules, latency
blocks, hardware compatibility failures, and stochastic seeds before any
embedding or role outcome is opened.

Decision: `READY_FOR_DATASET_AND_EMBEDDING_AUDIT_ONLY`. This permits only
canonical FiQA preparation and construction plus independent audit of the new
E5 cache. It does not permit calibration fitting, tune selection, protected
outcome access, retrieval evaluation, latency measurement, an LLM, or an
approximate index.

## Bound inputs and identities

The freeze validates all three inputs before writing output:

- source-audit fingerprint
  `99211145fdf7976fe58072d516c4f6c95c33ed880afe63189cb8e464aac15aa5`;
- role-witness fingerprint
  `dfccd57b074d8faa11a410f5d94a970133e29fcadac9b08e4972d747249fcaff`;
- certification power-plan fingerprint
  `f1bddbc072143ec13b23785775d7b7ebf97913146eb05022e7d43f0d12a644a2`.

It independently verifies each file SHA-256, every embedded fingerprint, the
source-audit-to-witness link, all ordered role hashes, and the exact primary
hypothesis family. A mismatch fails before creating the output directory.

The checked-in config fingerprint is
`47c602c777e9e4589597ae996a7d1459407ae916b376854699569c115ebdfc41`.
The resolved protocol fingerprint is
`cb3ef70f3ffc801c248f3269e0807480f0ee5a51cde41a52573a03f228a42368`.

## Dataset and five roles

The archive remains the pinned 57,638-item BEIR FiQA corpus with 6,648 eligible
external queries. The 38 source items whose title and text are both empty are
retained. Their deterministic prepared text becomes `[EMPTY_DOCUMENT]`, so the
exact E5 input is `passage: [EMPTY_DOCUMENT]`; silent deletion is forbidden.
All original qrels remain in force, including the 35/2/1 train/dev/test positive
references to those empty source items.

The final role counts are:

| Role | Count |
| --- | ---: |
| `query_cal` | 1,966 |
| `query_tune` | 1,967 |
| `query_cert` | 1,567 |
| `query_latency` | 500 |
| `query_test` | 648 |

`role_assignments.json` contains the exact ordered IDs and hashes. The source
audit found no normalized query-text duplicates, and the final roles are ID-
and normalized-text-disjoint. The assignment fingerprint is
`ae884e0001d92ad11ddc1e420ece5412846454864331843a25a7e5cccf445dfe`.
The initial guard-state fingerprint is
`6f1eb5a4ecef7cf4a0413c13de82cd31dc2024b1593ebd86c57c153c276f45b9`;
calibration, certification, latency, and test are all closed, no selection is
frozen, and there are no fit artifacts.

## Embedding and retrieval contract

The new cache must use `intfloat/e5-base-v2` at revision
`f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, float32 inputs/outputs, eager
attention, deterministic algorithms, disabled TF32, and L2 normalization before
projection. Corpus and query prefixes remain `passage: ` and `query: `.

Retrieval freezes:

- dense Gaussian projection entries with variance `1/m_prime`;
- fresh projection seed `83047` and `m_prime=192`;
- no projected-vector renormalization;
- squared L2 in normalized original and projected spaces;
- lexicographic document-ID tie breaks;
- `k_gt=10`, `k_ctx=5`, `M_pilot=64`, `s_lid=32`, and 16 minimum LID neighbors;
- one projected scan shared by pilot and expansion; and
- common coordinate work
  `(corpus_size + embedding_dimension) * m_prime + embedding_dimension * M`.

The deterministic binary-interleaved geometric budget grid is:

```text
[64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 57638]
```

The last value is the full corpus, so failure remains visible rather than
triggering an unregistered budget expansion.

## Candidates, selection, and certification

The common policy suite contains fixed, monotone-binned, Raw Tri-Predict,
LID-calibration-only, budget-residual-only, and full PDCTP. Fixed candidates use
all 21 budgets. Monotone candidates use 4/6/8 bins and frozen target levels.
The Cartesian full-PDCTP grid contains exactly 1,620 tuples from five Raw Tri
thresholds, four LID regularizers, three residual training levels, three
quantiles, three residual regularizers, and three nonnegative safety offsets
corresponding approximately to multiplicative factors 1.00/1.05/1.10.

Only `query_tune` may select. Candidates must first meet a 0.95 tune retention
lower-bound target and 0.02 candidate/final evidence noninferiority tolerances,
then minimize common coordinate work, breaking ties by lower mean budget and
canonical fingerprint. Comparator cost claims require the comparator to meet
the same quality constraints. The shuffled-profile diagnostic remains tune
only and uses seed `83059`.

Certification uses all 1,567 `query_cert` IDs and the already checked-in
worst-case power plan. Its six primary hypotheses use Bonferroni family-wise
alpha 0.05: absolute 0.95 embedding retention; candidate and final evidence
noninferiority at margin -0.02 against fixed; and normalized-budget superiority
against eligible fixed, monotone, and Raw Tri comparators. Any failure is
terminal with no retuning or budget-grid expansion.

## Paired latency contract

Latency remains closed until scientific certification is terminal. It then
uses all 500 label-free `query_latency` IDs, exact FAISS CPU first and one exact
A100 80 GB backend second, one thread, single-query warm-index execution, 10
warmups, 30 randomized paired repetitions, and method-order seed `83071`.

The three primary paired mean-latency comparisons are PDCTP versus eligible
fixed, monotone, and Raw Tri policies. They use one-sided paired Student-t upper
bounds with Bonferroni family-wise alpha 0.05. p50/p95/p99 are descriptive only.
All stage times, work/byte counts, RSS/GPU memory, and setup/index-build costs
must be saved separately.

The tested FAISS GPU contract caps `k` at 2,048 and consumes a 64-item stable-
boundary guard, so the maximum compatible selected budget is 1,984. If any
required selected policy exceeds it, the GPU latency gate fails terminally; a
smaller substitute budget is forbidden.

## Command, artifacts, and tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m \
  tri_rag_harness.pdctp_real_protocol \
  --config configs/pdctp_fiqa_real_protocol_freeze_v1.json \
  --source-audit artifacts/pdctp_fiqa_source_audit_v1/source_audit.json \
  --role-witness artifacts/pdctp_fiqa_source_audit_v1/role_feasibility_witness.json \
  --power-plan artifacts/pdctp_network_free/power_plan_v1.json \
  --output artifacts/pdctp_fiqa_real_protocol_v1
```

The four checked-in artifacts are `protocol_freeze.json`,
`role_assignments.json`, `protocol_state.json`, and `report.md`. Repeated local
generation is byte-identical. The complete network-free suite ran 135 tests:
134 passed, one optional real-FAISS conformance test skipped, and zero failed.

The next gate may prepare the canonical dataset and build a new E5 cache bound
to this freeze. It must independently verify source IDs, empty-document
replacement count and identities, role hashes, formatted text hashes,
normalization, array shapes/dtypes/norms, model snapshot identity, truncation,
and complete cache fingerprints before any `query_cal` fit is opened.
