# FiQA query_cal fitting gate

This gate is the first protected-outcome access in the fresh PDCTP protocol.
It is implemented by `tri_rag_harness.pdctp_fiqa_query_cal` and frozen by
`configs/pdctp_fiqa_query_cal_v1.json`.

## Scope

The runner validates the protocol freeze, initial five-role state, role
assignments, qrel-free dataset manifest, embedding request/cache, and accepted
embedding audit before it opens any calibration outcome. It then opens the
complete 1,966-ID `query_cal` role in its frozen order. No partial role or
reordered role is accepted.

The only allowed supervision is:

- oracle exact LID from original-space nearest-neighbor distances;
- exact original-space top-10 identities; and
- realized embedding retention of those identities in the projected ranking.

The runner does not open or parse a qrel member. It does not access
`query_tune`, `query_cert`, `query_latency`, or `query_test`, select a policy,
run a certificate or latency benchmark, use an approximate index, or invoke an
LLM.

## Numerical contract

Corpus and `query_cal` E5 vectors are read as the already audited normalized
float32 arrays and converted to float64 for exact computation. The frozen dense
Gaussian projection uses seed `83047`, dimension `192`, and NumPy standard
deviation `1/sqrt(192)` (variance `1/192`). Projected vectors are never
renormalized. Original and projected search both use squared L2 with
lexicographic document-ID ties.

Each query batch produces one full projected-distance block. The pilot top-64
and the projected ranks of the exact original top-10 are derived from that same
block, so pilot and expansion do not trigger a second projected scan. Full
projected rankings are not persisted. Query-level records retain the pilot
profile, exact top-10 identities, their projected ranks, every frozen-grid
retention value, and the smallest required training budget.

Because `k_gt=10`, retention changes in increments of `0.1`; consequently the
predeclared training levels `0.95`, `0.98`, and `1.0` all require 10/10
retention. The runner records and checks this fact without changing or dropping
any preregistered level.

## Candidate fitting and storage

All four LID regularization candidates are fit on valid `query_cal` pilot
features with valid oracle LID targets. Budget residuals are fit for every
predeclared Raw Tri threshold, LID candidate (or raw-pilot ablation), training
level, quantile, and regularization. Safety offsets are preregistered operating
parameters, not separately optimized fits.

The real-data residual solver minimizes the same exact pinball-plus-L2
objective as the foundation while avoiding the foundation solver's two slack
variables per query. Zero-regularization candidates use a sparse HiGHS linear
program; positive-regularization candidates use a coefficient-only SLSQP
formulation with an analytic pinball subgradient. The legacy foundation solver
name and behavior remain the default and are unchanged.

To avoid repeating roughly 1,966 ID strings in every residual model, the
candidate bundle stores the ordered fit IDs once. Every compact model can be
reconstructed by restoring that shared list and must pass the ordinary
calibrator schema/fingerprint loader. Each safety-offset operating point stores
the fingerprint of its fully reconstructed effective calibrator. The frozen
suite contains 1,620 full-PDCTP and 405 residual-only operating points. No
candidate is selected in this gate.

## Outputs

The runner writes eight artifacts in a new output directory:

- `query_cal_access.json`;
- `projection.json`;
- `query_cal_records.jsonl`;
- `lid_calibrator_candidates.json`;
- `residual_calibrator_candidates.json`;
- `protocol_state_after_query_cal.json`;
- `manifest.json`; and
- `report.md`.

Portable artifacts contain no timestamps or timings. The terminal gate
decision is `QUERY_CAL_FITS_FROZEN_READY_FOR_QUERY_TUNE`; it means the next
runner may open `query_tune`, not that any policy has been selected or any
scientific claim has passed.

## Command

Run from the repository root with the accepted cluster cache path substituted
for `<embedding-cache>`:

```bash
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

python3 -m tri_rag_harness.pdctp_fiqa_query_cal \
  --config configs/pdctp_fiqa_query_cal_v1.json \
  --real-protocol-config configs/pdctp_fiqa_real_protocol_freeze_v1.json \
  --protocol-freeze artifacts/pdctp_fiqa_real_protocol_v1/protocol_freeze.json \
  --protocol-state artifacts/pdctp_fiqa_real_protocol_v1/protocol_state.json \
  --role-assignments artifacts/pdctp_fiqa_real_protocol_v1/role_assignments.json \
  --embedding-audit artifacts/pdctp_fiqa_e5_v1/embedding_audit.json \
  --embedding-config configs/pdctp_fiqa_e5_base_v2_embeddings.json \
  --prepared data/pdctp-fiqa-text-375414-a \
  --embedding-cache <embedding-cache> \
  --output runs/pdctp-fiqa-query-cal-<job-id>
```

## Accepted execution

Slurm allocation `375414` on `a100-0` executed the committed implementation
over the complete frozen role. The cluster ran 146 tests with zero failures or
skips and wrote 1,966 query records. The returned 3,161,188-byte archive has
SHA-256
`686781b787c5a64a00b81996547594abfd5ffcd60107927844a40a552872089c`.
The manifest and post-calibration state fingerprints are
`140c3a5f9ac168e222f9ff7e7cd3edb7f75879a40aa358158a1cfac4b77b2be6`
and
`2fe1eeb55180198165b6d1b46f35bf4214b8711cda554e7a2ca211f5b193f481`.

Independent audit rehashed every file; replayed the projection, exact
retrieval, retention targets, role guards, and state transition; reconstructed
all 2,025 operating points; and performed a complete refit from the returned
records. The four-candidate LID bundle and 675-base-model residual bundle were
value-identical to the returned artifacts, with fingerprints
`4526c8b752325e3cae040d8b450c76cd0df77571b9e6d6080bd1a53ba4a56a1e`
and
`ed6cf0f7056fc1b7345b5303a7ad71815a2b8df6677c6c1c2c0e231bbf9c9f31`.
The fit set contains 1,933 valid feature/target records. The audit records 33
duplicate-original-distance feature failures, 15 duplicate-distance pilot-LID
failures, and seven duplicate-distance oracle-LID failures; the frozen fallback
handles them without changing role membership.

The projection, every retrieved identity and rank, retention/required-budget
map, deployable feature, target, and fitted artifact reproduced exactly. The
full record file is not byte-identical across the cluster and Mac: 60
diagnostic oracle-LID values and one saved projected-distance value differ only
in the last decimal, with maximum absolute differences approximately
`1.01e-10`. This does not affect a policy input or fit, and the deviation is
explicitly retained in the checked-in audit instead of being mislabeled as
byte identity.

The accepted independent-audit fingerprint is
`1e2e09d9bbc794682190ec0228ab4ce01f41aedf9a399d08a00aca80a84594a0`.
This authorizes implementation of the `query_tune` runner only. `query_tune`
has not been opened, selection remains unset, and every later role remains
closed. The post-audit offline suite reports 146 passes, one optional
real-FAISS skip, and zero failures across 147 tests.
