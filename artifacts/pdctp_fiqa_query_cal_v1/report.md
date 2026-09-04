# PDCTP FiQA query_cal independent audit

Decision: `ACCEPT_QUERY_CAL_FITS_READY_TO_IMPLEMENT_QUERY_TUNE`.

The returned 3,161,188-byte archive matches SHA-256
`686781b787c5a64a00b81996547594abfd5ffcd60107927844a40a552872089c`.
Its cluster log records 146 passing tests and a complete 1,966-query run.

All file and artifact fingerprints, query order, supervision guards,
single-projected-scan records, exact top-10 identities, projected ranks,
retention maps, required-budget maps, four LID candidates, 675 residual base
models, 1,620 full-PDCTP operating points, 405 residual-only operating points,
and the post-calibration protocol state passed structural reconstruction.

A local refit from the returned query records took 1,464.426 seconds and
reproduced both candidate bundles exactly. The fit sets contain 1,933 queries.
All 33 excluded feature profiles have duplicate original distances; 15 already
invalidate pilot LID, while the other 18 duplicate only in the wider pilot
profile. Seven oracle LID failures also have duplicate distances and fall
inside the excluded feature set. These records retain the frozen fallback and
are not silently removed from the query-level artifact.

Independent embedding-to-retrieval replay regenerated the projection exactly
and reproduced all candidate IDs, exact top-10 identities, projected ranks,
retention/required-budget maps, deployable feature vectors, and all fitted
candidate artifacts. The full query-record file is not cross-platform
byte-identical: 60 diagnostic oracle-LID values differ by at most approximately
`1.01e-10`, and one saved projected-distance value differs by approximately
`1.01e-10`. Neither field changes a deployable feature, fit artifact, ranking,
budget target, or operating point. This last-decimal portability deviation is
retained explicitly rather than reported as byte identity.

Only `query_cal` is open. Both calibration fit bundles are frozen; selection is
unset, and `query_tune`, `query_cert`, `query_latency`, and `query_test` remain
closed. No qrel, LLM, approximate index, certification, or latency measurement
was used.

Downstream dry validation corrected one audit-metadata transcription: the
returned `query_cal_records.jsonl` SHA-256 is
`ecc244b05846df73ed2f1ba6f7e9765d745d03a223f6a66fd6b5b5b9f98b35d9`.
The initially recorded value omitted two hexadecimal characters. This
correction matches the original run manifest and changes no returned artifact
or scientific result; the superseding audit fingerprint is
`e3cd09d125a868b685df02b23f0706926fa5786752d863f17bf13d6293de7884`.
