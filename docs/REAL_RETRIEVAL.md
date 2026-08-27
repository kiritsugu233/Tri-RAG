# Real SciFact retrieval baseline

## Scope and frozen inputs

The first real retrieval gate is an exact original-space quality reference on
`query_tune` only. It does not project vectors, choose `m_prime`, fit a policy,
run certification, or inspect retrieval outcomes for `query_cert` or
`query_test`. The config loader rejects either protected split.

`configs/real_scifact_original_exact_tune.json` freezes:

- repaired dataset fingerprint
  `4a73586d3a29a0567287e501ac3c06c998af661cdc74dbc589e7525a7924f903`;
- accepted E5 cache fingerprint
  `2ec53ce38e226129ba0feffcd28ba1da1081e0627ad8e54f4a60e430c341e914`;
- embedding config and request fingerprints;
- normalized original-space squared L2 in NumPy float64;
- lexicographic stable-document-ID tie breaking;
- cutoffs 1, 5, and 10, with `k_ctx=5` and `k_gt=10`;
- query batch size 32 and the frozen 403-query tune ID hash.

The baseline-config fingerprint is
`ff675fed06fc6506ed68a83426a021ee53a701f06af4144351b2172c2dbc19f6`.
Before search, the runner revalidates the dataset artifacts and the complete
embedding cache without loading the text model, then checks ordered corpus and
query IDs against the prepared JSONL rows.

## Command

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.real_original_baseline \
  --config configs/real_scifact_original_exact_tune.json \
  --dataset data/prepared/scifact-dedup-v2 \
  --embedding-config configs/real_scifact_e5_base_v2_embeddings.json \
  --embedding-cache data/embeddings/scifact-e5-base-v2-dedup-v2 \
  --output runs/scifact-original-exact-tune
```

The result directory contains:

- `per_query.jsonl`: query ID, positive qrels, exact top-10 document IDs, and
  evidence hit/recall/nDCG at every frozen cutoff;
- `summary.json`: aggregate tune-only evidence metrics;
- `manifest.json`: complete input identities, result artifact hashes, and a
  deterministic result fingerprint;
- `timings.json`: non-result index/search timing and work counts;
- `report.md`: concise interpretation-limited table.

Timings are intentionally excluded from the result fingerprint. This permits
byte comparison of retrieval results across machines without treating hardware
latency as scientific identity.

## Local correctness run

The independently accepted A100 arrays were consumed from the audit archive on
the Mac without copying them into the repository. Two fresh output directories
matched byte for byte for `manifest.json`, `per_query.jsonl`, `summary.json`,
and `report.md`. The deterministic result fingerprint is
`2921f39dc051bc3331da8bf9b0ddc6c584dcd1f043099d8dda353653a1926b1c`.

On 403 tune queries, the exact original-space reference produced:

| cutoff | evidence hit | evidence recall | nDCG |
| ---: | ---: | ---: | ---: |
| 1 | 0.630273 | 0.606493 | 0.630273 |
| 5 | 0.803970 | 0.786849 | 0.713828 |
| 10 | 0.866005 | 0.850124 | 0.735501 |

These are tune-only labeled-evidence results for the frozen E5 model. They are
not embedding-neighbor retention, a policy certificate, a test result, or an
answer-quality claim. A Genoa rerun should reproduce the result fingerprint;
its timings are reported separately as a systems observation.

## Genoa reproduction

Slurm job `373780` on Genoa reproduced the deterministic result fingerprint
`2921f39dc051bc3331da8bf9b0ddc6c584dcd1f043099d8dda353653a1926b1c`
twice at commit `37f68fc`. The two runs are byte-identical for the manifest,
403 query-level records, summary, and report. Their non-identity timings were
`0.5372` and `0.5336 ms/query`; this difference is normal measurement noise.
All 71 then-current tests passed apart from the expected optional real-FAISS
skip in the NumPy environment. The returned audit archive has SHA-256
`c91a402eea55de127381f82f21f3a988ced679959b88eed868604292fda1af6d`.

An independent local audit recomputed every result-artifact hash, the result
fingerprint, and every evidence aggregate from `per_query.jsonl`. The archive
code matches commit `37f68fc`; only Python `3.9.6` locally versus `3.9.23` on
Genoa changes the non-result manifest fingerprint.

## Fixed-dimension tune sweep

The next gate is implemented by
`configs/real_scifact_fixed_dimension_tune.json` and
`tri_rag_harness.real_dimension_sweep`. Its config fingerprint is
`3265e303c5249a6b90868f5234d333eca3f1fc4bc28c12cdb710382e2b71eabd`.
Before observing any projected real-data result, it froze:

- `query_tune` as the only accessible split;
- dense Gaussian entries with variance `1/m_prime`, base seed `27011`, and no
  projected-vector renormalization;
- nested, same-seed candidate projections at dimensions
  `[16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]`;
- `k_ctx=5`, `k_gt=10`, `M_pilot=32`, and the common grid
  `[32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072,
  4096, 5183]`;
- a tune empirical-Bernstein selection score with `alpha=0.05` and target
  `0.95`, explicitly not a certificate;
- the common absolute coordinate-work objective
  `(N+d)*m_prime + d*M`.

The objective counts one query projection (`d*m_prime`), one projected
full-corpus scan (`N*m_prime`), and exact original-space reranking (`d*M`). It
does not compare candidate saving against a different denominator at each
dimension. The terminal full-corpus budget makes failure visible rather than
silently expanding a candidate grid. Evidence labels and evidence metrics do
not enter selection.

Two local real-array runs produced byte-identical manifest, per-query,
selection, frozen-projection, summary, and report artifacts. An independent
reduction of all 4,836 query/dimension records reproduced every one of the 192
dimension/budget bounds and the final ordering. The selected tune operating
point is:

- `m_prime=192`, fixed reference `M=768`;
- mean tune embedding retention `0.985360`;
- tune lower bound `0.958051` against target `0.95`;
- coordinate work `1,732,416` versus `3,980,544` for an original full scan,
  a theoretical reduction of `56.48%`;
- selection fingerprint
  `093588a27e0d588b9407d02fe5c5ed7e46f6a5fdc02a1881738abbde4eda01fb`;
- frozen-projection fingerprint
  `8a9a1148527db16c43bc3fedf6da1ac79ae00c0d76f2a31321cd6d9fe049809e`;
- deterministic result fingerprint
  `5dcb0a5f17cc1f2f1684a38c71ede5c8dfc1de709f98496156959e14fbea7558`.

These are tune selection results, not an independent retention certificate,
measured latency saving, evidence result, test result, or answer-quality claim.
The runner saves monotone per-budget overlap and retention for every query and
dimension, plus a full-ranking row hash bound to the accepted corpus ID map.
Timings are saved separately and excluded from result identity.

Run the sweep with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.real_dimension_sweep \
  --config configs/real_scifact_fixed_dimension_tune.json \
  --dataset data/prepared/scifact-dedup-v2 \
  --embedding-config configs/real_scifact_e5_base_v2_embeddings.json \
  --embedding-cache data/embeddings/scifact-e5-base-v2-dedup-v2 \
  --original-baseline runs/scifact-original-exact-tune \
  --output runs/scifact-fixed-dimension-tune
```

Slurm job `373780` reproduced the sweep twice on Genoa at commit `07c28e1`.
Every deterministic artifact is byte-identical across the two runs and has the
same selection/result identities above. Independent local audit recomputed all
192 empirical-Bernstein bounds from the returned 4,836 query-level rows. The
archive SHA-256 is
`99cf703cd384555305d8d526224ccc01208c76539cd79acb45b8ea600b737b21`.

## Tune-only policy fitting

`configs/real_scifact_policy_tune.json` freezes policy selection at the accepted
`m_prime=192` projection. It permits only `query_tune`, binds every dataset,
embedding, baseline, dimension-selection, and projection fingerprint, and uses
the same complete budget grid and coordinate objective as dimension selection.
Evidence labels are excluded. Protocol v2 additionally freezes nine-decimal
LID canonicalization and separates scientific policy identity from the
platform-bound compiled lookup. Its config fingerprint is
`47d37917974869641951a0155e71ffbb76f676d8229ff606fef56fabc83ba812`.

The runner reconstructs and verifies every projected ranking and retention from
the frozen dimension run. Pilot LID uses original-space reranking of the first
32 projected candidates with `s_lid=20`; exact original-space LID is saved only
as a tune diagnostic. It evaluates every fixed budget, crosses predeclared
monotone-bin targets, and crosses dense analytic prediction targets with
tune-only residual corrections. The selected analytic policy is compiled into
fingerprint-bound adjacent-float64 LID intervals and checked against the
reference on every tune query. Tri-Predict policy schema version 2 also makes
the complete-corpus target-1 semantics part of the fingerprint rather than
allowing a finite-budget probability rounded to one to masquerade as exact.
The analytic policy is the portable scientific object. The compiled interval
table remains fingerprinted and must reference that analytic policy, but its
adjacent-float boundaries and file hash are deployment metadata rather than
inputs to tune selection or scientific result identity.

Two independent local runs produced byte-identical deterministic artifacts.
Independent reduction of `per_query.jsonl` reproduced both selected-policy
bounds and all result-artifact hashes:

| policy | mean M | mean retention | tune lower bound | candidate saving | coordinate saving |
| :--- | ---: | ---: | ---: | ---: | ---: |
| fixed `M=768` | 768.000 | 0.985360 | 0.958051 | 0.00% | 0.00% |
| monotone binned | 672.397 | 0.983623 | 0.956282 | 12.45% | 4.24% |
| Tri-Predict | 1092.548 | 0.979653 | 0.950276 | -42.26% | -14.39% |

The monotone policy freezes bin budgets `[384, 512, 768, 1024]`. Tri-Predict
needs target `0.99995` with no residual correction to reach the tune score. At
target `0.9998` it already averages `M=850.256`, more than fixed, but its lower
bound is only `0.938794`; therefore finer threshold interpolation cannot produce
an eligible analytic policy cheaper than fixed. This is a negative tune-only
Tri-Predict efficiency result, not a certificate. The binned improvement is
also only a tune candidate until independent certification.

The pilot/oracle clipped-LID MAE is `14.8902`, with all 403 pairs valid. This
does not yet prove that pilot error rather than the rank model or mean-field
stack causes the analytic inefficiency. Oracle-LID and actual-distance modes
remain diagnostics and were not allowed to choose the deployable policy.

The protocol-v2 result/selection fingerprints are
`2c31279a8f8038eebb049b0630548b2edd533ee5e1e01adb6cbd0a41e7e9bcb8`
and `2eaed81134b1621e9f2fd2f072a3c800cb6eaa8610bcaeb02f0a8465c34509f1`.
The monotone and analytic policy fingerprints are
`7734ac4efb84a66af837028a289422ff21ce77d1e9cfae68f484d52d10286f38`
and `7838e1be673932f38c5b4db9d1cea06e168565b0b7feba3014bef618f89d4423`.
The local compiled deployment fingerprint is
`4530a8a5bc9ef8d3c9858da8774490f98ffdf722fae943356205abd185dadd7b`;
it is not expected to equal a Genoa-local compilation fingerprint.

### Cross-platform audit and protocol-v2 repair

The first Genoa policy run at commit `516c1e4` completed twice on job `373780`.
All nine deterministic artifacts were byte-identical between the two Genoa
runs. Fixed, monotone, and analytic Tri-Predict policy identities, every chosen
budget, every retention value, and every tune lower bound also matched the Mac.
Only the asserted aggregate identities differed.

Independent comparison of archive
`scifact-policy-tune-373780-audit.tar.gz` (SHA-256
`a74ce1d5ad1a13f4c7851deccac9314bfbec378c5c085194d134eeac1fd3bb13`)
located 447 differing LID fields in 222 of 403 records. All were numerical tail
noise: the maximum absolute difference was `8.01e-12`, and every Mac/Genoa LID
pair becomes identical at nine decimals. Thirteen of fifteen compiled interval
boundaries also differed, with maximum absolute displacement `7.82e-14`; both
compiled artifacts referenced the same analytic policy, had the same 16 states,
and passed all 158 validation points with zero mismatches.

Replaying the selected monotone and analytic policies on every nine-decimal LID
changed zero of 403 decisions for either Mac or Genoa records. Protocol v2 thus
repairs identity without changing the selected operating point or its outcome:
the LID precision is explicit in the config and feature version, while compiled
lookup identity moves under a separate deployment section of the manifest.
Two fresh local v2 runs are byte-identical for every deterministic artifact,
and independent reduction reproduced the new result identity above.

### Accepted Genoa protocol-v2 gate

Slurm job `373780` on `genoa02` ran protocol v2 at commit `5389745` with
Python `3.9.23`, NumPy `1.26.4`, and SciPy `1.13.0`. The run completed after 85
tests passed and the one optional real-FAISS conformance test skipped. Its seven
scientific files—query records, selection, fixed grid, monotone policy,
analytic Tri-Predict policy, summary, and report—are byte-identical to the
independent Mac output. Consequently the portable result and selection
fingerprints remain
`2c31279a8f8038eebb049b0630548b2edd533ee5e1e01adb6cbd0a41e7e9bcb8`
and `2eaed81134b1621e9f2f072a3c800cb6eaa8610bcaeb02f0a8465c34509f1`.

The Genoa-local compiled deployment fingerprint is
`687d47f7fa1f93babaec6049ceaf825929445a92be52df48f26945ab38b42c30`.
It references analytic policy
`7838e1be673932f38c5b4db9d1cea06e168565b0b7feba3014bef618f89d4423`,
contains the expected 16 states and 15 transition boundaries, and passed all
158 validation points with zero mismatches. It differs from the Mac compiled
fingerprint only in the explicitly non-scientific platform deployment layer.

The returned archive `scifact-policy-v2-373780-audit.tar.gz` has SHA-256
`b091a1ac57fdaca61bbb0d849cdadf9e91507fe779d5b5f95c7937076841246c`.
Independent local audit rehashed every artifact, reconstructed the result
identity, verified the 403-query tune scope and ID hash, and recomputed all
three empirical-Bernstein statistics and the nine-decimal LID diagnostic. No
certification or test records were accessed during this gate.

Run policy fitting with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.real_policy_tune \
  --config configs/real_scifact_policy_tune.json \
  --dataset data/prepared/scifact-dedup-v2 \
  --embedding-config configs/real_scifact_e5_base_v2_embeddings.json \
  --embedding-cache data/embeddings/scifact-e5-base-v2-dedup-v2 \
  --dimension-selection runs/scifact-fixed-dimension-tune \
  --output runs/scifact-policy-tune
```

## Frozen certification-only runner

`configs/real_scifact_policy_certify.json` and
`tri_rag_harness.real_policy_certify` are implemented and frozen before any
real certification retrieval outcome is read. Config fingerprint
`e5545a4aa4c07a1bc188870538c7d346ff26faf38f135c03d4b32f4a18c7ce74`
binds the accepted dataset/cache, 404-query cert ID hash, fixed projection and
dimension, complete budget grid, fixed reference, both adaptive policies, and
the exact Genoa compiled policy fingerprint/file hash.

The loader verifies the policy manifest, every scientific artifact hash, the
reconstructed scientific result identity, selection, fixed-grid and selected
fixed-policy identities, monotone and analytic policy serialization, and the
compiled deployment binding before selecting or scoring `query_cert`. The
runtime reconstructs one non-renormalized Gaussian projection, performs one
exact projected full-corpus ranking per query, estimates deployable LID from
the first 32 candidates in original space, and reuses both projected ranking
and pilot original distances for expansion/reranking. The compiled Tri-Predict
decision must match the analytic policy on every query or the run aborts.

The three policies are predeclared standalone certificates rather than
candidates for post-cert selection. Each uses the empirical-Bernstein lower
bound with `alpha=0.05`, target `0.95`, and all 404 frozen queries. A FAIL is a
valid terminal artifact: the command still publishes the query-level result
and must not trigger retuning or budget expansion. Evidence qrels are not
loaded, oracle LID remains diagnostic only, and `query_test` is not evaluated.
Candidate and coordinate savings are work proxies rather than latency claims.

Four synthetic-only end-to-end tests verify protected-scope refusal, rejection
of policy tampering before protected data access, one projected scan, pilot
distance reuse accounting, candidate/rerank overlap equivalence, complete
query-level auditability, terminal failure semantics, and byte reproducibility.
The full local suite has 92 tests: 91 pass and the optional real-FAISS test
skips. No real certification output exists yet.

The one-time command is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.real_policy_certify \
  --config configs/real_scifact_policy_certify.json \
  --dataset data/prepared/scifact-dedup-v2 \
  --embedding-config configs/real_scifact_e5_base_v2_embeddings.json \
  --embedding-cache data/embeddings/scifact-e5-base-v2-dedup-v2 \
  --policy-run runs/slurm-scifact-policy-v2-373780 \
  --output runs/scifact-policy-cert
```

## Next gate

Run the frozen command exactly once on Genoa and archive the output and complete
log regardless of PASS/FAIL. Do not change any policy/configuration after the
result. Independently recompute artifact identities and all three bounds from
the returned query-level records before beginning untouched `query_test` work.
