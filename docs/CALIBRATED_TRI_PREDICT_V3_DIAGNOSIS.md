# Calibrated Tri-Predict v3: Step 3 causal diagnosis

Date: 2026-09-05

This is a diagnosis and a network-free repair boundary, not a certificate or a
positive v3 result. Raw Tri-Predict v1 and Calibrated Tri-Predict v2 are frozen.
No query-cert, query-latency, or query-test identity or outcome was opened, no
data was downloaded, and no LLM or answer generation was run.

## Input audit

All allowed archives were extracted into fresh directories under
`/private/tmp`; no repository artifact was overwritten. The source-audit,
protocol-freeze, E5-cache, query-cal, and query-tune returned-archive SHA-256
values independently matched, respectively:

- `098910e034dfb1790913699a3c3ad9e4c13106852722821dde8218539e31f46e`
- `36e8698f01777abc2f6dde5ee5e69385f1f9ca8298ca59e3fff47c6a421d165e`
- `87288fd7e913930474c9f764017780b729ab00404e93d88e2fb9ffc0359c1133`
- `686781b787c5a64a00b81996547594abfd5ffcd60107927844a40a552872089c`
- `3cfbeb6abd65b4e01991bc79065c2c244c8bc356f86deddae88a9ae4b7084969`

The audit then independently replayed all run-file hashes, 1,966 query-cal
record fingerprints, 2,025 residual operating points, 1,967 query-tune record
fingerprints, 11,802 selected-policy record fingerprints, and the complete
`2086 x 1967` int32 candidate-budget matrix. All 2,086 candidate evaluations
and all six family selections reconstructed exactly. The principal accepted
identities were:

| Boundary | Fingerprint |
|---|---|
| query-cal config | `7ff0bdf656ebc22026702622e933975ffe56b3814bb384b1db99effde51df36b` |
| query-cal manifest | `140c3a5f9ac168e222f9ff7e7cd3edb7f75879a40aa358158a1cfac4b77b2be6` |
| query-cal state | `2fe1eeb55180198165b6d1b46f35bf4214b8711cda554e7a2ca211f5b193f481` |
| LID candidate bundle | `4526c8b752325e3cae040d8b450c76cd0df77571b9e6d6080bd1a53ba4a56a1e` |
| residual candidate bundle | `ed6cf0f7056fc1b7345b5303a7ad71815a2b8df6677c6c1c2c0e231bbf9c9f31` |
| query-tune config | `06c647625bb01192b54ae0698e9e4150fe4fec0d2b4407858de74c763573d7d0` |
| query-tune manifest | `7cf01ee872a59ee28e6dd0a0c5ffa10ab556b9c0746e3b0a96b8454bdb31836e` |
| selection | `8db86a98eab28deaf6ba173ab78e8336a5bc23b2fc2916653cfbe6b2696cb9ee` |
| frozen suite | `ae0b21e565810853ca97add88fa593b686e6afee451b49ea7e3ee9fd4eb5aefd` |
| post-tune state | `55ecdf8e3cc53d554d6476569d34cd309e4e0f182a5a86e634741ef1a9dd97b5` |

The replay recovered the frozen result: fixed `M=768`; selected full-PDCTP
mean budget `1892.763599`, mean retention `0.980732079`, 31 feature-invalid
terminal fallbacks, and `+7.318153%` common coordinate work versus fixed.

One shell status command surfaced the pathname of an untracked protected
archive before the whitelist was tightened. Its contents, hash, member list,
metadata, identities, and outcomes were not read. Every subsequent command
used explicit allowed paths.

## Layer-by-layer matrix

Signed errors below are prediction minus the stated reference.

| Layer | Result | Evidence and direction |
|---|---|---|
| 1. Exact dense-Gaussian Tri-Law | PASS | Eight cases with 200,000 projections each had maximum absolute probability error `0.0015296` (`1.603` binomial standard errors). Projection-entry variance was `0.00520007` versus `1/192 = 0.00520833`. Finite-quantile conditional-to-marginal integration at `m_prime=192` differed by `-1.88e-14`. The first infinite-interval quadrature attempt sampled the concentrated integrand poorly and returned zero; finite chi-square quantile bounds removed that diagnostic integration error without changing Tri-Law. |
| 2. Finite-rank aggregation | PASS | Across 900 rows (`m_prime` 64/192/384, LID 2/8/20/50/100, six budgets, neighbor ranks 1--10), geometric quadrature minus exact finite summation had signed mean `-1.335e-6`, MAE `1.368e-6`, and maximum absolute error `7.599e-6`. This is not material. |
| 3. Rank-distance power law | FAIL: earliest | On the deterministic 16 lowest-SHA256 valid query-cal IDs, all top-k strict-gap checks passed. Model-minus-actual log-beta bias was `-0.086956`, MAE `0.104660`, RMSE `0.171680`, and reached `-0.392610` at competitor rank 57,637. Replacing modeled ranks by actual distances moved predicted-minus-realized retention to signed mean `-0.058420`, MAE `0.080380`, RMSE `0.132040`, maximum `0.385950`. Internal competitor ties were recorded separately and did not invalidate the top-k diagnostic. |
| 4. Mean field / geometry | PASS numerically; FAIL scientifically | On eight deterministic query-cal profiles, 128 direct dense-Gaussian projections and 512 independent-radial trials showed actual-distance mean-field minus independent-radial retention `-0.001582` (RMSE `0.005016`), but minus direct actual-vector projection `-0.048300` (RMSE `0.063860`). Bias was budget-shaped: `-0.105880`, `-0.082010`, `-0.040450`, `-0.012720`, and `-0.000449` at budgets 64/256/768/2048/8192. Exact actual-rho expected inversions averaged `154.665` versus `154.449` direct Monte Carlo; forcing `rho=0` added `105.701` inversions. The numerical mean-field step works under its independence model; real shared projection geometry violates that model. |
| 5. LID input | FAIL | Among 1,933 valid fit records, raw pilot minus oracle signed log bias was `-0.470563`, log RMSE `0.509170`, MAE `15.280450`, RMSE `17.568600`, signed raw bias `-15.277400`. The selected v2 calibrator had in-sample MAE/RMSE/log-RMSE `4.758/6.353/0.150890`; deterministic five-fold query-cal CV was `4.7885/6.3885/0.150890` with signed log bias `-0.000299`, so visible overfit is not the cause. Pilot distances exceeded the true same-rank distance in `84.04%` of comparisons; rank-32 log inflation averaged `0.032204`, top-10 overlap averaged `0.7297`, and 1,694 queries missed at least one exact top-10 item in the pilot set. Failures were 15 pilot LIDs, 33 features, and seven oracle LIDs. |
| 6. Calibration target | FAIL | Predicted-minus-realized curve bias/RMSE was `+0.03591/0.10548` for raw pilot LID, `-0.06028/0.14051` for oracle LID, `-0.05369/0.12808` for calibrated pilot LID, and `-0.00225/0.03697` for the per-query full-curve effective Tri-LID. Mean full/low/high effective LIDs were `29.47/29.58/7.108`; median high-minus-low was `-27.72`, and `78.32%` differed by more than five. Oracle geometric LID is therefore the wrong correction target, and one effective scalar cannot describe both curve regimes. |
| 7. Budget residual | FAIL as a causal repair | For the selected v2 residual model, target log budget ratios had mean/SD `-3.09193/1.35934`; predictions had `-2.46245/0.88737`, correlation `0.69577`, signed error `+0.62948`, MAE `0.95615`, RMSE `1.16316`. Continuous quantile coverage was `0.74909` for the 0.75 model, while grid operational coverage was `0.83859`; raw-anchor coverage was `0.98293`. Valid records had no terminal saturation. With `k_gt=10`, required-budget maps for retention 0.95, 0.98, and 1.0 were identical on every query, so those candidate axes do not define distinct events. The residual is a high-variance budget compensator, not a repair of the prediction model. |
| 8. Fallback and selection | FAIL | Of 1,967 tune queries, 1,936 were feature-valid with PDCTP mean budget `1000.148760`; 31 invalid records fell back to the full corpus and contributed `896.2735` of the `1124.7636` overall mean excess over fixed (`79.6855%`). There were no valid-query terminal/nonattainment or residual-saturation decisions. None of the 1,080 quality-eligible PDCTP candidates was cheaper than fixed in common work, yet the family rule selected the cheapest eligible PDCTP within its own family. At matched mean budget, valid-only PDCTP minus fixed-mixture retention was `+0.001490`, candidate evidence `-0.022842`, and final evidence `+0.001897`; full selected PDCTP minus matched fixed mixture was `-0.01097` retention and `-0.07363` candidate evidence. The frozen contract can admit an inefficient adaptive family member by design. |

## Causal conclusion

1. The exact Tri-Law implementation is not wrong.
2. Finite-rank quadrature is accurate enough and is not the failure source.
3. The scalar LID rank-distance model fails even when supplied oracle LID; this
   is the first failing approximation layer.
4. Pilot construction adds a large downward LID bias (`-0.470563` in log
   space), but it is not the first failure.
5. The v2 LID calibrator shows essentially equal train and five-fold query-cal
   error. Oracle LID is the wrong prediction target rather than an overfit
   explanation.
6. One scalar effective dimension fails to match low and high budget regimes.
7. The residual calibrator mainly compensates with a noisy multiplicative
   budget correction; it does not repair the curve model.
8. Full-corpus feature fallback explains `79.6855%` of the selected method's
   mean-budget excess over fixed.
9. The frozen selection rule checks quality eligibility and minimizes within
   each family, but does not require tune-side cost superiority over fixed; it
   can therefore accept an inefficient method by design.

These are scientific model failures, not implementation conformance failures.

## Smallest v3 repair and ablation

The minimum supported change is to replace only v2's oracle-scalar-LID target
with query-cal-only effective Tri-LID curve targets. The deployable calibrator
has a low-budget head (through `M=768`) and a high-budget head (from `M=1024`),
while a full-curve scalar-effective-LID mode is retained as the one-factor
ablation. Both modes use the unchanged float64 Raw Tri-Predict retention
calculation; the two-regime mode differs only in the effective-LID curve shape.
No residual correction, fallback redesign, or selection-rule change is folded
into this repair.

A deterministic five-fold query-cal precheck used SHA256(query ID) modulo five
and log-ridge regularization 1.0. Curve RMSE was `0.128081` for v2,
`0.104557` for the scalar-effective ablation, and `0.100478` for the two-regime
repair; high-regime RMSE improved from `0.04539` to `0.02440`. Target
predictability remained weak (scalar/high log-RMSE `1.056/1.20`), so this only
supports a network-free implementation direction. It is not policy selection
or evidence for a positive claim.

`tri_rag_harness.pdctp_v3` implements the isolated repair with new v3 names,
schemas, versions, and fingerprints. Its in-memory prediction cache keys on
the exact float64 LID bit pattern plus the complete frozen numerical problem;
cache state is absent from scientific serialization. The tiny complete-suite
test proves cached and uncached budgets, selection objects, artifact values,
and fingerprints are identical.

## Reproducibility boundary

Query-level real-data evidence remains in the accepted query-cal and query-tune
JSONL files and complete tune candidate matrix. Derived diagnostics use stable
record order or the lowest SHA256(query ID), explicit parameter grids, and the
formulas stated in the layer table. Tri-Law conformance also remains directly
reproducible through `tests/test_tri_law.py`, whose per-case seeds are `700+i`.
The v3 unit fixture is CPU-only and contains its full synthetic query-level
curves in `tests/test_pdctp_v3.py`.

A defensible real v3 result now requires a new dataset and a freshly frozen
cal/tune/cert/latency/test protocol. Step 3 stops before that gate and does not
reuse v2 tune, certification, latency, or test outcomes for v3 fitting,
selection, or claims.
