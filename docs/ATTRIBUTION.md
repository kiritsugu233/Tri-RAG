# Tri-Predict Error Attribution and Fresh Synthetic Repair

## Question

The Milestone 4 development run at `m_prime=16` passed its retention certificate but used more candidates than the smallest certified fixed budget. This experiment asks whether the main cause is pilot-LID estimation or later Tri-Predict approximations.

## Attribution modes

All three modes retain the orthogonal conditional chi-square branch, the structural surrogate, conditional independence, and mean-field thresholding.

| Mode | LID source / distance model | Deployable? |
|---|---|---|
| `pilot_lid_rank_model` | Pilot-rerank LID and `(l/j)^(2/lambda_q)` | Yes |
| `oracle_lid_rank_model` | Exact-neighbor LID and the same rank model | No |
| `actual_distance_beta` | Full exact original squared-distance ratios | No |

The pilot-versus-oracle comparison diagnoses substitution of the deployable LID estimator. Replacing the scalar rank model with actual distance ratios diagnoses the rank-distance approximation. Error remaining with actual ratios still contains orthogonality, structural, independence, and mean-field error, so the decomposition is diagnostic rather than mathematically additive.

## Command

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m tri_rag_harness.attribution \
  --config configs/synthetic_mvp.json \
  --output runs/attribution_m16
```

The command writes `attribution.json` and `attribution_per_query.jsonl`.

## Results at `m_prime=16`

Calibration errors are computed over every query-budget cell in the frozen grid.

| Split | Pilot-rank MAE | Oracle-rank MAE | Actual-beta MAE |
|---|---:|---:|---:|
| Tune | 0.1385 | 0.2369 | 0.1131 |
| Cert | 0.1369 | 0.2448 | 0.1132 |
| Test | 0.1337 | 0.2502 | 0.1103 |

On certification queries:

- replacing pilot LID with oracle LID worsens MAE by `0.1079` and raises mean `M` from `39.28` to `73.14`;
- replacing the pilot-LID rank model with actual distance ratios improves MAE by only `0.0236`;
- actual ratios still leave MAE `0.1132` and signed prediction bias `-0.0508`.

Therefore pilot-LID inaccuracy is not the primary failure. The exact-neighbor Hill estimate is a local statistic and is not necessarily a better scalar parameter for the full rank range. Pilot-LID underestimation happens to cancel part of the conservative downstream approximation. The scalar LID rank model contributes error, but most residual error remains after supplying actual distance ratios.

## Repair protocol

The original data/projection seed was treated as development data. Before inspecting a new result, the following configuration was frozen:

- new data seed `7301`, included in stable corpus/query IDs;
- new projection seed `8111`;
- `m_prime=4`;
- analytic predicted-retention threshold `0.89`;
- no safety correction;
- 256 tune, 512 certification, and 160 test queries;
- unchanged `M_grid=[12,20,32,48,80]`, `k_gt=5`, and `M_pilot=12`.

The smaller projection dimension makes the fixed baseline sufficiently expensive for query adaptation to have useful room. The threshold was frozen using the previous development problem; it was not changed after the fresh certification result.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m tri_rag_harness.run \
  --config configs/synthetic_attribution_fresh.json \
  --output runs/synthetic_attribution_fresh
```

Fresh result:

| Metric | Value |
|---|---:|
| Certification queries | 512 |
| Mean realized retention | 0.8535 |
| Empirical-Bernstein lower bound | 0.8130 |
| Certification target | 0.8000 |
| Mean adaptive `M` | 63.6953 |
| Smallest certified fixed `M` | 80 |
| Candidate saving | 20.38% |
| Saturated certification queries | 125 |

This meets the synthetic 20% efficiency criterion, but it does not establish a real-retrieval or RAG result. The high saturation rate and aggressive four-dimensional projection are risks to re-evaluate on a pinned external-query dataset.

