# Tune-only global `m_prime` sweep

Updated: 2026-08-19

## Purpose

The original synthetic development run fixed `m_prime=16` before testing the
analytic policy. Its Tri-Predict certificate passed, but its mean certification
budget was `39.2812` while the smallest certified fixed budget was `32`, giving
`-22.75%` candidate saving. That result cannot be repaired by choosing a new
dimension from an already inspected certification split.

`tri_rag_harness.mprime_sweep` implements a two-stage replacement:

1. evaluate every predeclared `(m_prime, Tri-Predict threshold)` pair using only
   `query_tune`;
2. write `selection.json` and `selected_config.json`, freezing one dimension and
   one threshold;
3. only then run the frozen configuration on the independent `query_cert` split;
4. report the fresh certificate without any retry or post-certification tuning.

The selector rejects any non-`query_tune` record. Its selection rule is serialized
in the selection artifact. Candidate dimensions, the threshold grid, tune lower
bound target, saturation cap, data seed, and projection seed are all predeclared
in the source config.

## Predeclared experiment

- config: `configs/synthetic_mprime_sweep_fresh.json`
- previously unused data seed: `12011`
- previously unused projection seed: `13007`
- candidates: `[4, 8, 12, 16, 24]`
- threshold grid: `0.80` through `0.98` in steps of `0.01`
- tune queries: `512`
- fresh certification queries: `768`
- test queries: `256`; never used by the selector
- tune eligibility: empirical-Bernstein lower bound at least `0.80`
- maximum tune saturation fraction: `0.40`
- certification: `alpha=0.05`, target `0.80`

Within each dimension, the selector minimizes mean candidate budget among
eligible thresholds. Across dimensions, it maximizes candidate saving against
the smallest tune-qualified fixed budget at that same dimension. This is a
candidate-count objective; it is not a wall-clock or FLOP claim.

## Tune-only results

| `m_prime` | eligible | threshold | tune lower bound | mean `M` | tune fixed `M` | saving | saturation |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | no | — | — | — | — | — | — |
| 8 | yes | 0.93 | 0.8043 | 52.0000 | 48 | -8.33% | 12.50% |
| 12 | yes | 0.95 | 0.8039 | 45.9453 | 48 | 4.28% | 9.18% |
| 16 | yes | 0.96 | 0.8069 | 39.9688 | 32 | -24.90% | 6.05% |
| 24 | yes | 0.91 | 0.8064 | 28.6484 | 32 | 10.47% | 1.37% |

`m_prime=4` was rejected rather than rescued: even fixed `M=80` had a tune lower
bound of only `0.7825`. The frozen winner was therefore:

- `m_prime=24`;
- Tri-Predict threshold `0.91`;
- selection fingerprint
  `bc35d0ef545efeb58f53f977679f1ac9c9c8ffe62a646044985e85ed93b1053c`.

## Fresh certification result

The first certification evaluation for data seed `12011` produced:

- `n=768`, exceeding the planned `n=253`;
- mean embedding retention `0.847917`;
- empirical-Bernstein radius `0.030683`;
- lower bound `0.817234` against target `0.80`: **PASS**;
- mean candidate budget `28.40625`;
- budget counts `{12: 235, 20: 221, 32: 161, 48: 83, 80: 68}`;
- 7 analytically saturated queries (`0.91%`);
- smallest certified fixed budget `M=32`, whose mean retention was `0.910938`
  and lower bound was `0.884052`;
- candidate saving `1 - 28.40625/32 = 11.2305%`.

The frozen Tri-Predict policy fingerprint is
`91d710d7a62837984e47cbe18ca4fe8d387495740f419b01a03af5551aaef42c`;
the certification split hash is
`07734ff9fd2503c2c1bb957eb12159ec90295a1665b2559f40875c69e68fe664`.

This repairs the negative *candidate-count* efficiency result on a fresh
synthetic split. It does not establish wall-clock speedup: increasing
`m_prime` from 16 to 24 makes projected-distance computation more expensive,
and exact-search timings must be measured separately. It also does not replace
the required real external-query experiment.

## Commands

Local or Genoa CPU execution:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m tri_rag_harness.mprime_sweep \
  --config configs/synthetic_mprime_sweep_fresh.json \
  --output runs/synthetic_mprime_sweep_fresh
```

Tests:

```bash
scripts/run_tests.sh
```

The important top-level artifacts are `selection.json`, `selected_config.json`,
`sweep_result.json`, and `sweep_report.md`. The full frozen run is under
`selected_run/`.
