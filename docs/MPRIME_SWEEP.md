# Tune-only global `m_prime` sweep

Updated: 2026-08-20

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

## Extended-range experiment

An additional, independently seeded experiment was predeclared in
`configs/synthetic_mprime_sweep_extended_fresh.json`. It keeps ambient dimension
`d=32`, expands the candidate set to
`[2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]`, and uses data/projection seeds
`16001`/`17011`. It has 768 tune, 1024 certification, and 320 test queries. The
threshold grid is `0.80` through `0.99`; all other selection rules are unchanged.

Tune-only results:

| `m_prime` | eligible | threshold | tune lower bound | mean `M` | fixed `M` | relative saving | saturation |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | no | — | — | — | — | — | — |
| 4 | yes | 0.93 | 0.8045 | 69.7552 | 80 | 12.81% | 39.58% |
| 6 | yes | 0.95 | 0.8025 | 61.9583 | 80 | 22.55% | 26.69% |
| 8 | yes | 0.95 | 0.8064 | 53.8229 | 80 | 32.72% | 17.06% |
| 10 | yes | 0.95 | 0.8070 | 47.4375 | 48 | 1.17% | 11.46% |
| 12 | yes | 0.95 | 0.8056 | 43.5052 | 48 | 9.36% | 8.59% |
| 14 | yes | 0.94 | 0.8024 | 38.5521 | 32 | -20.48% | 4.43% |
| 16 | yes | 0.92 | 0.8069 | 34.6875 | 32 | -8.40% | 3.12% |
| 20 | yes | 0.90 | 0.8039 | 29.6562 | 32 | 7.32% | 1.17% |
| 24 | yes | 0.89 | 0.8068 | 26.2552 | 20 | -31.28% | 0.78% |
| 28 | yes | 0.84 | 0.8025 | 21.0104 | 20 | -5.05% | 0.00% |
| 32 | yes | 0.83 | 0.8016 | 18.7760 | 20 | 6.12% | 0.00% |

The predeclared rule froze `m_prime=8`, threshold `0.95`, with selection
fingerprint
`655d8c9530366ee7dcf3bb9f0f784cf4ba20fabbb95454a8bce8c7ba07af7def`.
Fresh certification then produced:

- `n=1024`, planned `n=336`;
- mean retention `0.841602`, radius `0.024058`, lower bound `0.817544`:
  **PASS** against `0.80`;
- mean `M=54.863281`;
- smallest certified fixed budget `M=80`;
- relative candidate saving `31.4209%`;
- 186 saturated queries (`18.16%`);
- policy fingerprint
  `a503d27a2f1f2e8613f3986e7b8f7b79e6627325ffd84d37014343c51fc4361f`.

### What the expanded range reveals

The selected dimension is not stable across the two independent synthetic
experiments (`24` in the five-point sweep, `8` in the extended sweep). More
importantly, maximizing relative saving against a dimension-specific fixed
baseline is not a sound cross-dimension compute objective:

- absolute mean `M` decreases almost continuously as `m_prime` grows;
- the fixed baseline changes discontinuously from `80` to `48`, `32`, and `20`;
- those denominator jumps create the sawtooth relative-saving column and favor
  dimensions immediately before a fixed-baseline transition;
- at the selected `m_prime=8`, fixed `M=48` missed certification by only
  `0.000207` (`0.799793` versus `0.80`). Had it crossed that boundary, the same
  adaptive mean would imply `-14.30%` rather than `+31.42%` relative saving.

Therefore the extended certificate is valid for the frozen `m_prime=8` policy,
but its `31.42%` headline is fragile and must not be interpreted as global
compute optimality. Before another selection run, the protocol should
predeclare a cross-dimension objective that includes projected-search work and
original-space reranking, and should densify the fixed-`M` grid to reduce
threshold cliffs. Any such revised rule requires new tune/cert seeds; this
already inspected certification split cannot be reused for selection.

Extended local or Genoa command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m tri_rag_harness.mprime_sweep \
  --config configs/synthetic_mprime_sweep_extended_fresh.json \
  --output runs/synthetic_mprime_sweep_extended_fresh
```

### Genoa reproduction and timing audit

Slurm job `371643` reproduced the extended run on
`genoa04.cloud.r-ccs.riken.jp` at commit
`d5ec795abf0ca604c90ac2b5300708232874ef32`, using Python `3.9.23`, NumPy
`1.26.4`, and SciPy `1.13.0`. All 39 tests passed in `4.211 s`.

The following Slurm artifacts are byte-identical to the local run:

- `selection.json`;
- `selected_config.json`;
- `sweep_result.json`;
- `tri_predict_policy.json`;
- `tri_predict_certification.json`.

The only deterministic aggregate difference is a platform-level floating-point
rounding change in the test mean pilot/oracle LID gap (`8.252871586613274`
locally versus `8.25287158661327` on Genoa). The manifest's semantic contents
match; its timestamp, platform, and Python-version fields correctly differ.

Genoa timing for the frozen Tri-Predict policy was:

| component | mean ms/query |
| --- | ---: |
| pilot projected search | 0.006577 |
| expansion projected search | 0.006809 |
| original-space rerank | 0.020316 |
| analytic policy computation | 5.998815 |
| total retrieval path | 6.032517 |

The empirical binned policy path averaged `0.040392 ms/query` on the same run.
The analytic policy computation therefore dominates the synthetic latency and
makes the current Tri-Predict path roughly 149 times slower than the empirical
path, even though it uses fewer original-space candidate distances. These
timings are for a 160-item corpus and are not a serving benchmark, but they rule
out interpreting candidate-count saving as demonstrated latency saving.
