# Raw Tri-Predict v1 — terminal negative baseline

## Immutable version identity

Raw Tri-Predict v1 is closed at Git commit `fb09c00` and annotated tag
`raw-tri-predict-v1-terminal-negative`. The tag contains the complete accepted
SciFact retrieval, certification, descriptive test, and posthoc tune-diagnostic
record. It is the reference implementation and negative baseline for all later
calibrated methods.

The tag is immutable. New calibration code, schemas, configs, runs, or claims
must use a new method name and version. Do not rewrite a v1 artifact, reuse a v1
result fingerprint for a new output, move a v2 run into a v1 directory, or
silently retag a bug fix. A material v1 defect must be documented as an erratum
and must not move this tag.

Restore the exact source with:

```bash
git switch --detach raw-tri-predict-v1-terminal-negative
```

Development of the successor starts from the tag on branch
`codex/calibrated-tri-predict-v2`.

## What v1 established

The harness validated the exact dense-Gaussian Tri-Law separately from the
query-adaptive system, removed repeated pilot/expansion scans, compiled the
analytic policy into a negligible-cost lookup, passed exact NumPy/FAISS CPU/GPU
conformance gates, and executed a leakage-controlled real-data protocol with
external SciFact queries and pinned E5 embeddings.

The real policy result is terminal:

| split | policy | mean M | mean retention | retention status |
| :--- | :--- | ---: | ---: | :--- |
| tune | fixed `M=768` | 768.000 | 0.985360 | selection reference |
| tune | monotone-binned | 672.397 | 0.983623 | tune only |
| tune | Raw Tri-Predict | 1092.548 | 0.979653 | negative efficiency |
| cert | fixed `M=768` | 768.000 | 0.985396 | PASS, LCB 0.958311 |
| cert | monotone-binned | 673.901 | 0.980446 | PASS, LCB 0.952304 |
| cert | Raw Tri-Predict | 1119.515 | 0.972525 | FAIL, LCB 0.942480 |
| test | fixed `M=768` | 768.000 | 0.982667 | descriptive |
| test | monotone-binned | 698.453 | 0.982333 | descriptive |
| test | Raw Tri-Predict | 1211.613 | 0.974333 | descriptive |

On the terminal test split, Raw Tri-Predict uses 57.76% more candidates than
fixed and retains 25 fewer top-10 neighbors in aggregate. It must not be
repaired on the observed SciFact certification or test queries.

Answer generation was deliberately not run. v1 therefore makes no answer-
quality claim; embedding retention, qrel evidence metrics, budget, and measured
retrieval latency remain separate outcomes.

The accepted posthoc tune diagnostic separates retention from evidence:

| policy | candidate evidence recall | final evidence recall@5 |
| :--- | ---: | ---: |
| fixed `M=768` | 0.971464 | 0.786849 |
| monotone-binned | 0.972705 | 0.786849 |
| Raw Tri-Predict | 0.972705 | 0.786849 |

Shuffling deployable pilot LID over tune queries reduces mean embedding
retention by `0.007400` for monotone and `0.025826` for Raw Tri-Predict, and
reduces candidate evidence recall by `0.011926/0.017023`. It does not reduce
final evidence recall@5. Pilot distance therefore contains real allocation
signal, but that signal did not become a downstream final-context advantage in
v1.

## Two independent failure mechanisms

### 1. Pilot LID bias

Pilot LID is systematically below oracle LID. On test the means are
`21.91/36.68`, with clipped MAE `14.78`. On tune, substituting oracle LID into
the frozen Raw Tri map recovers 77 of the 82 top-10 neighbors missed under pilot
LID. Pilot bias is therefore the primary observed source of v1 retention loss.

### 2. Analytic LID-to-budget miscalibration

Oracle LID does not repair efficiency. In the same tune counterfactual it raises
mean Raw Tri budget from `1092.5` to `3198.3`, although the realized smallest
grid budget reaching unit top-10 retention averages only `420.1`. The oracle-
driven map overallocates 394 of 403 tune queries. The analytic rank/mean-field
aggregation is therefore the primary observed source of negative efficiency.

These mechanisms must remain separate in successor ablations. A single opaque
end-to-end model is insufficient evidence that both were fixed.

## Accepted v1 artifacts

- frozen policy: `scifact-policy-v2-373780-audit.tar.gz`, SHA-256
  `b091a1ac57fdaca61bbb0d849cdadf9e91507fe779d5b5f95c7937076841246c`;
- terminal certificate: `scifact-policy-cert-373780-audit.tar.gz`, SHA-256
  `4fd19b3b205c92d42596700e845da99b732261531d6c222d73375d57fc7ef12b`;
- terminal descriptive test: `scifact-policy-test-374032-audit.tar.gz`, SHA-256
  `39610254579876b77148d0045044aaa8bee1b950624dace646a8ae959ee22c76`;
- posthoc tune diagnostics: `scifact-tune-diagnostics-374032-audit.tar.gz`,
  SHA-256
  `a376a1cb484e1e57a726cce23afcf34004f3e403bfa5bc7c1d257ce17aa30804`.

The result fingerprints and detailed limitations remain in `STATUS.md` and
`docs/REAL_RETRIEVAL.md` at the tagged revision.

## Permitted use in v2

v2 may reuse v1 source code for exact Tri-Law, raw analytic prediction,
projection, retrieval, metrics, artifact hashing, and backend conformance. It
may use v1 outcomes to motivate the predeclared successor design and as a
historical baseline.

v2 must not use SciFact cert/test records to fit a feature, choose a model,
select a threshold, set a tolerance, decide a budget grid, or validate a new
claim. All successor calibration, selection, certification, latency comparison,
and test reporting require newly prepared data and fresh identities.
