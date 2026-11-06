# One PQ attribution check v1

2026-11-06. Actual consent: **“批准第19.4节八路径，仅执行这一次归因”**.
Scope is plan section19.4, not reopening the closed PQ refinement route.
Old science/config/roles/results are unchanged. No Tri gate3, adapter, predictor,
new retrieval, fit or raw-vector reads. Only one real closed-array run permitted.

## Frozen inputs and point diagnostics

Bind completed scientific3f82b40/index8eaf1d2e, old32 opened queries (16 per role),
nprobe64/256,6400 candidates,100 native outputs. Four existing seeds, m8/16/32;
primary2026110401/m32/nprobe64. No best-seed selection or fresh certification.
Only recover original verified NPZ; no rebuilding missing inputs or combining failures.

Compare original ADC order, saved ADC score re-sort, real reconstruction, that
reconstruction plus Gaussian h dot sketch/T times ||q||, and sidePQ32/64/128
corrections. Real reconstruction is explicitly prepaid diagnostic input; production
LUT/mapping cost is not hidden in point arithmetic. Stable ordering uses original
row IDs for native float32 clip(1-score,0,2) ties. At fixed1600/3200/4800/6400,
rerank with saved native scores, record exact candidate-top100 preservation,
benchmark cutoff+.001 and strict published-ID quality. No labels enter estimates.
For each summary also report quantiles of point native-distance minus stored
conservative lower-distance bound; this includes clip/storage/native guards, not
a newly inferred exact normal confidence radius. Point errors use saved real raw dot, not clipped native distances; quantization,
clip and native score differences therefore remain visible in ranking results.

The necessary headroom screen is positive only if primary Gaussian preserves
all observed query-top100 in BOTH roles at an existing M<=3200 while ADC and
sidePQ128 at that same cutoff each lose at least one query in EACH role. Report
all outcomes; no selecting a deployment cutoff. This is only necessary evidence,
not a full quality/cost result. C1 subset fees are conditional feasibility proxies
with different address patterns; no proportional row-to-latency extrapolation.

## Relative errors and cheap proxies

Within rank80–120 (inclusive), take all820 unordered pairs, separately for ADC
order and true order (oracle). Epsilon_i=estimated real score−saved real raw dot.
Report mean point bias/MSE, pair bias/MSE, strict inversions, and
mean((epsilon_i−epsilon_j)^2)/mean(epsilon_i^2+epsilon_j^2). A ratio below1
is observed cancellation against this uncorrelated-error reference, not a theorem
or net gain. Zero denominator is null with a distinct perfect-estimate flag.
No residual Gram is saved and four fixed seeds cannot certify covariance.

Conditional ideal Gaussian derivation: e_i=a_i*u+v_i, h=Ru, T=h dot h. Given h,
cov(epsilon_i/||q||,epsilon_j/||q||)=v_i dot v_j/(m*T), and difference variance
is ||v_i−v_j||²/(m*T). Check independently on synthetic shared-projection fixtures,
including equal/opposite errors. Do not apply Gaussian theory to sidePQ, claim
Tri novelty for this identity, or confuse observed error cancellation with a
rank/recall guarantee.

Fixed label-free proxies are ADC gap, sum of residual radii, and same-seed sketch
difference norm. Report Spearman association with squared pair error, the per-pair
cancellation ratio (exclude zero denominators), and proxy
means, separately by role, seed and boundary. These are descriptive, no fitted
classifier, thresholds, certified identification or automatic selection. Proxy
values never accept true errors; truth appears only in association evaluation.
Arithmetic, logical accesses and summary gather bytes are charged and labeled;
local timings exclude complete serving/mapping and physical traffic. Their
correlations alone cannot establish cheap deployability or a positive net gain.

## Exit, resource and closure

Any missing necessary point advantage ends current interface research. Relative
cancellation absent or not cheaply identifiable also ends it. A positive point
screen alone does not reopen the route. If clear relative/proxy evidence emerges,
review it against the C1 fee allowance before drafting only a single-mechanism
small experiment; require user review before any experiment. No auto next run.

CPU<=1200s, RSS<=4GiB, derived outputs<=256MiB. Old input copies accounted
separately. Verify hashes before consuming the unique real run. Runner fixes its
output namespace and refuses reuse; no arbitrary output/seed override. Check
resources between query groups, preserve failures; do not auto retry. No claim
of hard per-instruction memory/time enforcement. Archive every closed output
with SHA, including failure status. formal_certification=false throughout.

Run local synthetic tests and all old719 tests plus new fixtures, protection and
scope review, commit/push/verify before real run. Preflight reads/checks only:

```sh
OPENBLAS_NUM_THREADS=1 /private/tmp/route6-b0-local-env-20261103/bin/python \
  scripts/run_tls_rag_step5_pq_attribution_v1.py preflight
# Only after all verified old arrays are local, once:
OPENBLAS_NUM_THREADS=1 /private/tmp/route6-b0-local-env-20261103/bin/python \
  scripts/run_tls_rag_step5_pq_attribution_v1.py run
```
