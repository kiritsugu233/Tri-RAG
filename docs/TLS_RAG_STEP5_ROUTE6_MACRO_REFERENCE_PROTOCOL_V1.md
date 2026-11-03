# R6-M0 structured macro reference v1 (frozen 2026-11-03)

The user answered the exact nine-path question linked to exploration §46.8:

> 批准九路径、测试同步及一次M0计算（推荐）

This freezes the new namespace `tls-rag-route6-macro-reference-v1`, its six new
files, narrow test membership/register additions, full local regression and
synchronization, then one CPU reference run on the existing A800 node. It does
not reopen R6-A sampling, R6-B/C/D, VIBE, old A0, routes4/5 or real retrieval.
No new random projections, MC, dependencies, CUDA calls or calibration are used.

## Probability and geometry contract

For k fixed GTs G, n outsiders O, M>=k, and 1<=ell<=k, let L be omitted GTs.
With projected squared Euclidean norms S and no ties almost surely, define
t=k-ell+1, h=M-k+ell, T_t the t-th smallest projected GT distance. Then

```text
{L>=ell} = {sum_{j in O} 1{S_j<T_t} >= h}
         = {sum_{j in O} 1{sum_{i in G} 1{S_j<S_i}>=ell} >= h}.
```

GT retention is a prefix of its internal projected ordering, so the rank of
GT_(t) is t plus the outsider count before it. M=N is exactly impossible.
Stable-ID total orders extend the deterministic identity for equal distances;
the stochastic integral is confined to the positive definite continuous models.
It does not add epsilon to comparisons or change any old retrieval semantics.

The ideal Gram is G_ij=sigma_i*sigma_j*((1-c)*1{i=j}+c), with
sigma_i²=1+i/100, N64, k4/8, c0/.5, and m8/32/64. All Gaussian projection entries
have variance 1/m. Original normalization/unit lift and common positive scale
cancel in ranks; no projected renormalization is introduced.
Independently, Z0,Zi~N_m(0,I), and projected u_i equals
sigma_i/sqrt(m)*(sqrt(c)*Z0+sqrt(1-c)*Zi).

For c0, S_i=sigma_i²/m*chi2_m independently. For c.5, R=||Z0||²~chi2_m,
and conditional on R=r, S_i are independent with distribution
sigma_i²*(1-c)/m*ncx2_m(c*r/(1-c)). This follows because their joint conditional
kernel given Z0 depends only on its length. Negative correlations, c1, arbitrary
Gram and multiple heterogeneous factors are excluded, rather than approximated.
The generic mathematical core permits equal positive sigma² as an exchangeable
test fixture; the frozen old-input adapter still requires a strict original GT
boundary. All scientific dimensions and IDs must validate.

## Integral and numerical contract

At a given s,r, F_i=P(S_i<=s|r). Full nonnegative Bernoulli convolution gives
PMFs for GT count A and outsider count B. Direct coefficient tail sums give
F_Tt(s|r)=P(A>=t), g_h(s|r)=P(B>=h). Thus
theta(r)=integral g_h(s|r) dF_Tt(s|r), and Theta=E_R theta(R).
The independent model uses just the inner integral. A node evaluates all64 CDFs
and both PMFs, cached for all9 events of a model. No independent flip sampling
or multiplication of marginal Tri probabilities is substituted.

For any s partition including zero and infinity, interval mass w=F_T(right)-F_T(left)
gives lower=sum w*g(left), upper=sum w*g(right). Infinite endpoint CDFs are
analytic0/1; the tail mass is included. The inner algorithm splits the largest
event gap contribution, at a finite midpoint or a positive mean-scale/doubling
point for its infinite cell. No approximate quantile is assigned an exact weight.

The outer theta(r) need not be monotone. Noncentral chi-square stochastically
increases with r. Conditional GT at a, outsiders at b define theta(a,b).
For a<=r<=b, quantile coupling and event monotonicity give
theta(a,b)<=theta(r,r)<=theta(b,a). Each finite outer cell therefore uses the
lower inner bracket of theta(a,b) and upper inner bracket of theta(b,a), weighted
by its actual chi-square mass. The infinite outer cell contributes [0,tail_mass].
It is split at a chi-square probability midpoint; finite cells are also split
at their actual probability midpoint. No monotonicity of theta(r) is assumed.

Deterministic adaptation picks the outer cell with largest event gap. If the
sum of its two inner gaps, weighted by the outer mass, exceeds half the cell
gap, it refines the inner integral with larger maximum gap; otherwise it splits
the outer cell. New outer cells start with coarse inner partitions. Their raw
brackets can widen, so all successive valid global brackets are intersected.
Such reset widening is separately recorded; it is not mislabeled as FP drift.
Failed/incomplete cell refinements never replace the previous valid partition.
Final distance/factor partitions, progress brackets, counts and audits are saved.

In exact CDF arithmetic these are probability enclosures. SciPy FP64 CDF/ppf,
DP and matrix operations have no directed-rounding proof. Outputs are labeled
`discretization_bracket`, `roundoff_certified=false`; width success is only
`numerically_resolved`. The absolute goal is2^-14 for each event. PMF mass,
CDF mass and actual nested-bracket drift audit tolerance is2^-40. Deviations
within tolerance are retained, without PMF normalization or silent probability
clipping. Nonfinite/out-of-range CDFs, significant mass errors or reversed
intersections produce failure evidence, not a retuned tolerance. Node/time
exhaustion saves the current bracket as `unresolved`; its midpoint is not truth.

## Input, comparison and claim contract

Read only the released R6-A repair1 metadata from scientific commit
`bdd6cc80a67084ca72dded91bd7e50d944e14fd2`. Frozen index SHA is
`0ac8d55bbbf943ea6d7e28e0fafb825f1d095337788afba0c1708834c63bf271`.
The config binds its old config/source map and the exact33 selected paths:
source_freeze, four Gram/ID pairs,12 bounds and12 Gram unit JSONs. Check index,
file sizes/SHA, source/config, stable-ID row maps, strict original GT boundary,
model/event identity and old65536-trial Gram stream. Do not read raw bins,
external labels or VIBE data. Do not migrate old manifests or require that
historical registry bytes equal newly registered M0 bytes.

Old MC uses a floating-point unit-lift Gram. Align it to the ideal using the
fixed common scale Gactual[0,0]/sigma_0², without fitting risk labels. Report
entry error and positive-definite covariance diagnostic E=Gideal^-1/2*
(Gactual/scale-Gideal)*Gideal^-1/2. For rho=||E||2<1,
TV<=sqrt(m)*||E||F/sqrt(8*(1-rho)), capped at1, follows from Gaussian KL and
Pinsker. Its FP64 evaluation is diagnostic, not a formal rounding certificate
or arbitrary low-rank approximation theorem.

All108 original macro events are reported, grouped by ell1/ceil(k/2)/k.
Compare ideal brackets to old simultaneous Gram CP with the geometry diagnostic,
old first-moment/S1/S2 expressions, pair-only column reaggregation and the old
remote-expectation expression floor. These are development consistency checks,
not a new certification or unique attribution of every source of slack. Full
q_j integration, compression, scalar surrogates and static dimension selection
are deferred. Random-Pi risk is distinct from fixed-Pi new-query performance.

## Finite execution and provenance

Traverse c0/k4, c0/k8, c.5/k4, c.5/k8, each m8/32/64. For each model use
M=k,2k,4k and ell=unique(1,ceil(k/2),k). Max32768 distinct finite(s,r) CDF-vector
nodes per model, shared among9 events, with no MC-label-driven adaptation.
There are no stochastic components or seeds to select.

One Linux CPU worker, cumulative1200s, at most4 allocated CPU threads (BLAS
thread envs set to1), RSS4GiB including supervisor tree, artifact256MiB,
temporary128MiB, reserve12GiB. Reuse NumPy1.26.4/SciPy1.13.0 in the existing
environment; no install/upgrade. CPU-only tiny/complete local tests are timed
separately. Clean reviewed canonical commit is pushed and verified before
ff-only node synchronization. No auto retry, resume or resource enlargement.

Exclusive new run directory contains source/config/CPU environment/input
fingerprints, immutable model files, progress, comparison/result/delivery index,
worker log and external execution receipt. Existing read-only bounded writer and
process-tree monitor are reused with GPU monitoring disabled. Worker checks
parent PID, environment, source map every5s and a supervisor deadline (5s reserved
for saving). On failure it saves partial outputs/not-started identities when
possible. External supervision terminates/reaps only its owned child, preserving
overshoot and incomplete-artifact evidence. Sampled monitoring can overshoot
nominal resource limits; actual receipts take precedence over planned limits.
Source/input fingerprints are rechecked before completion. Dependency versions,
source files, governance checker and registry belong to the new source identity.

## Required offline validation and replay

Independent exhaustive small permutations; rational subset PB probabilities;
central exponential CDF; k1 pair agreement with unchanged Tri at c0/.5 and
m1/2/8/32; exchangeable hypergeometric tails; heterogeneous m2 exact5/78 fixture;
inner/outer nesting and crossed endpoints; infinite tails and budgets; invalid
domains, source/index/file/ID rejection; exclusive artifact bounds; real owned
CPU child stop. Full preexisting tests remain unchanged and must also pass.
Coarse mathematical unit-test widths are explicitly separate from frozen M0.

```bash
PYTHONPATH=src:. python scripts/run_tls_rag_step5_route6_macro_reference_v1.py self-test
PYTHONPATH=src:. python scripts/run_tls_rag_step5_route6_macro_reference_v1.py audit-inputs \
  --measurement-dir runs/route6-A-a800-20261103-repair1 \
  --index runs/route6-A-a800-repair1-preflight-20261103/full-run-file-index.json
PYTHONPATH=src:. python scripts/run_tls_rag_step5_route6_macro_reference_v1.py run \
  --measurement-dir runs/route6-A-a800-20261103-repair1 \
  --index runs/route6-A-a800-repair1-preflight-20261103/full-run-file-index.json \
  --run-dir runs/route6-M0-a800-20261103
```

Use the absolute existing interpreter recorded in STATUS on this node; shell
`python` may not exist. The final command is authorized once, after validation.
