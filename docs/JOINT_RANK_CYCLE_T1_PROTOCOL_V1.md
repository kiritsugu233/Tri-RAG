# Joint-rank cycle T1 v1 — approved deterministic verification protocol

Date: 2026-11-02, Asia/Shanghai. New independent theory namespace.
Protection: L0. This freezes verification semantics, not publication novelty.

## Authorization and scope

The immediately preceding question asked approval for the six paths in
[T0 section7](JOINT_RANK_THEORY_T0.md#7-最小-t1-受保护范围提案尚未批准所有新路径尚未创建).
The actual user reply was:

> 批准，你的意思是tri-law已经有人用过了吗

Proposal/source parent: `a74e45b34d28e756bf1ef61f9bb4c074804d75c7`.
Authorized paths: `mechanism/joint_rank_cycle_v1.py`,
`tests/test_joint_rank_cycle_v1.py`, this protocol, one new test-module entry in
`scripts/run_tests.sh`, and narrow registration/consent changes to
`docs/CODE_PROTECTION.md` and `docs/code_protection.json`.
Existing scientific implementations, tests, configurations, roles, source-bound
artifacts and dependency versions are unchanged. Shared test-entry membership
does not turn this into a continuation of the Step5 systems experiments.

No Monte Carlo, numerical integration, moment/state LP, real-data access,
retrieval/build experiment, environment installation or T2 study is authorized.
All seeds are `not_applicable`: there are no stochastic checks in this phase.

## Probability object and fixed finite cases

Geometry is fixed before projection: `a=x0-q`, `bj=xj-q`,
`0<||a||<||bj||`; all entries of a shared Pi are iid `N(0,1/m)`.
`Ij=1{||Pi*bj||²<||Pi*a||²}`, `Z=sum Ij`, `theta=P(Z>=M)`.
Target ties have Gaussian probability zero under the strict original-distance
assumption. Deterministic rank checks explicitly refuse target-distance ties;
competitor-competitor ties count their distinct identities.

The claim family has `m=2,M=n`, `a` orthogonal to all `bj`, and
`||a||²=1,||bj||²=beta`. Residual Grams R+ and R- have diagonal1 and an unchorded
n-cycle of absolute edge value r; only the closing edge changes sign.
The mathematical draft addresses n>=3, beta>1 and 0<r<1/2. The software/case
certificates in this phase are restricted to n=3/4, beta=2, r=0 or1/4.
The public arithmetic helpers may accept other exact beta>1 or 0<=r<1/2, but
no such extra scientific case is run under this protocol. Unsupported n is refused.

## Exact certificate APIs

- `signed_cycle_gram(n,r,cycle_sign=...)`: residual R, exact integers/Fractions;
  not an empirical covariance estimate. The domain excludes r>=1/2, without a
  claim that every matrix outside that sufficient domain is indefinite.
- `cycle_determinant_certificate(n,r)`: every coefficient/principal minor of
  `D±(s)=det(I+R±diag(s))`, plus actual diagonal-sign witnesses for every proper
  subset. Empty subset determinant is1. The sole possible nonzero difference
  coefficient is the full monomial, `4*(-1)^(n-1)*r^n`.
- `validate_gram_and_lift(G,scale_squared=...,ambient_dimension=...)`: exact PSD
  via all principal minors; exact rank/range solving; strict target geometry;
  minimum query norm and point-Gram certificate for a unit-sphere realization.
  Singular PSD inputs are classified explicitly; inconsistent range/length/
  ambient-space conditions produce an infeasible lift, with a reason. No jitter.
- `rank_tail_gap_lower_bound(n,beta,r)`: exact analytic lower bound only for the
  stated m=2/M=n family; output is not an estimate or the exact gap.
- `rank_count_from_squared_norms(...)`: `(Z,1+Z)` in exact arithmetic, with target
  ties/negative squared norms refused. Common positive scaling preserves output.

Float/bool inputs are refused at exact-rational boundaries. No epsilon changes
an event, PSD classification, rank, range test or scientific assertion. Coordinates
involving square roots are unnecessary: point Grams provide exact witnesses.

## Mathematical review obligations

1. SPD: each residual Gram has eigenvalues at least `ell=1-2r>0`.
   Full displacement G is `diag(1,beta R)` with rank n+1. All proper cycle
   subsets are forests; sign switching preserves their projected norm laws,
   including the independent shared-target length, hence all subset inversion laws.
2. Two real Gaussian rows give `L±(s)=1/D±(s)` for the competitor norm vector
   scaled by beta. The target squared norm is an independent `Exp(1)` variable.
3. Principal-minor expansion and permutation expansion independently check
   `D+-D-=4*(-1)^(n-1)*r^n*product(sj)`. Thus the Laplace transform of
   `F--F+` is `4*(-1)^(n-1)*r^n L-L+`. The inverse is the density h of the
   sum of **independent vectors** X-+X+, not independent competitors within Pi.
   Positive continuous densities on the open orthant justify taking diagonal values.
4. Therefore the absolute rank-tail gap is
   `4*r^n*beta*integral exp(-beta*t)*h(t,...,t) dt`.
   Polar-coordinate Gaussian density bounds give
   `f_R(x)>=exp(-sum(xj)/ell)` and `h(t,...,t)>=t^n*exp(-n*t/ell)`.
   Integrating this analytic lower function symbolically gives
   `4*n!*beta*r^n/(beta+n/ell)^(n+1)`.
5. With `c_i=-Gii/2`, solve `Gz=c` exactly. For common scale squared
   `epsilon²=1/(1+beta*n/ell)`, scaled minimum query norm squared is at most1/4.
   An extra orthogonal dimension makes all `q,q+epsilon*v_i` unit vectors in
   dimension n+2. Direct reconstruction from the point Gram must recover
   every original displacement product multiplied by epsilon².

The exact gap lower fixtures are `3/16384` for n=3 and `3/400000` for n=4,
beta2/r1/4. Determinant-difference coefficients are respectively1/16 and−1/64.
r=0 collapses the two distributions. A certificate checks finite algebra and
admissibility; validity of the analytic probability chain is a separate proof
review obligation, not something a test pass establishes automatically.

## Boundary fixtures and independent checks

Fixed boundary fixtures are the old n=2 independent/antipodal residual examples
(analytic probabilities1/6 and1/3), strict n=1 collinearity (zero reversal),
indefinite/asymmetric/invalid-distance Grams, consistent and inconsistent singular
Grams, minimum query norm above/equal/below1, insufficient ambient dimension,
invalid exact-number domains, target ties and competitor ties.
The antipodal example has c outside the Gram range and cannot be lifted by
positive common scaling. Duplicate residuals may instead be range-consistent.

All fixed cases use exact rational arithmetic. Independent determinant checks
use the Leibniz permutation formula rather than production elimination. Evaluation
on the Boolean grid for the formal variables s checks every multilinear coefficient;
this is an algebra check, not an added geometric/statistical parameter grid.
Every proper-subset sign witness is verified entry by entry. Exhaustive finite
ranking/order checks verify the omission event and scaling invariance.

## Resources, evidence and handoff

New T1 scientific checks: cumulative maximum300s, RSS1GiB, evidence32MiB.
On a limit or proof failure, stop and preserve partial output; do not add cases.
Existing full repository regression is separate validation work, with its cost
reported separately. Use only an already available fixed-version environment.
The known incomplete old FAISS environment must not be repaired/installed by
interpreting this six-path approval as environment-install consent.

Evidence directory: unique, exclusive-create
`runs/joint-rank-theory-v1/t1-<unique-id>`; do not overwrite an earlier result.
Save exact certificates, source/environment identities, approval/proof review,
commands, stdout/stderr, observed time/resources and actual validation status.
There are no per-projection records because no projection is sampled.
Any environment failure is retained and never relabeled as a full-suite pass.

Run the new offline unittest module and the complete repository suite; zero skips
are required for full implementation acceptance. Keep all old L0 byte hashes,
registry rows/reasons and historical scientific bindings except the expressly
approved append-only human-register hash update. Adding the test module changes
the new identity of `scripts/run_tests.sh`, which is bound by old route4 source
maps; do not refresh old manifests or rerun those experiments.

Implementation validation does not establish publication novelty. The signed-cycle
principal-minor mechanism is prior work, and the scalar law is equivalent by
polarization to Durrant–Kaban2013 Thm3.2. Retain the
[T0 prior-work qualifications](JOINT_RANK_THEORY_T0.md#3-先行文献对照与阅读深度),
including the inaccessible1977 full text. Any claim of a novel theorem or T2
numerical study needs a separate review and scope.
