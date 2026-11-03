# Scheme6 R6-A v1 — local implementation and deferred A800 verification

2026-11-02. New namespace `tls-rag-route6-A-v1`, independent of old A0 and routes4/5.
After the exact section42.9 nine-path proposal, the user instructed:

> 你先改repo的代码吧，改好之后我开启服务器让你进行测试

The automatic review initially rejected scientific tests/shared test entry as
insufficiently path-scoped. No rejected command was rerouted; the user then
explicitly approved the listed nine paths and replied:

> 批准上述九路径及本地测试、同步

That explicit scope authorizes the proposed R6-A code, offline CPU tests/full regression and
standing GitHub synchronization. A800 connection, environment installation and
scientific execution wait for the user's server release. R6-B/C/D, VIBE download,
real data, tree optimization and a deployed static selector remain out of scope.

On2026-11-03 the user opened the A800 node. Existing fixed CPU packages and
Torch2.8.0+cu128 were reused without installation. Linux32 tests and tiny CUDA
checks passed, but the first run on `dde6720` stopped after153.482226s: an empty
boundary complement produced eta=-2.22e-16 by subtracting reordered sums.
The original failed run/lock remain immutable. The user then approved section44's
exact five-path numerical repair, tests/synchronization and a fresh R6-A run:

> 批准五路径修复、测试同步及新一轮R6-A（推荐）

Numerical revision1 directly sums the complement; it never clips eta or relaxes
the existing negative-input guard. Mathematical v1 contract, seeds, grid and
resource settings remain fixed. The fresh run has its own source fingerprint,
environment lock and `route6-A-a800-20261103-repair1` directory, starts at sample0
and has a separately approved maximum3600s. Counts are not pooled with the failed
run. Recheck the reviewed source, full tests and independent GPU fixture first.

## Probability, geometry and fixed finite design

The complete finite design is in `configs/tls_rag_step5_route6_v1.json` and the
approved section42.7. Sixteen geometries: twelve single-target N5/k1 and four
macro N64/k4 or8; ambient dimension128. Single-target m2/8/32/64 and M1/2/4;
macro m8/32/64, M=k/2k/4k, ell=unique(1,ceil(k/2),k). Total60 geometry/m units.

Single residuals I/equicorrelation0.5/positivecycle0.25/negativeclosingcycle0.25,
beta2/alpha0 or0.5 produce8 cases. Three independent alpha0 gap cases use
beta1.05/1.25/4; one beta2 repeated residual case retains exact duplicate corpus
vectors as distinct IDs. Macro Gram has diagonal1+i/100 and I/equicorrelation0.5.
No target boundary ties; competitor ties are allowed. All points and queries
have unit norm before projection and are not renormalized after projection.

Lift c=-diag(G)/2 must lie in range of a declared-rank PSD G; use minimal norm
solution, scale squared1/[4(1+qmin²)], add one orthogonal query component.
Declared ranks are N except repeated residual rank2. Cholesky/full-rank factors
and analytic repeated factor preserve the intended geometry. Numerical audit
tolerances1e-11 relative/1e-13 absolute validate reconstruction, never change a
ranking event. No jitter/nearest-PSD repair. The Gram sampler factors the actual
reconstructed displacement Gram, retaining its declared rank and recording bytes.
Stable IDs are `route6:{geometry}:corpus:{row}` and a distinct query ID; row-based
binary artifacts refer to explicit ID-map files.

Each unit has two independent PCG64 streams with2^16 matrices/path: unit-coordinate
projection and full displacement-Gram projection. Both preserve shared Gaussian
rows across all competitors. Masters64101/64103; SHA256 of JSON
[namespace,master,geometry,m,path], separators comma/colon and ensure_ascii=True,
provides the full256-bit seed. Normal entries have standard deviation1/sqrt(m).
Draws are in C order, so a different memory batch does not change sample IDs.
Batch128; first32 samples per path audited on CPU FP64 and diagnosed on CPU FP32.
Job permutation seed64105. Every M/ell in one Pi shares the same sample.

## Mathematical bounds and information accounting

Primary omission L=|G_original_topk minus projected_exact_topM|. Across-cut
inversions satisfy C_cross>=L(M-k+L), hence Markov yields
P(L>=ell)<=sum p_ij/[ell(M-k+ell)], clipped at1, with impossible events exactly0.
This is a necessary-condition bound, not a recovered joint law or an ANN guarantee.

Classical marginal interval uses the best trimmed Markov upper and complement
lower over unrestricted Bernoulli couplings. Single-target counts are exact rank
counts. Macro target outsider counts at threshold M-k+1 give necessary-event upper
bounds only; macro lower omission probabilities cannot be inferred from them.
Boundary selection is the eight highest exact marginal risks per target (or all
if fewer), tied by corpus row, fixed before draws. Remaining eta is never discarded.
Evaluate all t=0,...,h-1 splits P(Zlocal>=h-t)+eta/(t+1), h=M-k+1; minimize valid
bounds, with same-input S1 baseline.

Conditional marginal CDF has variance beta-alpha² and noncentrality
alpha²*T/(beta-alpha²), with T~chi-square(m). Monotone enclosure requires
|alpha|<=1 and strict-farther beta>1. K8192 equal-probability Riemann bins;
endpoint0 at0 and safe upper endpoint1 at infinity. Mathematical discretization
bracket<=1/K per monotone integral; SciPy rounding is separately uncertified.
Classical and conditional upper bounds may be intersected. No conditional
independence is assumed by these marginal bounds.

For S2, coordinate-stream CP upper endpoints for selected boundary joint events
are explicitly extra statistical information. With local mean mu and sum upper
joint probabilities S, variance upper is mu+2S-mu². Negative variance produces an
inconsistent-input status and conservative bound1; it is not silently repaired.
Cantelli and order2 factorial bounds are combined with the marginal bound and
remote residual. These are not marginal-only predictions. Independently drawn
Gram-stream omission intervals assess them. Same-stream apparent agreement is
not claimed as independent validation. No Monte Carlo plug-in covariance is
labeled an exact analytic covariance. Full tree LP is not implemented.

Scalar law uses gamma and the regularized Beta identity. The small argument
(1-gamma)/2 is computed through a stable reciprocal threshold, not subtraction
from a rounded gamma=1. m1 has a Cauchy branch to retain representable extreme
small probabilities; exact collinearity alone branches to0. Chernoff power and
exponential upper envelopes are numerical evaluations of mathematical bounds.
The210 primary grid points include signed-rho symmetry checks. Compare against
the unchanged stable Tri implementation, atol1e-12+rtol1e-8*p; the grid is finite
numerical conformance, not a proof of uniform approximation/roundoff error.
No online lookup approximation, fitting or performance-optimized surrogate ships.

## Catalogue, uncertainty and recomputation

All geometries and the FULL event catalogue are sealed before the first
scientific Pi (and before the separate eight-sample integration self-check):
marginals for all target/outsider pairs, boundary pair joint events, each target's
omission at each M, and macro omissions at each M/ell, for both routes/all units.
Bonferroni family alpha0.05 spans that catalogue; each CP tail uses alpha/(2J).
Risk thresholds0.01/0.001 are predeclared. Upper endpoint<=delta passes,
lower>delta fails, otherwise undecided; zero events have positive upper bounds.
No extra samples/scenarios/seeds after inspection. A ring's proven small gap may
remain unresolved at this sample size; retain exact T1 evidence rather than
claiming equal tails or chasing significance.

FP64 distance envelopes use the standard IEEE dot-product forward-error model,
gamma_d=d*u/(1-d*u), including projection, coordinate subtraction, squares and
sums. Definite comparisons require disjoint intervals; possible comparisons
include overlap. No epsilon turns a strict tie into a flip. Definite/possible
rank counts yield lower/upper missing counts and Bernoulli event envelopes.
CP lower endpoints use definite counts, upper endpoints possible counts.
This assumes ordinary IEEE arithmetic, no overflow/underflow model violations
and the declared Gram factors; it is NOT formal directed-rounding certification.
Nonfinite distances stop. CPU/GPU prefix differences must fit combined envelopes.
FP32 is a separately labeled CPU diagnostic, never mixed into the main sample.

Raw binary records preserve sample IDs, rank envelopes, packed outsider inversion
masks and event masks. Every file has a dtype/layout/event-ID map; bitorder=little.
Each binary layout is sealed before its first sample write, so a partial file
remains self-describing. Each unit/path uses one append-only binary file with bounded writes, not tens of
thousands of tiny batch files. Counts/intervals can be recomputed from these bytes.
Projection matrices are not all persisted; fixed RNG/seed/geometry/environment
and sample IDs permit replay. CPU and GPU results need not be byte-identical;
scientific identity is the recorded artifact/environment, not cross-device FMA equality.

## CPU tests now and GPU lock later

CPU imports require only NumPy1.26.4/SciPy1.13.0. Torch imports are lazy.
No GPU test silently falls back to CPU or reports a skip as an A800 pass.
Local self-test uses8 samples/path for one fixed geometry and explicitly says
integration fixture only. CPU tiny tests may inject smaller configs; the production
CLI has no grid/sample override and only stage A exists.

A800 preflight later writes an exclusive environment lock containing exact
Python/NumPy/SciPy/PyTorch/CUDA/driver/device UUID/precision configuration and
torch extension SHA, plus source map/config fingerprint. No automatic install or
version substitution. Resource budgets are derived from actual cgroup memory,
CPU quota/affinity and free GPU memory. GPU calls require Linux and visible cuda:0
named A800. UUID must match nvidia-smi; older torch without a UUID property is accepted only
when exactly one physical and one visible GPU have matching names, recording
the unique physical UUID. Ambiguous mapping blocks execution. If a compatible wheel/environment cannot
be restored, report blocked; do not modify old dependency files or binaries.
GPU precision API is chosen by available PyTorch API family, never mixing old
allow_tf32 with new fp32_precision. Main path FP64; TF32 disabled for diagnostics. Deterministic torch algorithms
and an explicitly recorded cuBLAS workspace are required; no torch RNG is used.
After catalogue sealing, a separate eight-sample device self-check must pass
before the scientific sample loop.
All CUDA timing boundaries synchronize; RNG, H2D, kernels, D2H, counting, artifacts,
risk computation and overall wall times are separately recorded. No retrieval
latency or application speedup can be claimed from these diagnostic clocks.

## Resource supervision and source identity

Scientific wall<=3600s from source freeze, one worker/one GPU. Device allocator
limit=min(24GiB,60% preflight free), process-tree RSS=min(32GiB,60% cgroup allocation),
CPU threads<=8 and allocated cores. Outputs8GiB, tmp4GiB, disk reserve12GiB.
0.25s external process-tree/nvidia-smi monitoring includes the supervisor and its
worker, including parent CUDA context memory; bounded1MiB writes check budget
and reserve before writing. Sampling/shutdown/receipt-size overshoot is recorded,
not denied. TERM then5s grace and KILL. Monitor errors also kill owned workers
and produce failure receipts. GPU allocator's own cap supplements external process
memory accounting, which includes context memory. No automatic stage transition,
resume, resource enlargement, deletion of old artifacts or source downloads.

Source/config/old Tri/guard/registry map is frozen; clean tracked checkout and
protection checker pass before scientific execution. Worker belongs to its parent
supervisor and cumulative deadline. Source hashes are checked at least every5s
at checkpoints and after completion; deadline checked each batch. CPU/full tests
are run before source commit/push; the A800 later uses that exact reviewed commit
through fetch/ff-only without resetting server changes.

Delivery index contains immutable scientific artifacts. Supervisor source freeze,
streaming worker log and final execution receipt are separate orchestration
evidence; final receipt reports sampled artifact bytes before writing itself.
A successful process means only finite measurement completed. Statistical anomaly
candidates/uncertain cells must be reviewed before passing any scientific gate.

## Replay

Local, without torch or server access:

```bash
PYTHONPATH=src /private/tmp/joint-rank-t1-test-env-20261102/bin/python \
  scripts/run_tls_rag_step5_route6_v1.py self-test
```

After the user opens the A800 node, source/environment preflight and explicit
review of the actual lock (no installation is implied):

```bash
python scripts/run_tls_rag_step5_route6_v1.py preflight --output /new/path/environment-lock.json
python scripts/run_tls_rag_step5_route6_v1.py gpu-check \
  --environment-lock /new/path/environment-lock.json --output /new/path/gpu-check.json
python scripts/run_tls_rag_step5_route6_v1.py run --stage A \
  --environment-lock /new/path/environment-lock.json --run-dir runs/route6-A-UNIQUE
```

Commands are implemented interfaces. They are not evidence of an executed A800
check, server authorization, complete scientific results or RAG headroom.
