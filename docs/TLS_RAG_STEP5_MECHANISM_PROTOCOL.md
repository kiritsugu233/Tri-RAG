# Step 5 mechanism study v1

Approved on 2026-10-24, reaffirmed 2026-10-25: “我申请了A800节点，你开始工作吧，
你连入ssh进去之后执行这次的两阶段测试”. The reviewed small-study section of
IMPLEMENTATION_PLAN.md and configs/tls_rag_step5_mechanism_v1.json define this
new namespace. Existing scientific sources, original source-bound inputs, ANN
protocol and results remain frozen. This document does not authorize another run.

## Data and access

Reuse all 1,000,000 normalized 384-dimensional corpus vectors, the saved float64
128-dimensional Gaussian projection (seed 55117), original exact top-10 truth and
the opened ANN trajectories. No embedding/ground-truth regeneration or HNSW build.
The new reader verifies the anchored parent manifest, old source hashes, index
binding and anchored manifests of the three already-open roles. It checks only
allowed, used files; it never invokes the legacy all-role loader or scans unopened
role directories. Opaque manifest hashes/counts are not query/label access.

Sample by SHA-256 of canonical JSON ["step5-mechanism-v1",55141,role,query_id],
then stable string ID. Fit512/calibration256/selection256/evaluation512 come from
cal_model_fit/cal_bound_fit/tune/tune. All IDs are disjoint. Lexical-family
separation is inherited from the sealed producer; unavailable per-query family
keys are not reconstructed. History exclusion and semantic independence remain
unestablished. Parent tune has already been inspected: evaluation is isolated
from this fit/selection, not a fresh independent sample. formal_certification=false.

Shared parent truth JSONL and label JSON contain other opened-parent records.
Only the trusted reader scans/parses and filters those shared files; it returns
the requested subset only. Evaluation vectors, truth and labels are unavailable
through the group API until the selection seal is written and verified. All
sampling IDs, models, margins, selection, config, code and analysis are sealed
before that opening. This is an auditable interface boundary, not an OS sandbox.
Original cert/latency/test arrays, IDs, labels and records remain closed.

## A: geometry and saved backend decomposition

Use Selection256 and original M=64/256 anchors. E64 is direct float64 squared-L2
exact search with stable string-ID ties, in blocks <=8192. E32 uses FAISS1.8.0
IndexFlatL2 on the same float32 inputs as old HNSW, one query/search thread. Detect
cutoff ties and include the full cutoff group via range search before string-ID
tie resolution. This reference is not claimed bitwise identical to every HNSW
internal distance kernel. Attribute the residual conservatively as combined
backend/precision/tie error, never wholly to graph navigation.

H is each saved raw all-trajectory HNSW call at its actual requested M; C is its
persistent accepted pool. Do not truncate larger ANN calls. Verify for each query
1-R_H = |T\\E64|/10 + |(T intersect E64)\\H|/10 - |(T intersect H)\\E64|/10.
Save omitted/rescued IDs and E64->E32->H->C edges. Count each true neighbor's
projected rank=1+Z+W and verify inclusion at both anchors; compare W with j-1.
Save query-level sets/ranks, quantiles, paired development intervals and difficult
query IDs. A nonzero backend discrepancy is not a failure of the triplet law.
Integrity/equality failure stops B. A has a durable completion checkpoint.

## B: matched pure projection and features

Use exact nested grid [32,64,128,256]. A current-prefix session scores only new
original candidates and caches only these scores. Features/controller never see
future prefixes, exact truth, qrels or realized recall. A maximum exact shortlist
may be retrieved once for adaptive methods; fixed methods extract their own M.
No features or quality prediction are computed at terminal256; it is a fallback.

B preserves the reviewed 13 base features. D adds context/shell log-distortion
q90; T adds boundary/frontier triplet mean/max, using the existing exact Tri-Law
helper and matched sample32+context5+shell8, LID20 geometry. Feature order is B,D,T.
B and B+D execute no triplet calculations. Base validity requires positive original
distances and finite features; D requires valid positive distortion samples in
both groups; T requires valid pairs in both groups, no zero/nonfinite/collinear
required pairs. T does not compute hidden D features. The full family requires all.

Fit four stagewise standardized ridge models, penalty0.01, intercept unpenalized,
scale_floor1e-12, minimum32 valid Fit rows. Residual calibration uses only the
separate calibration group: max(0, higher median of raw prediction minus retention).
The decision score clips raw prediction minus that margin to [0,1]. Invalid states,
missing fits/margins expand. Common thresholds [.80,.85,.90,.95] do not change the
quality targets. Selection chooses the fastest measured eligible fixed budget;
per-family candidates meet mean retention>=.90, Hit@5 delta vs exact>=-.02,
retention delta vs fixed>=-.01, then minimize original distances, tie by policy hash.
No candidate is a valid negative outcome; no substitution or retuning.

Evaluation512 is opened once. The primary prediction metric is adjusted-score
MAE, equal-weight across32/64/128. Raw MAE is secondary. Report RMSE, ten equal-width
score-bin calibration error, and invalid counts. Paired primary MAE uses queries
with all three predictions valid in both B+D+T and B+D; positive evidence requires
full paired coverage. Missing predictions therefore cannot improve a positive
claim by selectively deleting hard queries. B+T vs B is supporting only.

Save per-query states, outcomes, stop budgets, pilot/expansion/rerank counts and
headroom components. Use 5000 paired whole-query-block percentile bootstrap draws,
seed55147, nominal95%; stages/methods/repeats stay in the same sampled block. These
are descriptive development intervals with no formal population guarantee.

## Timing, resources and decision

Measure fixed Selection pipelines directly (one call per budget/query), balanced
rotations of a seeded random method order. Cached trajectory replay is not latency.
On the first64 fixed Selection queries, compare mandatory full-family pilot feature
and controller time with a conservative removable-work upper bound: all fixed256
selection/merge, candidate gathering/scoring, reranking and remaining pipeline
overhead (total minus projection and scan, adding selection back). The shared exact
distance scan is not removable. If the paired95% lower bound of the mean difference
is positive, retain no_headroom and omit the expensive systems-benefit branch.
Continue bounded prediction/quality diagnostics. This is a component screen, not
a measured online adaptive latency. It is deliberately generous to potential
candidate savings, including all fixed shortlist selection as removable work.

Otherwise, if both primary controllers and a fixed comparator qualify, run complete
pipelines on Evaluation512, 16 Selection warmup queries, 3 repeats, one CPU thread,
seed55143 balanced rotations and shuffled query order. Include projection, complete
exact scan/selection, features, controller, candidate access/scoring and reranking.
Reset application caches per method/query. Verify timed quality/budget against
sealed-policy replay. Report warm cloud-backed I/O/page-cache conditions, setup,
mean and pooled-call p95 separately. This is not deployed HNSW serving performance.

Two-hour cumulative process ceiling (including verification/setup), RSS<=16GiB,
new output<=2GiB and >=4GiB data-disk reserve. The wrapper enforces an external
two-hour timeout; internal checks preserve resource-stop state. Forecast throughput
at the first64 A queries and first64 Selection fixed timing queries. No automatic
sample/corpus reduction, second experiment, hardware switch, retry/resume or ANN
integration. An exclusive parent reservation prevents creating a replacement run.
Compact new outputs go to a unique sibling under the Step5 input root; never write
the old bundle/run_archive or restage a full duplicate to the data disk. Verify a
new output SHA manifest on termination. Preserve partial/failing reservations.

Positive evidence requires primary MAE reduction>=5% and paired upper ratio<1,
mean retention>=.90, Hit delta>=-.02, paired retention lower difference>=-.01 and
original-distance upper ratio<=.90 versus both B+D and fixed, complete-pipeline
mean upper ratio<1 and p95 upper ratio<=1.05 versus both. Failure is scoped to this
controller/data/condition, not Tri-Law in general. Prediction-only evidence is
reported as such. A positive result only permits proposing a separately reviewed
HNSW study; full RAG and correlation/novelty experiments remain deferred.

## Replay entry

After local offline checks, push the continuing branch and verify GitHub HEAD,
then synchronize the server at that commit and run server offline checks. Invoke:

```sh
STEP5_PYTHON=/root/autodl-tmp/conda-envs/tri-rag-step5/bin/python \
sh scripts/run_tls_rag_step5_mechanism.sh \
  --bundle INPUT/preparation-ov283f6o/bundle \
  --archive INPUT/experiment-aynq9z39/run_archive \
  --output INPUT/mechanism-UNIQUE
```

INPUT denotes /root/autodl-fs/tls-rag-step5-input-dPpTfj. OUTPUT must not exist;
there is no retry option. A single durable SSH-launched process is this one study,
not a recurring automation. Git bundle fallback is allowed only for already-pushed,
verified commits if direct GitHub pull cannot connect; retain origin unchanged.
