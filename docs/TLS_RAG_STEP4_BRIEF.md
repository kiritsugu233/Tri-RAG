# TLS-RAG Step 4 frozen transfer protocol

This milestone freezes a protocol and tests its guards using generated synthetic
inputs only. It does not run real transfer, select a method, certify performance,
measure latency, or start Step 5. The machine-readable specification is
`configs/tls_rag_step4_protocol_v1.json`; changing any field requires a new
protocol identity. Real data bindings are deliberately absent, not wildcards.

## Scientific question and frozen mechanism

Does the frozen two-action TLS-RAG controller meet preregistered evidence and
retention targets on fresh external queries, and what retrieval work does it
require? Rows 1--4 diagnose the contribution of each feature family. No claim
that Tri-Law alone guarantees evidence or answers is part of this protocol.

Step 2, Step 3, and exact Tri-Law source/config/tests remain unchanged. The
transfer preserves normalized original embeddings, one dense Gaussian matrix
with variance `1/m_prime`, seed 24103, `m_prime=2`, no projected renormalization,
exact squared-L2 search and string-ID ties, `[3,6,12]` budgets, pilot 3, `k_gt=2`,
`k_ctx=2`, two expansions, prefix/cache reuse and original reranking. These small
budgets are deliberately a literal transfer of the frozen mechanism; failure
on a larger corpus is admissible and cannot trigger silent enlargement. The
original embedding dimension and model must be bound before any real role opens.

Row 1 contains fixed 3/6/12 references. Rows 2--4 use exactly the Step 3 feature
registries: base, then observed-pair geometry, then deterministic plan/facet
features. Step 4 has distinct outer candidate IDs, schemas and artifacts; any
call to the upstream controller uses its original candidate identity. The
current-prefix risk profiles, ridge scores, quantiles, validity rules, median
bins with right assignment, minimum cell size 8, candidate-specific reachability,
vacuous `[0,1]` intervals, and dual thresholds 0.65/0.15 are preserved. Invalid
states with finite, correctly typed inputs expand only to the next grid entry.
Malformed or nonfinite deployable inputs are rejected before a decision,
preserving the upstream input contract without imputation. At the last entry, STOP records
nonattainment or invalidity; it does not assert success or grow the grid.

## Statistical interpretation agreed for Step 4

The unchanged internal Clopper--Pearson calculation remains a controller
calibration mechanism. Its nominal alpha 0.2 is divided over
`3 candidates * 3 stages * 2 bins * 2 outcomes = 36` cells. Only the remaining
event upper and sufficiency lower sides are used for stopping. This allocation
does not establish simultaneous population coverage for the full data-adaptive
binning/reachability procedure. Step 4 makes no such claim and does not treat
these values as per-query posteriors. This interpretation leaves all Step 3
behavior and historical files intact.

Formal acceptance, if later authorized, uses a separate independent query role
after the entire controller is frozen. Its alpha budget is separate from the
internal calibration alpha. Readiness tests exercise protocol guards, not the
validity of a new statistical theorem.

## Roles, opening order and independence

The fixed order is `query_cal_model_fit` (256), `query_cal_bound_fit` (512),
`query_tune` (256), `query_cert` (1024), `query_latency` (256), `query_test` (1024).
The first two are mutually exclusive subdivisions of `query_cal`. Fit the score
models only on the first; build the unchanged internal tables on the second.
Freeze each output before the next role opens. On tune, cert and test, close and
serialize all role Phase A decisions before opening that role's labels. Latency
never has a label-opening transition. No calibration query is reused elsewhere.

Use stable string IDs and one query per independently sampled question family.
Query and family IDs must be globally unique; query IDs are external to corpus
IDs. Sort families by SHA-256 of the split seed 44109 and family ID, then assign
the fixed counts in the specified order before labels are inspected. Related
paraphrases cannot be counted as independent observations. Syntactic ID checks
prove only mechanical disjointness; a future sampling and semantic-duplication
audit must establish the iid question-family assumption from the frozen target
population. If that audit cannot support it, this protocol cannot certify the
dataset. No query-stage pseudoreplication is permitted.

## Selection, acceptance and terminal failure

Future selection on tune considers exactly the three adaptive candidates. Keep
candidates meeting all three empirical targets below, choose the smallest mean
number of newly computed original distances, and break ties by ascending Step 4
candidate ID. This is a work objective, with no latency implication. If none
qualify, end with failure. Freeze a receipt binding the chosen candidate, models,
tables, role assignment, protocol and code before opening cert inputs.

On the complete 1024-query cert role, evaluate exactly three primary endpoints
for that single selected controller. Set `alpha_cert=0.05`, with one-sided
`a=0.05/3` per endpoint, and accept only if all three pass:

| Endpoint, one value per query | Prespecified bound and gate |
| --- | --- |
| Early stop with later useful evidence | Binary CP upper <= 0.05 |
| Terminal context insufficient | Binary CP upper <= 0.10 |
| Exact original top-2 retention | Bounded-mean Hoeffding lower >= 0.90 |

The first event is `budget < 12` AND the frozen Step 3 later-context event at
that stop stage. Its denominator is all cert queries, including those that
expand to 12. Always expanding makes that event zero without demonstrating any
work saving; report early-stop frequency and work separately. The second event
is the negation of the unchanged context-sufficiency label at termination. The
third is the fraction of the original exact top-2 retained in the candidates
(equivalently after exact original top-2 reranking). Invalid input/failure rows
are retained and counted conservatively as insufficient with zero retention;
unresolvable early-stop events are counted as failures. Do not omit them.

For binary event count x among n independent queries, the one-sided CP upper
is 1 if x=n, otherwise Beta quantile `q(1-a; x+1,n-x)`. For retention values in
`[0,1]`, use `max(0, mean - sqrt(log(1/a)/(2*n)))`. The iid premise is required;
these formulas are protocol definitions, not executed certification here.
The targets are design choices fixed before data, not estimates or promises.
No correction over internal stages or discarded tune candidates is needed for
these three endpoints of one controller fixed independently of cert. Other
Rows 1--4 results are descriptive and cannot select a replacement on cert.

Any failed bound, missing query, invalid binding, or insufficient independence
audit yields failure/inconclusive, never automatic retry, sample top-up or
budget growth. After cert inspection, any policy change requires a new version
and fresh independent roles. Failure does not open latency or test as a way to
rescue a claim. Successful cert is followed only by separately specified,
authorized label-free paired systems measurement and final descriptive test
reporting; those later runners are outside this milestone.

## Identity chain and real-data gate

Before any real data operation, a separate authorization must name the minimum
adapter, corpus, query/role-manifest and label paths. This milestone has neither
discovered nor read those paths. A later preregistration must bind target
population/sampling, dataset/corpus, embedding model revision/dimension,
normalization/projection, external IDs and independent units, annotation
semantics/provenance, deployable plan/facet generator, code/environment/seeds,
this protocol, and complete split identity. Unknown or placeholder identities
cannot pass. If evidence annotations cannot express the frozen Step 3 events,
stop and version the protocol before outcomes are opened.

Readiness has no real adapter, data loader, arbitrary input path or enable-real
switch. It verifies the pinned protocol before generating synthetic records,
uses distinct synthetic namespaces, validates strict deployable input shapes,
and simulates role transitions with hash-bound receipts. It is an input guard,
not an OS security boundary or a completed real runner. Real binding and exact
path authorization require a later explicit implementation/review; nothing in
this protocol grants access to real data.

Synthetic output contains only protocol identity, generated role assignment,
transition and denial records, label-free controller probes, and a fingerprint
manifest. No timing, selected method, empirical quality result or certification
result is emitted. Same-host byte equality is required. Step 3 raw artifact
identities remain host observations; its complete frozen Step 2 semantic
identities remain the cross-host compatibility gates.
