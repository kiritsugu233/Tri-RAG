# Tri-Law numerical v2 repair — 2026-09-07

## Authorization and maintained checkout

User approval in this task:

> 修复你发现的Tri-law问题，然后告诉我修改后的最新版本代码repo在哪，我感觉github上的branch过多了，之后的工作推进从你审查并改完的版本开始。

This authorizes the discovered R1/R2 numerical repairs, corresponding conformance
and specification updates, the approved protection baseline, and continuing work
from the repaired review branch. It does not authorize deleting historical branches,
force-pushing or changing the GitHub default branch.

Repair parent: `c40b7b0132b77a70e11fa2e5425fad3dc77b1116`.
Repair backup: `codex/repo-review-code-protection` at `62377d854ddd183925c4b62ca81653e9c14d9c34`.
The user subsequently reserved this branch as a backup. Current Step 4 development
continues on the existing `codex/tls-rag-step4-retention-v3`; see [current entry](../START_HERE.md).
[GitHub code](https://github.com/kiritsugu233/Tri-RAG/tree/codex/repo-review-code-protection).
Local checkout: `/Users/guanghongxu/Query-Adaptive-Tri-RAG`.
Future authorized steps continue there; no automatic new branch per step.
`TRI_LAW_NUMERICAL_VERSION = 2` identifies this numerical implementation, not a
new Tri-Law theorem or a change to the TLS-RAG retention v3 protocol version.

## Defects and change

The original [review](REPOSITORY_REVIEW.md) and
[failed evidence](review_numerical_evidence.json) remain historical evidence.
The [post-repair evidence](tri_law_numerical_fix_evidence.json) contains the same
four independent reproductions after repair.

- R1: simultaneous near-tie and near-collinearity cancelled the discriminant.
  At beta=1.0000000000000002, rho=0.9999999999999999, m=1, the old probability
  was 1. The corrected threshold is 1.0000000149011612 and probability is
  approximately 0.4999999976284065.
- R2: large finite beta caused inaccurate thresholds or overflow; orthogonal
  beta=1e16 and 1e200 now return exactly beta. Conditional y=beta=1e308, m=4
  now returns 0.5939941502901616 instead of 1.

In `src/tri_rag_harness/tri_law.py`, `_sqrt_threshold` uses
`t=(beta-1)/(2 sqrt(beta) sqrt((1-|rho|)(1+|rho|)))` and
`sqrt(r)=hypot(1,t)+t`. This is algebraically the same law without subtraction
of nearly equal terms or squaring beta. Orthogonal thresholds retain r=beta
exactly. The m=1 Cauchy-square and m=2 reciprocal tails retain representable
small probabilities, including cases where r itself overflows float64. For
m>=3 the existing F survival function is used. Conditional evaluation divides
before multiplication only when m*y overflows. See [specification](TRI_LAW_SPEC.md).

An L1 wrapper cannot repair the authoritative exported numerical functions or
protect their other callers; the approved L0 implementation edit was necessary.
Public signatures, broadcasting, validation tolerance, collinear zero boundary,
existing Monte Carlo tolerances and scientific formulas remain unchanged.
`tests/test_tri_law.py` adds five boundary/high-precision/property regressions;
no existing test is weakened. The audit script only changes its explanatory
docstring. Other implementation modules, configs and launchers are unchanged.

Approved protected paths: `src/tri_rag_harness/tri_law.py`,
`tests/test_tri_law.py`, `docs/TRI_LAW_SPEC.md`, `AGENTS.md`,
`docs/CODE_PROTECTION.md`, `docs/code_protection.json`, and
`docs/TLS_RAG_STEP4_RETENTION_V3.md`. Governance changes record the repaired
baseline and continuing-branch instruction; prior-approval controls remain.

## Validation and replay

From the canonical checkout:

```bash
python3 scripts/check_code_protection.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/audit_numerical_contracts.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -p test_tri_law.py -v
sh scripts/run_tls_rag_step4_readiness.sh /private/tmp/tri-law-numerical-v2-final-20260907
```

Use a fresh output directory for another readiness run. Observed locally:

- Independent audit: all four formerly failing cases pass, exit 0.
- Focused Tri-Law suite: 10/10 pass, including Decimal precision 100 reference
  grid, float64 maximum/subnormals, sign symmetry, broadcasting and monotonicity.
- Two full rounds: 270 tests each, 269 pass, one optional real-FAISS skip;
  31.058 s and 31.031 s. Two v1 focused rounds: 26/26 each.
- Protection inventory/hash guard passes: 178 files (119 L0, 56 L1, 3 L2).
  All 52 registered Markdown files pass relative-link and fence checks;
  `git diff --check` passes.
- Six synthetic readiness outputs are byte-identical between runs;
  real-data gate remains closed.
- Additional deterministic Decimal-100 threshold sweep: 2,000 cases, seed
  9072026, maximum relative error 6.661338147750939e-16 (bound 3e-15).
  Betas alternate exp(uniform(0,709)) and 1+10**uniform(-15,3); rhos use
  uniform values or 1-10**uniform(-15,-1). This exploratory sweep supplements
  the committed deterministic conformance grid.

Temporary full log: `/tmp/tri-law-numerical-v2-final.log`; readiness artifacts:
`/private/tmp/tri-law-numerical-v2-final-20260907/{a,b}`. Durable audit output is
linked above. These tests do not prove correctness for every possible input or
establish real-data effectiveness, certification or latency improvement.
No cluster, GPU, model-download or protected real-role evaluation was run.

## Source binding and historical replay

`tls_rag_step4_probe.CODE_FILES` binds `tri_law.py`; retention v3 uses that
parent loader. An old source-bound real bundle is therefore expected to fail
code-hash validation on this repaired checkout. Do not refresh its hashes,
weaken the loader, or claim the old bundle was evaluated by numerical v2.
The new constant does not automatically migrate caches or bundle schemas.
Risk values and downstream fingerprints can change. The existing v1 synthetic
readiness equality is not a promise of bitwise equality for every old artifact.

Replay historical experiments at their original exact commit, including the
pre-repair baseline `248c29e2243c60b3793458c8ee723eefba5a5855`, in a separate
Linux historical checkout. New repaired experiments need a reviewed source
binding/run namespace and applicable independent evaluation roles; cert/probe
inspection is not undone by changing code. Historical archives remain untouched.
For a separate Linux development checkout after synchronization:

```bash
git clone --branch codex/repo-review-code-protection --single-branch https://github.com/kiritsugu233/Tri-RAG.git Tri-RAG-reviewed
cd Tri-RAG-reviewed
git rev-parse HEAD
python3 scripts/check_code_protection.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/audit_numerical_contracts.py
```

Compare the printed SHA with the delivered repaired commit before replay;
install the repository's declared dependencies in the cluster environment.
These are replay instructions, not reported cluster results.
