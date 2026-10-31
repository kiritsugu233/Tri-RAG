#!/bin/sh
set -eu
repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repository_root"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "${TRI_RAG_TEST_PYTHON:-python3}" - "$@" <<'PY'
import argparse
import json
from pathlib import Path
import sys
import unittest


# Explicit file membership: new files require a reviewed classification, while
# new methods in an existing file are automatically included by discovery.
STEP5_MODULES = frozenset("""
test_attribution
test_certification
test_code_protection
test_config_projection
test_end_to_end
test_index_retrieval
test_lid_policy
test_mprime_sweep
test_retrieval_benchmark
test_tls_rag_step2
test_tls_rag_step3
test_tls_rag_step5_ann
test_tls_rag_step5_ann_comparators_v2
test_tls_rag_step5_ann_continuation_v1
test_tls_rag_step5_ann_continuation_v2
test_tls_rag_step5_ann_enhancement
test_tls_rag_step5_baseline_headroom
test_tls_rag_step5_mechanism
test_tls_rag_step5_reuse_probe
test_tls_rag_step5_route4_screen_v1
test_tri_law
test_tri_predict
""".split())
LEGACY_MODULES = frozenset("""
test_beir_dataset
test_text_embeddings
test_real_dimension_sweep
test_real_original_baseline
test_real_policy_certify
test_real_policy_test
test_real_policy_tune
test_real_tune_diagnostics
test_pdctp_calibration
test_pdctp_dataset_audit
test_pdctp_features
test_pdctp_fiqa_embedding_gate
test_pdctp_fiqa_query_cal
test_pdctp_fiqa_query_cert
test_pdctp_fiqa_query_tune
test_pdctp_foundation
test_pdctp_policies
test_pdctp_protocol_statistics
test_pdctp_real_protocol
test_pdctp_v3
test_tls_rag_step4
test_tls_rag_step4_a800_replay
test_tls_rag_step4_probe
test_tls_rag_step4_repaired_acceptance
test_tls_rag_step4_retention_v3
""".split())


def fail(message):
    print(f"Test collection error: {message}", file=sys.stderr)
    raise SystemExit(2)


def leaves(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from leaves(item)
        else:
            yield item


parser = argparse.ArgumentParser(prog="scripts/run_tests.sh")
parser.add_argument("suite", choices=("full", "step5", "legacy"), nargs="?", default="full")
mode = parser.parse_args().suite
if STEP5_MODULES & LEGACY_MODULES:
    fail("step5 and legacy file memberships overlap")
classified = STEP5_MODULES | LEGACY_MODULES
expected_paths = {f"tests/{name}.py" for name in classified}
actual_paths = {path.as_posix() for path in Path("tests").rglob("test_*.py")}
if actual_paths != expected_paths:
    fail(json.dumps({"unclassified_files": sorted(actual_paths - expected_paths),
                     "missing_files": sorted(expected_paths - actual_paths)}))

# Always import/discover the entire repository before filtering, so a partial
# run cannot hide an import failure in the other cohort.
loader = unittest.TestLoader()
discovered = loader.discover("tests")
if loader.errors:
    fail("\n".join(loader.errors))
tests = list(leaves(discovered))
ids = [test.id() for test in tests]
if len(ids) != len(set(ids)):
    fail("duplicate test IDs")
collected_modules = {test.__class__.__module__ for test in tests}
if collected_modules != classified:
    fail(json.dumps({"unclassified_modules": sorted(collected_modules - classified),
                     "modules_without_tests": sorted(classified - collected_modules)}))
cohorts = {
    "step5": [test for test in tests if test.__class__.__module__ in STEP5_MODULES],
    "legacy": [test for test in tests if test.__class__.__module__ in LEGACY_MODULES],
}
selected = tests if mode == "full" else cohorts[mode]
print(json.dumps({
    "suite": mode,
    "collected": len(tests),
    "selected": len(selected),
    "cohorts": {name: len(items) for name, items in cohorts.items()},
    "not_selected": {name: len(items) for name, items in cohorts.items()
                     if mode != "full" and name != mode},
    "full_repository_scope": mode == "full",
}), flush=True)
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(selected))
print(json.dumps({
    "suite": mode,
    "executed": result.testsRun,
    "successful": result.wasSuccessful(),
    "skipped": len(result.skipped),
    "full_repository_validation_passed": (
        mode == "full" and result.wasSuccessful() and not result.skipped
        and not result.expectedFailures and result.testsRun == len(tests)
    ),
}), flush=True)
raise SystemExit(0 if result.wasSuccessful() else 1)
PY
