#!/bin/sh
# Explicit exit handling: never terminate the caller's interactive shell.
if [ "$(uname -s)" = Darwin ] && [ "$(pwd -P)" != /Users/guanghongxu/Query-Adaptive-Tri-RAG ]; then
  echo 'Use /Users/guanghongxu/Query-Adaptive-Tri-RAG for local work.' >&2
  exit 2
fi
case "${1-}" in
  a800-prepare) [ "$#" -eq 2 ] || exit 2 ;;
  a800-replay) [ "$#" -eq 3 ] || exit 2 ;;
  *) if [ "$#" -ne 2 ]; then
       echo 'Usage: script PREVIOUS_V2_RUN ABSENT_OUTPUT | a800-prepare ABSENT_INPUT | a800-replay INPUT ABSENT_OUTPUT' >&2
       exit 2
     fi ;;
esac
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 scripts/check_code_protection.py
protection_rc=$?
echo "Protection check exit code: ${protection_rc}"
if [ "$protection_rc" -ne 0 ]; then exit "$protection_rc"; fi
case "$1" in
  a800-prepare)
    python3 -m tri_rag_harness.tls_rag_step4_a800_replay prepare --output "$2"
    run_rc=$?
    echo "A800 preparation exit code: ${run_rc}"
    exit "$run_rc" ;;
  a800-replay)
    python3 -m tri_rag_harness.tls_rag_step4_a800_replay replay --input "$2" --output "$3"
    run_rc=$?
    report="$3/report.md" ;;
  *)
    python3 -m tri_rag_harness.tls_rag_step4_repaired_acceptance --previous-run "$1" --output "$2"
    run_rc=$?
    report="$2/report.md" ;;
esac
echo "Evaluation exit code: ${run_rc}"
if [ "$run_rc" -ne 0 ]; then exit "$run_rc"; fi
cat "$report"
report_rc=$?
echo "Report read exit code: ${report_rc}"
exit "$report_rc"
