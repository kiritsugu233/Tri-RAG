#!/bin/sh
set -eu
if [ "$(uname -s)" = Darwin ] && [ "$(pwd -P)" != /Users/guanghongxu/Query-Adaptive-Tri-RAG ]; then
  echo 'Use /Users/guanghongxu/Query-Adaptive-Tri-RAG for local work.' >&2
  exit 2
fi
if [ "$#" -ne 2 ]; then
  echo 'Usage: sh scripts/run_tls_rag_step4_retention_v3.sh PREVIOUS_V2_RUN ABSENT_V3_OUTPUT' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -m tri_rag_harness.tls_rag_step4_retention_v3 --previous-run "$1" --output "$2"
cat "$2/report.md"
