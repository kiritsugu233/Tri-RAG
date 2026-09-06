#!/bin/sh
set -eu
if [ "$(uname -s)" = Darwin ] && [ "$(pwd -P)" != /Users/guanghongxu/Query-Adaptive-Tri-RAG ]; then
  echo 'Use /Users/guanghongxu/Query-Adaptive-Tri-RAG for all local steps.' >&2
  exit 2
fi
if [ "$#" -ne 2 ]; then
  echo 'Usage: sh scripts/run_tls_rag_step4_probe.sh prepare|evaluate NEW_RUN_ROOT' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
case "$1" in
  prepare) python3 -m tri_rag_harness.tls_rag_step4_nfcorpus --device cuda --output "$2" ;;
  evaluate) python3 -m tri_rag_harness.tls_rag_step4_probe --bundle "$2/bundle" --output "$2/evaluation" ;;
  *) echo 'Expected prepare or evaluate.' >&2; exit 2 ;;
esac
