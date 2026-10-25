#!/bin/sh
# One approved development study; output must not exist. No retry/resume mode.
set -eu
task_repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$task_repo"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$task_repo/src:$task_repo"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
task_python=${STEP5_PYTHON:-python3}
"$task_python" scripts/check_code_protection.py
git diff --exit-code --quiet
git diff --cached --exit-code --quiet
exec timeout --signal=TERM --kill-after=15s 7200s \
  "$task_python" -u -m mechanism.tls_rag_step5_audit "$@"
