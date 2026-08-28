#!/bin/bash
#SBATCH --job-name=pdctp-foundation
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/pdctp-foundation-%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate tri-rag

scripts/run_tests.sh

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m tri_rag_harness.pdctp_foundation \
  --config configs/pdctp_network_free_foundation_v1.json \
  --output "runs/slurm-pdctp-foundation-${SLURM_JOB_ID}"
