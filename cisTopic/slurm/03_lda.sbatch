#!/bin/bash
#SBATCH --job-name=cistopic_03_lda
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00

# NOTE: MALLET training on ~1M regions x 10K cells, 500 iterations, K in [10..60]
# is genuinely long. If your scheduler allows it, consider submitting one job
# per K (--n-topics <K>) as a job array instead of training the whole grid
# serially in one job.

set -euo pipefail

# Resolve project root from SLURM_SUBMIT_DIR (BASH_SOURCE points to
# /var/spool/slurmd on compute nodes and won't find scripts/configs).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/scripts" && -d "${SUBMIT_DIR}/configs" ]]; then
  ROOT_DIR="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/../scripts" && -d "${SUBMIT_DIR}/../configs" ]]; then
  ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${ROOT_DIR}"

CFG="${CFG:-${ROOT_DIR}/configs/default.yaml}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic

export MALLET_MEMORY="${MALLET_MEMORY:-200G}"

python scripts/03_run_lda_mallet.py --config "$CFG"
