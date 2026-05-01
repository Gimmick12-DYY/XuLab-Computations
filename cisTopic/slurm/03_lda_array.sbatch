#!/bin/bash
#SBATCH --job-name=cistopic_03_lda_array
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --array=0-4   # edit to match the length of K_LIST below

# One MALLET training per K, parallel across the cluster. Edit K_LIST to the
# same values as lda.n_topics in your config.

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

K_LIST=(10 15 20 30 50)
K=${K_LIST[$SLURM_ARRAY_TASK_ID]}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic
export MALLET_MEMORY="${MALLET_MEMORY:-200G}"

python scripts/03_run_lda_mallet.py --config "$CFG" --n-topics "$K"
