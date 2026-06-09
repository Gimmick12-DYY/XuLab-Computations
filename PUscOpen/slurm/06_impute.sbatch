#!/bin/bash
#SBATCH --job-name=pusc_06_impute
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# NOTE: memory scales with impute.source_threshold_mode = sparsity_match:K
# (output nnz ~ K * raw_nnz). 64G was sized for K=1 (~50k bins). 128G is for
# K=4. For higher K, override on the command line (sbatch args beat #SBATCH):
#   sbatch --mem=256G --time=08:00:00 slurm/06_impute.sbatch
# If it still OOMs, lower K or restrict impute.propagate.target_bed instead of
# chasing the whole (near-dense) genome.

set -euo pipefail

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
CONDA_ENV="${CONDA_ENV:-puscopen}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python scripts/06_impute.py --config "$CFG"
