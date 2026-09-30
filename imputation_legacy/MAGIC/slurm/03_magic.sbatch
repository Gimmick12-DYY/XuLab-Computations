#!/bin/bash
#SBATCH --job-name=magic_03_magic
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00

# MAGIC builds a cell-cell affinity (PCA-projected) and applies M^t @ X.
# Memory peaks at ~(n_cells x n_regions x 4 bytes) for the dense imputed
# output. For 10k cells x 100k regions that is ~4 GB; for ~600k regions
# (post drop-zero on the CTCF dataset) ~24 GB. Bump --mem if your input is
# bigger.

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
CONDA_ENV="${CONDA_ENV:-magic}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python scripts/03_run_magic.py --config "$CFG"
