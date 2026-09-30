#!/bin/bash
#SBATCH --job-name=pusc_06_impute
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00

# Memory is dominated by the DENSE imputed matrix over the modeled subspace:
# (refined_mask rows) x cells x ~16 bytes, plus a copy during the propagation
# add. With refined_mask_threshold=top_k:200000 that is ~200k x 10410 ~= 17 GB,
# so 128G is ample. It scales LINEARLY with refined_mask_threshold's row count:
# top_k:300000 ~= 25 GB, "all" (~592k rows) ~= 49 GB and OOMs even at 256G via
# the old COO round-trip (now fixed) -- so raise N only as far as your RAM (and
# every downstream consumer that load_npz's this matrix) can hold.

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
