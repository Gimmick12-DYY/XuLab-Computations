#!/bin/bash
#SBATCH --job-name=pusc_06_impute
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=08:00:00

# NOTE: the dominant cost is building the DENSE W*H reconstruction of the
# modeled subspace (modeled_regions x cells); at ~592k x 10410 that is ~6.2e9
# nonzeros ~= 48 GB held resident, plus the source_mask copy + propagation +
# output assembly. 128G OOM'd at the final assembly step for K=4, so 256G.
# source_threshold_mode = sparsity_match:K sets how many entries survive
# (output nnz ~ K * raw_nnz) -> more propagation/assembly memory at higher K.
# If 256G still OOMs: lower K (sparsity_match:2), or restrict
# impute.propagate.target_bed to a candidate BED instead of "off".

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
