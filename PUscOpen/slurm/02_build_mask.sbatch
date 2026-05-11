#!/bin/bash
#SBATCH --job-name=pusc_02_mask
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

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

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
module load r/4.4.0 || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

Rscript scripts/02_build_mask.R --config "$CFG"
