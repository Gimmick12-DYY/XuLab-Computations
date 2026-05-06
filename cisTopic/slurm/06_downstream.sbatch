#!/bin/bash
#SBATCH --job-name=cistopic_06_down
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00

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
# Force-skip UMAP even if config is changed (optional): NO_UMAP=1 sbatch ...
NO_UMAP="${NO_UMAP:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic

CMD=(python scripts/06_downstream.py --config "$CFG")
if [[ "${NO_UMAP}" == "1" ]]; then CMD+=(--no-umap); fi
echo "Running: ${CMD[*]}"
"${CMD[@]}"
