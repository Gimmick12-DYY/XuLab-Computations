#!/bin/bash
#SBATCH --job-name=cistopic_10_nonzero
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00

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
THRESHOLDS="${THRESHOLDS:-0,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4}"
CHUNK_ROWS="${CHUNK_ROWS:-2000}"
OUT_NAME="${OUT_NAME:-nonzero_complexity}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic

CMD=(
  python scripts/10_nonzero_complexity.py
  --config "$CFG"
  --thresholds "$THRESHOLDS"
  --chunk-rows "$CHUNK_ROWS"
  --out-name "$OUT_NAME"
)

if [[ -n "${IMPUTED_H5:-}" ]]; then CMD+=(--imputed-h5 "$IMPUTED_H5"); fi
if [[ -n "${MM_DIR:-}" ]]; then CMD+=(--mm-dir "$MM_DIR"); fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
