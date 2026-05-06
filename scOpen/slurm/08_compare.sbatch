#!/bin/bash
#SBATCH --job-name=scopen_08_compare
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

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
CONDA_ENV="${CONDA_ENV:-scopen}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(python scripts/08_compare_imputed.py --config "$CFG")
if [[ -n "${IMPUTED_H5:-}" ]];      then CMD+=(--imputed-h5  "$IMPUTED_H5");      fi
if [[ -n "${MM_DIR:-}" ]];          then CMD+=(--mm-dir      "$MM_DIR");          fi
if [[ -n "${REGIONS_BED:-}" ]];     then CMD+=(--regions-bed "$REGIONS_BED");     fi
if [[ -n "${COMPARE_SPACE:-}" ]];   then CMD+=(--compare-space "$COMPARE_SPACE"); fi
if [[ "${EXPORT_FILLED_MTX:-0}" == "1" ]]; then CMD+=(--export-filled-mtx); fi
if [[ -n "${FILL_THRESHOLD:-}" ]];  then CMD+=(--fill-threshold "$FILL_THRESHOLD"); fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
