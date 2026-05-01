#!/bin/bash
#SBATCH --job-name=cistopic_09_visualize
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=04:00:00

# Reproduce cisTopic Supp. Fig. 15 style heatmap (original vs imputed)
# and report concrete density / per-cell numbers.

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
N_REGIONS="${N_REGIONS:-1000}"
N_CELLS="${N_CELLS:-5000}"
SPACE="${SPACE:-full}"
REGION_MODE="${REGION_MODE:-original_top_cells}"
SEED="${SEED:-42}"
OUT_NAME="${OUT_NAME:-visualize_imputation}"
NOCLUSTER="${NOCLUSTER:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic

CMD=(
  python scripts/09_visualize_imputation.py
  --config "$CFG"
  --n-regions "$N_REGIONS"
  --n-cells "$N_CELLS"
  --space "$SPACE"
  --region-mode "$REGION_MODE"
  --seed "$SEED"
  --out-name "$OUT_NAME"
)

if [[ -n "${IMPUTED_H5:-}" ]]; then CMD+=(--imputed-h5 "$IMPUTED_H5"); fi
if [[ -n "${MM_DIR:-}" ]];     then CMD+=(--mm-dir     "$MM_DIR");     fi
if [[ "${NOCLUSTER}" == "1" ]]; then CMD+=(--no-cluster); fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
