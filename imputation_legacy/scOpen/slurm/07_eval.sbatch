#!/bin/bash
#SBATCH --job-name=scopen_07_eval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
N_SAMPLES="${N_SAMPLES:-100000}"
SEED="${SEED:-42}"
OUT_NAME="${OUT_NAME:-heldout_eval}"
SAMPLE_SPACE="${SAMPLE_SPACE:-modeled}"
EVAL_MODE="${EVAL_MODE:-reconstruction}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(
  python scripts/07_eval_heldout.py
  --config "$CFG"
  --n-samples "$N_SAMPLES"
  --seed "$SEED"
  --out-name "$OUT_NAME"
  --sample-space "$SAMPLE_SPACE"
)

case "$EVAL_MODE" in
  reconstruction)
    ;;
  prepare_holdout)
    CMD+=(--prepare-holdout)
    ;;
  strict_holdout)
    if [[ -z "${HOLDOUT_SPLIT:-}" || -z "${HOLDOUT_IMPUTED_H5:-}" ]]; then
      echo "EVAL_MODE=strict_holdout requires HOLDOUT_SPLIT and HOLDOUT_IMPUTED_H5." >&2
      exit 1
    fi
    CMD+=(--holdout-split "$HOLDOUT_SPLIT" --imputed-h5 "$HOLDOUT_IMPUTED_H5")
    ;;
  *)
    echo "Unknown EVAL_MODE: $EVAL_MODE" >&2
    exit 1
    ;;
esac

echo "Running: ${CMD[*]}"
"${CMD[@]}"
