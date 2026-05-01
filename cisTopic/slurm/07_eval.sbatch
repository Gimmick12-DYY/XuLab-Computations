#!/bin/bash
#SBATCH --job-name=cistopic_07_eval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Compare imputed accessibility against observed matrix entries.
#
# Modes (set via EVAL_MODE env var):
#   EVAL_MODE=reconstruction  -> fit-on-train diagnostic (default)
#   EVAL_MODE=prepare_holdout -> create masked matrix + saved split
#   EVAL_MODE=strict_holdout  -> evaluate held-out run with saved split
#
# Strict held-out example:
#   sbatch --export=ALL,EVAL_MODE=prepare_holdout slurm/07_eval.sbatch
#   # rerun 02-05 with a config whose paths.mm points to <work_dir>/mm_holdout
#   sbatch --export=ALL,EVAL_MODE=strict_holdout,\
#HOLDOUT_SPLIT=/path/to/orig_work/eval/holdout_split.npz,\
#HOLDOUT_IMPUTE_DIR=/path/to/holdout_work/impute slurm/07_eval.sbatch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CFG="${CFG:-${ROOT_DIR}/configs/default.yaml}"
N_SAMPLES="${N_SAMPLES:-100000}"
SEED="${SEED:-42}"
OUT_NAME="${OUT_NAME:-heldout_eval}"
EVAL_MODE="${EVAL_MODE:-reconstruction}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic

CMD=(
  python scripts/07_eval_heldout.py
  --config "$CFG"
  --n-samples "$N_SAMPLES"
  --seed "$SEED"
  --out-name "$OUT_NAME"
)

case "$EVAL_MODE" in
  reconstruction)
    ;;
  prepare_holdout)
    CMD+=(--prepare-holdout)
    ;;
  strict_holdout)
    if [[ -z "${HOLDOUT_SPLIT:-}" || -z "${HOLDOUT_IMPUTE_DIR:-}" ]]; then
      echo "EVAL_MODE=strict_holdout requires HOLDOUT_SPLIT and HOLDOUT_IMPUTE_DIR." >&2
      exit 1
    fi
    CMD+=(--holdout-split "$HOLDOUT_SPLIT" --impute-dir "$HOLDOUT_IMPUTE_DIR")
    ;;
  *)
    echo "Unknown EVAL_MODE: $EVAL_MODE" >&2
    echo "Valid values: reconstruction, prepare_holdout, strict_holdout" >&2
    exit 1
    ;;
esac

echo "Running: ${CMD[*]}"
"${CMD[@]}"
