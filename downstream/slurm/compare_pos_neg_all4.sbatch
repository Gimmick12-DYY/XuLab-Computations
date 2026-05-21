#!/bin/bash
#SBATCH --job-name=ds_posneg_all8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00

# Build bins (if missing) and compare 8 matrices:
#   raw + cisTopic + FITS + scOpen + MAGIC + PUscOpen + cicero + scbasset
#
# Imputed inputs default to each pipeline's native <work>/ctcf/impute/.
# Override *_DIR or INPUTS to point elsewhere.
#
# Optional overrides:
#   ROOT_DIR, CONDA_ENV, MM_DIR, BINS_TSV, BINS_OUT_DIR,
#   CACHE_DIR, POSNEG_OUT_DIR, CISTOPIC_DIR, FITS_DIR, SCOPEN_DIR, MAGIC_DIR,
#   PUSCOPEN_DIR, CICERO_DIR, SCBASSET_DIR, OUT_NAME, N_TOP, SEED, INPUTS

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
CONDA_ENV="${CONDA_ENV:-cistopic}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
  ROOT_DIR="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/../downstream" ]]; then
  ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
fi
DOWN_DIR="${ROOT_DIR}/downstream"
# shellcheck source=_pipeline_paths.sh
source "${DOWN_DIR}/slurm/_pipeline_paths.sh"

BINS_OUT_DIR="${BINS_OUT_DIR:-${ROOT_DIR}/downstream/bins}"
BINS_TSV="${BINS_TSV:-${BINS_OUT_DIR}/pos_neg_bins.tsv}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/downstream/cache}"

POSNEG_OUT_DIR="${POSNEG_OUT_DIR:-${ROOT_DIR}/downstream/pos_neg_all8}"
OUT_NAME="${OUT_NAME:-compare_pos_neg_all8}"

N_TOP="${N_TOP:-10000}"
SEED="${SEED:-2026}"

if [[ -z "${INPUTS:-}" ]]; then
  default_inputs all
fi
pipeline_paths_native

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${BINS_OUT_DIR}" "${CACHE_DIR}" "${POSNEG_OUT_DIR}"

if [[ ! -f "${BINS_TSV}" ]]; then
  echo "[$(date)] BINS_TSV not found; generating: ${BINS_TSV}"
  Rscript "${ROOT_DIR}/downstream/prepare_pos_neg_bins.R" \
    --mm-dir "${MM_DIR}" \
    --out-dir "${BINS_OUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --n-top "${N_TOP}" \
    --seed "${SEED}"
fi

echo "[$(date)] Running 8-way pos/neg comparison"
echo "INPUTS=${INPUTS}"
CMD=(
  python "${DOWN_DIR}/compare_pos_neg.py"
  --bins-tsv "${BINS_TSV}"
  --out-dir "${POSNEG_OUT_DIR}"
  --out-name "${OUT_NAME}"
)
for pair in ${INPUTS}; do
  CMD+=(--input "${pair}")
done
"${CMD[@]}"

echo "[$(date)] Done."
echo "Outputs:"
echo "  ${POSNEG_OUT_DIR}/${OUT_NAME}.png"
echo "  ${POSNEG_OUT_DIR}/${OUT_NAME}_auroc.png"
echo "  ${POSNEG_OUT_DIR}/${OUT_NAME}.json"
echo "  ${POSNEG_OUT_DIR}/${OUT_NAME}.tsv"
