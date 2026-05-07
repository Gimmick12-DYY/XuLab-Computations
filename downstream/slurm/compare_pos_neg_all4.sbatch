#!/bin/bash
#SBATCH --job-name=ds_posneg_all4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00

# Build bins (if missing) and compare 4 matrices:
#   raw + cisTopic + FITS + scOpen
#
# Each pipeline now writes its imputed output as a small directory
# (factors.npz / matrix.npy + regions.tsv + barcodes.tsv + meta.json),
# so we point compare_pos_neg.py at those directories directly.
#
# Optional overrides:
#   ROOT_DIR, CONDA_ENV, MM_DIR, BINS_TSV, BINS_OUT_DIR, CACHE_DIR,
#   POSNEG_OUT_DIR, CISTOPIC_DIR, FITS_DIR, SCOPEN_DIR,
#   OUT_NAME, N_TOP, SEED

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
CONDA_ENV="${CONDA_ENV:-cistopic}"

MM_DIR="${MM_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm}"
BINS_OUT_DIR="${BINS_OUT_DIR:-${ROOT_DIR}/downstream/bins}"
BINS_TSV="${BINS_TSV:-${BINS_OUT_DIR}/pos_neg_bins.tsv}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/downstream/cache}"

POSNEG_OUT_DIR="${POSNEG_OUT_DIR:-${ROOT_DIR}/downstream/pos_neg_all4}"
OUT_NAME="${OUT_NAME:-compare_pos_neg_all4}"

# New compact-output directories produced by each pipeline's 05_impute step.
CISTOPIC_DIR="${CISTOPIC_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/impute}"
FITS_DIR="${FITS_DIR:-${ROOT_DIR}/FITS/work/ctcf/impute}"
SCOPEN_DIR="${SCOPEN_DIR:-${ROOT_DIR}/scOpen/work/ctcf/impute}"

N_TOP="${N_TOP:-10000}"
SEED="${SEED:-2026}"

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

echo "[$(date)] Running 4-way pos/neg comparison"
python "${ROOT_DIR}/downstream/compare_pos_neg.py" \
  --bins-tsv "${BINS_TSV}" \
  --out-dir "${POSNEG_OUT_DIR}" \
  --out-name "${OUT_NAME}" \
  --input "raw=${MM_DIR}" \
  --input "cisTopic=${CISTOPIC_DIR}" \
  --input "FITS=${FITS_DIR}" \
  --input "scOpen=${SCOPEN_DIR}"

echo "[$(date)] Done."
echo "Outputs:"
echo "  ${POSNEG_OUT_DIR}/${OUT_NAME}.png"
echo "  ${POSNEG_OUT_DIR}/${OUT_NAME}_auroc.png"
echo "  ${POSNEG_OUT_DIR}/${OUT_NAME}.json"
echo "  ${POSNEG_OUT_DIR}/${OUT_NAME}.tsv"
echo "  ${POSNEG_OUT_DIR}/${OUT_NAME}_modeled.png"
echo "  ${POSNEG_OUT_DIR}/${OUT_NAME}_modeled_auroc.png"
echo "  ${POSNEG_OUT_DIR}/${OUT_NAME}_modeled.json"
echo "  ${POSNEG_OUT_DIR}/${OUT_NAME}_auroc_full_vs_modeled.png"
