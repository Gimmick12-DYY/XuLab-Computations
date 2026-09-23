#!/bin/bash
#SBATCH --job-name=ctcf_drop5_eval
#SBATCH --output=/work/users/d/y/dyy12/XuLab/downstream/logs/ctcf_drop5_eval-%j.out
#SBATCH --error=/work/users/d/y/dyy12/XuLab/downstream/logs/ctcf_drop5_eval-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL
#
# Score CTCF raw+unified against majority-2/3 peaks after dropping the lowest
# 5% by ATAC FC. Reuses already-called MACS peaks from all_universe.

set -euo pipefail
ROOT_DIR="${XULAB:-/work/users/d/y/dyy12/XuLab}"
DOWN="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"
mkdir -p "${DOWN}/logs"

TAG="${TAG:-ctcf_maj2of3_drop5}"
BINS_TSV="${BINS_TSV:-${DOWN}/bins_${TAG}/pos_neg_bins.tsv}"
MM="${ROOT_DIR}/unified/work/ctcf/mm"
IMPUTE="${ROOT_DIR}/unified/work/ctcf/impute"
INPUTS="raw=${MM} unified=${IMPUTE}"
REUSE="${DOWN}/peak_coverage_ctcf_all_universe"

if [[ ! -s "${BINS_TSV}" ]]; then
  echo "missing ${BINS_TSV}; run build_ctcf_maj2of3_drop5_bins.py first" >&2
  exit 1
fi

CONDA_ENV="${CONDA_ENV:-cistopic}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export BINS_TSV INPUTS
export OUT_DIR="${DOWN}/compare_pos_neg_${TAG}"
export OUT_NAME="compare_pos_neg_${TAG}"
echo "[$(date)] compare_pos_neg"
python "${DOWN}/compare_pos_neg.py" \
  --bins-tsv "${BINS_TSV}" \
  --out-dir "${OUT_DIR}" \
  --out-name "${OUT_NAME}" \
  --input "raw=${MM}" \
  --input "unified=${IMPUTE}"

export OUT_DIR="${DOWN}/bin_coverage_${TAG}"
export OUT_NAME="bin_coverage_${TAG}"
echo "[$(date)] bin_coverage"
python "${DOWN}/bin_sensitivity_specificity.py" \
  --bins-tsv "${BINS_TSV}" \
  --out-dir "${OUT_DIR}" \
  --out-name "${OUT_NAME}" \
  --input "raw=${MM}" \
  --input "unified=${IMPUTE}"

export PEAK_COV_OUT_DIR="${DOWN}/peak_coverage_${TAG}"
export OUT_NAME="peak_coverage_${TAG}"
echo "[$(date)] peak_coverage (reuse ${REUSE})"
python "${DOWN}/peak_coverage.py" \
  --bins-tsv "${BINS_TSV}" \
  --out-dir "${PEAK_COV_OUT_DIR}" \
  --out-name "${OUT_NAME}" \
  --reuse-peaks-from "${REUSE}" \
  --target-fragments 150000000 \
  --ref-quantile 0.5 \
  --frags-at-ref 100 \
  --bg-quantile 0.5 \
  --bg-scale 1.0 \
  --max-frags-per-bin 500 \
  --spread-bp 150 \
  --macs-q 0.05 \
  --extsize 200 \
  --input "raw=${MM}" \
  --input "unified=${IMPUTE}"

echo "[$(date)] done"
echo "  ${DOWN}/compare_pos_neg_${TAG}/${OUT_NAME}.json"
echo "  ${DOWN}/bin_coverage_${TAG}/bin_coverage_${TAG}.json"
echo "  ${DOWN}/peak_coverage_${TAG}/peak_coverage_${TAG}_table.txt"
