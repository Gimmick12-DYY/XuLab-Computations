#!/bin/bash
#SBATCH --job-name=tf_chromhmm_x5
#SBATCH --output=tf_chromhmm_x5-%j.out
#SBATCH --error=tf_chromhmm_x5-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=FAIL
#
# ChromHMM-18 split into N_SPLIT E1-quantile subclasses (default 5 -> 90 units),
# then the same RPKM + Pearson workflow as 04_chromhmm_rpkm.sbatch.
#
#   sbatch tf_complex/slurm/04_chromhmm_x5.sbatch
#   N_SPLIT=3 sbatch tf_complex/slurm/04_chromhmm_x5.sbatch
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
TFC="${XULAB}/tf_complex"
WORK_ROOT="${WORK_ROOT:-${XULAB}/unified/work}"
CHROMHMM_BED="${CHROMHMM_BED:-${XULAB}/data/HEK293T_chromHMM18.bed.gz}"
AB_BED="${AB_BED:-${XULAB}/hic/work/compartments/compartments_25000.AB.bed}"
N_SPLIT="${N_SPLIT:-5}"
MATRIX_DIR="${MATRIX_DIR:-${TFC}/work/chromhmm_x${N_SPLIT}}"
RESULTS="${RESULTS:-${TFC}/results_chromhmm_x${N_SPLIT}}"
SOURCE="${SOURCE:-raw}"
TRANSFORM="${TRANSFORM:-rpkm}"
N_CLUSTERS="${N_CLUSTERS:-0}"              # 0 = silhouette-chosen k
CELL_META="${CELL_META:-${XULAB}/data/TF1000cells.meta.csv}"
REBUILD="${REBUILD:-1}"
CONDA_ENV="${CONDA_ENV:-tf_complex}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
[[ -s "${CHROMHMM_BED}" ]] || { echo "missing ChromHMM BED: ${CHROMHMM_BED}" >&2; exit 1; }
[[ -s "${AB_BED}" ]] || { echo "missing A/B BED: ${AB_BED}" >&2; exit 1; }
mkdir -p "${MATRIX_DIR}" "${RESULTS}"

SPLIT_BED="${MATRIX_DIR}/chromhmm18_x${N_SPLIT}.bed"
if [[ "${REBUILD}" == "1" || ! -s "${SPLIT_BED}" ]]; then
  echo "[$(date)] split each ChromHMM state into ${N_SPLIT} E1-quantile classes"
  python "${TFC}/scripts/split_chromhmm_states.py" \
    --chromhmm-bed "${CHROMHMM_BED}" --ab-bed "${AB_BED}" \
    --n-split "${N_SPLIT}" --out "${SPLIT_BED}"
fi

if [[ "${REBUILD}" == "1" || ! -s "${MATRIX_DIR}/chromhmm_matrix.npz" ]]; then
  echo "[$(date)] build ChromHMM×${N_SPLIT} RPKM matrix (source=${SOURCE})"
  python "${TFC}/scripts/build_chromhmm_matrix.py" \
    --work-root "${WORK_ROOT}" --chromhmm-bed "${SPLIT_BED}" \
    --out-dir "${MATRIX_DIR}" --source "${SOURCE}"
fi

echo "[$(date)] correlate + plot"
python "${TFC}/scripts/chromhmm_correlation.py" \
  --matrix-dir "${MATRIX_DIR}" --out-dir "${RESULTS}" \
  --transform "${TRANSFORM}" --n-clusters "${N_CLUSTERS}" --cell-meta "${CELL_META}"
echo "[$(date)] done -> ${RESULTS}"
