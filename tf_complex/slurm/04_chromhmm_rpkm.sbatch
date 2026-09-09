#!/bin/bash
#SBATCH --job-name=tf_chromhmm
#SBATCH --output=tf_chromhmm-%j.out
#SBATCH --error=tf_chromhmm-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=FAIL
#
# "TF chromHMM18 rpkm ratio pearson cor": per-TF RPKM across the 18 ChromHMM states
# -> Pearson TF x TF -> cluster -> 3-panel figure (heatmap + cell-count bars +
# per-cluster ChromHMM state composition). RPKM normalizes by state LENGTH and per-TF
# DEPTH -> fixes both the size and cell-count confounds at once (unlike CPM).
#
#   CHROMHMM_BED=/path/HEK293T_chromHMM18.bed sbatch tf_complex/slurm/04_chromhmm_rpkm.sbatch
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
TFC="${XULAB}/tf_complex"
WORK_ROOT="${WORK_ROOT:-${XULAB}/unified/work}"
CHROMHMM_BED="${CHROMHMM_BED:-${XULAB}/data/HEK293T_chromHMM18.bed}"   # <-- supply this
MATRIX_DIR="${MATRIX_DIR:-${TFC}/work/chromhmm}"
RESULTS="${RESULTS:-${TFC}/results_chromhmm}"
SOURCE="${SOURCE:-raw}"                   # raw (mm) | imputed
TRANSFORM="${TRANSFORM:-rpkm}"           # rpkm (reads/kb/million -- already a ratio) | log | ratio
N_CLUSTERS="${N_CLUSTERS:-3}"
CELL_META="${CELL_META:-${XULAB}/data/TF1000cells.meta.csv}"
REBUILD="${REBUILD:-1}"
CONDA_ENV="${CONDA_ENV:-tf_complex}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
[[ -s "${CHROMHMM_BED}" ]] || { echo "missing ChromHMM-18 BED: ${CHROMHMM_BED} (set CHROMHMM_BED=...)" >&2; exit 1; }
mkdir -p "${MATRIX_DIR}" "${RESULTS}"

if [[ "${REBUILD}" == "1" || ! -s "${MATRIX_DIR}/chromhmm_matrix.npz" ]]; then
  echo "[$(date)] build ChromHMM RPKM matrix (source=${SOURCE})"
  python "${TFC}/scripts/build_chromhmm_matrix.py" \
    --work-root "${WORK_ROOT}" --chromhmm-bed "${CHROMHMM_BED}" \
    --out-dir "${MATRIX_DIR}" --source "${SOURCE}"
fi

echo "[$(date)] correlate + plot (transform=${TRANSFORM}, k=${N_CLUSTERS})"
python "${TFC}/scripts/chromhmm_correlation.py" \
  --matrix-dir "${MATRIX_DIR}" --out-dir "${RESULTS}" \
  --transform "${TRANSFORM}" --n-clusters "${N_CLUSTERS}" --cell-meta "${CELL_META}"
echo "[$(date)] done -> ${RESULTS}"
