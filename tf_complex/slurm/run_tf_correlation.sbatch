#!/bin/bash
#SBATCH --job-name=tf_corr
#SBATCH --output=tf_corr-%j.out
#SBATCH --error=tf_corr-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --mail-type=FAIL
#
# TF co-occupancy correlation -> candidate protein complexes (A/B-stratified).
#   1. build_tf_matrix.py : RAW per-TF pseudobulk -> bins x TFs matrix (run once)
#   2. tf_correlation.py  : TF x TF correlation (genome/A/B) + clustering -> complexes
#
#   sbatch tf_complex/slurm/run_tf_correlation.sbatch
#   REBUILD=0 METRIC=jaccard CORR_THRESHOLD=0.4 sbatch tf_complex/slurm/run_tf_correlation.sbatch
#
# Uses RAW binding (unified/work/<tf>/mm) -- NOT imputed (imputed is open-chromatin
# masked and would confound co-occupancy). A/B from the Hi-C compartments (step 05).

set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
TFC="${XULAB}/tf_complex"
WORK_ROOT="${WORK_ROOT:-${XULAB}/unified/work}"
MATRIX_DIR="${MATRIX_DIR:-${TFC}/work}"
RESULTS="${RESULTS:-${TFC}/results}"
AB_BED="${AB_BED:-${XULAB}/hic/work/compartments/compartments_25000.AB.bed}"

REBUILD="${REBUILD:-1}"
METRIC="${METRIC:-cooccur}"        # cooccur (log2 obs/exp overlap) | spearman | pearson | jaccard
AGG_BP="${AGG_BP:-10000}"          # aggregate 1kb bins -> this window before correlating
MIN_TFS_PER_BIN="${MIN_TFS_PER_BIN:-2}"
CLUSTER_THRESHOLD="${CLUSTER_THRESHOLD:-1.0}"   # complex if score>=this (cooccur log2 fold; 1.0=2x)

CONDA_ENV="${CONDA_ENV:-tf_complex}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

[[ -s "${AB_BED}" ]] || { echo "missing A/B calls: ${AB_BED} (run hic/slurm/05_classify_compartments.sbatch)" >&2; exit 1; }
mkdir -p "${MATRIX_DIR}" "${RESULTS}"

if [[ "${REBUILD}" == "1" || ! -s "${MATRIX_DIR}/tf_matrix.npz" ]]; then
  echo "[$(date)] build_tf_matrix (RAW pseudobulk)"
  python "${TFC}/scripts/build_tf_matrix.py" --work-root "${WORK_ROOT}" --out-dir "${MATRIX_DIR}"
else
  echo "[$(date)] reuse ${MATRIX_DIR}/tf_matrix.npz"
fi

echo "[$(date)] tf_correlation (metric=${METRIC}, A/B-stratified)"
python "${TFC}/scripts/tf_correlation.py" \
  --matrix-dir "${MATRIX_DIR}" --ab-bed "${AB_BED}" --out-dir "${RESULTS}" \
  --metric "${METRIC}" --agg-bp "${AGG_BP}" --min-tfs-per-bin "${MIN_TFS_PER_BIN}" \
  --cluster-threshold "${CLUSTER_THRESHOLD}" --plot
echo "[$(date)] done -> ${RESULTS}"
