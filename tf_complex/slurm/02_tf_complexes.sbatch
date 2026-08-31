#!/bin/bash
#SBATCH --job-name=tf_complexes
#SBATCH --output=tf_complexes-%j.out
#SBATCH --error=tf_complexes-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --mail-type=FAIL
#
# Step 2: union per-TF peaks -> loci x TF matrix, remove HOT loci + blacklist,
# PCA-correlate TFs, cluster -> candidate protein complexes (Partridge 2020 style).
# A/B compartments annotate the complexes (they do NOT define co-binding).
#
#   sbatch tf_complex/slurm/02_tf_complexes.sbatch
#   METRIC=jaccard HOT_FRAC=0.4 CLUSTER_THRESHOLD=0.5 sbatch tf_complex/slurm/02_tf_complexes.sbatch
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
TFC="${XULAB}/tf_complex"
PEAKS_DIR="${PEAKS_DIR:-${TFC}/work/peaks}"
MATRIX_DIR="${MATRIX_DIR:-${TFC}/work}"
RESULTS="${RESULTS:-${TFC}/results}"
AB_BED="${AB_BED:-${XULAB}/hic/work/compartments/compartments_25000.AB.bed}"
BLACKLIST="${BLACKLIST:-${TFC}/ref/hg38-blacklist.v2.bed}"   # optional; skipped if missing

METRIC="${METRIC:-pca}"                 # pca (Partridge-style, default) | pearson | jaccard
N_PC="${N_PC:-25}"
HOT_FRAC="${HOT_FRAC:-0.5}"             # drop loci bound by > this fraction of TFs (HOT artifacts)
MIN_OCC="${MIN_OCC:-2}"                 # keep loci bound by >= this many TFs
CLUSTER="${CLUSTER:-hierarchical}"      # hierarchical | threshold
CLUSTER_THRESHOLD="${CLUSTER_THRESHOLD:-0.5}"   # set from printed p90/p95
REBUILD="${REBUILD:-1}"                 # 0 = reuse peak_matrix.npz

CONDA_ENV="${CONDA_ENV:-tf_complex}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
[[ -s "${AB_BED}" ]] || { echo "missing A/B calls: ${AB_BED}" >&2; exit 1; }
mkdir -p "${MATRIX_DIR}" "${RESULTS}"

if [[ "${REBUILD}" == "1" || ! -s "${MATRIX_DIR}/peak_matrix.npz" ]]; then
  echo "[$(date)] build_peak_matrix (union per-TF peaks -> loci x TF)"
  python "${TFC}/scripts/build_peak_matrix.py" \
    --peaks-dir "${PEAKS_DIR}" --out-dir "${MATRIX_DIR}" --ab-bed "${AB_BED}"
else
  echo "[$(date)] reuse ${MATRIX_DIR}/peak_matrix.npz"
fi

BL_ARG=()
[[ -s "${BLACKLIST}" ]] && BL_ARG=(--blacklist "${BLACKLIST}") || echo "[warn] no blacklist at ${BLACKLIST}; skipping blacklist filter"

echo "[$(date)] tf_complexes (metric=${METRIC}, hot_frac=${HOT_FRAC})"
python "${TFC}/scripts/tf_complexes.py" \
  --matrix-dir "${MATRIX_DIR}" --out-dir "${RESULTS}" \
  --metric "${METRIC}" --n-pc "${N_PC}" \
  --hot-frac "${HOT_FRAC}" --min-occ "${MIN_OCC}" \
  --cluster "${CLUSTER}" --cluster-threshold "${CLUSTER_THRESHOLD}" \
  "${BL_ARG[@]}" --plot
echo "[$(date)] done -> ${RESULTS}"
