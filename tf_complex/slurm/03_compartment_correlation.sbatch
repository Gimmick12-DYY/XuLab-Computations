#!/bin/bash
#SBATCH --job-name=tf_compartcorr
#SBATCH --output=tf_compartcorr-%j.out
#SBATCH --error=tf_compartcorr-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=FAIL
#
# A/B compartment co-binding correlation (the intended pipeline):
#   1. build_compartment_matrix.py : per-TF raw pseudobulk reads summed into each A/B
#      compartment DOMAIN  -> (domains x TFs) read-count matrix.
#   2. compartment_correlation.py  : CPM-normalize per TF (read FRACTIONS) -> Pearson
#      TF x TF over ALL / A-only (A-A) / B-only (B-B) domains, + A-B differential.
#
#   sbatch tf_complex/slurm/03_compartment_correlation.sbatch
#   INPUT=count sbatch tf_complex/slurm/03_compartment_correlation.sbatch   # raw counts instead of CPM
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
TFC="${XULAB}/tf_complex"
WORK_ROOT="${WORK_ROOT:-${XULAB}/unified/work}"
UNIT="${UNIT:-bin}"                      # bin (25kb compartment bins, uniform size, default) | domain (Mb segments)
if [[ "${UNIT}" == "bin" ]]; then
  DOMAINS_BED="${DOMAINS_BED:-${XULAB}/hic/work/compartments/compartments_25000.AB.bed}"
else
  DOMAINS_BED="${DOMAINS_BED:-${XULAB}/hic/work/compartments/compartments_25000.domains.bed}"
fi
MATRIX_DIR="${MATRIX_DIR:-${TFC}/work/compartment}"
SOURCE="${SOURCE:-raw}"                  # raw (mm) | imputed
INPUT="${INPUT:-tf_fraction}"            # tf_fraction (per-TF: reads/TF-total, default) | domain_fraction | count
RESULTS="${RESULTS:-${TFC}/results_compartment_${INPUT}}"   # _cpm and _count coexist
CELL_META="${CELL_META:-${XULAB}/data/TF1000cells.meta.csv}"
CLUSTER_THRESHOLD="${CLUSTER_THRESHOLD:-0.3}"   # cluster -> candidate_complexes; set near printed p95
REBUILD="${REBUILD:-1}"
CONDA_ENV="${CONDA_ENV:-tf_complex}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
[[ -s "${DOMAINS_BED}" ]] || { echo "missing domains: ${DOMAINS_BED} (run hic step 05)" >&2; exit 1; }
mkdir -p "${MATRIX_DIR}" "${RESULTS}"

if [[ "${REBUILD}" == "1" || ! -s "${MATRIX_DIR}/compartment_matrix.npz" ]]; then
  echo "[$(date)] build compartment x TF read matrix (source=${SOURCE})"
  python "${TFC}/scripts/build_compartment_matrix.py" \
    --work-root "${WORK_ROOT}" --domains-bed "${DOMAINS_BED}" --unit "${UNIT}" \
    --out-dir "${MATRIX_DIR}" --source "${SOURCE}"
fi

echo "[$(date)] correlate (input=${INPUT})"
python "${TFC}/scripts/compartment_correlation.py" \
  --matrix-dir "${MATRIX_DIR}" --out-dir "${RESULTS}" \
  --input "${INPUT}" --cell-meta "${CELL_META}" \
  --cluster-threshold "${CLUSTER_THRESHOLD}" --plot
echo "[$(date)] done -> ${RESULTS}"
