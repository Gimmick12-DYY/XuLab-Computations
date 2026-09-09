#!/bin/bash
#SBATCH --job-name=tf_compartrpkm
#SBATCH --output=tf_compartrpkm-%j.out
#SBATCH --error=tf_compartrpkm-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=FAIL
#
# "TF compartment rpkm pearson cor": the validated ChromHMM-18 RPKM workflow
# (04_chromhmm_rpkm.sbatch) with the unit swapped from a chromatin state to an A/B
# compartment domain. RPKM normalizes by domain LENGTH and per-TF DEPTH at once,
# which is what the earlier CPM / tf_fraction / domain_fraction attempts could not
# do (per-TF fraction is Pearson scale-invariant, so it left the size confound in).
#
#   1. build_compartment_matrix.py       reads -> (units x TFs) RPKM
#   2. compartment_rpkm_correlation.py   Pearson TF x TF over genome / A-only / B-only,
#                                        cluster, heatmap + cell-count bars
#
#   sbatch tf_complex/slurm/05_compartment_rpkm.sbatch
#   UNIT=bin sbatch tf_complex/slurm/05_compartment_rpkm.sbatch    # 25 kb bins (RPKM == CPM)
#   REBUILD=0 sbatch tf_complex/slurm/05_compartment_rpkm.sbatch   # reuse the matrix
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
TFC="${XULAB}/tf_complex"
WORK_ROOT="${WORK_ROOT:-${XULAB}/unified/work}"
UNIT="${UNIT:-domain}"                    # domain (contiguous A/B segments, variable bp) | bin (25 kb)
if [[ "${UNIT}" == "bin" ]]; then
  DOMAINS_BED="${DOMAINS_BED:-${XULAB}/hic/work/compartments/compartments_25000.AB.bed}"
else
  DOMAINS_BED="${DOMAINS_BED:-${XULAB}/hic/work/compartments/compartments_25000.domains.bed}"
fi
MATRIX_DIR="${MATRIX_DIR:-${TFC}/work/compartment_rpkm_${UNIT}}"
RESULTS="${RESULTS:-${TFC}/results_compartment_rpkm_${UNIT}}"
SOURCE="${SOURCE:-raw}"                   # raw (mm) | imputed
TRANSFORM="${TRANSFORM:-rpkm}"            # rpkm | log (log1p of the RPKM matrix)
N_CLUSTERS="${N_CLUSTERS:-3}"
CELL_META="${CELL_META:-${XULAB}/data/TF1000cells.meta.csv}"
REBUILD="${REBUILD:-1}"
CONDA_ENV="${CONDA_ENV:-tf_complex}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
[[ -s "${DOMAINS_BED}" ]] || { echo "missing compartments: ${DOMAINS_BED} (run hic step 05)" >&2; exit 1; }
mkdir -p "${MATRIX_DIR}" "${RESULTS}"

if [[ "${REBUILD}" == "1" || ! -s "${MATRIX_DIR}/compartment_matrix.npz" ]]; then
  echo "[$(date)] build compartment RPKM matrix (unit=${UNIT}, source=${SOURCE})"
  python "${TFC}/scripts/build_compartment_matrix.py" \
    --work-root "${WORK_ROOT}" --domains-bed "${DOMAINS_BED}" --unit "${UNIT}" \
    --out-dir "${MATRIX_DIR}" --source "${SOURCE}" --transform rpkm
fi

echo "[$(date)] correlate + plot (transform=${TRANSFORM}, k=${N_CLUSTERS})"
python "${TFC}/scripts/compartment_rpkm_correlation.py" \
  --matrix-dir "${MATRIX_DIR}" --out-dir "${RESULTS}" \
  --transform "${TRANSFORM}" --n-clusters "${N_CLUSTERS}" --cell-meta "${CELL_META}"
echo "[$(date)] done -> ${RESULTS}"
