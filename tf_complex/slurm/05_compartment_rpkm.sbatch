#!/bin/bash
#SBATCH --job-name=tf_compartrpkm
#SBATCH --output=tf_compartrpkm-%j.out
#SBATCH --error=tf_compartrpkm-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=FAIL
#
# TF compartment Pearson on A/B domains. No ChromHMM.
# TRANSFORM=rpkm     reads / domain_kb / (TF_total_reads/1e6)
# TRANSFORM=per_cell reads / domain_kb / n_cells   (n_cells from TF1000 metadata)
# Pearson is unchanged by swapping those two per-TF constants. The /kb term
# is the one that changes the correlation. Cell-count bars are kept.
#
# UNIT=domain (default): each smoothed compartment participates in the correlation.
# UNIT=class: 18 E1-quantile pools (ChromHMM analog) — NOT the intended analysis.
# UNIT=bin: one unit per 25 kb bin.
#
#   sbatch tf_complex/slurm/05_compartment_rpkm.sbatch               # domain + SMOOTH=2
#   SMOOTH=0 sbatch tf_complex/slurm/05_compartment_rpkm.sbatch       # raw unsmoothed domains
#   UNIT=class sbatch tf_complex/slurm/05_compartment_rpkm.sbatch     # 18 E1 classes
#   REBUILD=0 sbatch tf_complex/slurm/05_compartment_rpkm.sbatch      # reuse the matrix
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
TFC="${XULAB}/tf_complex"
WORK_ROOT="${WORK_ROOT:-${XULAB}/unified/work}"
UNIT="${UNIT:-domain}"                    # domain (one unit per compartment) | class | bin
AB_BED="${AB_BED:-${XULAB}/hic/work/compartments/compartments_25000.AB.bed}"
if [[ "${UNIT}" == "bin" ]]; then
  DOMAINS_BED="${DOMAINS_BED:-${AB_BED}}"
else
  DOMAINS_BED="${DOMAINS_BED:-${XULAB}/hic/work/compartments/compartments_25000.domains.bed}"
fi
PER_ARM="${PER_ARM:-9}"                   # classes per arm; total units = 2*PER_ARM
SMOOTH="${SMOOTH:-2}"                     # absorb sandwiched flips of <= N bins (0 = raw calls)
MIN_BP="${MIN_BP:-0}"                     # drop domains shorter than this (0 = keep all)
TAG="${TAG:-${UNIT}$([[ "${SMOOTH}" != "0" ]] && echo "_sm${SMOOTH}")$([[ "${MIN_BP}" != "0" ]] && echo "_${MIN_BP}bp")}"
TRANSFORM="${TRANSFORM:-rpkm}"            # rpkm | per_cell | log (log1p after the matrix is built)
NORM="${NORM:-${TRANSFORM}}"
if [[ "${NORM}" == "log" ]]; then NORM="rpkm"; fi
MATRIX_DIR="${MATRIX_DIR:-${TFC}/work/compartment_${NORM}_${TAG}}"
RESULTS="${RESULTS:-${TFC}/results_compartment_${NORM}_${TAG}}"
SOURCE="${SOURCE:-raw}"                   # raw (mm) | imputed
N_CLUSTERS="${N_CLUSTERS:-3}"
CELL_META="${CELL_META:-${XULAB}/data/TF1000cells.meta.csv}"
REBUILD="${REBUILD:-1}"
CONDA_ENV="${CONDA_ENV:-tf_complex}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
mkdir -p "${MATRIX_DIR}" "${RESULTS}"

# Optional: absorb short sandwiched compartment flips before anything else. 48.5% of
# the raw 25 kb domains are a single bin and most of those sit between two
# opposite-label neighbours -- eigenvector flicker, not 25 kb compartments.
if [[ "${SMOOTH}" != "0" ]]; then
  SM="${MATRIX_DIR}/compartments_smoothed.bed"
  if [[ "${REBUILD}" == "1" || ! -s "${SM}" ]]; then
    echo "[$(date)] smooth compartments (absorb sandwiched runs <= ${SMOOTH} bins)"
    python "${TFC}/scripts/smooth_compartments.py" \
      --ab-bed "${AB_BED}" --min-bins "${SMOOTH}" --out "${SM}"
  fi
  AB_BED="${SM}"
  DOMAINS_BED="${SM%.bed}.domains.bed"
fi

if [[ "${UNIT}" == "class" ]]; then
  [[ -s "${AB_BED}" ]] || { echo "missing compartments: ${AB_BED} (run hic step 05)" >&2; exit 1; }
  MAT="${MATRIX_DIR}/chromhmm_matrix.npz"
  if [[ "${REBUILD}" == "1" || ! -s "${MAT}" ]]; then
    echo "[$(date)] build ${PER_ARM}x2 compartment classes from E1"
    python "${TFC}/scripts/build_compartment_classes.py" \
      --ab-bed "${AB_BED}" --per-arm "${PER_ARM}" \
      --out "${MATRIX_DIR}/compartment_classes.bed"
    echo "[$(date)] build compartment-class RPKM matrix (source=${SOURCE})"
    python "${TFC}/scripts/build_chromhmm_matrix.py" \
      --work-root "${WORK_ROOT}" --chromhmm-bed "${MATRIX_DIR}/compartment_classes.bed" \
      --out-dir "${MATRIX_DIR}" --source "${SOURCE}"
  fi
else
  [[ -s "${DOMAINS_BED}" ]] || { echo "missing compartments: ${DOMAINS_BED} (run hic step 05)" >&2; exit 1; }
  if [[ "${REBUILD}" == "1" || ! -s "${MATRIX_DIR}/compartment_matrix.npz" ]]; then
    echo "[$(date)] build compartment matrix (unit=${UNIT}, source=${SOURCE}, norm=${NORM})"
    python "${TFC}/scripts/build_compartment_matrix.py" \
      --work-root "${WORK_ROOT}" --domains-bed "${DOMAINS_BED}" --unit "${UNIT}" \
      --out-dir "${MATRIX_DIR}" --source "${SOURCE}" --transform "${NORM}" \
      --cell-meta "${CELL_META}"
  fi
fi

PLOT_TRANSFORM="rpkm"
if [[ "${TRANSFORM}" == "log" ]]; then PLOT_TRANSFORM="log"; fi
echo "[$(date)] correlate + plot (norm=${NORM}, plot=${PLOT_TRANSFORM}, k=${N_CLUSTERS})"
python "${TFC}/scripts/compartment_rpkm_correlation.py" \
  --matrix-dir "${MATRIX_DIR}" --out-dir "${RESULTS}" \
  --transform "${PLOT_TRANSFORM}" --n-clusters "${N_CLUSTERS}" --cell-meta "${CELL_META}" \
  --min-bp "${MIN_BP}"
echo "[$(date)] done -> ${RESULTS}"
