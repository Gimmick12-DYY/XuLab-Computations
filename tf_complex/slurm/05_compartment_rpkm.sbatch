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
# (04_chromhmm_rpkm.sbatch) with the unit swapped from a chromatin state to a
# compartment unit. Cell-count bars are kept; there is no state-composition panel.
#
# UNIT=class (default) is the one to trust: 2x9 eigenvector-strength compartment
# classes pooled genome-wide, matching ChromHMM-18's dimensionality and reads/unit,
# built by build_compartment_classes.py and then the *same* build_chromhmm_matrix.py.
# UNIT=domain|bin puts one unit per compartment domain / 25 kb bin; at that
# granularity shallow TFs are Poisson-dominated and mean r tracks cell count at
# rho=+0.98 (vs +0.22 for ChromHMM-18), so those runs are for comparison only.
#
#   1. build_compartment_classes.py      A/B + E1 -> 18 compartment-class BED   [class]
#      build_chromhmm_matrix.py          reads -> (classes x TFs) RPKM          [class]
#      build_compartment_matrix.py       reads -> (domains x TFs) RPKM          [domain|bin]
#   2. compartment_rpkm_correlation.py   Pearson TF x TF, cluster, heatmap + cell bars
#
#   sbatch tf_complex/slurm/05_compartment_rpkm.sbatch
#   PER_ARM=5 sbatch tf_complex/slurm/05_compartment_rpkm.sbatch      # 10 classes instead of 18
#   UNIT=domain sbatch tf_complex/slurm/05_compartment_rpkm.sbatch    # per-domain (confounded)
#   UNIT=bin sbatch tf_complex/slurm/05_compartment_rpkm.sbatch       # 25 kb bins (RPKM == CPM)
#   REBUILD=0 sbatch tf_complex/slurm/05_compartment_rpkm.sbatch      # reuse the matrix
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
TFC="${XULAB}/tf_complex"
WORK_ROOT="${WORK_ROOT:-${XULAB}/unified/work}"
UNIT="${UNIT:-class}"                     # class (E1-strength classes, recommended) | domain | bin
AB_BED="${AB_BED:-${XULAB}/hic/work/compartments/compartments_25000.AB.bed}"
if [[ "${UNIT}" == "bin" ]]; then
  DOMAINS_BED="${DOMAINS_BED:-${AB_BED}}"
else
  DOMAINS_BED="${DOMAINS_BED:-${XULAB}/hic/work/compartments/compartments_25000.domains.bed}"
fi
PER_ARM="${PER_ARM:-9}"                   # classes per arm; total units = 2*PER_ARM
SMOOTH="${SMOOTH:-2}"                     # absorb sandwiched flips of <= N bins (0 = raw calls)
TAG="${TAG:-${UNIT}$([[ "${SMOOTH}" != "0" ]] && echo "_sm${SMOOTH}")}"
MATRIX_DIR="${MATRIX_DIR:-${TFC}/work/compartment_rpkm_${TAG}}"
RESULTS="${RESULTS:-${TFC}/results_compartment_rpkm_${TAG}}"
SOURCE="${SOURCE:-raw}"                   # raw (mm) | imputed
TRANSFORM="${TRANSFORM:-rpkm}"            # rpkm | log (log1p of the RPKM matrix)
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
    echo "[$(date)] build compartment RPKM matrix (unit=${UNIT}, source=${SOURCE})"
    python "${TFC}/scripts/build_compartment_matrix.py" \
      --work-root "${WORK_ROOT}" --domains-bed "${DOMAINS_BED}" --unit "${UNIT}" \
      --out-dir "${MATRIX_DIR}" --source "${SOURCE}" --transform rpkm
  fi
fi

echo "[$(date)] correlate + plot (transform=${TRANSFORM}, k=${N_CLUSTERS})"
python "${TFC}/scripts/compartment_rpkm_correlation.py" \
  --matrix-dir "${MATRIX_DIR}" --out-dir "${RESULTS}" \
  --transform "${TRANSFORM}" --n-clusters "${N_CLUSTERS}" --cell-meta "${CELL_META}"
echo "[$(date)] done -> ${RESULTS}"
