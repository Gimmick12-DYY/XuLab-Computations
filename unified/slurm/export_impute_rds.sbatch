#!/bin/bash
#SBATCH --job-name=impute_to_rds
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Convert unified impute/matrix_csr.npz -> Kangli-style dgCMatrix .rds
# for sharing. Default: CTCF MAZ NFYA ZNF282.
#
# Usage:
#   sbatch unified/slurm/export_impute_rds.sbatch
#   TFS="ctcf maz" sbatch unified/slurm/export_impute_rds.sbatch

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/scripts" && -d "${SUBMIT_DIR}/configs" ]]; then
  UNI_DIR="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/unified/scripts" ]]; then
  UNI_DIR="${SUBMIT_DIR}/unified"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  UNI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
ROOT="$(cd "${UNI_DIR}/.." && pwd)"
cd "${ROOT}"

TFS="${TFS:-ctcf maz nfya znf282}"
OUT_DIR="${OUT_DIR:-${ROOT}/data/share}"
CONDA_ENV="${CONDA_ENV:-unified}"
CONDA_ENV_R="${CONDA_ENV_R:-cistopic}"

mkdir -p "${OUT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"

# Need scipy (unified) + R Matrix (cistopic). Prefer unified for python,
# point Rscript at cistopic's R if needed.
conda activate "${CONDA_ENV}"
RSCRIPT="$(command -v Rscript || true)"
if [[ -z "${RSCRIPT}" ]] || ! "${RSCRIPT}" -e 'library(Matrix)' >/dev/null 2>&1; then
  conda activate "${CONDA_ENV_R}"
  RSCRIPT="$(command -v Rscript)"
  # Still need scipy for the python step — activate unified for python path
  PY="$(conda run -n "${CONDA_ENV}" which python)"
else
  PY="$(command -v python)"
fi

echo "[$(date)] OUT_DIR=${OUT_DIR}"
echo "[$(date)] python=${PY} Rscript=${RSCRIPT}"

for tf in ${TFS}; do
  TF_UP="$(echo "${tf}" | tr '[:lower:]' '[:upper:]')"
  IMP="${ROOT}/unified/work/${tf}/impute"
  OUT="${OUT_DIR}/${TF_UP}_unified_imputed.rds"
  if [[ ! -f "${IMP}/matrix_csr.npz" ]]; then
    echo "ERROR: missing ${IMP}/matrix_csr.npz" >&2
    exit 1
  fi
  echo "[$(date)] === ${TF_UP}: ${IMP} -> ${OUT} ==="
  "${PY}" "${UNI_DIR}/scripts/03_export_impute_rds.py" \
    --impute-dir "${IMP}" \
    --out "${OUT}" \
    --rscript "${RSCRIPT}"
done

echo "[$(date)] Done. Shared RDS files:"
ls -lh "${OUT_DIR}"/*_unified_imputed.rds
