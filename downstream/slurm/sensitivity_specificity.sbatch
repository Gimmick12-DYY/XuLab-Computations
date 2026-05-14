#!/bin/bash
#SBATCH --job-name=ds_sens_spec
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# Compute sensitivity / specificity / AUROC / AUPR / max-F1 for each imputation
# pipeline against the raw observed (binarised) matrix.
#
# Defaults match compare_pos_neg_all4.sbatch (this XuLab tree). Override any path
# by exporting the variable before sbatch. For union CSRs, set each pipeline
# *_DIR to the directory that holds the post-union matrix_csr.npz (see compare_pos_neg_all4.sbatch header).
#
# Optional env vars (defaults shown):
#   ROOT_DIR      repo root (default: auto from submit dir, else /work/users/d/y/dyy12/XuLab)
#   IMPUTE_UNION_ROOT  parent of per-pipeline union dirs (default: .../impute_union_mm)
#   MM_DIR        raw matrix dir: matrix.mtx.gz + regions.tsv.gz + barcodes.tsv.gz
#                 (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm)
#   OUT_DIR       output directory (default: ${ROOT_DIR}/downstream/sensitivity_specificity)
#   INPUTS        space-separated LABEL=PATH pairs; if unset, built from the dirs below
#   CISTOPIC_DIR  (default: ${IMPUTE_UNION_ROOT}/cisTopic)
#   FITS_DIR      (default: ${IMPUTE_UNION_ROOT}/FITS)
#   SCOPEN_DIR    (default: ${IMPUTE_UNION_ROOT}/scOpen)
#   MAGIC_DIR     (default: ${IMPUTE_UNION_ROOT}/MAGIC)
#   PUSCOPEN_DIR  (default: ${IMPUTE_UNION_ROOT}/PUscOpen)
#   CONDA_ENV     conda env (default: cistopic; FITS/scopen/magic envs work too)
#   OUT_NAME      basename for output files (default: sensitivity_specificity)
#   THRESHOLDS    comma-separated absolute thresholds (default: auto per-pipeline)
#   N_THRESHOLDS  number of auto thresholds per pipeline (default 48)
#   CHUNK_ROWS    streaming chunk size in rows (default 2000)
#   SEED          RNG seed for the threshold-sample step (default 42)

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
    ROOT_DIR="${SUBMIT_DIR}"
  elif [[ -f "${SUBMIT_DIR}/sensitivity_specificity.py" ]]; then
    ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"

DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

MM_DIR="${MM_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/downstream/sensitivity_specificity}"

IMPUTE_UNION_ROOT="${IMPUTE_UNION_ROOT:-${ROOT_DIR}/downstream/impute_union_mm}"
CISTOPIC_DIR="${CISTOPIC_DIR:-${IMPUTE_UNION_ROOT}/cisTopic}"
FITS_DIR="${FITS_DIR:-${IMPUTE_UNION_ROOT}/FITS}"
SCOPEN_DIR="${SCOPEN_DIR:-${IMPUTE_UNION_ROOT}/scOpen}"
MAGIC_DIR="${MAGIC_DIR:-${IMPUTE_UNION_ROOT}/MAGIC}"
PUSCOPEN_DIR="${PUSCOPEN_DIR:-${IMPUTE_UNION_ROOT}/PUscOpen}"

if [[ -z "${INPUTS:-}" ]]; then
  INPUTS="cisTopic=${CISTOPIC_DIR} FITS=${FITS_DIR} scOpen=${SCOPEN_DIR} MAGIC=${MAGIC_DIR} PUscOpen=${PUSCOPEN_DIR}"
fi

if [[ -z "${MM_DIR:-}" || -z "${OUT_DIR:-}" || -z "${INPUTS:-}" ]]; then
  echo "Required: MM_DIR=<mm/>, OUT_DIR=<dir>, INPUTS=\"label1=path1 label2=path2 ...\" (or rely on defaults under ROOT_DIR)" >&2
  exit 1
fi
CONDA_ENV="${CONDA_ENV:-cistopic}"
OUT_NAME="${OUT_NAME:-sensitivity_specificity}"
N_THRESHOLDS="${N_THRESHOLDS:-48}"
CHUNK_ROWS="${CHUNK_ROWS:-2000}"
SEED="${SEED:-42}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(
  python "${DOWN_DIR}/sensitivity_specificity.py"
  --mm-dir       "${MM_DIR}"
  --out-dir      "${OUT_DIR}"
  --out-name     "${OUT_NAME}"
  --n-thresholds "${N_THRESHOLDS}"
  --chunk-rows   "${CHUNK_ROWS}"
  --seed         "${SEED}"
)
if [[ -n "${THRESHOLDS:-}" ]]; then
  CMD+=(--thresholds "${THRESHOLDS}")
fi
for pair in ${INPUTS}; do
  CMD+=(--input "${pair}")
done

echo "Running: ${CMD[*]}"
"${CMD[@]}"
