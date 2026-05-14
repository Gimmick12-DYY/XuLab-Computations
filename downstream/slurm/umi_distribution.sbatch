#!/bin/bash
#SBATCH --job-name=ds_umi_dist
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Per-cell UMI / signal-sum distribution across pipelines.
#
# Defaults match compare_pos_neg_all4.sbatch (this XuLab tree). Override any
# path by exporting the variable before sbatch. Point pipeline dirs at union
# matrix_csr.npz locations when comparing union outputs (same as compare_pos_neg).
#
# Optional env vars (defaults shown):
#   ROOT_DIR      repo root (default: auto from submit dir)
#   IMPUTE_UNION_ROOT  parent of per-pipeline union dirs (default: .../impute_union_mm)
#   OUT_DIR       output directory (default: ${ROOT_DIR}/downstream/umi_distribution)
#   INPUTS        space-separated LABEL=PATH pairs; if unset, built from below.
#   MM_DIR        raw matrix dir
#                 (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm)
#   CISTOPIC_DIR  (default: ${IMPUTE_UNION_ROOT}/cisTopic)
#   FITS_DIR      (default: ${IMPUTE_UNION_ROOT}/FITS)
#   SCOPEN_DIR    (default: ${IMPUTE_UNION_ROOT}/scOpen)
#   MAGIC_DIR     (default: ${IMPUTE_UNION_ROOT}/MAGIC)
#   PUSCOPEN_DIR  (default: ${IMPUTE_UNION_ROOT}/PUscOpen)
#   ALIGN_TO      input label whose barcode order anchors the comparison
#                 (default: raw). Pass "none" to skip alignment entirely.
#   CONDA_ENV     conda env (default: cistopic; FITS/scopen/magic envs work too)
#   OUT_NAME      basename for output files (default: umi_distribution)

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
    ROOT_DIR="${SUBMIT_DIR}"
  elif [[ -f "${SUBMIT_DIR}/umi_distribution.py" ]]; then
    ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"

DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/downstream/umi_distribution}"
MM_DIR="${MM_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm}"
IMPUTE_UNION_ROOT="${IMPUTE_UNION_ROOT:-${ROOT_DIR}/downstream/impute_union_mm}"
CISTOPIC_DIR="${CISTOPIC_DIR:-${IMPUTE_UNION_ROOT}/cisTopic}"
FITS_DIR="${FITS_DIR:-${IMPUTE_UNION_ROOT}/FITS}"
SCOPEN_DIR="${SCOPEN_DIR:-${IMPUTE_UNION_ROOT}/scOpen}"
MAGIC_DIR="${MAGIC_DIR:-${IMPUTE_UNION_ROOT}/MAGIC}"
PUSCOPEN_DIR="${PUSCOPEN_DIR:-${IMPUTE_UNION_ROOT}/PUscOpen}"

if [[ -z "${INPUTS:-}" ]]; then
  INPUTS="raw=${MM_DIR} cisTopic=${CISTOPIC_DIR} FITS=${FITS_DIR} scOpen=${SCOPEN_DIR} MAGIC=${MAGIC_DIR} PUscOpen=${PUSCOPEN_DIR}"
fi

CONDA_ENV="${CONDA_ENV:-cistopic}"
OUT_NAME="${OUT_NAME:-umi_distribution}"
ALIGN_TO="${ALIGN_TO:-raw}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(
  python "${DOWN_DIR}/umi_distribution.py"
  --out-dir   "${OUT_DIR}"
  --out-name  "${OUT_NAME}"
)
if [[ -n "${ALIGN_TO}" ]]; then
  CMD+=(--align-to "${ALIGN_TO}")
fi
for pair in ${INPUTS}; do
  CMD+=(--input "${pair}")
done

echo "Running: ${CMD[*]}"
"${CMD[@]}"
