#!/bin/bash
#SBATCH --job-name=ds_umi_dist
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Per-cell UMI / signal-sum distribution across pipelines.
#
# Defaults: native impute dirs + raw mm. Override any path before sbatch.
#
# Optional env vars (defaults shown):
#   ROOT_DIR      repo root (default: auto from submit dir)
#   OUT_DIR       output directory (default: ${ROOT_DIR}/downstream/umi_distribution)
#   INPUTS        space-separated LABEL=PATH pairs; if unset, built from below.
#   MM_DIR        raw matrix dir
#                 (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm)
#   CISTOPIC_DIR  (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/impute)
#   FITS_DIR      (default: ${ROOT_DIR}/FITS/work/ctcf/impute)
#   SCOPEN_DIR    (default: ${ROOT_DIR}/scOpen/work/ctcf/impute)
#   MAGIC_DIR     (default: ${ROOT_DIR}/MAGIC/work/ctcf/impute)
#   PUSCOPEN_DIR  (default: ${ROOT_DIR}/PUscOpen/work/ctcf/impute)
#   CICERO_DIR    (default: ${ROOT_DIR}/Cicero/work/ctcf/impute)
#   SCBASSET_DIR  (default: ${ROOT_DIR}/scBasset/work/ctcf/impute)
#   SCBASSET_PUSCOPEN_DIR  (default: ${ROOT_DIR}/scBasset_PUscOpen/work/ctcf/impute)
#   SCBASSET_CISTOPIC_DIR  (default: ${ROOT_DIR}/scBasset_cisTopic/work/ctcf/impute)
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
DOWN_DIR="${ROOT_DIR}/downstream"
# shellcheck source=_pipeline_paths.sh
source "${DOWN_DIR}/slurm/_pipeline_paths.sh"
if [[ -z "${INPUTS:-}" ]]; then
  default_inputs all
fi
pipeline_paths_native

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
