#!/bin/bash
#SBATCH --job-name=ds_bin_sens_spec
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Bin-level sensitivity / specificity using the cCRE-derived ground truth
# (positive / negative bin sets from prepare_pos_neg_bins.R, which reproduce
# Bin_posVSneg.ipynb). Companion to the entry-level sensitivity_specificity.sbatch.
#
# Defaults match compare_pos_neg_all4.sbatch (this XuLab tree). Override any path
# by exporting the variable before sbatch.
#
# Optional env vars (defaults shown):
#   ROOT_DIR      repo root (default: auto from submit dir, else /work/users/d/y/dyy12/XuLab)
#   BINS_TSV      output of prepare_pos_neg_bins.R
#                 (default: ${ROOT_DIR}/downstream/bins/pos_neg_bins.tsv)
#   OUT_DIR       output directory
#                 (default: ${ROOT_DIR}/downstream/bin_sensitivity_specificity)
#   INPUTS        space-separated LABEL=PATH pairs; if unset, built from the dirs below
#   MM_DIR        raw matrix dir for the "raw" input
#                 (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm)
#   CISTOPIC_DIR  (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/impute)
#   FITS_DIR      (default: ${ROOT_DIR}/FITS/work/ctcf/impute)
#   SCOPEN_DIR    (default: ${ROOT_DIR}/scOpen/work/ctcf/impute)
#   MAGIC_DIR     (default: ${ROOT_DIR}/MAGIC/work/ctcf/impute)
#   CONDA_ENV     conda env (default: cistopic; FITS/scopen/magic envs work too)
#   OUT_NAME      basename for output files (default: bin_sensitivity_specificity)
#   THRESHOLDS    comma-separated absolute thresholds (default: auto per-pipeline)
#   N_THRESHOLDS  number of auto thresholds per pipeline (default 64)

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
    ROOT_DIR="${SUBMIT_DIR}"
  elif [[ -f "${SUBMIT_DIR}/bin_sensitivity_specificity.py" ]]; then
    ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"

DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

BINS_TSV="${BINS_TSV:-${ROOT_DIR}/downstream/bins/pos_neg_bins.tsv}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/downstream/bin_sensitivity_specificity}"

MM_DIR="${MM_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm}"
CISTOPIC_DIR="${CISTOPIC_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/impute}"
FITS_DIR="${FITS_DIR:-${ROOT_DIR}/FITS/work/ctcf/impute}"
SCOPEN_DIR="${SCOPEN_DIR:-${ROOT_DIR}/scOpen/work/ctcf/impute}"
MAGIC_DIR="${MAGIC_DIR:-${ROOT_DIR}/MAGIC/work/ctcf/impute}"

if [[ -z "${INPUTS:-}" ]]; then
  INPUTS="raw=${MM_DIR} cisTopic=${CISTOPIC_DIR} FITS=${FITS_DIR} scOpen=${SCOPEN_DIR} MAGIC=${MAGIC_DIR}"
fi

if [[ -z "${BINS_TSV:-}" || -z "${OUT_DIR:-}" || -z "${INPUTS:-}" ]]; then
  echo "Required: BINS_TSV=<tsv>, OUT_DIR=<dir>, INPUTS=\"label1=path1 ...\" (or rely on defaults under ROOT_DIR)" >&2
  exit 1
fi
CONDA_ENV="${CONDA_ENV:-cistopic}"
OUT_NAME="${OUT_NAME:-bin_sensitivity_specificity}"
N_THRESHOLDS="${N_THRESHOLDS:-64}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(
  python "${DOWN_DIR}/bin_sensitivity_specificity.py"
  --bins-tsv     "${BINS_TSV}"
  --out-dir      "${OUT_DIR}"
  --out-name     "${OUT_NAME}"
  --n-thresholds "${N_THRESHOLDS}"
)
if [[ -n "${THRESHOLDS:-}" ]]; then
  CMD+=(--thresholds "${THRESHOLDS}")
fi
for pair in ${INPUTS}; do
  CMD+=(--input "${pair}")
done

echo "Running: ${CMD[*]}"
"${CMD[@]}"
