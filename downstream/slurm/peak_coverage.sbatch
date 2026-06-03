#!/bin/bash
#SBATCH --job-name=ds_peak_cov
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# MACS2 peak calling on pseudo-bulk BEDs from each pipeline matrix, then
# reference pos/neg bin coverage. Requires macs2 (cistopic env; pip build works
# on Longleaf — bioconda macs2 in data_prep hits __log_finite on glibc 2.34+).
#
# Default INPUTS: raw + all ten imputed pipelines (same as compare_pos_neg_all10).
# Override INPUTS or individual *_DIR before sbatch.
#
# Optional env vars:
#   ROOT_DIR, CONDA_ENV, MM_DIR, BINS_TSV, PEAK_COV_OUT_DIR,
#   OUT_NAME, TARGET_FRAGMENTS, INPUTS,
#   CISTOPIC_DIR, FITS_DIR, SCOPEN_DIR, MAGIC_DIR, PUSCOPEN_DIR, CICERO_DIR,
#   SCBASSET_DIR, SCBASSET_PUSCOPEN_DIR, SCBASSET_CISTOPIC_DIR,
#   SCBASSET_CISTOPIC_PUSCOPEN_DIR

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
    ROOT_DIR="${SUBMIT_DIR}"
  elif [[ -f "${SUBMIT_DIR}/peak_coverage.py" ]]; then
    ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"

DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

# shellcheck source=_pipeline_paths.sh
source "${DOWN_DIR}/slurm/_pipeline_paths.sh"

BINS_TSV="${BINS_TSV:-${ROOT_DIR}/downstream/bins/pos_neg_bins.tsv}"
PEAK_COV_OUT_DIR="${PEAK_COV_OUT_DIR:-${ROOT_DIR}/downstream/peak_coverage_ctcf}"
OUT_NAME="${OUT_NAME:-peak_coverage}"
TARGET_FRAGMENTS="${TARGET_FRAGMENTS:-50000000}"
CONDA_ENV="${CONDA_ENV:-cistopic}"

if [[ -z "${INPUTS:-}" ]]; then
  default_inputs all
fi
pipeline_paths_native

if [[ ! -f "${BINS_TSV}" ]]; then
  echo "ERROR: BINS_TSV not found: ${BINS_TSV}" >&2
  echo "Run: sbatch downstream/slurm/prepare_pos_neg_bins.sbatch" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if ! command -v macs2 >/dev/null 2>&1; then
  echo "ERROR: macs2 not on PATH in env ${CONDA_ENV}" >&2
  exit 1
fi

mkdir -p "${PEAK_COV_OUT_DIR}"

echo "[$(date)] peak_coverage: TARGET_FRAGMENTS=${TARGET_FRAGMENTS}"
echo "BINS_TSV=${BINS_TSV}"
echo "OUT_DIR=${PEAK_COV_OUT_DIR}"
echo "INPUTS=${INPUTS}"

CMD=(
  python "${DOWN_DIR}/peak_coverage.py"
  --bins-tsv "${BINS_TSV}"
  --out-dir "${PEAK_COV_OUT_DIR}"
  --out-name "${OUT_NAME}"
  --target-fragments "${TARGET_FRAGMENTS}"
)
for pair in ${INPUTS}; do
  CMD+=(--input "${pair}")
done
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "[$(date)] Done."
echo "  ${PEAK_COV_OUT_DIR}/${OUT_NAME}.tsv"
