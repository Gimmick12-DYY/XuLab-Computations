#!/bin/bash
#SBATCH --job-name=ds_complexity_gain
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00

# Per-cell complexity gain: for each pipeline, count bins above a per-pipeline
# threshold per cell; compare to raw nonzero count per cell.
#
# Threshold mode default: sparsity_match:2 (matches cisTopic impute when using
# K=2). Reads native impute outputs (not pre-merged union CSRs).
# Set UNION_WITH_RAW=1 to pass --union-with-raw (max(raw, imputed) counting).
# Use quantile:0.99 / fixed:0.1 for other operating points.
#
# Defaults match compare_pos_neg_all4.sbatch (this XuLab tree). Override any
# path by exporting the variable before sbatch.
#
# Optional env vars (defaults shown):
#   ROOT_DIR             repo root (default: auto from submit dir)
#   MM_DIR               raw matrix dir
#                        (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm)
#   OUT_DIR              output directory
#                        (default: ${ROOT_DIR}/downstream/complexity_gain)
#   INPUTS               space-separated LABEL=PATH pairs; if unset, built below.
#   CISTOPIC_DIR / FITS_DIR / SCOPEN_DIR / MAGIC_DIR / PUSCOPEN_DIR
#   CONDA_ENV       conda env (default: cistopic)
#   OUT_NAME        basename for output files (default: complexity_gain)
#   THRESHOLD_MODE  sparsity_match[:K] | quantile:Q | fixed:T
#                   (default: sparsity_match:2 — matches cisTopic/configs/default.yaml)
#   UNION_WITH_RAW  if 1/true, pass --union-with-raw (default 0: native imputed only).
#   CHUNK_ROWS      streaming chunk size (default 2000)
#   SAMPLE_SIZE     random sample size for threshold quantile estimation
#                   (default 4000000)
#   SEED            RNG seed (default 42)

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
    ROOT_DIR="${SUBMIT_DIR}"
  elif [[ -f "${SUBMIT_DIR}/complexity_gain.py" ]]; then
    ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"

DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/downstream/complexity_gain}"
DOWN_DIR="${ROOT_DIR}/downstream"
# shellcheck source=_pipeline_paths.sh
source "${DOWN_DIR}/slurm/_pipeline_paths.sh"
if [[ -z "${INPUTS:-}" ]]; then
  default_inputs imputed
fi
pipeline_paths_native

CONDA_ENV="${CONDA_ENV:-cistopic}"
OUT_NAME="${OUT_NAME:-complexity_gain}"
THRESHOLD_MODE="${THRESHOLD_MODE:-sparsity_match:2}"
UNION_WITH_RAW="${UNION_WITH_RAW:-0}"
CHUNK_ROWS="${CHUNK_ROWS:-2000}"
SAMPLE_SIZE="${SAMPLE_SIZE:-4000000}"
SEED="${SEED:-42}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(
  python "${DOWN_DIR}/complexity_gain.py"
  --mm-dir         "${MM_DIR}"
  --out-dir        "${OUT_DIR}"
  --out-name       "${OUT_NAME}"
  --threshold-mode "${THRESHOLD_MODE}"
  --chunk-rows     "${CHUNK_ROWS}"
  --sample-size    "${SAMPLE_SIZE}"
  --seed           "${SEED}"
)
case "${UNION_WITH_RAW}" in
  0|false|FALSE|no|No|NO|off|OFF) ;;
  *) CMD+=(--union-with-raw) ;;
esac
for pair in ${INPUTS}; do
  CMD+=(--input "${pair}")
done

echo "Running: ${CMD[*]}"
"${CMD[@]}"
