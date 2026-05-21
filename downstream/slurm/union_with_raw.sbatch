#!/bin/bash
#SBATCH --job-name=ds_union_raw
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# OPTIONAL / legacy: materialize max(raw, imputed) CSRs (downstream/union_with_raw.py).
# Not used by default downstream workflow — comparators read native impute/ dirs.
# Writes <OUT_ROOT>/<label>/matrix_csr.npz, regions.tsv, barcodes.tsv, meta.json
# for each LABEL=IMPUTE_DIR pair.
#
# Optional env vars:
#   ROOT_DIR         repo root (default: auto from submit dir)
#   MM_DIR           raw mm (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm)
#   OUT_ROOT         parent directory for per-label outputs (= IMPUTE_UNION_ROOT
#                    for downstream defaults; default ${ROOT_DIR}/downstream/impute_union_mm)
#   INPUTS           space-separated LABEL=PATH impute dirs (default: 7 imputed pipelines)
#   THRESHOLD_MODE   (default: sparsity_match:2)
#   FROM_META        if 1, pass --from-meta (use impute meta.json threshold)
#   CHUNK_ROWS, SAMPLE_SIZE, SEED
#   NO_UNION         if 1, pass --no-union (thresholded imputed only)
#   REPLACEMENT_MODE if 1, pass --replacement-mode (K*raw_nnz before union)
#   CONDA_ENV        (default: cistopic)

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
    ROOT_DIR="${SUBMIT_DIR}"
  elif [[ -f "${SUBMIT_DIR}/union_with_raw.py" ]]; then
    ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"

DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/downstream/impute_union_mm}"
DOWN_DIR="${ROOT_DIR}/downstream"
# shellcheck source=_pipeline_paths.sh
source "${DOWN_DIR}/slurm/_pipeline_paths.sh"
if [[ -z "${INPUTS:-}" ]]; then
  default_inputs imputed
fi
pipeline_paths_native

CONDA_ENV="${CONDA_ENV:-cistopic}"
THRESHOLD_MODE="${THRESHOLD_MODE:-sparsity_match:2}"
CHUNK_ROWS="${CHUNK_ROWS:-2000}"
SAMPLE_SIZE="${SAMPLE_SIZE:-4000000}"
SEED="${SEED:-42}"
FROM_META="${FROM_META:-0}"
NO_UNION="${NO_UNION:-0}"
REPLACEMENT_MODE="${REPLACEMENT_MODE:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${OUT_ROOT}"

for pair in ${INPUTS}; do
  label="${pair%%=*}"
  impdir="${pair#*=}"
  echo "[$(date)] union_with_raw: label=${label} impute-dir=${impdir}"
  CMD=(
    python "${DOWN_DIR}/union_with_raw.py"
    --mm-dir "${MM_DIR}"
    --impute-dir "${impdir}"
    --out-dir "${OUT_ROOT}"
    --label "${label}"
    --threshold-mode "${THRESHOLD_MODE}"
    --chunk-rows "${CHUNK_ROWS}"
    --sample-size "${SAMPLE_SIZE}"
    --seed "${SEED}"
  )
  if [[ "${FROM_META}" =~ ^(1|true|yes|on)$ ]]; then
    CMD+=(--from-meta)
  fi
  if [[ "${NO_UNION}" =~ ^(1|true|yes|on)$ ]]; then
    CMD+=(--no-union)
  fi
  if [[ "${REPLACEMENT_MODE}" =~ ^(1|true|yes|on)$ ]]; then
    CMD+=(--replacement-mode)
  fi
  echo "Running: ${CMD[*]}"
  "${CMD[@]}"
done

echo "[$(date)] Done. Union CSRs under: ${OUT_ROOT}/<label>/"
