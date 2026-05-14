#!/bin/bash
#SBATCH --job-name=ds_union_raw
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# Materialize max(raw, imputed) CSRs for one or more impute directories (same
# semantics as downstream/union_with_raw.py + sparsity_match:K addition mode).
# Writes
#   <OUT_ROOT>/<label>/matrix_csr.npz, regions.tsv, barcodes.tsv, meta.json
# for each LABEL=IMPUTE_DIR pair. Downstream sbatch scripts default
# IMPUTE_UNION_ROOT to this same path so compare_pos_neg / complexity_gain / …
# read ${IMPUTE_UNION_ROOT}/cisTopic, /FITS, /scOpen, etc. by default.
#
# Optional env vars:
#   ROOT_DIR         repo root (default: auto from submit dir)
#   MM_DIR           raw mm (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm)
#   OUT_ROOT         parent directory for per-label outputs (= IMPUTE_UNION_ROOT
#                    for downstream defaults; default ${ROOT_DIR}/downstream/impute_union_mm)
#   INPUTS           space-separated LABEL=PATH impute dirs (default: 5 pipelines)
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

MM_DIR="${MM_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/downstream/impute_union_mm}"

CISTOPIC_DIR="${CISTOPIC_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/impute}"
FITS_DIR="${FITS_DIR:-${ROOT_DIR}/FITS/work/ctcf/impute}"
SCOPEN_DIR="${SCOPEN_DIR:-${ROOT_DIR}/scOpen/work/ctcf/impute}"
MAGIC_DIR="${MAGIC_DIR:-${ROOT_DIR}/MAGIC/work/ctcf/impute}"
PUSCOPEN_DIR="${PUSCOPEN_DIR:-${ROOT_DIR}/PUscOpen/work/ctcf/impute}"

if [[ -z "${INPUTS:-}" ]]; then
  INPUTS="cisTopic=${CISTOPIC_DIR} FITS=${FITS_DIR} scOpen=${SCOPEN_DIR} MAGIC=${MAGIC_DIR} PUscOpen=${PUSCOPEN_DIR}"
fi

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
