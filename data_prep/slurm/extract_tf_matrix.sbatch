#!/bin/bash
#SBATCH --job-name=extract_tf
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Build a per-TF unified-pipeline input from the combined TF1000cells data:
#   1. subset TF1000cells.bin1000.mtx.rds to one TF's cells -> <TF>_bin1000_mtx.rds
#      (same dgCMatrix format as CTCF_bin1000_mtx.rds; a drop-in paths.input_rds)
#   2. (optional) generate unified/configs/<tf>.yaml from default.yaml with
#      input_rds + work_dir repointed, so the pipeline is ready to run.
#
# The SCRIPT lives in the tracked repo (data_prep/); the DATA (.rds + per-cell
# metadata) lives under the gitignored data/ dir, default <repo>/data.
#
# Required:
#   TF            TF symbol to extract, e.g. NFYA or ZNF768 (must match meta$TF)
# Optional env:
#   DATA_DIR      dir holding the rds + metadata (default: <repo>/data)
#   MATRIX        combined bins x all-cells rds (default: $DATA_DIR/TF1000cells.bin1000.mtx.rds)
#   META          PER-CELL metadata CSV with 'barcode' + 'TF' columns
#                 (default: $DATA_DIR/TF1000cells.meta.csv). Must be the full
#                 per-cell file, NOT a TF->count pivot summary.
#   MIN_CELLS_PER_BIN  drop bins seen in < this many of the TF's cells (default 0)
#   MAKE_UNIFIED_CONFIG 1 (default) -> also write unified/configs/<tf>.yaml
#   CONDA_ENV     env with R + Matrix (default: cistopic)
#
# Usage:
#   TF=NFYA   sbatch data_prep/slurm/extract_tf_matrix.sbatch
#   TF=ZNF768 sbatch data_prep/slurm/extract_tf_matrix.sbatch

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/data_prep" && -d "${SUBMIT_DIR}/unified" ]]; then
  ROOT_DIR="${SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${TF:-}" ]]; then echo "ERROR: set TF=<symbol>, e.g. TF=NFYA" >&2; exit 1; fi
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
MATRIX="${MATRIX:-${DATA_DIR}/TF1000cells.bin1000.mtx.rds}"
META="${META:-${DATA_DIR}/TF1000cells.meta.csv}"
MIN_CELLS_PER_BIN="${MIN_CELLS_PER_BIN:-0}"
MAKE_UNIFIED_CONFIG="${MAKE_UNIFIED_CONFIG:-1}"
CONDA_ENV="${CONDA_ENV:-cistopic}"
R_SCRIPT="${ROOT_DIR}/data_prep/extract_tf_matrix.R"
TF_LC="$(echo "${TF}" | tr '[:upper:]' '[:lower:]')"
OUT_RDS="${DATA_DIR}/${TF}_bin1000_mtx.rds"

for f in "${R_SCRIPT}" "${MATRIX}" "${META}"; do
  if [[ ! -f "${f}" ]]; then echo "ERROR: missing ${f}" >&2; exit 1; fi
done

# Guard: META must be the PER-CELL file (has a 'barcode' and 'TF' column), not
# the TF->count pivot ('Row Labels,Count of cell'). Fail loudly if it's wrong.
HDR="$(head -1 "${META}")"
if ! echo "${HDR}" | grep -qiE '(^|,)barcode(,|$)' || ! echo "${HDR}" | grep -qiE '(^|,)TF(,|$)'; then
  echo "ERROR: ${META} is not the per-cell metadata." >&2
  echo "  header was: ${HDR}" >&2
  echo "  Need the full per-cell file with 'barcode' and 'TF' columns" >&2
  echo "  (this looks like a TF->count summary). Restore it or set META=." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "[$(date)] Extracting TF=${TF} from ${MATRIX}"
Rscript "${R_SCRIPT}" \
  --matrix "${MATRIX}" \
  --meta   "${META}" \
  --tf     "${TF}" \
  --out    "${OUT_RDS}" \
  --min-cells-per-bin "${MIN_CELLS_PER_BIN}"

echo "[$(date)] Wrote ${OUT_RDS}"

if [[ "${MAKE_UNIFIED_CONFIG}" == "1" ]]; then
  TEMPLATE="${ROOT_DIR}/unified/configs/default.yaml"
  CFG_OUT="${ROOT_DIR}/unified/configs/${TF_LC}.yaml"
  WORK_DIR="${ROOT_DIR}/unified/work/${TF_LC}"
  cp "${TEMPLATE}" "${CFG_OUT}"
  # Repoint only the two paths: keys (leading 2 spaces), preserving all comments.
  sed -i -E "s|^  input_rds:.*|  input_rds:   ${OUT_RDS}|" "${CFG_OUT}"
  sed -i -E "s|^  work_dir:.*|  work_dir:    ${WORK_DIR}|" "${CFG_OUT}"
  echo "[$(date)] Wrote unified config ${CFG_OUT}"
  echo "  input_rds -> ${OUT_RDS}"
  echo "  work_dir  -> ${WORK_DIR}"
  echo
  echo "Run the unified pipeline for ${TF}:"
  echo "  CFG=${CFG_OUT} sbatch unified/slurm/99_full_pipeline.sbatch"
fi

echo "[$(date)] Done."
