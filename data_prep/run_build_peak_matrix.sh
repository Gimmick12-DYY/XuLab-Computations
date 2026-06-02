#!/usr/bin/env bash
# Build a peak x cell RDS for one TF (see config.yaml).
#
#   bash data_prep/run_build_peak_matrix.sh CTCF
#   bash data_prep/run_build_peak_matrix.sh ZNF282
#
set -euo pipefail

TF="${1:-CTCF}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CFG="${CFG:-${SCRIPT_DIR}/config.yaml}"
CONDA_ENV="${CONDA_ENV:-data_prep}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

read_cfg() {
  python - <<PY
import sys
from pathlib import Path
import yaml

cfg_path = Path("${CFG}")
tf = "${TF}"
with cfg_path.open() as fh:
    cfg = yaml.safe_load(fh)
root = Path("${ROOT_DIR}")
entry = cfg["inputs"][tf]
frag = (root / entry["fragments"]).resolve()
out = (root / entry["out"]).resolve()
defs = cfg.get("defaults") or {}
print(frag)
print(out)
print(defs.get("min_cells_per_peak", 2))
print(defs.get("min_peaks_per_cell", 200))
PY
}

mapfile -t _vals < <(read_cfg)
FRAGMENTS="${_vals[0]}"
OUT="${_vals[1]}"
MIN_CELLS="${_vals[2]}"
MIN_PEAKS="${_vals[3]}"

if [[ ! -f "${FRAGMENTS}" ]]; then
  echo "fragments not found: ${FRAGMENTS}" >&2
  exit 1
fi

WORK_DIR="${OUT%.rds}.work"
mkdir -p "${WORK_DIR}"

# build_peak_matrix.py expects 5-col BED; repo fragments include a strand column.
FRAGMENTS_5="${WORK_DIR}/fragments.5col.bed.gz"
if [[ ! -s "${FRAGMENTS_5}" || "${FRAGMENTS}" -nt "${FRAGMENTS_5}" ]]; then
  echo "[run_build_peak_matrix] ${TF}: chr,start,end,barcode,count -> ${FRAGMENTS_5}"
  if [[ "${FRAGMENTS}" == *.gz ]]; then
    zcat "${FRAGMENTS}" | cut -f1-5 | gzip -c > "${FRAGMENTS_5}"
  else
    cut -f1-5 "${FRAGMENTS}" | gzip -c > "${FRAGMENTS_5}"
  fi
fi

python "${SCRIPT_DIR}/build_peak_matrix.py" \
  --fragments "${FRAGMENTS_5}" \
  --out "${OUT}" \
  --tf-name "${TF}" \
  --work-dir "${WORK_DIR}" \
  --min-cells-per-peak "${MIN_CELLS}" \
  --min-peaks-per-cell "${MIN_PEAKS}"
