#!/bin/bash
#SBATCH --job-name=igv_union_bw
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-78

# BigWig tracks for unified imputed matrices after the ATAC-union open-chromatin
# mask (HEK293T_ATAC_union3). One array task per TF under unified/work/*/impute.
#
# Convention (same as igv_panel7 imputed tracks):
#   imputed = per-bin mean over cells (occupancy fraction; binary CSR)
#   --normalize p99 for comparable IGV y-scales
#
# Outputs:
#   downstream/tracks/igv_unified_union/<TF>_imputed_mean_normp99.bw
#
# Usage:
#   cd /work/users/d/y/dyy12/XuLab
#   sbatch downstream/slurm/igv_unified_union_imputed_bw.sbatch
#   # or one TF:
#   TF=ctcf sbatch --array=0 downstream/slurm/igv_unified_union_imputed_bw.sbatch

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-cistopic}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/downstream/cache}"
CHROM_SIZES="${CHROM_SIZES:-${CACHE_DIR}/hg38.chrom.sizes}"
TRACKS_DIR="${TRACKS_DIR:-${ROOT_DIR}/downstream/tracks/igv_unified_union}"
NORMALIZE="${NORMALIZE:-p99}"
TRANSFORM="${TRANSFORM:-none}"
CENTER="${CENTER:-none}"
CHUNK_COLS="${CHUNK_COLS:-500}"
CHUNK_ROWS="${CHUNK_ROWS:-10000}"
PIP_EXTRAS="${PIP_EXTRAS:-1}"
AGGREGATE="${AGGREGATE:-mean}"

mapfile -t TFS < <(
  for d in "${ROOT_DIR}"/unified/work/*/impute/matrix_csr.npz; do
    basename "$(dirname "$(dirname "$d")")"
  done | sort
)

if [[ -n "${TF:-}" ]]; then
  TF="$(echo "${TF}" | tr '[:upper:]' '[:lower:]')"
else
  IDX="${SLURM_ARRAY_TASK_ID:-0}"
  if (( IDX < 0 || IDX >= ${#TFS[@]} )); then
    echo "ERROR: array index ${IDX} out of range 0-$((${#TFS[@]} - 1))" >&2
    exit 1
  fi
  TF="${TFS[$IDX]}"
fi

TF_UP="$(echo "${TF}" | tr '[:lower:]' '[:upper:]')"
INPUT_DIR="${ROOT_DIR}/unified/work/${TF}/impute"

SUF=""
[[ "${CENTER}" != "none" ]] && SUF+="_c${CENTER}"
[[ "${NORMALIZE}" != "none" ]] && SUF+="_norm${NORMALIZE}"
[[ "${TRANSFORM}" != "none" ]] && SUF+="_${TRANSFORM}"
OUTPUT_BW="${TRACKS_DIR}/${TF_UP}_imputed_${AGGREGATE}${SUF}.bw"

mkdir -p "${TRACKS_DIR}" "${CACHE_DIR}"

echo "[$(date)] TF=${TF_UP} n_tfs=${#TFS[@]}"
echo "[$(date)] INPUT_DIR=${INPUT_DIR}"
echo "[$(date)] OUTPUT_BW=${OUTPUT_BW}"
echo "[$(date)] AGGREGATE=${AGGREGATE} NORMALIZE=${NORMALIZE}"

if [[ ! -f "${INPUT_DIR}/matrix_csr.npz" ]]; then
  echo "ERROR: missing ${INPUT_DIR}/matrix_csr.npz" >&2
  exit 1
fi

# Confirm this impute dir is the union-masked matrix.
python3 - <<PY
import json, sys
from pathlib import Path
meta = Path("${INPUT_DIR}") / "meta.json"
if not meta.is_file():
    print("WARN: no meta.json; proceeding without mask check", flush=True)
    sys.exit(0)
m = json.loads(meta.read_text())
bed = m.get("open_chromatin_bed") or ""
print(f"[meta] open_chromatin_bed={bed}", flush=True)
print(f"[meta] open_bins={m.get('open_chromatin_n_open_bins')}", flush=True)
if "ATAC_union" not in bed and "union" not in bed.lower():
    print(f"ERROR: expected ATAC-union mask, got: {bed!r}", file=sys.stderr)
    sys.exit(1)
PY

if [[ ! -f "${CHROM_SIZES}" ]]; then
  echo "[$(date)] downloading hg38.chrom.sizes -> ${CHROM_SIZES}"
  wget -q -O "${CHROM_SIZES}" \
    'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes'
fi

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if ! python -c "import pyBigWig" 2>/dev/null; then
  if [[ "${PIP_EXTRAS}" == "1" ]]; then
    python -m pip install --upgrade-strategy only-if-needed pyBigWig
  else
    echo "ERROR: pyBigWig missing" >&2
    exit 1
  fi
fi

extra=()
[[ "${CENTER}" != "none" ]] && extra+=(--center "${CENTER}")
[[ "${NORMALIZE}" != "none" ]] && extra+=(--normalize "${NORMALIZE}")
[[ "${TRANSFORM}" != "none" ]] && extra+=(--transform "${TRANSFORM}")

python "${ROOT_DIR}/downstream/imputed_matrix_to_bigwig.py" \
  --input "${INPUT_DIR}" \
  --output "${OUTPUT_BW}" \
  --chrom-sizes "${CHROM_SIZES}" \
  --aggregate "${AGGREGATE}" \
  --chunk-cols "${CHUNK_COLS}" \
  --chunk-rows "${CHUNK_ROWS}" \
  "${extra[@]}"

echo "[$(date)] Done: ${OUTPUT_BW}"
