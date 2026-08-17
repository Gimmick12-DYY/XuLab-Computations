#!/bin/bash
#SBATCH --job-name=igv_panel7_bw
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-13

# BigWig tracks for the 7-TF bulk-reference panel (unified raw + imputed).
# 14 array tasks: 7 TFs × {raw, imputed}.
#
# Convention (matches sc vs bulk validation):
#   raw     = per-bin sum over cells  (fragment-count pseudobulk)
#   imputed = per-bin mean over cells (occupancy fraction; binary CSR)
# Display: --normalize p99 so IGV y-scales are comparable within a TF.
#
# Outputs:
#   downstream/tracks/igv_panel7/<TF>_raw_sum_normp99.bw
#   downstream/tracks/igv_panel7/<TF>_imputed_mean_normp99.bw
#
# Usage:
#   cd /work/users/d/y/dyy12/XuLab
#   sbatch downstream/slurm/igv_panel7_raw_imputed_bw.sbatch

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-cistopic}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/downstream/cache}"
CHROM_SIZES="${CHROM_SIZES:-${CACHE_DIR}/hg38.chrom.sizes}"
TRACKS_DIR="${TRACKS_DIR:-${ROOT_DIR}/downstream/tracks/igv_panel7}"
NORMALIZE="${NORMALIZE:-p99}"
TRANSFORM="${TRANSFORM:-none}"
CENTER="${CENTER:-none}"
CHUNK_COLS="${CHUNK_COLS:-500}"
CHUNK_ROWS="${CHUNK_ROWS:-10000}"
PIP_EXTRAS="${PIP_EXTRAS:-1}"

TFS=(ctcf maz zic2 zbtb7a znf777 nfya znf282)
IDX="${SLURM_ARRAY_TASK_ID}"
TF_IDX=$((IDX / 2))
KIND_IDX=$((IDX % 2))
TF="${TFS[$TF_IDX]}"
TF_UP="$(echo "${TF}" | tr '[:lower:]' '[:upper:]')"

if [[ "${KIND_IDX}" -eq 0 ]]; then
  KIND="raw"
  INPUT_DIR="${ROOT_DIR}/unified/work/${TF}/mm"
  AGGREGATE="sum"
else
  KIND="imputed"
  INPUT_DIR="${ROOT_DIR}/unified/work/${TF}/impute"
  AGGREGATE="mean"
fi

SUF=""
[[ "${CENTER}" != "none" ]] && SUF+="_c${CENTER}"
[[ "${NORMALIZE}" != "none" ]] && SUF+="_norm${NORMALIZE}"
[[ "${TRANSFORM}" != "none" ]] && SUF+="_${TRANSFORM}"
OUTPUT_BW="${TRACKS_DIR}/${TF_UP}_${KIND}_${AGGREGATE}${SUF}.bw"

mkdir -p "${TRACKS_DIR}" "${CACHE_DIR}"

echo "[$(date)] task=${IDX} TF=${TF_UP} kind=${KIND}"
echo "[$(date)] INPUT_DIR=${INPUT_DIR}"
echo "[$(date)] OUTPUT_BW=${OUTPUT_BW}"
echo "[$(date)] AGGREGATE=${AGGREGATE} NORMALIZE=${NORMALIZE}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "ERROR: missing input dir ${INPUT_DIR}" >&2
  exit 1
fi

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
