#!/bin/bash
#SBATCH --job-name=ds_impute2bw
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# SLURM wrapper for downstream/imputed_matrix_to_bigwig.py
# (IGV / UCSC BigWig; same role as Bioconductor rtracklayer::export(..., format="BigWig")).
#
# Required (or use defaults below):
#   INPUT_DIR   impute/ or mm/ directory (matrix_csr.npz, matrix.npy, factors.npz, or mm)
#   OUTPUT_BW   path to output .bw
#
# Optional overrides (export before sbatch or use #SBATCH --export=ALL,VAR=...):
#   ROOT_DIR       XuLab root (auto-detected from submit dir if unset)
#   CONDA_ENV      conda env with numpy/scipy/pandas (default: cistopic)
#   CHROM_SIZES    UCSC chrom.sizes file (default: ${ROOT_DIR}/downstream/cache/hg38.chrom.sizes)
#                  If missing, it is downloaded once from UCSC (needs network).
#   AGGREGATE      sum | mean | max (default: sum; max = factored/cisTopic only)
#   CENTER         none | median | p25 (default: none) — subtract then clip ≥0
#   NORMALIZE      none | max | p99 (default: none)
#   TRANSFORM      none | log1p (default: none)
#   CHUNK_COLS     --chunk-cols for dense/mm (default: 500)
#   CHUNK_ROWS     --chunk-rows for CSR (default: 10000)
#   PIP_EXTRAS     if set to 1, run pip install for pyBigWig when import fails (default: 1)
#
# Examples:
#   sbatch downstream/slurm/imputed_to_bigwig.sbatch
#   INPUT_DIR=/path/to/impute OUTPUT_BW=/path/to/out.bw sbatch downstream/slurm/imputed_to_bigwig.sbatch
#   sbatch --export=ALL,INPUT_DIR=${HOME}/XuLab/MAGIC/work/ctcf/impute,OUTPUT_BW=${HOME}/tracks/magic.bw \\
#          downstream/slurm/imputed_to_bigwig.sbatch
#   sbatch --export=ALL,INPUT_DIR=${HOME}/XuLab/scBasset/work/ctcf/impute, \\
#          OUTPUT_BW=${HOME}/XuLab/scBasset/work/ctcf/bigwig/scbasset_ctcf_mean.bw, \\
#          AGGREGATE=mean,NORMALIZE=p99 downstream/slurm/imputed_to_bigwig.sbatch
#   sbatch --export=ALL,INPUT_DIR=${HOME}/XuLab/scBasset_PUscOpen/work/ctcf/impute, \\
#          OUTPUT_BW=${HOME}/XuLab/downstream/tracks/igv_ctcf/ctcf_scbasset_puscopen_mean.bw, \\
#          AGGREGATE=mean,NORMALIZE=p99,CHUNK_ROWS=5000 downstream/slurm/imputed_to_bigwig.sbatch
#   sbatch downstream/slurm/igv_ctcf_scbasset_tracks.sbatch   # all three scBasset-family .bw

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
    ROOT_DIR="${SUBMIT_DIR}"
  elif [[ -d "${SUBMIT_DIR}/../downstream" ]]; then
    ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-cistopic}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/downstream/cache}"
CHROM_SIZES="${CHROM_SIZES:-${CACHE_DIR}/hg38.chrom.sizes}"
AGGREGATE="${AGGREGATE:-sum}"
CENTER="${CENTER:-none}"
NORMALIZE="${NORMALIZE:-none}"
TRANSFORM="${TRANSFORM:-none}"
CHUNK_COLS="${CHUNK_COLS:-500}"
CHUNK_ROWS="${CHUNK_ROWS:-10000}"
PIP_EXTRAS="${PIP_EXTRAS:-1}"

# Defaults: PUscOpen CTCF impute (override INPUT_DIR / OUTPUT_BW)
INPUT_DIR="${INPUT_DIR:-${ROOT_DIR}/PUscOpen/work/ctcf/impute}"
OUTPUT_BW="${OUTPUT_BW:-${ROOT_DIR}/downstream/tracks/PUscOpen_ctcf_${AGGREGATE}.bw}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi

echo "[$(date)] ROOT_DIR=${ROOT_DIR}"
echo "[$(date)] CONDA_ENV=${CONDA_ENV}"
echo "[$(date)] INPUT_DIR=${INPUT_DIR}"
echo "[$(date)] OUTPUT_BW=${OUTPUT_BW}"
echo "[$(date)] AGGREGATE=${AGGREGATE} CENTER=${CENTER} NORMALIZE=${NORMALIZE} TRANSFORM=${TRANSFORM}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "ERROR: INPUT_DIR is not a directory: ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_BW}")"
mkdir -p "${CACHE_DIR}"

if [[ ! -f "${CHROM_SIZES}" ]]; then
  echo "[$(date)] CHROM_SIZES not found; downloading hg38.chrom.sizes to ${CHROM_SIZES}"
  wget -q -O "${CHROM_SIZES}" \
    'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes' \
    || { echo "ERROR: wget failed; place hg38.chrom.sizes at CHROM_SIZES manually." >&2; exit 1; }
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "[$(date)] Python: $(command -v python)"
python - <<'PY'
import sys
for m in ("numpy", "scipy", "pandas"):
    __import__(m)
print("OK:", "numpy/scipy/pandas")
PY

if ! python -c "import pyBigWig" 2>/dev/null; then
  if [[ "${PIP_EXTRAS}" == "1" ]]; then
    echo "[$(date)] Installing pyBigWig into conda env ${CONDA_ENV} ..."
    python -m pip install --upgrade-strategy only-if-needed pyBigWig
  else
    echo "ERROR: pyBigWig not importable; set PIP_EXTRAS=1 or: pip install pyBigWig" >&2
    exit 1
  fi
fi
python -c "import pyBigWig; print('pyBigWig OK')"

echo "[$(date)] Running imputed_matrix_to_bigwig.py"
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
echo "Load in IGV: genome hg38, then File -> Load from File -> ${OUTPUT_BW}"
