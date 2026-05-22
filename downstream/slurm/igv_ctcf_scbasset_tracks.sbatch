#!/bin/bash
#SBATCH --job-name=igv_scbasset_bw
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=12:00:00

# BigWig tracks for the scBasset family (IGV / UCSC):
#   scbasset                    sequence-only impute
#   scbasset_puscopen           scBasset → PUscOpen cascade (very dense CSR; needs high mem)
#   scbasset_cistopic           scBasset → cisTopic (LDA-denoise + union)
#   scbasset_cistopic_puscopen  union of the two above (additive)
#
# Uses downstream/imputed_matrix_to_bigwig.py (same as imputed_to_bigwig.sbatch).
#
# Optional env:
#   ROOT_DIR, CONDA_ENV (default: cistopic)
#   SCBASSET_DIR, SCBASSET_PUSCOPEN_DIR, SCBASSET_CISTOPIC_DIR,
#   SCBASSET_CISTOPIC_PUSCOPEN_DIR                          (from _pipeline_paths.sh)
#   TRACKS_DIR      default: ${ROOT_DIR}/downstream/tracks/igv_ctcf
#   CHROM_SIZES, CHUNK_ROWS (default 10000; scbasset_pu may need 5000 if OOM)
#   AGGREGATE       default: mean  (fraction of cells per bin for binary CSR)
#   NORMALIZE       default: none  (raw per-bin units; set p99 or max to
#                                   force a per-track relative scaling)
#   TRANSFORM       default: none
#   CENTER          default: none
#   PIP_EXTRAS      default: 1
#
# Example:
#   cd /work/users/d/y/dyy12/XuLab && sbatch downstream/slurm/igv_ctcf_scbasset_tracks.sbatch

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

DOWN_DIR="${ROOT_DIR}/downstream"
# shellcheck source=_pipeline_paths.sh
source "${DOWN_DIR}/slurm/_pipeline_paths.sh"
pipeline_paths_native

CONDA_ENV="${CONDA_ENV:-cistopic}"
CACHE_DIR="${CACHE_DIR:-${DOWN_DIR}/cache}"
CHROM_SIZES="${CHROM_SIZES:-${CACHE_DIR}/hg38.chrom.sizes}"
TRACKS_DIR="${TRACKS_DIR:-${DOWN_DIR}/tracks/igv_ctcf}"
AGGREGATE="${AGGREGATE:-mean}"
NORMALIZE="${NORMALIZE:-none}"
TRANSFORM="${TRANSFORM:-none}"
CENTER="${CENTER:-none}"
CHUNK_ROWS="${CHUNK_ROWS:-10000}"
PIP_EXTRAS="${PIP_EXTRAS:-1}"

SUF=""
[[ "${CENTER}" != "none" ]] && SUF+="_c${CENTER}"
[[ "${NORMALIZE}" != "none" ]] && SUF+="_norm${NORMALIZE}"
[[ "${TRANSFORM}" != "none" ]] && SUF+="_${TRANSFORM}"

OUT_SCB="${TRACKS_DIR}/ctcf_scbasset_${AGGREGATE}${SUF}.bw"
OUT_SCB_PU="${TRACKS_DIR}/ctcf_scbasset_puscopen_${AGGREGATE}${SUF}.bw"
OUT_SCB_CIS="${TRACKS_DIR}/ctcf_scbasset_cistopic_${AGGREGATE}${SUF}.bw"
OUT_SCB_CIS_PU="${TRACKS_DIR}/ctcf_scbasset_cistopic_puscopen_${AGGREGATE}${SUF}.bw"

mkdir -p "${TRACKS_DIR}" "${CACHE_DIR}"

if [[ ! -f "${CHROM_SIZES}" ]]; then
  echo "[$(date)] CHROM_SIZES not found; downloading hg38.chrom.sizes to ${CHROM_SIZES}"
  wget -q -O "${CHROM_SIZES}" \
    'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes' \
    || { echo "ERROR: wget failed; place hg38.chrom.sizes at CHROM_SIZES manually." >&2; exit 1; }
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python - <<'PY'
for m in ("numpy", "scipy", "pandas"):
    __import__(m)
print("OK: numpy/scipy/pandas")
PY

if ! python -c "import pyBigWig" 2>/dev/null; then
  if [[ "${PIP_EXTRAS}" == "1" ]]; then
    echo "[$(date)] Installing pyBigWig ..."
    python -m pip install --upgrade-strategy only-if-needed pyBigWig
  else
    echo "ERROR: pyBigWig missing; set PIP_EXTRAS=1" >&2
    exit 1
  fi
fi
python -c "import pyBigWig; print('pyBigWig OK')"

run_bw() {
  local indir="$1"
  local outbw="$2"
  local label="$3"
  echo "[$(date)] --- ${label} (aggregate=${AGGREGATE}, center=${CENTER}, normalize=${NORMALIZE}, transform=${TRANSFORM}) ---"
  if [[ ! -d "${indir}" ]]; then
    echo "ERROR: missing directory (${label}): ${indir}" >&2
    exit 1
  fi
  local -a extra=()
  extra+=(--aggregate "${AGGREGATE}" --chunk-rows "${CHUNK_ROWS}")
  [[ "${CENTER}" != "none" ]] && extra+=(--center "${CENTER}")
  [[ "${NORMALIZE}" != "none" ]] && extra+=(--normalize "${NORMALIZE}")
  [[ "${TRANSFORM}" != "none" ]] && extra+=(--transform "${TRANSFORM}")
  python "${DOWN_DIR}/imputed_matrix_to_bigwig.py" \
    --input "${indir}" \
    --output "${outbw}" \
    --chrom-sizes "${CHROM_SIZES}" \
    "${extra[@]}"
}

run_bw "${SCBASSET_DIR}" "${OUT_SCB}" "scbasset"
run_bw "${SCBASSET_PUSCOPEN_DIR}" "${OUT_SCB_PU}" "scbasset_puscopen"
run_bw "${SCBASSET_CISTOPIC_DIR}" "${OUT_SCB_CIS}" "scbasset_cistopic"
run_bw "${SCBASSET_CISTOPIC_PUSCOPEN_DIR}" "${OUT_SCB_CIS_PU}" "scbasset_cistopic_puscopen"

echo "[$(date)] Done. Load in IGV (hg38):"
echo "  ${OUT_SCB}"
echo "  ${OUT_SCB_PU}"
echo "  ${OUT_SCB_CIS}"
echo "  ${OUT_SCB_CIS_PU}"
