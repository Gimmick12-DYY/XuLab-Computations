#!/bin/bash
#SBATCH --job-name=igv_ctcf_bw
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# One job: BigWig tracks for IGV — **raw** (from data/), **cisTopic** imputed,
# **PUscOpen** imputed. Raw counts are exported from the shared RDS via
# cisTopic/scripts/01_export_rds_to_mm.R into ${RAW_MM_DIR} when matrix.mtx.gz
# is not already present.
#
# Defaults (override with export or sbatch --export=ALL,...):
#   ROOT_DIR          XuLab root
#   CONDA_ENV         Python env with numpy/scipy/pandas + pyBigWig (default: cistopic)
#   RAW_RDS           default: ${ROOT_DIR}/data/CTCF_bin1000_mtx.rds
#   RAW_MM_DIR        default: ${ROOT_DIR}/data/mm   (RDS → mm export lands here)
#   EXPORT_RAW_MM     if 1 and matrix.mtx.gz missing under RAW_MM_DIR, run R export (default: 1)
#   CISTOPIC_IMPUTE   default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/impute
#   PUSCOPEN_IMPUTE   default: ${ROOT_DIR}/PUscOpen/work/ctcf/impute
#   TRACKS_DIR        default: ${ROOT_DIR}/downstream/tracks/igv_ctcf
#   CHROM_SIZES, AGGREGATE, CHUNK_COLS, CHUNK_ROWS, PIP_EXTRAS  (same as imputed_to_bigwig.sbatch)
#   R_MODULE          optional: e.g. R/4.4.0 — loaded if Rscript is missing after conda activate
#   TRANSFORM_RAW     none | log1p (default: none) — passed to imputed_matrix_to_bigwig.py
#   TRANSFORM_CISTOPIC none | log1p (default: none) — log1p can flatten mid-level signal;
#                        cisTopic plateau is usually better handled with CENTER_CISTOPIC.
#   TRANSFORM_PUSCOPEN none | log1p (default: none)
#   CENTER_RAW        none | median | p25 (default: none)
#   CENTER_CISTOPIC   none | median | p25 (default: median) — subtract then clip at 0
#                        (display-only; removes uniform LDA floor in IGV).
#   CENTER_PUSCOPEN   none | median | p25 (default: none)
#   AGGREGATE_RAW / AGGREGATE_CISTOPIC / AGGREGATE_PUSCOPEN — default: inherit AGGREGATE (sum).
#                        Set AGGREGATE_CISTOPIC=max for peakier factored tracks (cisTopic only).
#   NORMALIZE_RAW     none | max | p99 (default: p99) — display-only; divides scores
#                        so tracks are comparable in IGV (each file scaled to itself).
#   NORMALIZE_CISTOPIC none | max | p99 (default: p99)
#   NORMALIZE_PUSCOPEN none | max | p99 (default: p99)
#   Set any to **none** to write raw pseudobulk/count units (e.g. match compare_pos_neg).
#
# Example:
#   cd /work/users/d/y/dyy12/XuLab && sbatch downstream/slurm/igv_ctcf_cistopic_puscopen_raw.sbatch
#   NORMALIZE_CISTOPIC=none TRANSFORM_CISTOPIC=none sbatch ...   # linear cisTopic .bw

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
CHUNK_COLS="${CHUNK_COLS:-500}"
CHUNK_ROWS="${CHUNK_ROWS:-10000}"
PIP_EXTRAS="${PIP_EXTRAS:-1}"

RAW_RDS="${RAW_RDS:-${ROOT_DIR}/data/CTCF_bin1000_mtx.rds}"
RAW_MM_DIR="${RAW_MM_DIR:-${ROOT_DIR}/data/mm}"
EXPORT_RAW_MM="${EXPORT_RAW_MM:-1}"
CISTOPIC_IMPUTE="${CISTOPIC_IMPUTE:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/impute}"
PUSCOPEN_IMPUTE="${PUSCOPEN_IMPUTE:-${ROOT_DIR}/PUscOpen/work/ctcf/impute}"
TRACKS_DIR="${TRACKS_DIR:-${ROOT_DIR}/downstream/tracks/igv_ctcf}"

TRANSFORM_RAW="${TRANSFORM_RAW:-none}"
TRANSFORM_CISTOPIC="${TRANSFORM_CISTOPIC:-none}"
TRANSFORM_PUSCOPEN="${TRANSFORM_PUSCOPEN:-none}"

CENTER_RAW="${CENTER_RAW:-none}"
CENTER_CISTOPIC="${CENTER_CISTOPIC:-median}"
CENTER_PUSCOPEN="${CENTER_PUSCOPEN:-none}"

NORMALIZE_RAW="${NORMALIZE_RAW:-p99}"
NORMALIZE_CISTOPIC="${NORMALIZE_CISTOPIC:-p99}"
NORMALIZE_PUSCOPEN="${NORMALIZE_PUSCOPEN:-p99}"

AGGREGATE_RAW="${AGGREGATE_RAW:-${AGGREGATE}}"
AGGREGATE_CISTOPIC="${AGGREGATE_CISTOPIC:-${AGGREGATE}}"
AGGREGATE_PUSCOPEN="${AGGREGATE_PUSCOPEN:-${AGGREGATE}}"

SUF_RAW=""
[[ "${CENTER_RAW}" != "none" ]] && SUF_RAW+="_c${CENTER_RAW}"
[[ "${NORMALIZE_RAW}" != "none" ]] && SUF_RAW+="_norm${NORMALIZE_RAW}"
[[ "${TRANSFORM_RAW}" != "none" ]] && SUF_RAW+="_${TRANSFORM_RAW}"
SUF_CIS=""
[[ "${CENTER_CISTOPIC}" != "none" ]] && SUF_CIS+="_c${CENTER_CISTOPIC}"
[[ "${NORMALIZE_CISTOPIC}" != "none" ]] && SUF_CIS+="_norm${NORMALIZE_CISTOPIC}"
[[ "${TRANSFORM_CISTOPIC}" != "none" ]] && SUF_CIS+="_${TRANSFORM_CISTOPIC}"
SUF_PU=""
[[ "${CENTER_PUSCOPEN}" != "none" ]] && SUF_PU+="_c${CENTER_PUSCOPEN}"
[[ "${NORMALIZE_PUSCOPEN}" != "none" ]] && SUF_PU+="_norm${NORMALIZE_PUSCOPEN}"
[[ "${TRANSFORM_PUSCOPEN}" != "none" ]] && SUF_PU+="_${TRANSFORM_PUSCOPEN}"

OUT_RAW="${TRACKS_DIR}/ctcf_raw_${AGGREGATE_RAW}${SUF_RAW}.bw"
OUT_CISTOPIC="${TRACKS_DIR}/ctcf_cistopic_${AGGREGATE_CISTOPIC}${SUF_CIS}.bw"
OUT_PUSCOPEN="${TRACKS_DIR}/ctcf_puscopen_${AGGREGATE_PUSCOPEN}${SUF_PU}.bw"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi

echo "[$(date)] ROOT_DIR=${ROOT_DIR}"
echo "[$(date)] RAW_RDS=${RAW_RDS}"
echo "[$(date)] RAW_MM_DIR=${RAW_MM_DIR}"
echo "[$(date)] CISTOPIC_IMPUTE=${CISTOPIC_IMPUTE}"
echo "[$(date)] PUSCOPEN_IMPUTE=${PUSCOPEN_IMPUTE}"
echo "[$(date)] TRACKS_DIR=${TRACKS_DIR}"
echo "[$(date)] TRANSFORM raw=${TRANSFORM_RAW} cistopic=${TRANSFORM_CISTOPIC} puscopen=${TRANSFORM_PUSCOPEN}"
echo "[$(date)] CENTER raw=${CENTER_RAW} cistopic=${CENTER_CISTOPIC} puscopen=${CENTER_PUSCOPEN}"
echo "[$(date)] NORMALIZE raw=${NORMALIZE_RAW} cistopic=${NORMALIZE_CISTOPIC} puscopen=${NORMALIZE_PUSCOPEN}"
echo "[$(date)] AGGREGATE raw=${AGGREGATE_RAW} cistopic=${AGGREGATE_CISTOPIC} puscopen=${AGGREGATE_PUSCOPEN}"

mkdir -p "${TRACKS_DIR}" "${CACHE_DIR}"

if [[ ! -f "${CHROM_SIZES}" ]]; then
  echo "[$(date)] CHROM_SIZES not found; downloading hg38.chrom.sizes to ${CHROM_SIZES}"
  wget -q -O "${CHROM_SIZES}" \
    'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes' \
    || { echo "ERROR: wget failed; place hg38.chrom.sizes at CHROM_SIZES manually." >&2; exit 1; }
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [[ -n "${R_MODULE:-}" ]]; then
  echo "[$(date)] module load ${R_MODULE}"
  module load "${R_MODULE}"
fi

echo "[$(date)] Python: $(command -v python)"
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
  local xf="${4:-none}"
  local norm="${5:-none}"
  local agg="${6:-sum}"
  local cen="${7:-none}"
  echo "[$(date)] --- ${label} (aggregate=${agg}, center=${cen}, normalize=${norm}, transform=${xf}) ---"
  if [[ ! -d "${indir}" ]]; then
    echo "ERROR: missing directory (${label}): ${indir}" >&2
    exit 1
  fi
  local -a extra=()
  extra+=(--aggregate "${agg}")
  if [[ "${cen}" != "none" ]]; then
    extra+=(--center "${cen}")
  fi
  if [[ "${norm}" != "none" ]]; then
    extra+=(--normalize "${norm}")
  fi
  if [[ "${xf}" != "none" ]]; then
    extra+=(--transform "${xf}")
  fi
  python "${ROOT_DIR}/downstream/imputed_matrix_to_bigwig.py" \
    --input "${indir}" \
    --output "${outbw}" \
    --chrom-sizes "${CHROM_SIZES}" \
    --chunk-cols "${CHUNK_COLS}" \
    --chunk-rows "${CHUNK_ROWS}" \
    "${extra[@]}"
}

# ---- Raw: ensure mm under data/mm from data/*.rds --------------------------------
if [[ ! -f "${RAW_MM_DIR}/matrix.mtx.gz" || ! -f "${RAW_MM_DIR}/regions.tsv.gz" ]]; then
  if [[ "${EXPORT_RAW_MM}" != "1" ]]; then
    echo "ERROR: raw mm missing under ${RAW_MM_DIR} and EXPORT_RAW_MM!=1" >&2
    exit 1
  fi
  if [[ ! -f "${RAW_RDS}" ]]; then
    echo "ERROR: RAW_RDS not found: ${RAW_RDS}" >&2
    exit 1
  fi
  if ! command -v Rscript >/dev/null 2>&1; then
    echo "ERROR: Rscript not on PATH; need R to export RDS → mm (or pre-populate ${RAW_MM_DIR})" >&2
    exit 1
  fi
  mkdir -p "${RAW_MM_DIR}"
  echo "[$(date)] Exporting ${RAW_RDS} → ${RAW_MM_DIR} (01_export_rds_to_mm.R)"
  Rscript "${ROOT_DIR}/cisTopic/scripts/01_export_rds_to_mm.R" \
    --input "${RAW_RDS}" \
    --outdir "${RAW_MM_DIR}"
else
  echo "[$(date)] Using existing raw mm under ${RAW_MM_DIR}"
fi

run_bw "${RAW_MM_DIR}" "${OUT_RAW}" "raw (data/mm)" "${TRANSFORM_RAW}" "${NORMALIZE_RAW}" "${AGGREGATE_RAW}" "${CENTER_RAW}"
run_bw "${CISTOPIC_IMPUTE}" "${OUT_CISTOPIC}" "cisTopic imputed" "${TRANSFORM_CISTOPIC}" "${NORMALIZE_CISTOPIC}" "${AGGREGATE_CISTOPIC}" "${CENTER_CISTOPIC}"
run_bw "${PUSCOPEN_IMPUTE}" "${OUT_PUSCOPEN}" "PUscOpen imputed" "${TRANSFORM_PUSCOPEN}" "${NORMALIZE_PUSCOPEN}" "${AGGREGATE_PUSCOPEN}" "${CENTER_PUSCOPEN}"

echo "[$(date)] Done."
echo "IGV (hg38): load these three tracks:"
echo "  ${OUT_RAW}"
echo "  ${OUT_CISTOPIC}"
echo "  ${OUT_PUSCOPEN}"
