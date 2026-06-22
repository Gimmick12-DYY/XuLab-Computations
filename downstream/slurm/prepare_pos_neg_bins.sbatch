#!/bin/bash
#SBATCH --job-name=ds_prepare_bins
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Build positive/negative CTCF bin indices from SCREEN cCRE annotations.
#
# Required env vars:
#   MM_DIR        directory with regions.tsv.gz (e.g. <work>/mm)
#   OUT_DIR       output dir for pos_neg_bins.tsv + meta.json
# Optional env vars:
#   CACHE_DIR     where to download cCRE/blacklist files (default: $OUT_DIR/cache)
#   N_TOP         top-N CTCF cCREs by score (default: 10000)
#   SEED          RNG seed for negative sampling (random mode only; default: 2026)
#   NEGATIVE_MODE 'desert' (rank candidate bins by bulk CTCF coverage, strictest =
#                 absolute deserts) or 'random' (legacy). Default: desert if
#                 BULK_CTCF_BAM is set, else random.
#   BULK_CTCF_BAM coordinate-sorted + indexed bulk CTCF ChIP-seq BAM. Required
#                 for desert mode; per-bin coverage via `samtools bedcov`.
#   SAMTOOLS      samtools executable (default: samtools; must provide `bedcov`)
#   CONDA_ENV     conda env with R + Bioconductor (default: cistopic; FITS env works too)

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
  ROOT_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/prepare_pos_neg_bins.R" ]]; then
  ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

if [[ -z "${MM_DIR:-}" || -z "${OUT_DIR:-}" ]]; then
  echo "Required: MM_DIR=<mm/>, OUT_DIR=<output dir>" >&2
  exit 1
fi
CACHE_DIR="${CACHE_DIR:-${OUT_DIR}/cache}"
N_TOP="${N_TOP:-10000}"
SEED="${SEED:-2026}"
SAMTOOLS="${SAMTOOLS:-samtools}"
BULK_CTCF_BAM="${BULK_CTCF_BAM:-}"
# Default to desert mode iff a bulk BAM is provided; else legacy random.
if [[ -n "${BULK_CTCF_BAM}" ]]; then
  NEGATIVE_MODE="${NEGATIVE_MODE:-desert}"
else
  NEGATIVE_MODE="${NEGATIVE_MODE:-random}"
fi
CONDA_ENV="${CONDA_ENV:-cistopic}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [[ "${NEGATIVE_MODE}" == "desert" ]]; then
  if [[ -z "${BULK_CTCF_BAM}" ]]; then
    echo "ERROR: NEGATIVE_MODE=desert requires BULK_CTCF_BAM=<bulk CTCF ChIP-seq BAM>" >&2
    exit 1
  fi
  if ! command -v "${SAMTOOLS}" >/dev/null 2>&1; then
    # The cistopic env may not ship samtools. Try the cluster module before
    # giving up, so desert mode works without rebuilding the conda env.
    echo "[$(date)] samtools not on PATH; trying 'module load samtools'"
    module load samtools 2>/dev/null || true
  fi
  if ! command -v "${SAMTOOLS}" >/dev/null 2>&1; then
    echo "ERROR: samtools ('${SAMTOOLS}') not on PATH in env ${CONDA_ENV} and no samtools module." >&2
    echo "  Fix (pick one):" >&2
    echo "   - conda install -n ${CONDA_ENV} -c bioconda samtools" >&2
    echo "   - module load samtools   (then resubmit)" >&2
    echo "   - SAMTOOLS=/path/to/samtools sbatch ..." >&2
    exit 1
  fi
  echo "[$(date)] using samtools: $(command -v "${SAMTOOLS}")"
  echo "[$(date)] Negative mode: desert (bulk BAM=${BULK_CTCF_BAM})"
else
  echo "[$(date)] Negative mode: random (seed=${SEED})"
fi

RARGS=(
  --mm-dir   "${MM_DIR}"
  --out-dir  "${OUT_DIR}"
  --cache-dir "${CACHE_DIR}"
  --n-top    "${N_TOP}"
  --seed     "${SEED}"
  --negative-mode "${NEGATIVE_MODE}"
  --samtools "${SAMTOOLS}"
)
[[ -n "${BULK_CTCF_BAM}" ]] && RARGS+=(--bulk-ctcf-bam "${BULK_CTCF_BAM}")

Rscript "${DOWN_DIR}/prepare_pos_neg_bins.R" "${RARGS[@]}"
