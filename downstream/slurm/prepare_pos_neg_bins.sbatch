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
#   NEGATIVE_MODE 'all_candidates' (use the ENTIRE no-cCRE candidate pool ->
#                 genome-wide FP rate; recommended), 'desert' (rank by bulk CTCF
#                 coverage, take the strictest -- WARNING: the extreme tail is
#                 dead chromatin where every method scores 0, uninformative), or
#                 'random' (legacy: matched to #positives). Default: desert if
#                 BULK_CTCF_BAM is set, else random.
#   MAX_NEG       cap on #negatives in all_candidates mode (0 = no cap; default 0)
#   BULK_CTCF_BAM coordinate-sorted + indexed bulk CTCF ChIP-seq BAM. Required
#                 for desert mode only; per-bin coverage via `samtools bedcov`.
#   SAMTOOLS      samtools executable (default: samtools; desert mode only)
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
MAX_NEG="${MAX_NEG:-0}"
# Balanced-set targets (bin counts). N_POS>0 selects top-N positive bins by CTCF
# z-score; N_NEG>0 random-samples N_NEG negatives from the candidate pool. Set
# both equal for a balanced benchmark, e.g. N_POS=100000 N_NEG=100000.
N_POS="${N_POS:-0}"
N_NEG="${N_NEG:-0}"
SAMTOOLS="${SAMTOOLS:-samtools}"
# Bulk BAM for desert ranking. BULK_BAM is the TF-agnostic name; BULK_CTCF_BAM
# kept for back-compat. Either works.
BULK_BAM="${BULK_BAM:-${BULK_CTCF_BAM:-}}"
# GENERIC (non-CTCF) mode: define positives from a TF ChIP-seq narrowPeak instead
# of CTCF SCREEN cCREs. Set TF_PEAKS_BED to switch on. PEAK_SIGNAL_COL ranks
# positive bins (narrowPeak col 7 = signalValue); EXCLUDE_FLANK_BP drops bins
# within N bp of a peak from the negative pool.
TF_PEAKS_BED="${TF_PEAKS_BED:-}"
PEAK_SIGNAL_COL="${PEAK_SIGNAL_COL:-7}"
EXCLUDE_FLANK_BP="${EXCLUDE_FLANK_BP:-0}"
# Default negative mode: desert iff a bulk BAM is given, else random (random
# matches #negatives to #positives -> a balanced set with no BAM needed).
if [[ -n "${BULK_BAM}" ]]; then
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

if [[ -n "${TF_PEAKS_BED}" ]]; then
  if [[ ! -f "${TF_PEAKS_BED}" ]]; then
    echo "ERROR: TF_PEAKS_BED not found: ${TF_PEAKS_BED}" >&2; exit 1
  fi
  echo "[$(date)] GENERIC mode: positives from TF peaks ${TF_PEAKS_BED} (signal col ${PEAK_SIGNAL_COL})"
fi

if [[ "${NEGATIVE_MODE}" == "desert" ]]; then
  if [[ -z "${BULK_BAM}" ]]; then
    echo "ERROR: NEGATIVE_MODE=desert requires BULK_BAM=<bulk ChIP-seq BAM>" >&2
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
  echo "[$(date)] Negative mode: desert (bulk BAM=${BULK_BAM})"
elif [[ "${NEGATIVE_MODE}" == "all_candidates" || "${NEGATIVE_MODE}" == "all" ]]; then
  echo "[$(date)] Negative mode: all_candidates (max_neg=${MAX_NEG})"
else
  echo "[$(date)] Negative mode: random (seed=${SEED})"
fi

RARGS=(
  --mm-dir   "${MM_DIR}"
  --out-dir  "${OUT_DIR}"
  --cache-dir "${CACHE_DIR}"
  --n-top    "${N_TOP}"
  --seed     "${SEED}"
  --max-neg  "${MAX_NEG}"
  --n-pos    "${N_POS}"
  --n-neg    "${N_NEG}"
  --negative-mode "${NEGATIVE_MODE}"
  --samtools "${SAMTOOLS}"
)
[[ -n "${BULK_BAM}" ]]      && RARGS+=(--bulk-bam "${BULK_BAM}")
[[ -n "${TF_PEAKS_BED}" ]] && RARGS+=(--tf-peaks-bed "${TF_PEAKS_BED}" \
                                      --peak-signal-col "${PEAK_SIGNAL_COL}" \
                                      --exclude-flank-bp "${EXCLUDE_FLANK_BP}")

Rscript "${DOWN_DIR}/prepare_pos_neg_bins.R" "${RARGS[@]}"
