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
# Positive benchmark definition (generic TF mode). True negatives are always:
#   exclude TF-bound cCREs -> bedcov zero-read filter -> random sample to match #pos.
#   fixed_n     top N_POS TF-bound cCRE bins by peak signal (default N_POS=5000)
#   all_bound   every TF-bound cCRE bin
#   peak_cutoff bins overlapping bulk peaks with q<PEAK_Q_MAX and p<PEAK_P_MAX
POS_MODE="${POS_MODE:-all_bound}"
PEAK_Q_MAX="${PEAK_Q_MAX:-0.01}"
PEAK_P_MAX="${PEAK_P_MAX:-1e-5}"
# Generic path: exclude ALL cCREs from the negative universe? Default 0 (NO) so
# the universe matches the standard TF benchmark (exclude only TF-bound cCREs,
# then rely on the bedcov zero-read filter). Set 1 to also drop all cCREs.
EXCLUDE_ALL_CCRE="${EXCLUDE_ALL_CCRE:-0}"
# CTCF/cCRE-path negative universe + coverage filter.
#   NEG_UNIVERSE=accessible_unbound -> cCRE bins not CTCF-bound (hard negatives)
#   NEG_MAX_COVERAGE=0              -> keep only zero-bulk-read bins (needs BULK_BAM)
NEG_UNIVERSE="${NEG_UNIVERSE:-non_ctcf_bound}"
NEG_MAX_COVERAGE="${NEG_MAX_COVERAGE:-0}"
# Default negative mode is always 'random' (matches #negatives to #positives).
# A BULK_BAM no longer forces 'desert' -- the BAM is also used by the coverage
# filter (NEG_MAX_COVERAGE) with random selection. Set NEGATIVE_MODE=desert
# explicitly if you want desert ranking.
NEGATIVE_MODE="${NEGATIVE_MODE:-random}"
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
  if [[ "${POS_MODE}" == "fixed_n" && "${N_POS}" == "0" ]]; then
    N_POS=5000
  fi
  echo "[$(date)] GENERIC mode: positives from TF peaks ${TF_PEAKS_BED} (POS_MODE=${POS_MODE}, signal col ${PEAK_SIGNAL_COL})"
fi

# Does this run need samtools? (desert ranking OR the coverage filter)
COV_FILTER=0
if awk -v x="${NEG_MAX_COVERAGE}" 'BEGIN{exit !(x>=0)}'; then COV_FILTER=1; fi
NEED_SAMTOOLS=0
[[ "${NEGATIVE_MODE}" == "desert" || "${COV_FILTER}" == "1" ]] && NEED_SAMTOOLS=1

if [[ "${NEED_SAMTOOLS}" == "1" ]]; then
  if [[ -z "${BULK_BAM}" ]]; then
    echo "ERROR: desert mode and NEG_MAX_COVERAGE>=0 both require BULK_BAM=<bulk ChIP-seq BAM>" >&2
    exit 1
  fi
  if ! command -v "${SAMTOOLS}" >/dev/null 2>&1; then
    echo "[$(date)] samtools not on PATH; trying 'module load samtools'"
    module load samtools 2>/dev/null || true
  fi
  if ! command -v "${SAMTOOLS}" >/dev/null 2>&1; then
    echo "ERROR: samtools ('${SAMTOOLS}') not on PATH in env ${CONDA_ENV} and no samtools module." >&2
    echo "  Fix: conda install -n ${CONDA_ENV} -c bioconda samtools | module load samtools | SAMTOOLS=/path" >&2
    exit 1
  fi
  echo "[$(date)] using samtools: $(command -v "${SAMTOOLS}")"
fi

echo "[$(date)] Negative universe: ${NEG_UNIVERSE}$([[ "${COV_FILTER}" == "1" ]] && echo " ; coverage filter <= ${NEG_MAX_COVERAGE} reads (BAM=${BULK_BAM})")"
echo "[$(date)] Negative mode: ${NEGATIVE_MODE}"

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
  --neg-universe "${NEG_UNIVERSE}"
  --neg-max-coverage "${NEG_MAX_COVERAGE}"
  --samtools "${SAMTOOLS}"
)
[[ -n "${BULK_BAM}" ]]      && RARGS+=(--bulk-bam "${BULK_BAM}")
[[ -n "${TF_PEAKS_BED}" ]] && RARGS+=(--tf-peaks-bed "${TF_PEAKS_BED}" \
                                      --peak-signal-col "${PEAK_SIGNAL_COL}" \
                                      --exclude-flank-bp "${EXCLUDE_FLANK_BP}")
[[ -n "${TF_PEAKS_BED}" ]] && RARGS+=(--positive-mode "${POS_MODE}" \
                                      --peak-q-max "${PEAK_Q_MAX}" \
                                      --peak-p-max "${PEAK_P_MAX}")
[[ "${EXCLUDE_ALL_CCRE}" != "1" ]] && RARGS+=(--no-exclude-ccre)

Rscript "${DOWN_DIR}/prepare_pos_neg_bins.R" "${RARGS[@]}"
