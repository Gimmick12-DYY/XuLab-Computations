#!/bin/bash
# -----------------------------------------------------------------------------
# run_tf_benchmark_modes.sh
#
# Submit the three TF benchmark positive definitions for one TF. True negatives
# are identical across modes:
#   1. exclude TF-bound cCRE bins (peak ∩ cCRE registry)
#   2. samtools bedcov on the bulk ChIP BAM; keep zero-read bins only
#   3. randomly sample negatives to match the positive count
#
# Positive modes (one job each):
#   fixed_n     top N_POS (default 5000) TF-bound cCRE bins by peak signal
#   all_bound   every TF-bound cCRE bin
#   peak_cutoff bins overlapping bulk peaks with q < PEAK_Q_MAX and p < PEAK_P_MAX
#
# Usage:
#   bash downstream/slurm/run_tf_benchmark_modes.sh MAZ
#   MM_DIR=/work/.../unified/work/maz/mm \
#     bash downstream/slurm/run_tf_benchmark_modes.sh MAZ
#
# Outputs:
#   downstream/bins_<tf>_<mode>/pos_neg_bins.tsv
# -----------------------------------------------------------------------------
set -euo pipefail

TF="${1:-}"
if [[ -z "${TF}" ]]; then
  echo "Usage: bash downstream/slurm/run_tf_benchmark_modes.sh <TF>" >&2
  exit 1
fi
TF_LOWER="$(echo "${TF}" | tr '[:upper:]' '[:lower:]')"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
SBATCH="${SBATCH:-${SCRIPT_DIR}/prepare_pos_neg_bins.sbatch}"

# Resolve HEK293 vs K562; CTCF bulk BAM may live as ENCFF139DCW.bam (legacy name).
LINE="HEK293"
[[ "${TF}" == "NFYA" ]] && LINE="K562"
if [[ "${TF}" == "CTCF" ]]; then
  PEAKS_BED="${TF_PEAKS_BED:-${DATA_DIR}/CTCF_${LINE}_peaks.bed}"
  if [[ -f "${DATA_DIR}/ENCFF139DCW.bam" ]]; then
    BULK_BAM="${BULK_BAM:-${DATA_DIR}/ENCFF139DCW.bam}"
  else
    BULK_BAM="${BULK_BAM:-${DATA_DIR}/CTCF_${LINE}.bam}"
  fi
else
  PEAKS_BED="${TF_PEAKS_BED:-${DATA_DIR}/${TF}_${LINE}_peaks.bed}"
  BULK_BAM="${BULK_BAM:-${DATA_DIR}/${TF}_${LINE}.bam}"
fi
MM_DIR="${MM_DIR:-${ROOT_DIR}/unified/work/${TF_LOWER}/mm}"
N_POS="${N_POS:-5000}"
PEAK_Q_MAX="${PEAK_Q_MAX:-0.01}"
PEAK_P_MAX="${PEAK_P_MAX:-1e-5}"
SEED="${SEED:-2026}"

for f in "${PEAKS_BED}" "${BULK_BAM}" "${MM_DIR}/regions.tsv.gz"; do
  if [[ ! -e "${f}" ]]; then
    echo "ERROR: required file missing: ${f}" >&2
    exit 1
  fi
done

submit_mode() {
  local mode="$1"
  local out="${ROOT_DIR}/downstream/bins_${TF_LOWER}_${mode}"
  echo "[submit] ${TF} POS_MODE=${mode} -> ${out}"
  sbatch --export=ALL,\
MM_DIR="${MM_DIR}",\
OUT_DIR="${out}",\
TF_PEAKS_BED="${PEAKS_BED}",\
BULK_BAM="${BULK_BAM}",\
POS_MODE="${mode}",\
N_POS="${N_POS}",\
PEAK_Q_MAX="${PEAK_Q_MAX}",\
PEAK_P_MAX="${PEAK_P_MAX}",\
NEG_MAX_COVERAGE=0,\
NEGATIVE_MODE=random,\
SEED="${SEED}" \
    "${SBATCH}"
}

submit_mode fixed_n
submit_mode all_bound
submit_mode peak_cutoff

echo "[submit] Done. Three jobs queued for ${TF}."
