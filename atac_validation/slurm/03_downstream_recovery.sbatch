#!/bin/bash
#SBATCH --job-name=av_recovery
#SBATCH --output=av_recovery-%j.out
#SBATCH --error=av_recovery-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL
#
# Step 3 (downstream check): per-TF open-positive recovery under each mask +
# the union. Shows how much the union expands credible (open) positives vs the
# current OmniATAC mask, i.e. what the all_universe_openpos eval would gain.
#
# Prereq: 02_overlap_union.sbatch (union BED) + per-TF positive bins on disk.
# Point BINS_DIR / MODE at whatever pos_neg bins you want scored.
#
#   sbatch atac_validation/slurm/03_downstream_recovery.sbatch
#   TFS="ctcf maz zic2" sbatch atac_validation/slurm/03_downstream_recovery.sbatch

set -euo pipefail
source "$(dirname "$0")/../scripts/lib.sh"
cd "${AV}/scripts"

BINS_DIR="${BINS_DIR:-${XULAB}/downstream}"
MODE="${MODE:-all_universe}"                       # bins_<tf>_<MODE>/pos_neg_bins.tsv
BINS_FILE="${BINS_FILE:-pos_neg_bins.tsv}"
TFS="${TFS:-ctcf maz zic2 zbtb7a znf777 nfya znf282}"

GSE283384_PEAKS="${XULAB}/data/HEK293T_ATAC_gse283384_wt_MACS2.narrowPeak.gz"
GSE152177_PEAKS="${XULAB}/data/HEK293T_ATAC_gse152177_wt_MACS2.narrowPeak.gz"
UNION_BED="${XULAB}/data/HEK293T_ATAC_union3.bed.gz"

POS_ARGS=()
for tf in ${TFS}; do
  f="${BINS_DIR}/bins_${tf}_${MODE}/${BINS_FILE}"
  if [[ -s "$f" ]]; then
    POS_ARGS+=(--pos "${tf}=${f}")
  else
    echo "WARNING: no bins for ${tf} at ${f}; skipping" >&2
  fi
done
if (( ${#POS_ARGS[@]} == 0 )); then
  echo "no positive bins found under ${BINS_DIR} for mode ${MODE}" >&2
  exit 1
fi

mkdir -p "${RESULTS}"
"${PY}" downstream_open_positive_recovery.py \
  "${POS_ARGS[@]}" \
  --mask "omniatac=${OMNIATAC_PEAKS}" \
  --mask "gse283384=${GSE283384_PEAKS}" \
  --mask "gse152177=${GSE152177_PEAKS}" \
  --mask "union=${UNION_BED}" \
  --out "${RESULTS}/open_positive_recovery_${MODE}.tsv"

echo "[$(date)] done -> ${RESULTS}/open_positive_recovery_${MODE}.tsv"
