#!/bin/bash
#SBATCH --job-name=av_macs_only
#SBATCH --output=av_macs_only-%A_%a.out
#SBATCH --error=av_macs_only-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#
# Resume OCR validation from existing pooled BAMs: index → MACS2 → blacklist →
# copy narrowPeak into data/. Skips download/align/pool.
#
# Usage (2 datasets):
#   sbatch --array=0-1 atac_validation/slurm/01b_macs_from_pooled.sbatch

set -euo pipefail
source "${XULAB:-/work/users/d/y/dyy12/XuLab}/atac_validation/scripts/lib.sh"

DATASETS=(gse283384_wt gse152177_wt)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#DATASETS[@]} )); then
  echo "array index ${IDX} out of range (n=${#DATASETS[@]})" >&2
  exit 1
fi
DATASET="${DATASETS[$IDX]}"

BAMDIR="${WORK}/${DATASET}/bam"
PKDIR="${WORK}/${DATASET}/peaks"
mkdir -p "${PKDIR}"

POOLED="${BAMDIR}/${DATASET}.pooled.bam"
if [[ ! -s "${POOLED}" ]]; then
  echo "missing pooled BAM: ${POOLED}" >&2
  echo "Re-run 01_align_call_peaks.sbatch for this dataset first." >&2
  exit 1
fi

if [[ ! -s "${POOLED}.bai" ]]; then
  echo "[$(date)] ${DATASET}: indexing pooled BAM"
  samtools index -@ "${THREADS}" "${POOLED}"
fi

echo "[$(date)] ${DATASET}: MACS2 callpeak (BAMPE, q=${MACS_Q}) on ${POOLED}"
macs2 callpeak -t "${POOLED}" -f BAMPE -g hs -n "${DATASET}" \
      --outdir "${PKDIR}" -q "${MACS_Q}" --keep-dup all

NP="${PKDIR}/${DATASET}_peaks.narrowPeak"
if [[ ! -s "${NP}" ]]; then
  echo "MACS2 did not write ${NP}" >&2
  exit 1
fi

if [[ -s "${BLACKLIST_BED}" ]]; then
  echo "[$(date)] ${DATASET}: removing blacklist peaks (${BLACKLIST_BED})"
  bedtools intersect -v -a "${NP}" -b "${BLACKLIST_BED}" > "${NP}.tmp" && mv "${NP}.tmp" "${NP}"
else
  echo "[$(date)] WARNING blacklist not found (${BLACKLIST_BED}); skipping" >&2
fi

pigz -f -k "${NP}"
mkdir -p "${XULAB}/data"
cp -f "${NP}.gz" "${XULAB}/data/HEK293T_ATAC_${DATASET}_MACS2.narrowPeak.gz"

echo "[$(date)] done ${DATASET}: $(wc -l < "${NP}") peaks -> data/HEK293T_ATAC_${DATASET}_MACS2.narrowPeak.gz"
