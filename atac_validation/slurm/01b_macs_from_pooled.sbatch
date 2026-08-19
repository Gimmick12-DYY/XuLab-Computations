#!/bin/bash
#SBATCH --job-name=av_macs_only
#SBATCH --output=av_macs_only-%A_%a.out
#SBATCH --error=av_macs_only-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#
# Resume OCR validation from existing per-run filtered BAMs:
#   rebuild pool from keep=yes rows -> coordinate sort -> index -> MACS2 ->
#   blacklist -> copy narrowPeak into data/.
#
# Always re-pools from samples.tsv when FORCE_REPOOL=1 (or no usable pool);
# otherwise reuses an existing coordinate-sorted ${dataset}.pooled.bam+.bai.
# MACS2 in the atac env must be a local source build (bioconda binary fails on
# RHEL9 with undefined symbol __log_finite).
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
mkdir -p "${BAMDIR}" "${PKDIR}"
export TMPDIR="${TMPDIR:-${BAMDIR}}"

FILT_BAMS=()
while IFS=$'\t' read -r _GSM SRR; do
  [[ -z "${SRR}" || "${SRR}" == "NA" ]] && continue
  FILT="${BAMDIR}/${SRR}.filt.bam"
  if [[ ! -s "${FILT}" ]]; then
    echo "missing filtered BAM: ${FILT}" >&2
    echo "Re-run 01_align_call_peaks.sbatch for this dataset first." >&2
    exit 1
  fi
  FILT_BAMS+=("${FILT}")
done < <(dataset_runs "${DATASET}")

if (( ${#FILT_BAMS[@]} == 0 )); then
  echo "no keep=yes runs for ${DATASET}" >&2
  exit 1
fi

POOLED="${BAMDIR}/${DATASET}.pooled.bam"
MERGED="${BAMDIR}/${DATASET}.pooled.merged.bam"
# Reuse an already-sorted+indexed pool unless FORCE_REPOOL=1.
if [[ "${FORCE_REPOOL:-0}" != "1" && -s "${POOLED}" && -s "${POOLED}.bai" ]]; then
  echo "[$(date)] ${DATASET}: reusing existing pooled BAM (${POOLED})"
else
  echo "[$(date)] ${DATASET}: pooling ${#FILT_BAMS[@]} keep=yes run(s) -> sort"
  rm -f "${POOLED}.bai" "${MERGED}" "${MERGED}.bai"
  if (( ${#FILT_BAMS[@]} == 1 )); then
    cp -f "${FILT_BAMS[0]}" "${MERGED}"
  else
    samtools merge -f -@ "${THREADS}" "${MERGED}" "${FILT_BAMS[@]}"
  fi
  # Force a true coordinate sort so samtools index / MACS2 see contiguous chrom blocks.
  samtools sort -@ "${THREADS}" -o "${POOLED}" "${MERGED}"
  rm -f "${MERGED}"
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
