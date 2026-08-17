#!/bin/bash
#SBATCH --job-name=av_align
#SBATCH --output=av_align-%A_%a.out
#SBATCH --error=av_align-%A_%a.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#
# One array task per dataset: trim -> bowtie2 -> filter/dedup each run, pool the
# runs, then MACS2 (BAMPE) -> narrowPeak. Peaks are the WT open-chromatin call
# used for overlap/union against the OmniATAC IDR mask.
#
# Prereq: 00_download.sbatch (fastq) + build_bowtie2_index.sh (index).
#
# Usage (2 datasets):
#   sbatch --array=0-1 atac_validation/slurm/01_align_call_peaks.sbatch
#
# Output:
#   work/<dataset>/bam/<srr>.filt.bam          per-run filtered BAM
#   work/<dataset>/bam/<dataset>.pooled.bam    merged
#   work/<dataset>/peaks/<dataset>_peaks.narrowPeak(.gz)
#   data/HEK293T_ATAC_<dataset>_MACS2.narrowPeak.gz   (copy, drop-in mask)

set -euo pipefail
source "$(dirname "$0")/../scripts/lib.sh"

DATASETS=(gse283384_wt gse152177_wt)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#DATASETS[@]} )); then
  echo "array index ${IDX} out of range (n=${#DATASETS[@]})" >&2
  exit 1
fi
DATASET="${DATASETS[$IDX]}"

FQDIR="${WORK}/${DATASET}/fastq"
BAMDIR="${WORK}/${DATASET}/bam"
PKDIR="${WORK}/${DATASET}/peaks"
TRIMDIR="${WORK}/${DATASET}/trim"
mkdir -p "${BAMDIR}" "${PKDIR}" "${TRIMDIR}"

if [[ ! -s "${BT2_INDEX}.1.bt2" && ! -s "${BT2_INDEX}.1.bt2l" ]]; then
  echo "bowtie2 index missing at ${BT2_INDEX}; run scripts/build_bowtie2_index.sh first" >&2
  exit 1
fi

FILT_BAMS=()
while IFS=$'\t' read -r GSM SRR; do
  [[ -z "${SRR}" ]] && continue
  R1="${FQDIR}/${SRR}_1.fastq.gz"
  R2="${FQDIR}/${SRR}_2.fastq.gz"
  FILT="${BAMDIR}/${SRR}.filt.bam"
  FILT_BAMS+=("${FILT}")
  if [[ -s "${FILT}" ]]; then
    echo "[$(date)] ${SRR}: filtered BAM exists, skipping"
    continue
  fi
  if [[ ! -s "${R1}" || ! -s "${R2}" ]]; then
    echo "missing paired FASTQ for ${SRR} in ${FQDIR}" >&2
    exit 1
  fi

  echo "[$(date)] ${DATASET}/${SRR}: fastp trim"
  fastp -i "${R1}" -I "${R2}" \
        -o "${TRIMDIR}/${SRR}_1.trim.fastq.gz" -O "${TRIMDIR}/${SRR}_2.trim.fastq.gz" \
        --detect_adapter_for_pe --thread "${THREADS}" \
        -j "${TRIMDIR}/${SRR}.fastp.json" -h "${TRIMDIR}/${SRR}.fastp.html"

  echo "[$(date)] ${SRR}: bowtie2"
  # --very-sensitive, ATAC fragment size window up to 2 kb.
  bowtie2 -x "${BT2_INDEX}" \
          -1 "${TRIMDIR}/${SRR}_1.trim.fastq.gz" -2 "${TRIMDIR}/${SRR}_2.trim.fastq.gz" \
          -X 2000 --very-sensitive -p "${THREADS}" 2> "${BAMDIR}/${SRR}.bowtie2.log" \
    | samtools view -@ "${THREADS}" -bS - > "${BAMDIR}/${SRR}.raw.bam"

  echo "[$(date)] ${SRR}: fixmate + markdup"
  samtools sort -n -@ "${THREADS}" -o "${BAMDIR}/${SRR}.nsort.bam" "${BAMDIR}/${SRR}.raw.bam"
  samtools fixmate -m -@ "${THREADS}" "${BAMDIR}/${SRR}.nsort.bam" "${BAMDIR}/${SRR}.fix.bam"
  samtools sort -@ "${THREADS}" -o "${BAMDIR}/${SRR}.csort.bam" "${BAMDIR}/${SRR}.fix.bam"
  samtools markdup -r -@ "${THREADS}" "${BAMDIR}/${SRR}.csort.bam" "${BAMDIR}/${SRR}.dedup.bam"
  samtools index "${BAMDIR}/${SRR}.dedup.bam"

  echo "[$(date)] ${SRR}: filter (proper pair, MAPQ>=30, main chroms, drop chrM/dup/secondary)"
  # -f 2 proper pair; -F 1804 drop unmapped/mate-unmapped/secondary/dup/supplementary.
  samtools view -@ "${THREADS}" -b -f 2 -q 30 -F 1804 \
      "${BAMDIR}/${SRR}.dedup.bam" ${MAIN_CHROMS} > "${FILT}"
  samtools index "${FILT}"

  rm -f "${BAMDIR}/${SRR}.raw.bam" "${BAMDIR}/${SRR}.nsort.bam" \
        "${BAMDIR}/${SRR}.fix.bam" "${BAMDIR}/${SRR}.csort.bam"
done < <(dataset_runs "${DATASET}")

if (( ${#FILT_BAMS[@]} == 0 )); then
  echo "no keep=yes runs for ${DATASET}" >&2
  exit 1
fi

POOLED="${BAMDIR}/${DATASET}.pooled.bam"
echo "[$(date)] ${DATASET}: pooling ${#FILT_BAMS[@]} run(s)"
if (( ${#FILT_BAMS[@]} == 1 )); then
  cp "${FILT_BAMS[0]}" "${POOLED}"
else
  samtools merge -f -@ "${THREADS}" "${POOLED}" "${FILT_BAMS[@]}"
fi
samtools index "${POOLED}"

echo "[$(date)] ${DATASET}: MACS2 callpeak (BAMPE, q=${MACS_Q})"
macs2 callpeak -t "${POOLED}" -f BAMPE -g hs -n "${DATASET}" \
      --outdir "${PKDIR}" -q "${MACS_Q}" --keep-dup all

NP="${PKDIR}/${DATASET}_peaks.narrowPeak"
# blacklist filter (optional)
if [[ -s "${BLACKLIST_BED}" ]]; then
  echo "[$(date)] ${DATASET}: removing blacklist peaks"
  bedtools intersect -v -a "${NP}" -b "${BLACKLIST_BED}" > "${NP}.tmp" && mv "${NP}.tmp" "${NP}"
else
  echo "[$(date)] WARNING blacklist not found (${BLACKLIST_BED}); skipping blacklist filter" >&2
fi

pigz -f -k "${NP}"
mkdir -p "${XULAB}/data"
cp -f "${NP}.gz" "${XULAB}/data/HEK293T_ATAC_${DATASET}_MACS2.narrowPeak.gz"

echo "[$(date)] done ${DATASET}: $(wc -l < "${NP}") peaks"
