#!/bin/bash
#SBATCH --job-name=av_download
#SBATCH --output=av_download-%A_%a.out
#SBATCH --error=av_download-%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --mail-type=FAIL
#
# Download one SRA run per array task -> gzipped paired FASTQ.
#
# Usage (10 keep=yes runs across both datasets):
#   N=$(awk -F'\t' '$1 !~ /^#/ && $6=="yes"' atac_validation/metadata/samples.tsv | wc -l)
#   sbatch --array=0-$((N-1)) atac_validation/slurm/00_download.sbatch
#
# Output: atac_validation/work/<dataset>/fastq/<srr>_{1,2}.fastq.gz

set -euo pipefail
# $0 is a spool copy under SLURM; always source via the Longleaf checkout.
source "${XULAB:-/work/users/d/y/dyy12/XuLab}/atac_validation/scripts/lib.sh"

mapfile -t ROWS < <(all_runs)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#ROWS[@]} )); then
  echo "array index ${IDX} out of range (n=${#ROWS[@]})" >&2
  exit 1
fi
IFS=$'\t' read -r DATASET GSM SRR <<< "${ROWS[$IDX]}"

FQDIR="${WORK}/${DATASET}/fastq"
mkdir -p "${FQDIR}"
cd "${FQDIR}"

if [[ -s "${SRR}_1.fastq.gz" ]]; then
  echo "[$(date)] ${SRR} already downloaded, skipping"
  exit 0
fi

echo "[$(date)] ${DATASET} ${GSM} ${SRR}: prefetch"
prefetch --max-size 100G -O "${FQDIR}" "${SRR}"

echo "[$(date)] ${SRR}: fasterq-dump"
fasterq-dump --split-files --threads "${THREADS}" -O "${FQDIR}" "${FQDIR}/${SRR}/${SRR}.sra"

echo "[$(date)] ${SRR}: gzip"
pigz -p "${THREADS}" -f "${SRR}"_*.fastq

# tidy the prefetched .sra directory
rm -rf "${FQDIR:?}/${SRR}"

echo "[$(date)] done ${SRR}"
ls -la "${FQDIR}/${SRR}"_*.fastq.gz
