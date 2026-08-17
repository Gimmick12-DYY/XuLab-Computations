#!/bin/bash
#SBATCH --job-name=av_bt2index
#SBATCH --output=av_bt2index-%j.out
#SBATCH --error=av_bt2index-%j.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --mail-type=FAIL
#
# One-time bowtie2 index build from the shared hg38 FASTA.
# Run once before 01_align_call_peaks.sbatch (sbatch or bash, both work).
#
#   sbatch atac_validation/scripts/build_bowtie2_index.sh

set -euo pipefail
source "${XULAB:-/work/users/d/y/dyy12/XuLab}/atac_validation/scripts/lib.sh"

mkdir -p "${REF_DIR}"
if [[ -s "${BT2_INDEX}.1.bt2" || -s "${BT2_INDEX}.1.bt2l" ]]; then
  echo "bowtie2 index already present at ${BT2_INDEX}, nothing to do"
  exit 0
fi
if [[ ! -s "${GENOME_FA}" ]]; then
  echo "genome FASTA not found: ${GENOME_FA}" >&2
  exit 1
fi

echo "[$(date)] building bowtie2 index -> ${BT2_INDEX}"
bowtie2-build --threads "${THREADS}" "${GENOME_FA}" "${BT2_INDEX}"
echo "[$(date)] done"
