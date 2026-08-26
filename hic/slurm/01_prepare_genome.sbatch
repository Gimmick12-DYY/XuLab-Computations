#!/bin/bash
#SBATCH --job-name=hic_genome
#SBATCH --output=hic_genome-%j.out
#SBATCH --error=hic_genome-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --mail-type=FAIL
#
# Build the runHiC genome folder: data/hg38/hg38.fa (main chromosomes only, so
# alt/scaffold multi-mapping does not distort Hi-C), + .fai + chrom.sizes + the
# aligner index. Main-chrom reference matters for clean A/B compartments.
#
#   sbatch hic/slurm/01_prepare_genome.sbatch
#   ALIGNER=chromap sbatch hic/slurm/01_prepare_genome.sbatch   # also build chromap index

set -euo pipefail
source "${XULAB:-/work/users/d/y/dyy12/XuLab}/hic/scripts/lib.sh"

mkdir -p "${GENOME_DIR}"

# 1. main-chromosome FASTA from the shared cache.
if [[ ! -s "${GENOME_FA}" ]]; then
  [[ -s "${SRC_GENOME_FA}" ]] || { echo "source FASTA missing: ${SRC_GENOME_FA}" >&2; exit 1; }
  [[ -s "${SRC_GENOME_FA}.fai" ]] || samtools faidx "${SRC_GENOME_FA}"
  echo "[$(date)] extracting main chromosomes -> ${GENOME_FA}"
  samtools faidx "${SRC_GENOME_FA}" ${MAIN_CHROMS} > "${GENOME_FA}"
fi
samtools faidx "${GENOME_FA}"
cut -f1,2 "${GENOME_FA}.fai" > "${CHROMSIZES}"
echo "[$(date)] chrom.sizes: $(wc -l < "${CHROMSIZES}") chromosomes"

# 2. aligner index.
if [[ "${ALIGNER}" == "bwa-mem" || "${BUILD_BWA:-1}" == "1" ]]; then
  if [[ ! -s "${GENOME_FA}.bwt" ]]; then
    echo "[$(date)] bwa index (~1 h)"
    bwa index "${GENOME_FA}"
  else
    echo "[$(date)] bwa index present"
  fi
fi
if [[ "${ALIGNER}" == "chromap" ]]; then
  if [[ ! -s "${GENOME_DIR}/${GENOME}.chromap.index" ]]; then
    echo "[$(date)] chromap index"
    chromap -i -r "${GENOME_FA}" -o "${GENOME_DIR}/${GENOME}.chromap.index"
  fi
fi
echo "[$(date)] genome ready under ${GENOME_DIR}"
