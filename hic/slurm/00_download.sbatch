#!/bin/bash
#SBATCH --job-name=hic_download
#SBATCH --output=hic_download-%j.out
#SBATCH --error=hic_download-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --mail-type=FAIL
#
# Download the HEK293T Hi-C SRA run (330M read pairs, ~large) into the runHiC
# data tree at data/HiC-SRA/<SRR>.sra. runHiC (-F SRA) extracts FASTQ internally.
#
#   sbatch hic/slurm/00_download.sbatch

set -euo pipefail
source "${XULAB:-/work/users/d/y/dyy12/XuLab}/hic/scripts/lib.sh"

SRA_DIR="${DATA}/HiC-SRA"
mkdir -p "${SRA_DIR}"

if [[ -s "${SRA_DIR}/${SRR}.sra" ]]; then
  echo "[$(date)] ${SRR}.sra already present, skipping"
  exit 0
fi

echo "[$(date)] prefetch ${SRR} (this is a deep Hi-C run; expect tens of GB)"
prefetch --max-size 200G -O "${SRA_DIR}" "${SRR}"

# prefetch writes ${SRA_DIR}/${SRR}/${SRR}.sra -- flatten to ${SRA_DIR}/${SRR}.sra
if [[ -s "${SRA_DIR}/${SRR}/${SRR}.sra" ]]; then
  mv -f "${SRA_DIR}/${SRR}/${SRR}.sra" "${SRA_DIR}/${SRR}.sra"
  rmdir "${SRA_DIR}/${SRR}" 2>/dev/null || true
fi
[[ -s "${SRA_DIR}/${SRR}.sra" ]] || { echo "download failed: ${SRA_DIR}/${SRR}.sra" >&2; exit 1; }
echo "[$(date)] done: $(ls -la "${SRA_DIR}/${SRR}.sra")"
