#!/bin/bash
#SBATCH --job-name=znf282_bulk_bw
#SBATCH --output=znf282_bulk_bw-%j.out
#SBATCH --error=znf282_bulk_bw-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL

# Convert ZNF282 HEK293 bulk ChIP-exo BAM -> BigWig for IGV.
#
# Usage:
#   sbatch downstream/slurm/znf282_bulk_to_bigwig.sbatch

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
BAM="${BAM:-${ROOT_DIR}/data/ZNF282_HEK293.bam}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/downstream/tracks/igv_panel7}"
OUT_BW="${OUT_BW:-${OUT_DIR}/ZNF282_bulk_CPM.bw}"
BAMCOVERAGE="${BAMCOVERAGE:-${ROOT_DIR}/.venv_deeptools/bin/bamCoverage}"
BIN_SIZE="${BIN_SIZE:-10}"
NORMALIZE="${NORMALIZE:-CPM}"
# hg38 effective genome size (deepTools default table)
EFFECTIVE_GENOME_SIZE="${EFFECTIVE_GENOME_SIZE:-2913022398}"

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"

if [[ ! -f "${BAM}" ]]; then
  echo "ERROR: missing BAM ${BAM}" >&2
  exit 1
fi
if [[ ! -f "${BAM}.bai" ]]; then
  echo "ERROR: missing BAM index ${BAM}.bai" >&2
  exit 1
fi
if [[ ! -x "${BAMCOVERAGE}" ]]; then
  echo "ERROR: bamCoverage not found at ${BAMCOVERAGE}" >&2
  exit 1
fi

export PATH="/nas/longleaf/rhel9/apps/samtools/1.24/bin:${PATH:-}"

echo "[$(date)] BAM=${BAM}"
echo "[$(date)] OUT_BW=${OUT_BW}"
echo "[$(date)] binSize=${BIN_SIZE} normalize=${NORMALIZE}"

"${BAMCOVERAGE}" \
  -b "${BAM}" \
  -o "${OUT_BW}" \
  --outFileFormat bigwig \
  --binSize "${BIN_SIZE}" \
  --normalizeUsing "${NORMALIZE}" \
  --effectiveGenomeSize "${EFFECTIVE_GENOME_SIZE}" \
  --ignoreDuplicates \
  --minMappingQuality 5 \
  -p "${SLURM_CPUS_PER_TASK:-8}"

ls -lh "${OUT_BW}"
echo "[$(date)] Done. Load in IGV (hg38): ${OUT_BW}"
