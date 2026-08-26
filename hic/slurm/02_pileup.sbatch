#!/bin/bash
#SBATCH --job-name=hic_pileup
#SBATCH --output=hic_pileup-%j.out
#SBATCH --error=hic_pileup-%j.err
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=FAIL
#
# runHiC pileup: mapping -> filtering -> binning(+ICE) in one shot, producing
# .pairs and balanced .cool/.mcool under the workspace. This is the heavy step
# (330M read pairs). bwa-mem is standard but slow at this depth; ALIGNER=chromap
# is much faster and fine for compartment-scale analysis.
#
#   sbatch hic/slurm/02_pileup.sbatch
#   ALIGNER=chromap sbatch hic/slurm/02_pileup.sbatch
#
# Prereqs: 00_download (SRA), 01_prepare_genome (fa + index). The aligner index
# for ${ALIGNER} must exist (rerun 01 with matching ALIGNER if you switch).

set -euo pipefail
source "${XULAB:-/work/users/d/y/dyy12/XuLab}/hic/scripts/lib.sh"

[[ -s "${DATA}/HiC-SRA/${SRR}.sra" ]] || { echo "missing SRA: ${DATA}/HiC-SRA/${SRR}.sra (run 00)" >&2; exit 1; }
[[ -s "${GENOME_FA}" ]] || { echo "missing genome: ${GENOME_FA} (run 01)" >&2; exit 1; }
command -v runHiC >/dev/null 2>&1 || { echo "runHiC not on PATH in env ${CONDA_ENV}" >&2; exit 1; }

mkdir -p "${WORK}"
cp -f "${SAMPLES_TSV}" "${WORK}/datasets.tsv"
cd "${WORK}"

echo "[$(date)] runHiC pileup  aligner=${ALIGNER} threads=${THREADS} enzyme=${ENZYME}"
runHiC pileup \
  -p "${DATA}" \
  -g "${GENOME}" \
  -f HiC-SRA \
  -F SRA \
  -A "${ALIGNER}" \
  -t "${THREADS}" \
  --include-readid \
  --drop-seq \
  --chunkSize "${CHUNK_SIZE}" \
  --logFile "${WORK}/runHiC-pileup.log"

echo "[$(date)] done. Outputs under ${WORK} (coolers-${GENOME}/*.mcool):"
ls -la "${WORK}/coolers-${GENOME}"/ 2>/dev/null || true
