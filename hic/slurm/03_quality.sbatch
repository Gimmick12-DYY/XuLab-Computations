#!/bin/bash
#SBATCH --job-name=hic_quality
#SBATCH --output=hic_quality-%j.out
#SBATCH --error=hic_quality-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --mail-type=FAIL
#
# runHiC quality: library-level QC (valid-pair ratio, cis/trans, redundancy)
# from the .pairs statistics produced by pileup.
#
#   sbatch hic/slurm/03_quality.sbatch
#
# NOTE: confirm exact flags for your runHiC version with `runHiC quality -h`
# (the metadata/label flags have varied between releases).

set -euo pipefail
source "${XULAB:-/work/users/d/y/dyy12/XuLab}/hic/scripts/lib.sh"

command -v runHiC >/dev/null 2>&1 || { echo "runHiC not on PATH in env ${CONDA_ENV}" >&2; exit 1; }
cd "${WORK}"
cp -f "${SAMPLES_TSV}" "${WORK}/datasets.tsv"

echo "[$(date)] runHiC quality"
runHiC quality -m datasets.tsv -L "${GENOME}" --logFile "${WORK}/runHiC-quality.log"
echo "[$(date)] done; QC tables/plots under ${WORK}"
