#!/bin/bash
#SBATCH --job-name=av_overlap
#SBATCH --output=av_overlap-%j.out
#SBATCH --error=av_overlap-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL
#
# Steps 2 + 3: bin-level overlap of the three ATAC masks, then the union mask.
# Prereq: 01_align_call_peaks.sbatch produced both validation narrowPeaks in data/.
#
#   sbatch atac_validation/slurm/02_overlap_union.sbatch

set -euo pipefail
source "${XULAB:-/work/users/d/y/dyy12/XuLab}/atac_validation/scripts/lib.sh"
cd "${AV}/scripts"

GSE283384_PEAKS="${XULAB}/data/HEK293T_ATAC_gse283384_wt_MACS2.narrowPeak.gz"
GSE152177_PEAKS="${XULAB}/data/HEK293T_ATAC_gse152177_wt_MACS2.narrowPeak.gz"
UNION_BED="${XULAB}/data/HEK293T_ATAC_union3.bed.gz"

for f in "${OMNIATAC_PEAKS}" "${GSE283384_PEAKS}" "${GSE152177_PEAKS}" "${REGIONS_TSV}"; do
  [[ -s "$f" ]] || { echo "missing required input: $f" >&2; exit 1; }
done

mkdir -p "${RESULTS}"

echo "[$(date)] pairwise overlap"
"${PY}" peak_overlap.py \
  --regions "${REGIONS_TSV}" \
  --mask "omniatac=${OMNIATAC_PEAKS}" \
  --mask "gse283384=${GSE283384_PEAKS}" \
  --mask "gse152177=${GSE152177_PEAKS}" \
  --out-dir "${RESULTS}" \
  --plot

echo "[$(date)] union mask"
"${PY}" build_union_mask.py \
  --mask "omniatac=${OMNIATAC_PEAKS}" \
  --mask "gse283384=${GSE283384_PEAKS}" \
  --mask "gse152177=${GSE152177_PEAKS}" \
  --regions "${REGIONS_TSV}" \
  --out-bed "${UNION_BED}" \
  --out-dir "${RESULTS}"

echo "[$(date)] done -> ${RESULTS}"
