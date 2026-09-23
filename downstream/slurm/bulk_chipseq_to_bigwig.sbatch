#!/bin/bash
#SBATCH --job-name=bulk_chip_bw
#SBATCH --output=bulk_chip_bw-%A_%a.out
#SBATCH --error=bulk_chip_bw-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#
# Coverage bigWig for each bulk ChIP-seq BAM (IGV overlay vs the matching
# peak BED — the check for assembly / liftOver misalignment).
#
#   sbatch --array=0-7 downstream/slurm/bulk_chipseq_to_bigwig.sbatch
#   TF=MAZ sbatch downstream/slurm/bulk_chipseq_to_bigwig.sbatch
#
# Outputs: downstream/tracks/bulk_chipseq/<TF>_bulk_CPM.bw
#          plus a symlink to the peak BED in the same folder.
#
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/downstream/tracks/bulk_chipseq}"
BAMCOVERAGE="${BAMCOVERAGE:-/nas/longleaf/rhel9/apps/deeptools/3.5.6/miniconda3/envs/deeptools/bin/bamCoverage}"
[[ -x "${BAMCOVERAGE}" ]] || BAMCOVERAGE="${ROOT_DIR}/.venv_deeptools/bin/bamCoverage"
BIN_SIZE="${BIN_SIZE:-10}"
NORMALIZE="${NORMALIZE:-CPM}"
EFFECTIVE_GENOME_SIZE="${EFFECTIVE_GENOME_SIZE:-2913022398}"
FORCE="${FORCE:-0}"

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"

# TF  BAM  PEAKS
MANIFEST=(
  "CTCF|${ROOT_DIR}/data/ENCFF139DCW.bam|${ROOT_DIR}/data/CTCF_HEK293_peaks.bed"
  "MAZ|${ROOT_DIR}/data/MAZ_HEK293.bam|${ROOT_DIR}/data/MAZ_HEK293_peaks.bed"
  "NFYA|${ROOT_DIR}/data/NFYA_K562.bam|${ROOT_DIR}/data/NFYA_K562_peaks.bed"
  "SETDB1|${ROOT_DIR}/data/SETDB1_HEK293.bam|${ROOT_DIR}/data/SETDB1_HEK293_peaks.bed"
  "ZBTB7A|${ROOT_DIR}/data/ZBTB7A_HEK293.bam|${ROOT_DIR}/data/ZBTB7A_HEK293_peaks.bed"
  "ZIC2|${ROOT_DIR}/data/ZIC2_HEK293.bam|${ROOT_DIR}/data/ZIC2_HEK293_peaks.bed"
  "ZNF282|${ROOT_DIR}/data/ZNF282_HEK293.bam|${ROOT_DIR}/data/ZNF282_HEK293_peaks.bed"
  "ZNF777|${ROOT_DIR}/data/ZNF777_HEK293.bam|${ROOT_DIR}/data/ZNF777_HEK293_peaks.bed"
)

if [[ -n "${TF:-}" ]]; then
  IDX=-1
  for i in "${!MANIFEST[@]}"; do
    t="${MANIFEST[$i]%%|*}"
    if [[ "${t}" == "${TF}" ]]; then IDX=$i; break; fi
  done
  (( IDX >= 0 )) || { echo "unknown TF=${TF}" >&2; exit 1; }
else
  IDX="${SLURM_ARRAY_TASK_ID:-0}"
fi
(( IDX >= 0 && IDX < ${#MANIFEST[@]} )) || { echo "array index ${IDX} out of range" >&2; exit 1; }

IFS='|' read -r TF BAM PEAKS <<<"${MANIFEST[$IDX]}"
OUT_BW="${OUT_DIR}/${TF}_bulk_CPM.bw"
PEAK_LINK="${OUT_DIR}/${TF}_peaks.bed"

echo "[$(date)] TF=${TF}"
echo "[$(date)] BAM=${BAM}"
echo "[$(date)] PEAKS=${PEAKS}"
echo "[$(date)] OUT=${OUT_BW}"

[[ -f "${BAM}" ]] || { echo "missing BAM ${BAM}" >&2; exit 1; }
[[ -f "${BAM}.bai" ]] || { echo "missing index ${BAM}.bai" >&2; exit 1; }
[[ -x "${BAMCOVERAGE}" ]] || { echo "bamCoverage not found: ${BAMCOVERAGE}" >&2; exit 1; }

if [[ -s "${PEAKS}" ]]; then
  ln -sfn "${PEAKS}" "${PEAK_LINK}"
fi

# Reuse the existing ZNF282 track if it is already CPM hg38.
EXISTING_ZNF="${ROOT_DIR}/downstream/tracks/igv_panel7/ZNF282_bulk_CPM.bw"
if [[ "${TF}" == "ZNF282" && -s "${EXISTING_ZNF}" && "${FORCE}" != "1" ]]; then
  ln -sfn "${EXISTING_ZNF}" "${OUT_BW}"
  echo "[$(date)] reused ${EXISTING_ZNF}"
  ls -lh "${OUT_BW}"
  exit 0
fi

if [[ -s "${OUT_BW}" && "${FORCE}" != "1" ]]; then
  echo "[$(date)] exists $(ls -lh "${OUT_BW}" | awk '{print $5}') — skip (FORCE=1 to redo)"
  exit 0
fi

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
echo "[$(date)] Done. IGV hg38: ${OUT_BW}  +  ${PEAK_LINK}"
