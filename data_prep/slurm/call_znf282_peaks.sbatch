#!/bin/bash
#SBATCH --job-name=znf282_macs_peaks
#SBATCH --output=znf282_macs_peaks-%j.out
#SBATCH --error=znf282_macs_peaks-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Call hg38 ZNF282 peaks directly from the aligned bulk BAM (replaces liftOver).
#
# Inputs:
#   BAM       default: data/ZNF282_HEK293.bam
# Outputs:
#   data/geo_znf282/macs/ZNF282_HEK293_peaks.narrowPeak
#   data/ZNF282_HEK293_peaks.bed          (installed narrowPeak; prior liftOver
#                                          archived to *.liftOver.bed once)
#
# Usage:
#   sbatch data_prep/slurm/call_znf282_peaks.sbatch

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/data_prep" && -d "${SUBMIT_DIR}/data" ]]; then
  ROOT_DIR="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/../data_prep" ]]; then
  ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  ROOT_DIR="/work/users/d/y/dyy12/XuLab"
fi
cd "${ROOT_DIR}"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
BAM="${BAM:-${DATA_DIR}/ZNF282_HEK293.bam}"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/geo_znf282/macs}"
NAME="${NAME:-ZNF282_HEK293}"
OUT_PEAKS="${OUT_PEAKS:-${DATA_DIR}/ZNF282_HEK293_peaks.bed}"
ARCHIVE="${ARCHIVE:-${DATA_DIR}/ZNF282_HEK293_peaks.liftOver.bed}"
QVALUE="${QVALUE:-1e-10}"
CONDA_ENV="${CONDA_ENV:-data_prep}"
# Optional post-filter on -log10(q) column (narrowPeak col 9). Default matches
# QVALUE=1e-10 so installed peaks stay near author scale (~7–9k). Set 0 to skip.
MIN_LOG10Q="${MIN_LOG10Q:-10}"

if [[ ! -f "${BAM}" ]]; then
  echo "ERROR: missing BAM ${BAM}" >&2
  exit 1
fi
if [[ ! -f "${BAM}.bai" ]]; then
  echo "ERROR: missing BAM index ${BAM}.bai" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "[$(date)] MACS3 callpeak on ${BAM}"
echo "  outdir=${OUT_DIR}  q=${QVALUE}  (SE ChIP-exo / BAM)"

# --keep-dup auto (MACS default): ChIP-exo without input over-calls with --keep-dup all.
macs3 callpeak \
  -t "${BAM}" \
  -f BAM \
  -g hs \
  -n "${NAME}" \
  --outdir "${OUT_DIR}" \
  -q "${QVALUE}"

NARROW="${OUT_DIR}/${NAME}_peaks.narrowPeak"
if [[ ! -s "${NARROW}" ]]; then
  echo "ERROR: MACS3 did not write ${NARROW}" >&2
  exit 1
fi

n_peaks="$(wc -l < "${NARROW}")"
echo "[$(date)] called ${n_peaks} peaks -> ${NARROW}"
cp -a "${NARROW}" "${OUT_DIR}/${NAME}_peaks.full.narrowPeak"

INSTALL_SRC="${NARROW}"
if awk -v m="${MIN_LOG10Q}" 'BEGIN{exit !(m+0 > 0)}'; then
  FILTERED="${OUT_DIR}/${NAME}_peaks.qfilt.narrowPeak"
  awk -v m="${MIN_LOG10Q}" -F'\t' '($9+0) >= m' "${NARROW}" > "${FILTERED}"
  echo "[$(date)] filter -log10(q)>=${MIN_LOG10Q}: $(wc -l < "${FILTERED}") / ${n_peaks}"
  INSTALL_SRC="${FILTERED}"
fi

# Archive prior liftOver peaks once (do not overwrite archive).
if [[ -f "${OUT_PEAKS}" && ! -f "${ARCHIVE}" ]]; then
  echo "[$(date)] archive liftOver peaks -> ${ARCHIVE}"
  cp -a "${OUT_PEAKS}" "${ARCHIVE}"
fi

# Install as the canonical peaks BED (full narrowPeak = 10 cols).
cp -a "${INSTALL_SRC}" "${OUT_PEAKS}"
echo "[$(date)] installed ${OUT_PEAKS} ($(wc -l < "${OUT_PEAKS}") peaks)"
echo "Done."
