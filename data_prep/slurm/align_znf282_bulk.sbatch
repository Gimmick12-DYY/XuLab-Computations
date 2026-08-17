#!/bin/bash
#SBATCH --job-name=znf282_bulk_bam
#SBATCH --output=znf282_bulk_bam-%j.out
#SBATCH --error=znf282_bulk_bam-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# Align Imbeault et al. ZNF282 ChIP-exo (SRR5197098 ± SRR5197099) to hg38.
# Matches paper settings as closely as practical: Bowtie2 --very-sensitive-local.
#
# Env overrides:
#   DATA_DIR   default: <repo>/data
#   WORK_DIR   default: <DATA_DIR>/geo_znf282/align
#   RUNS       space-separated SRR ids (default: both replicates)
#   BT2_INDEX  bowtie2 index prefix (default: Longleaf hg38_UCSC)
#   MERGE      1 = merge replicates into one BAM (default 1)

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/data_prep" ]]; then
  ROOT_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/fetch_znf282_bulk.sh" ]]; then
  ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
WORK_DIR="${WORK_DIR:-${DATA_DIR}/geo_znf282/align}"
BT2_INDEX="${BT2_INDEX:-/proj/seq/data/hg38_UCSC/Sequence/Bowtie2Index/genome}"
OUT_BAM="${OUT_BAM:-${DATA_DIR}/ZNF282_HEK293.bam}"
# Default: GSM2466515 only (rep1). Add SRR5197099 for rep2 if download works.
RUNS="${RUNS:-SRR5197098}"
MERGE="${MERGE:-1}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"

BOWTIE2="${BOWTIE2:-/nas/longleaf/rhel9/apps/bowtie2/2.5.4/bin/bowtie2}"
SAMTOOLS="${SAMTOOLS:-samtools}"

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
module load samtools 2>/dev/null || true
command -v "${SAMTOOLS}" >/dev/null 2>&1 || { echo "ERROR: samtools not found"; exit 1; }
[[ -x "${BOWTIE2}" ]] || { echo "ERROR: bowtie2 not found at ${BOWTIE2}"; exit 1; }
[[ -f "${BT2_INDEX}.1.bt2" || -f "${BT2_INDEX}.1.bt2l" ]] || {
  echo "ERROR: bowtie2 index missing: ${BT2_INDEX}"; exit 1;
}

echo "[$(date)] ZNF282 bulk align -> ${OUT_BAM}"
echo "[$(date)] runs=${RUNS} threads=${THREADS} index=${BT2_INDEX}"

# ENA FASTQ layout for 7-digit SRR accessions:
#   .../fastq/SRR519/008/SRR5197098/SRR5197098.fastq.gz
# (subdir = "00" + last digit). 6-digit use last-3; we only need 7-digit here.
ena_fastq_url() {
  local srr="$1"
  local prefix="${srr:0:6}"
  local digits="${srr#SRR}"
  local sub
  if [[ ${#digits} -eq 7 ]]; then
    sub=$(printf '00%s' "${digits: -1}")
  else
    sub=$(printf '%03d' "$((10#${digits: -3}))")
  fi
  printf 'https://ftp.sra.ebi.ac.uk/vol1/fastq/%s/%s/%s/%s.fastq.gz' \
    "${prefix}" "${sub}" "${srr}" "${srr}"
}

download_fastq() {
  # stdout MUST be only the FASTQ path (captured by caller). Logs go to stderr.
  local srr="$1"
  local fq="${WORK_DIR}/${srr}.fastq.gz"
  if [[ -s "${fq}" ]]; then
    echo "[$(date)] keep ${fq}" >&2
    printf '%s\n' "${fq}"
    return 0
  fi
  local url
  url="$(ena_fastq_url "${srr}")"
  echo "[$(date)] curl ${url}" >&2
  if ! curl -L --fail --retry 5 --retry-delay 10 -o "${fq}.partial" "${url}"; then
    # NCBI HTTPS fallback via ENA mirror sometimes fails; try NCBI gallery
    local ncbi="https://sra-pub-src-1.s3.amazonaws.com/${srr}/${srr}.fastq.gz"
    echo "[$(date)] ENA failed; trying ${ncbi}" >&2
    curl -L --fail --retry 5 --retry-delay 10 -o "${fq}.partial" "${ncbi}" || {
      echo "ERROR: could not download FASTQ for ${srr}" >&2
      echo "  Install sratoolkit and re-run with: prefetch ${srr} && fasterq-dump ${srr}" >&2
      exit 1
    }
  fi
  mv "${fq}.partial" "${fq}"
  printf '%s\n' "${fq}"
}

BAM_LIST=()
for SRR in ${RUNS}; do
  FQ="$(download_fastq "${SRR}")"
  # Strip accidental whitespace/newlines from capture
  FQ="$(printf '%s' "${FQ}" | tr -d '\r' | tail -n1)"
  [[ -f "${FQ}" ]] || { echo "ERROR: FASTQ path invalid: '${FQ}'" >&2; exit 1; }
  BAM="${WORK_DIR}/${SRR}.hg38.bam"
  if [[ -s "${BAM}" ]]; then
    echo "[$(date)] keep ${BAM}"
  else
    echo "[$(date)] bowtie2 align ${SRR} <- ${FQ}"
    "${BOWTIE2}" --very-sensitive-local -p "${THREADS}" -x "${BT2_INDEX}" -U "${FQ}" \
      2> "${WORK_DIR}/${SRR}.bowtie2.log" \
      | "${SAMTOOLS}" view -b -@ "${THREADS}" - \
      | "${SAMTOOLS}" sort -@ "${THREADS}" -o "${BAM}" -
    "${SAMTOOLS}" index "${BAM}"
  fi
  BAM_LIST+=("${BAM}")
done

if [[ "${MERGE}" == "1" && ${#BAM_LIST[@]} -gt 1 ]]; then
  echo "[$(date)] merge -> ${OUT_BAM}"
  "${SAMTOOLS}" merge -f -@ "${THREADS}" "${OUT_BAM}" "${BAM_LIST[@]}"
else
  echo "[$(date)] copy ${BAM_LIST[0]} -> ${OUT_BAM}"
  cp -f "${BAM_LIST[0]}" "${OUT_BAM}"
fi
"${SAMTOOLS}" index "${OUT_BAM}"
"${SAMTOOLS}" quickcheck "${OUT_BAM}"
echo "[$(date)] done: ${OUT_BAM}"
"${SAMTOOLS}" idxstats "${OUT_BAM}" | head -5
