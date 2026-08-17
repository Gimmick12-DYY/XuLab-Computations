#!/bin/bash
# -----------------------------------------------------------------------------
# fetch_znf282_bulk.sh
#
# ZNF282 is not on ENCODE. Bulk ChIP-exo reference from Imbeault et al. 2017
# (GEO GSM2466515 / GSE78099):
#   BAM   : SRA SRR5197098 (+ optional rep2) aligned to hg38
#   peaks : prefer MACS3 called on that BAM (data_prep/slurm/call_znf282_peaks.sbatch).
#           This script's default path is the older author BED (GRCh37) -> liftOver
#           -> hg38 narrowPeak-like; keep for provenance / fallback only.
#
# Usage:
#   bash data_prep/fetch_znf282_bulk.sh            # peaks only (fast)
#   bash data_prep/fetch_znf282_bulk.sh --align    # peaks + submit BAM align job
#   PEAKS_ONLY=1 bash data_prep/fetch_znf282_bulk.sh
#
# Outputs:
#   data/ZNF282_HEK293_peaks.bed
#   data/ZNF282_HEK293.bam(.bai)   # after align job finishes
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
GEO_DIR="${GEO_DIR:-${DATA_DIR}/geo_znf282}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/downstream/cache}"
DO_ALIGN=0
for arg in "$@"; do
  case "${arg}" in
    --align) DO_ALIGN=1 ;;
    --peaks-only) DO_ALIGN=0 ;;
    *) echo "Unknown arg: ${arg}" >&2; exit 1 ;;
  esac
done

mkdir -p "${GEO_DIR}" "${CACHE_DIR}" "${DATA_DIR}"
cd "${ROOT_DIR}"

PEAK_GZ="${GEO_DIR}/GSM2466515_ZNF282_peaks_processed_score_signal_exo.bed.gz"
PEAK_URL='https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2466nnn/GSM2466515/suppl/GSM2466515_ZNF282_peaks_processed_score_signal_exo.bed.gz'
CHAIN_GZ="${CACHE_DIR}/hg19ToHg38.over.chain.gz"
CHAIN_URL='https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz'
LIFTOVER="${CACHE_DIR}/liftOver"
LIFTOVER_URL='http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver'
OUT_PEAKS="${DATA_DIR}/ZNF282_HEK293_peaks.bed"

download() {
  local url="$1" dest="$2"
  if [[ -s "${dest}" ]]; then
    echo "[znf282] keep existing $(basename "${dest}")"
    return 0
  fi
  echo "[znf282] download ${url}"
  curl -L --fail -o "${dest}" "${url}"
}

download "${PEAK_URL}" "${PEAK_GZ}"
download "${CHAIN_URL}" "${CHAIN_GZ}"
if [[ ! -x "${LIFTOVER}" ]]; then
  download "${LIFTOVER_URL}" "${LIFTOVER}"
  chmod +x "${LIFTOVER}"
fi

HG19_BED="${GEO_DIR}/ZNF282_peaks_hg19.bed"
HG38_BED="${GEO_DIR}/ZNF282_peaks_hg38.bed"
UNMAPPED="${GEO_DIR}/ZNF282_peaks_hg19_unmapped.bed"

echo "[znf282] prepare hg19 BED"
zcat "${PEAK_GZ}" \
  | awk 'BEGIN{OFS="\t"} NF>=5 {print $1,$2,$3,$4,$5,($6=="+"||$6=="-")?$6:"."}' \
  > "${HG19_BED}"
echo "[znf282]   $(wc -l < "${HG19_BED}") peaks (hg19)"

if [[ ! -s "${HG38_BED}" || "${HG19_BED}" -nt "${HG38_BED}" ]]; then
  echo "[znf282] liftOver hg19 -> hg38"
  # Prefer uncompressed chain (some liftOver builds dislike .gz).
  CHAIN="${CACHE_DIR}/hg19ToHg38.over.chain"
  if [[ ! -s "${CHAIN}" || "${CHAIN_GZ}" -nt "${CHAIN}" ]]; then
    gunzip -c "${CHAIN_GZ}" > "${CHAIN}"
  fi
  "${LIFTOVER}" -bedPlus=6 "${HG19_BED}" "${CHAIN}" "${HG38_BED}" "${UNMAPPED}"
fi
echo "[znf282]   lifted=$(wc -l < "${HG38_BED}") unmapped=$(wc -l < "${UNMAPPED}")"

python3 - "${HG38_BED}" "${OUT_PEAKS}" <<'PY'
import sys
from pathlib import Path
inp, out = Path(sys.argv[1]), Path(sys.argv[2])
n = 0
with inp.open() as fin, out.open("w") as fout:
    for line in fin:
        p = line.rstrip("\n").split("\t")
        if len(p) < 5:
            continue
        chrom, start, end, name = p[0], p[1], p[2], p[3]
        sig = float(p[4])
        strand = p[5] if len(p) > 5 else "."
        score = min(1000, max(0, int(round(sig))))
        # narrowPeak-like: signalValue in col 7 (default PEAK_SIGNAL_COL)
        fout.write(
            f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\t"
            f"{sig:.6f}\t-1.00000\t-1.00000\t-1\n"
        )
        n += 1
print(f"[znf282] wrote {n} peaks -> {out}")
PY

echo "[znf282] peaks ready: ${OUT_PEAKS}"

if [[ "${DO_ALIGN}" == "1" ]]; then
  echo "[znf282] submitting hg38 BAM align job"
  sbatch "${ROOT_DIR}/data_prep/slurm/align_znf282_bulk.sbatch"
else
  cat <<EOF

[znf282] Peaks only. To build data/ZNF282_HEK293.bam (needed for zero-read negatives):
  bash data_prep/fetch_znf282_bulk.sh --align
  # or:
  sbatch data_prep/slurm/align_znf282_bulk.sbatch

EOF
fi
