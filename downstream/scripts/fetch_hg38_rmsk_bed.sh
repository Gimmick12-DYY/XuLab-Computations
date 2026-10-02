#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# fetch_hg38_rmsk_bed.sh   (Novel Motif Finding, Step 2 — TE/repeat check)
#
# Download UCSC hg38 RepeatMasker (rmsk table) and convert to a sorted BED:
#   chrom  start  end  repName;repClass;repFamily  .  strand
# Run on a host with outbound UCSC access (e.g. Longleaf). Output default:
#   downstream/cache/hg38.rmsk.bed.gz   (-> set REPEAT_BED to this in rank_motifs.sbatch)
#
#   bash downstream/scripts/fetch_hg38_rmsk_bed.sh [OUT.bed.gz]
# -----------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${HERE}/../cache/hg38.rmsk.bed.gz}"
URL="${RMSK_URL:-https://hgdownload.soc.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz}"
mkdir -p "$(dirname "${OUT}")"
RAW="$(dirname "${OUT}")/rmsk.txt.gz"

echo "[rmsk] downloading ${URL}"
# UCSC also serves over hgdownload.cse.ucsc.edu and rsync:// if this host is blocked
if ! curl -fSL --retry 3 -o "${RAW}" "${URL}"; then
  echo "[rmsk] curl failed; try: rsync -avzP rsync://hgdownload.soc.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz ${RAW}" >&2
  exit 1
fi

echo "[rmsk] rmsk.txt.gz -> BED (genoName/Start/End + repName;repClass;repFamily)"
# rmsk cols (with bin): 6=genoName 7=genoStart(0-based) 8=genoEnd 10=strand
#                       11=repName 12=repClass 13=repFamily
zcat "${RAW}" \
  | awk 'BEGIN{OFS="\t"} $6 ~ /^chr/ {print $6,$7,$8,$11";"$12";"$13,".",$10}' \
  | sort -k1,1 -k2,2n \
  | gzip -c > "${OUT}"

n=$(zcat "${OUT}" | wc -l | tr -d '[:space:]')
echo "[rmsk] wrote ${n} repeat intervals -> ${OUT}"
echo "[rmsk] use: REPEAT_BED=${OUT} ... (+ INSTANCES_BED from FIMO) in rank_motifs.sbatch"
