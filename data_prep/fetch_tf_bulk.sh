#!/bin/bash
# -----------------------------------------------------------------------------
# fetch_tf_bulk.sh
#
# Download ENCODE bulk ChIP-seq references (peaks BED + alignments BAM) for the
# TFs we benchmark, into the data dir, and index the BAMs. Accessions are baked
# in (resolved from the ENCODE portal, GRCh38). Peaks = positives; BAM = deserts.
#
# All HEK293-matched except NFYA (K562, cross-line -- the only abundant TF
# without a 293 reference).
#
# Usage:
#   bash data_prep/fetch_tf_bulk.sh ZBTB7A ZIC2 MAZ ZNF777
#   bash data_prep/fetch_tf_bulk.sh all
#   DATA_DIR=/work/.../data bash data_prep/fetch_tf_bulk.sh MAZ
#
# ZNF282 is not on ENCODE — use data_prep/fetch_znf282_bulk.sh (GEO GSM2466515).
#
# Output per TF <T>:  <DATA_DIR>/<T>_<LINE>_peaks.bed  and  <T>_<LINE>.bam(.bai)
# -----------------------------------------------------------------------------
set -euo pipefail

DATA_DIR="${DATA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}"
SAMTOOLS="${SAMTOOLS:-samtools}"
BASE="https://www.encodeproject.org/files"

# TF -> "peaks_ENCFF bam_ENCFF cellline"  (all HEK293 except NFYA = K562)
declare -A REF=(
  [CTCF]="ENCFF282PGY ENCFF139DCW HEK293"
  [ZNF768]="ENCFF489EFG ENCFF921GNM HEK293"
  [ZBTB7A]="ENCFF114DGV ENCFF559CGO HEK293"
  [ZIC2]="ENCFF359GIY ENCFF156XLC HEK293"
  [MAZ]="ENCFF529XTF ENCFF085SQS HEK293"
  [ZNF777]="ENCFF735CAY ENCFF436UNS HEK293"
  [NFYA]="ENCFF765GYT ENCFF873PQC K562"
)

if [[ "${1:-}" == "all" ]]; then
  TFS=(CTCF ZNF768 ZBTB7A ZIC2 MAZ ZNF777 NFYA)
else
  TFS=("$@")
fi
if [[ ${#TFS[@]} -eq 0 ]]; then
  echo "Usage: bash data_prep/fetch_tf_bulk.sh <TF> [<TF> ...] | all" >&2; exit 1
fi

command -v "${SAMTOOLS}" >/dev/null 2>&1 || { echo "ERROR: samtools not on PATH (module load samtools / conda)"; exit 1; }
mkdir -p "${DATA_DIR}"; cd "${DATA_DIR}"

for TF in "${TFS[@]}"; do
  if [[ -z "${REF[$TF]:-}" ]]; then echo "[fetch] SKIP ${TF}: no known accession"; continue; fi
  read -r PK BM LINE <<<"${REF[$TF]}"
  if [[ "${PK}" == ENCFF000NONE ]]; then
    echo "[fetch] ${TF}: handled by the SCREEN/cCRE path, nothing to download."; continue
  fi
  echo "[fetch] === ${TF} (${LINE}) : peaks=${PK} bam=${BM} ==="
  PK_OUT="${TF}_${LINE}_peaks.bed"
  BM_OUT="${TF}_${LINE}.bam"

  if [[ ! -f "${PK_OUT}" ]]; then
    wget -q -O "${PK_OUT}.gz" "${BASE}/${PK}/@@download/${PK}.bed.gz"
    gunzip -f "${PK_OUT}.gz"
  fi
  echo "[fetch]   peaks: $(wc -l < "${PK_OUT}") rows -> ${PK_OUT}"

  if [[ ! -f "${BM_OUT}" ]]; then
    wget -q -O "${BM_OUT}" "${BASE}/${BM}/@@download/${BM}.bam"
  fi
  "${SAMTOOLS}" quickcheck "${BM_OUT}" || { echo "ERROR: ${BM_OUT} truncated"; exit 1; }
  [[ -f "${BM_OUT}.bai" ]] || "${SAMTOOLS}" index "${BM_OUT}"
  echo "[fetch]   bam chroms: $("${SAMTOOLS}" idxstats "${BM_OUT}" | head -1 | cut -f1) ..."
done
echo "[fetch] Done. Files in ${DATA_DIR}"
