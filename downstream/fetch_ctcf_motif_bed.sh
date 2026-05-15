#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# fetch_ctcf_motif_bed.sh
#
# Download a CTCF motif BED for hg38 (JASPAR MA0139.1, predicted binding sites)
# into downstream/cache/. The cisTopic and PUscOpen propagation steps will
# auto-detect this file when impute.propagate.target_bed is left null.
#
# Output: <CACHE_DIR>/CTCF_motif_hg38.bed   (3-column BED: chrom start end)
#
# Optional env vars:
#   CACHE_DIR   default <repo-root>/downstream/cache
#   JASPAR_ID   default MA0139.1   (CTCF; alt MA0139.2 or MA0139.3 in newer JASPAR)
#   GENOME      default hg38
#   FORCE       set =1 to re-download even if the file exists
#   SOURCE_URL  override the JASPAR API URL outright; overrides JASPAR_ID/GENOME
#
# Sources tried (first successful one wins):
#   1. JASPAR REST API:
#        https://jaspar.genereg.net/api/v1/matrix/<ID>/predicted-binding-sites/<GENOME>/?format=bed
#   2. UCSC JASPAR 2024 hg38 per-TF bigBed (requires bigBedToBed in PATH):
#        https://expdata.cmmt.ubc.ca/JASPAR/downloads/UCSC_tracks/2024/hg38/<ID>.bb
#
# If both fail the script prints fallback instructions for manual download.
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/downstream/cache}"
JASPAR_ID="${JASPAR_ID:-MA0139.1}"
GENOME="${GENOME:-hg38}"
FORCE="${FORCE:-0}"
OUT_BED="${CACHE_DIR}/CTCF_motif_${GENOME}.bed"

mkdir -p "${CACHE_DIR}"

if [[ -f "${OUT_BED}" && "${FORCE}" != "1" ]]; then
  n_lines=$(wc -l < "${OUT_BED}" | tr -d '[:space:]')
  echo "[fetch_ctcf_motif] ${OUT_BED} already exists (${n_lines} lines)."
  echo "[fetch_ctcf_motif] Pass FORCE=1 to re-download."
  exit 0
fi

PRIMARY_URL="${SOURCE_URL:-https://jaspar.genereg.net/api/v1/matrix/${JASPAR_ID}/predicted-binding-sites/${GENOME}/?format=bed}"
UCSC_URL="https://expdata.cmmt.ubc.ca/JASPAR/downloads/UCSC_tracks/2024/${GENOME}/${JASPAR_ID}.bb"

tmp_bed="${OUT_BED}.tmp"
rm -f "${tmp_bed}"

attempt_jaspar_api() {
  echo "[fetch_ctcf_motif] Trying JASPAR API: ${PRIMARY_URL}"
  if curl -fsSL --connect-timeout 30 --max-time 600 "${PRIMARY_URL}" -o "${tmp_bed}"; then
    if [[ -s "${tmp_bed}" ]]; then
      head -1 "${tmp_bed}" | grep -E '^[A-Za-z0-9_]+\s' >/dev/null \
        || head -1 "${tmp_bed}" | grep -E '^chr' >/dev/null
      return 0
    fi
  fi
  return 1
}

attempt_ucsc_bigbed() {
  if ! command -v bigBedToBed >/dev/null 2>&1; then
    echo "[fetch_ctcf_motif] bigBedToBed not found in PATH; skipping UCSC fallback."
    return 1
  fi
  echo "[fetch_ctcf_motif] Trying UCSC JASPAR2024 bigBed: ${UCSC_URL}"
  local tmp_bb="${OUT_BED}.bb.tmp"
  if curl -fsSL --connect-timeout 30 --max-time 600 "${UCSC_URL}" -o "${tmp_bb}"; then
    if bigBedToBed "${tmp_bb}" "${tmp_bed}"; then
      rm -f "${tmp_bb}"
      return 0
    fi
    rm -f "${tmp_bb}"
  fi
  return 1
}

if attempt_jaspar_api; then
  :
elif attempt_ucsc_bigbed; then
  :
else
  echo "[fetch_ctcf_motif] All download attempts failed." >&2
  echo "" >&2
  echo "Manual fallback options:" >&2
  echo "  1. Browse JASPAR -> Search 'CTCF' -> 'Genome browser tracks' -> hg38 -> 'BED'" >&2
  echo "       https://jaspar.genereg.net/matrix/${JASPAR_ID}/" >&2
  echo "  2. Use FIMO from MEME suite to scan hg38.fa for the JASPAR PWM:" >&2
  echo "       fimo --oc fimo_out --thresh 1e-4 MA0139.1.meme hg38.fa" >&2
  echo "       awk -v OFS='\\t' 'NR>1 {print \$3, \$4-1, \$5}' fimo_out/fimo.tsv > ${OUT_BED}" >&2
  echo "  3. Use HOMER's scanMotifGenomeWide.pl with the CTCF JASPAR motif." >&2
  echo "" >&2
  echo "Save the resulting 3-column BED at:" >&2
  echo "    ${OUT_BED}" >&2
  echo "and re-run the propagation step (cisTopic 05_impute.py / PUscOpen 06_impute.py)." >&2
  exit 1
fi

# Normalise to a strict 3-column BED (chrom start end), keep only canonical chroms.
awk -v OFS='\t' '!/^#/ && !/^track/ && !/^browser/ && NF>=3 {
    chrom = $1
    if (chrom !~ /^chr/) { chrom = "chr" chrom }
    print chrom, $2, $3
}' "${tmp_bed}" \
  | LC_ALL=C sort -k1,1 -k2,2n -u > "${OUT_BED}"

rm -f "${tmp_bed}"

n_lines=$(wc -l < "${OUT_BED}" | tr -d '[:space:]')
echo "[fetch_ctcf_motif] Wrote ${OUT_BED} (${n_lines} intervals)."
echo "[fetch_ctcf_motif] Now point cisTopic / PUscOpen impute.propagate.target_bed at:"
echo "    ${OUT_BED}"
echo "(or leave it null -- both pipelines auto-detect this default cache path.)"
