#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# fetch_ctcf_motif_bed.sh
#
# Download a CTCF motif BED for hg38 (JASPAR MA0139.x predicted binding sites)
# into downstream/cache/. The cisTopic and PUscOpen propagation steps will
# auto-detect this file when impute.propagate.target_bed is left null.
#
# Output: <CACHE_DIR>/CTCF_motif_hg38.bed   (3-column BED: chrom start end)
#
# Optional env vars:
#   CACHE_DIR        default <repo-root>/downstream/cache
#   JASPAR_ID        default MA0139.2 (CTCF in JASPAR 2024; MA0139.1 is retired)
#   JASPAR_RELEASE   default 2024   (mencius mirror release folder)
#   GENOME           default hg38
#   FORCE            set =1 to re-download even if the file exists
#   SOURCE_URL       override download URL outright (tsv.gz or plain BED)
#
# Sources tried (first successful one wins):
#   1. JASPAR TFBS mirror (per-matrix TSV.gz, no UCSC tools needed):
#        https://mencius.uio.no/JASPAR/JASPAR_TFBSs/<RELEASE>/<GENOME>/<ID>.tsv.gz
#      For CTCF (MA0139.*) also tries MA0139.1 and MA0139.3 if the primary ID 404s.
#   2. JASPAR REST API (often 404 on elixir.no; kept for custom mirrors):
#        https://jaspar.elixir.no/api/v1/matrix/<ID>/predicted-binding-sites/<GENOME>/?format=bed
#   3. UCSC JASPAR bigBed per-TF track (requires bigBedToBed; Longleaf: ucsctools/320):
#        https://expdata.cmmt.ubc.ca/JASPAR/downloads/UCSC_tracks/<RELEASE>/<GENOME>/<ID>.bb
#
# If all fail the script prints fallback instructions for manual download.
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/downstream/cache}"
JASPAR_ID="${JASPAR_ID:-MA0139.2}"
JASPAR_RELEASE="${JASPAR_RELEASE:-2024}"
GENOME="${GENOME:-hg38}"
FORCE="${FORCE:-0}"
OUT_BED="${CACHE_DIR}/CTCF_motif_${GENOME}.bed"
MENCIUS_BASE="https://mencius.uio.no/JASPAR/JASPAR_TFBSs"

mkdir -p "${CACHE_DIR}"

if [[ -f "${OUT_BED}" && "${FORCE}" != "1" ]]; then
  n_lines=$(wc -l < "${OUT_BED}" | tr -d '[:space:]')
  echo "[fetch_ctcf_motif] ${OUT_BED} already exists (${n_lines} lines)."
  echo "[fetch_ctcf_motif] Pass FORCE=1 to re-download."
  exit 0
fi

PRIMARY_URL="${SOURCE_URL:-https://jaspar.elixir.no/api/v1/matrix/${JASPAR_ID}/predicted-binding-sites/${GENOME}/?format=bed}"
UCSC_URL="https://expdata.cmmt.ubc.ca/JASPAR/downloads/UCSC_tracks/${JASPAR_RELEASE}/${GENOME}/${JASPAR_ID}.bb"

tmp_bed="${OUT_BED}.tmp"
rm -f "${tmp_bed}"

ensure_ucsc_tools() {
  if command -v bigBedToBed >/dev/null 2>&1; then
    return 0
  fi
  # Longleaf: ucsctools module is not loaded in login shells by default.
  local d
  for d in /nas/longleaf/apps/ucsctools/*/bin; do
    if [[ -x "${d}/bigBedToBed" ]]; then
      export PATH="${d}:${PATH}"
      echo "[fetch_ctcf_motif] Prepended ${d} to PATH for bigBedToBed."
      return 0
    fi
  done
  return 1
}

jaspar_ids_to_try() {
  if [[ -n "${SOURCE_URL:-}" ]]; then
    echo "${JASPAR_ID}"
    return
  fi
  if [[ "${JASPAR_ID}" == MA0139.* ]]; then
    printf '%s\n' "${JASPAR_ID}" MA0139.2 MA0139.1 MA0139.3 | awk '!seen[$0]++'
  else
    echo "${JASPAR_ID}"
  fi
}

attempt_mencius_tsv() {
  local id url tmp_gz
  tmp_gz="${OUT_BED}.tsv.gz.tmp"
  while IFS= read -r id; do
    [[ -z "${id}" ]] && continue
    url="${MENCIUS_BASE}/${JASPAR_RELEASE}/${GENOME}/${id}.tsv.gz"
    echo "[fetch_ctcf_motif] Trying JASPAR TFBS mirror: ${url}"
    rm -f "${tmp_gz}"
    if curl -fsSL --connect-timeout 30 --max-time 1800 "${url}" -o "${tmp_gz}"; then
      if [[ -s "${tmp_gz}" ]] && gzip -t "${tmp_gz}" 2>/dev/null; then
        if zcat "${tmp_gz}" > "${tmp_bed}"; then
          rm -f "${tmp_gz}"
          if [[ -s "${tmp_bed}" ]]; then
            head -1 "${tmp_bed}" | grep -E '^chr[0-9A-Za-z_]+\s' >/dev/null && return 0
          fi
        fi
      fi
    fi
    rm -f "${tmp_gz}" "${tmp_bed}"
  done < <(jaspar_ids_to_try)
  return 1
}

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
  ensure_ucsc_tools || true
  if ! command -v bigBedToBed >/dev/null 2>&1; then
    echo "[fetch_ctcf_motif] bigBedToBed not found in PATH; skipping UCSC bigBed fallback."
    echo "[fetch_ctcf_motif] On Longleaf: module load ucsctools/320"
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

if [[ -n "${SOURCE_URL:-}" ]]; then
  echo "[fetch_ctcf_motif] Using SOURCE_URL: ${SOURCE_URL}"
  if [[ "${SOURCE_URL}" == *.gz ]]; then
    curl -fsSL --connect-timeout 30 --max-time 1800 "${SOURCE_URL}" | gzip -dc > "${tmp_bed}" || exit 1
  else
    curl -fsSL --connect-timeout 30 --max-time 1800 "${SOURCE_URL}" -o "${tmp_bed}" || exit 1
  fi
elif attempt_mencius_tsv; then
  :
elif attempt_jaspar_api; then
  :
elif attempt_ucsc_bigbed; then
  :
else
  echo "[fetch_ctcf_motif] All download attempts failed." >&2
  echo "" >&2
  echo "Manual fallback options:" >&2
  echo "  1. Download per-matrix TSV.gz from the JASPAR TFBS mirror, then re-run with SOURCE_URL:" >&2
  echo "       https://mencius.uio.no/JASPAR/JASPAR_TFBSs/${JASPAR_RELEASE}/${GENOME}/" >&2
  echo "       e.g. .../MA0139.2.tsv.gz for CTCF (hg38, JASPAR 2024)" >&2
  echo "  2. Browse JASPAR -> Search 'CTCF' -> genome browser tracks -> hg38 -> BED" >&2
  echo "       https://jaspar.elixir.no/matrix/${JASPAR_ID}/" >&2
  echo "  3. Use FIMO from MEME suite to scan hg38.fa for the JASPAR PWM:" >&2
  echo "       fimo --oc fimo_out --thresh 1e-4 ${JASPAR_ID}.meme hg38.fa" >&2
  echo "       awk -v OFS='\\t' 'NR>1 {print \$3, \$4-1, \$5}' fimo_out/fimo.tsv > ${OUT_BED}" >&2
  echo "  4. Use HOMER's scanMotifGenomeWide.pl with the CTCF JASPAR motif." >&2
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
