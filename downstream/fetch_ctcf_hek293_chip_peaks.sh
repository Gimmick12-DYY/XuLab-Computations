#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# fetch_ctcf_hek293_chip_peaks.sh
#
# Install a *genuine* HEK293 CTCF ChIP-seq peak set as the CTCF positive source
# for the downstream all_universe / all_bound eval, replacing the previously used
# data/CTCF_HEK293_peaks.bed (which produced ~322k positive bins -- ~6x a real
# CTCF ChIP and far above every other TF -- because it was not an equivalent
# HEK293 ChIP peak call).
#
# Source: ENCODE experiment ENCSR000DTW  "CTCF ChIP-seq on human HEK293"
#   file  ENCFF314ZAL  GRCh38 "optimal IDR thresholded peaks" (narrowPeak, 51,527)
#   -> same experiment as the CTCF bulk BAM already used downstream (ENCFF139DCW,
#      ENCSR000DTW rep1), so peaks and coverage stay consistent.
#
# Output: data/CTCF_HEK293_peaks.bed   (10-col narrowPeak, main chroms only)
#         data/CTCF_HEK293_peaks.bed.provenance.txt
# Backup: data/CTCF_HEK293_peaks.bed.pre_encff314zal  (if a file was already there)
#
# Optional env vars:
#   DATA_DIR    default <repo-root>/data
#   ENCODE_ACC  default ENCFF314ZAL
#   FORCE       set =1 to reinstall even if the target already came from this file
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
ENCODE_ACC="${ENCODE_ACC:-ENCFF314ZAL}"
FORCE="${FORCE:-0}"
OUT_BED="${DATA_DIR}/CTCF_HEK293_peaks.bed"
PROV="${OUT_BED}.provenance.txt"
BACKUP="${OUT_BED}.pre_encff314zal"
URL="https://www.encodeproject.org/files/${ENCODE_ACC}/@@download/${ENCODE_ACC}.bed.gz"

# Autosomes + chrX/Y (drops _random / chrUn_* / chrM scaffolds the bin universe lacks).
# Anchored with $ because $1 is the whole chrom field; POSIX awk has no \b.
MAIN_CHROMS_RE='^(chr[1-9]|chr1[0-9]|chr2[0-2]|chrX|chrY)$'

mkdir -p "${DATA_DIR}"

if [[ -f "${PROV}" && "${FORCE}" != "1" ]] && grep -q "${ENCODE_ACC}" "${PROV}" 2>/dev/null; then
  echo "[fetch_ctcf_chip] ${OUT_BED} already installed from ${ENCODE_ACC}."
  echo "[fetch_ctcf_chip] Pass FORCE=1 to reinstall."
  exit 0
fi

tmp_gz="${OUT_BED}.dl.gz.tmp"
tmp_bed="${OUT_BED}.tmp"
rm -f "${tmp_gz}" "${tmp_bed}"

echo "[fetch_ctcf_chip] downloading ${ENCODE_ACC} from ENCODE"
curl -fsSL --connect-timeout 30 --max-time 1800 "${URL}" -o "${tmp_gz}"
if [[ ! -s "${tmp_gz}" ]] || ! gzip -t "${tmp_gz}" 2>/dev/null; then
  echo "[fetch_ctcf_chip] download failed or not gzip: ${URL}" >&2
  rm -f "${tmp_gz}"
  exit 1
fi

# Keep full narrowPeak columns (so POS_MODE=peak_cutoff can still use p/q), main chroms only.
zcat "${tmp_gz}" \
  | awk -v re="${MAIN_CHROMS_RE}" '$1 ~ re' \
  | LC_ALL=C sort -k1,1 -k2,2n > "${tmp_bed}"
rm -f "${tmp_gz}"

n_new=$(wc -l < "${tmp_bed}" | tr -d '[:space:]')
if [[ "${n_new}" -lt 1000 ]]; then
  echo "[fetch_ctcf_chip] refusing to install: only ${n_new} peaks after chrom filter" >&2
  rm -f "${tmp_bed}"
  exit 1
fi

if [[ -f "${OUT_BED}" && ! -f "${BACKUP}" ]]; then
  n_old=$(wc -l < "${OUT_BED}" | tr -d '[:space:]')
  cp -p "${OUT_BED}" "${BACKUP}"
  echo "[fetch_ctcf_chip] backed up existing peaks (${n_old} lines) -> ${BACKUP}"
fi

mv -f "${tmp_bed}" "${OUT_BED}"

cat > "${PROV}" <<EOF
CTCF HEK293 ChIP-seq peaks installed by fetch_ctcf_hek293_chip_peaks.sh
date_utc:      $(date -u +%Y-%m-%dT%H:%M:%SZ)
encode_file:   ${ENCODE_ACC}
encode_expt:   ENCSR000DTW  (CTCF ChIP-seq on human HEK293)
assembly:      GRCh38
output_type:   optimal IDR thresholded peaks (narrowPeak)
n_peaks:       ${n_new}
matched_bulk:  ENCFF139DCW.bam (ENCSR000DTW rep1) -- already used for CTCF coverage/negatives
url:           ${URL}
replaces:      prior data/CTCF_HEK293_peaks.bed (inflated ~322k positive bins)
EOF

echo "[fetch_ctcf_chip] wrote ${OUT_BED} (${n_new} peaks) + ${PROV}"
echo "[fetch_ctcf_chip] Next: rebuild CTCF bins + eval (+ openpos / slide patch):"
echo "    bash downstream/slurm/run_all_universe_panel.sh CTCF"
echo "    # then openpos bins/eval + downstream/update_slide_ctcf_rows.py"
