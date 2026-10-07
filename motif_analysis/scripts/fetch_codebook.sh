#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# fetch_codebook.sh   (Novel Motif Finding, Steps 1 + 2 — Codebook data)
#
# Download Codebook / MEX motif resources (Hughes et al., Nature 2026;
# "An expanded codebook of human TF DNA-binding specificity") from Zenodo
# 10.5281/zenodo.15667805 and build the DBs our pipeline uses:
#   cache/motifdb/codebook_artifacts.meme  = MEX artifact/contaminant set (Step 2)
#   cache/motifdb/codebook_top1.meme       = 1 curated motif per TF (Step 1 re-annot)
# Then merge v12 + Codebook for TOMTOM (prints the command).
#
# Optional: CHS=1 also pulls MEX.CHS.tar (1.4 GB ChIP-seq peaks, Step 4).
#
#   bash motif_analysis/scripts/fetch_codebook.sh
# -----------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${HERE}/../.." && pwd)}"
CACHE="${CACHE:-${ROOT}/motif_analysis/cache}"
DB="${CACHE}/motifdb"; RAW="${CACHE}/codebook_raw"
Z="https://zenodo.org/records/15667805/files"
mkdir -p "${DB}" "${RAW}"

get() { local f="$1"; [[ -s "${RAW}/${f}" ]] || curl -fSL --retry 3 -o "${RAW}/${f}" "${Z}/${f}?download=1"; }

echo "[codebook] artifact motifs (MEME, ready) + curated top1 + metadata"
get "MEX_artifacts_formatted.tgz"
get "MEX_top1.zip"
get "metadata_complete_motif.zip"

# 1. artifact DB (ready-made MEME, 37 motifs: poly-G, Alu/repeat, CAC/GGAA, NFI, ...)
tar xzf "${RAW}/MEX_artifacts_formatted.tgz" -C "${RAW}"
ART=$(find "${RAW}" -name "MEX-ARTIFACTS_meme_format.meme" | head -1)
cp "${ART}" "${DB}/codebook_artifacts.meme"
# Zenodo export sometimes leaves "w= " blank; fill widths from matrix row counts.
python - "${DB}/codebook_artifacts.meme" <<'PY'
import re, sys
from pathlib import Path
p = Path(sys.argv[1]); lines = p.read_text().splitlines(); out=[]
i=0
while i < len(lines):
    ln = lines[i]
    m = re.match(r"(letter-probability matrix: alength= 4 )w=\s*(nsites=.*)", ln)
    if m:
        j=i+1; rows=0
        while j < len(lines) and re.match(r"^[\d.eE+\-\s\t]+$", lines[j].strip()) and lines[j].strip():
            rows += 1; j += 1
        ln = f"{m.group(1)}w= {rows} {m.group(2)}"
    out.append(ln); i += 1
p.write_text("\n".join(out)+"\n")
PY
echo "[codebook] -> ${DB}/codebook_artifacts.meme ($(grep -c '^MOTIF' "${DB}/codebook_artifacts.meme") motifs)"

# 2. curated Codebook motifs (1 per TF) -> MEME
TOP1="${RAW}/MEX_top1"; mkdir -p "${TOP1}"; unzip -oq "${RAW}/MEX_top1.zip" -d "${TOP1}"
python "${HERE}/ppm_to_meme.py" --in-dir "${TOP1}" --out "${DB}/codebook_top1.meme"

if [[ "${CHS:-0}" == "1" ]]; then
  echo "[codebook] MEX.CHS.tar (1.4 GB ChIP-seq peaks, Step 4)"; get "MEX.CHS.tar"
  tar xf "${RAW}/MEX.CHS.tar" -C "${RAW}" && echo "[codebook] CHS extracted under ${RAW}"
fi

echo
echo "[codebook] Step-1 merged DB (add HOCOMOCO v12 you already have):"
echo "  python ${HERE}/merge_meme_db.py \\"
echo "    --in HOCOMOCOv12_H12CORE_meme_format.meme=h12 ${DB}/codebook_top1.meme=cdbk \\"
echo "    --out ${DB}/merged_human_motifs.meme"
echo "[codebook] Step-2 artifact DB ready: ${DB}/codebook_artifacts.meme (ARTIFACT_DB in rank_motifs.sbatch)"
