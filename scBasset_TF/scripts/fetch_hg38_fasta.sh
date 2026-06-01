#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# fetch_hg38_fasta.sh
#
# Download the GRCh38/hg38 primary assembly FASTA (UCSC, ~3 GB compressed) and
# build the samtools .fai index that pyfaidx needs. Cached so re-runs are no-op.
#
# Usage:
#   ./fetch_hg38_fasta.sh                       # -> downstream/cache/hg38.fa(.fai)
#   FA_DIR=/scratch/genomes ./fetch_hg38_fasta.sh
#   SOURCE_URL=... ./fetch_hg38_fasta.sh
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FA_DIR="${FA_DIR:-${REPO_ROOT}/downstream/cache}"
SOURCE_URL="${SOURCE_URL:-https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz}"
FORCE="${FORCE:-0}"

mkdir -p "${FA_DIR}"
OUT_FA="${FA_DIR}/hg38.fa"
OUT_GZ="${OUT_FA}.gz"

if [[ "${FORCE}" == "0" && -s "${OUT_FA}" && -s "${OUT_FA}.fai" ]]; then
  echo "hg38 FASTA already present at ${OUT_FA} (with .fai). Skipping."
  echo "Set FORCE=1 to re-download."
  exit 0
fi

echo "Downloading ${SOURCE_URL} -> ${OUT_GZ}"
curl -L --fail -o "${OUT_GZ}" "${SOURCE_URL}"

echo "Decompressing -> ${OUT_FA}"
gunzip -f "${OUT_GZ}"

if ! command -v samtools >/dev/null 2>&1; then
  echo "ERROR: samtools not found on PATH; cannot build .fai index." >&2
  echo "       conda activate scbasset    (or any env with samtools) and re-run." >&2
  exit 1
fi

echo "samtools faidx ${OUT_FA}"
samtools faidx "${OUT_FA}"

echo "Done."
echo "  FASTA: ${OUT_FA}"
echo "  Index: ${OUT_FA}.fai"
echo
echo "Point scBasset/configs/default.yaml -> paths.genome_fa: ${OUT_FA}"
