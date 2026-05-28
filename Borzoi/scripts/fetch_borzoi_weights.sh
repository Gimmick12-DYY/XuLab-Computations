#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# fetch_borzoi_weights.sh
#
# Download Borzoi human checkpoints + targets_human.txt into
# <repo>/downstream/cache/borzoi/, mirroring the layout that 03_borzoi_predict.py
# expects:
#   <cache>/fold0/   (and optionally fold1..fold3 for ensembling)
#   <cache>/targets_human.txt
#
# borzoi-pytorch fetches weights from HuggingFace Hub on demand, so the
# essential thing this script does is:
#   1. snapshot_download each fold repo into <cache>/fold<i>/, and
#   2. pull targets_human.txt from the Calico borzoi repo.
#
# Usage:
#   ./fetch_borzoi_weights.sh                # default: fold0 only
#   FOLDS="0 1 2 3" ./fetch_borzoi_weights.sh
#   CACHE_DIR=/scratch/borzoi ./fetch_borzoi_weights.sh
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/downstream/cache/borzoi}"
FOLDS="${FOLDS:-0}"
TARGETS_URL="${TARGETS_URL:-https://raw.githubusercontent.com/calico/borzoi/main/examples/targets_human.txt}"
FORCE="${FORCE:-0}"

mkdir -p "${CACHE_DIR}"

# ---- targets_human.txt -------------------------------------------------------
TARGETS_PATH="${CACHE_DIR}/targets_human.txt"
if [[ "${FORCE}" == "0" && -s "${TARGETS_PATH}" ]]; then
  echo "targets_human.txt already present (${TARGETS_PATH}). Skipping."
else
  echo "Downloading ${TARGETS_URL} -> ${TARGETS_PATH}"
  curl -L --fail -o "${TARGETS_PATH}" "${TARGETS_URL}"
fi

# ---- Folds via huggingface_hub.snapshot_download -----------------------------
if ! python -c "import huggingface_hub" 2>/dev/null; then
  echo "ERROR: huggingface_hub Python package not available." >&2
  echo "       conda activate borzoi    (or pip install huggingface_hub) and re-run." >&2
  exit 1
fi

for f in ${FOLDS}; do
  DEST="${CACHE_DIR}/fold${f}"
  if [[ "${FORCE}" == "0" && -s "${DEST}/config.json" ]]; then
    echo "fold${f} already present (${DEST}). Skipping."
    continue
  fi
  REPO="johahi/borzoi-replicate-${f}"
  echo "Downloading ${REPO} -> ${DEST}"
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${REPO}", local_dir="${DEST}",
                  local_dir_use_symlinks=False)
PY
done

echo
echo "Done."
echo "  Cache: ${CACHE_DIR}"
echo "  Targets: ${TARGETS_PATH}"
echo "  Folds: ${FOLDS}"
echo
echo "Point Borzoi/configs/default.yaml -> paths.borzoi_weights: ${CACHE_DIR}"
