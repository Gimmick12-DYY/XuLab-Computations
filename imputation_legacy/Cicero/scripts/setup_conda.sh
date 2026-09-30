#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# setup_conda.sh — create/update the cicero conda env, then install cicero from GitHub.
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${MODULEPATH:-}" ]] || type module &>/dev/null; then
  module unload r 2>/dev/null || true
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

export R_INSTALL_STAGED=false

RECREATE="${RECREATE:-0}"
if [[ "${RECREATE}" == "1" ]]; then
  echo "[setup_conda] Removing env cicero (RECREATE=1)"
  conda env remove -n cicero -y 2>/dev/null || conda env remove -n cicero || true
fi

if conda env list | awk '{print $1}' | grep -qx cicero; then
  echo "[setup_conda] Updating env cicero (pinned R 4.4 + bioconda strict)"
  if ! conda env update -n cicero -f environment.yml --prune; then
    echo "[setup_conda] conda env update failed."
    echo "  If you saw GenomeInfoDbData / sub-architecture 'R' errors, recreate:"
    echo "    RECREATE=1 bash scripts/setup_conda.sh"
    exit 1
  fi
else
  echo "[setup_conda] Creating env cicero"
  conda env create -f environment.yml
fi

conda activate cicero
echo "[setup_conda] R: $(which Rscript)"
Rscript -e 'cat("R.version=", as.character(getRversion()), "\n")'

# Belt-and-suspenders: cicero needs glasso + VGAM; monocle3 does not pull them in.
if ! conda list -n cicero r-glasso 2>/dev/null | grep -q r-glasso; then
  conda install -n cicero -c conda-forge r-glasso r-vgam
fi

Rscript scripts/setup_r.R
Rscript scripts/check_env.R
