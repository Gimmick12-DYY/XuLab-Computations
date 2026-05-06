#!/bin/bash
#SBATCH --job-name=ds_prepare_bins
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Build positive/negative CTCF bin indices from SCREEN cCRE annotations.
#
# Required env vars:
#   MM_DIR        directory with regions.tsv.gz (e.g. <work>/mm)
#   OUT_DIR       output dir for pos_neg_bins.tsv + meta.json
# Optional env vars:
#   CACHE_DIR     where to download cCRE/blacklist files (default: $OUT_DIR/cache)
#   N_TOP         top-N CTCF cCREs by score (default: 10000)
#   SEED          RNG seed for negative sampling (default: 2026)
#   CONDA_ENV     conda env with R + Bioconductor (default: cistopic; FITS env works too)

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
  ROOT_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/prepare_pos_neg_bins.R" ]]; then
  ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

if [[ -z "${MM_DIR:-}" || -z "${OUT_DIR:-}" ]]; then
  echo "Required: MM_DIR=<mm/>, OUT_DIR=<output dir>" >&2
  exit 1
fi
CACHE_DIR="${CACHE_DIR:-${OUT_DIR}/cache}"
N_TOP="${N_TOP:-10000}"
SEED="${SEED:-2026}"
CONDA_ENV="${CONDA_ENV:-cistopic}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
module load r/4.4.0 || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

Rscript "${DOWN_DIR}/prepare_pos_neg_bins.R" \
  --mm-dir   "${MM_DIR}" \
  --out-dir  "${OUT_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --n-top    "${N_TOP}" \
  --seed     "${SEED}"
