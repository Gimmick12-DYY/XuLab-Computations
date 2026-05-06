#!/bin/bash
#SBATCH --job-name=ds_compare_posneg
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Compare per-bin signal at CTCF positive vs negative bins across multiple
# input matrices (raw mm + imputed HDF5 from each pipeline).
#
# Required env vars:
#   BINS_TSV      output of prepare_pos_neg_bins.R
#   OUT_DIR       output directory for plots/JSON/TSV
#   INPUTS        space-separated LABEL=PATH pairs, e.g.
#                 INPUTS="raw=/work/.../mm cisTopic=/work/.../impute/x.h5 FITS=/work/.../impute/y.h5"
# Optional env vars:
#   CONDA_ENV     conda env (default: cistopic; FITS env works too)
#   OUT_NAME      basename for output files (default: compare_pos_neg)

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
  ROOT_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/compare_pos_neg.py" ]]; then
  ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

if [[ -z "${BINS_TSV:-}" || -z "${OUT_DIR:-}" || -z "${INPUTS:-}" ]]; then
  echo "Required: BINS_TSV=<tsv>, OUT_DIR=<dir>, INPUTS=\"label1=path1 label2=path2 ...\"" >&2
  exit 1
fi
CONDA_ENV="${CONDA_ENV:-cistopic}"
OUT_NAME="${OUT_NAME:-compare_pos_neg}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(
  python "${DOWN_DIR}/compare_pos_neg.py"
  --bins-tsv "${BINS_TSV}"
  --out-dir  "${OUT_DIR}"
  --out-name "${OUT_NAME}"
)
for pair in ${INPUTS}; do
  CMD+=(--input "${pair}")
done

echo "Running: ${CMD[*]}"
"${CMD[@]}"
