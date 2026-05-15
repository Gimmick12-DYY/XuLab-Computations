#!/bin/bash
#SBATCH --job-name=ds_cell_kgraph
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Cell-graph kNN propagation: for every cell c, pull raw signal from c's k
# nearest cell-graph neighbours to fill in bins where c is currently zero but
# its neighbours have signal. Complementary axis to the bin-axis genomic-window
# propagation that lives inside cisTopic 05_impute.py and PUscOpen 06_impute.py.
#
# Optional env vars (defaults shown):
#   ROOT_DIR              repo root (auto-detect from submit dir,
#                         falls back to /work/users/d/y/dyy12/XuLab)
#   MM_DIR                raw matrix dir
#                         (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm)
#   EMBEDDINGS            cell embeddings (default: scOpen factors.npz directory)
#   EMBEDDINGS_BARCODES   optional barcodes file aligned to EMBEDDINGS rows
#   OUT_DIR               output directory
#                         (default: ${ROOT_DIR}/downstream/cell_kgraph)
#   OUT_NAME              output basename (default cell_kgraph)
#   K                     number of neighbours per cell (default 20)
#   MIN_NEIGHBOURS        # neighbours with raw>0 required to propagate (default 3)
#   WEIGHT                multiplier on propagated value (default 1.0)
#   METRIC                cosine | euclidean | manhattan (default cosine)
#   CONDA_ENV             conda env (default cistopic; scopen/magic envs work too)

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
    ROOT_DIR="${SUBMIT_DIR}"
  elif [[ -f "${SUBMIT_DIR}/cell_graph_kneighbour.py" ]]; then
    ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"

DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

MM_DIR="${MM_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm}"
EMBEDDINGS="${EMBEDDINGS:-${ROOT_DIR}/scOpen/work/ctcf/impute}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/downstream/cell_kgraph}"
OUT_NAME="${OUT_NAME:-cell_kgraph}"
K="${K:-20}"
MIN_NEIGHBOURS="${MIN_NEIGHBOURS:-3}"
WEIGHT="${WEIGHT:-1.0}"
METRIC="${METRIC:-cosine}"
CONDA_ENV="${CONDA_ENV:-cistopic}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(
  python "${DOWN_DIR}/cell_graph_kneighbour.py"
  --mm-dir         "${MM_DIR}"
  --embeddings     "${EMBEDDINGS}"
  --out-dir        "${OUT_DIR}"
  --out-name       "${OUT_NAME}"
  --k              "${K}"
  --min-neighbours "${MIN_NEIGHBOURS}"
  --weight         "${WEIGHT}"
  --metric         "${METRIC}"
)
if [[ -n "${EMBEDDINGS_BARCODES:-}" ]]; then
  CMD+=(--embeddings-barcodes "${EMBEDDINGS_BARCODES}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo
echo "Output: ${OUT_DIR}/matrix_csr.npz (full mm shape, raw + propagated)"
echo "Add to comparisons via:"
echo "  --input cell_kgraph=${OUT_DIR}"
