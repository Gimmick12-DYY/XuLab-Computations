#!/bin/bash
#SBATCH --job-name=ds_bin_posneg_bl
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# Baseline overlap: per-mm-row "any signal" + total stored nnz; bin-rows where
# imputed CSR has signal but raw row was all-zero (see JSON imputed_vs_raw).
# Runs downstream/bin_pos_neg_baseline_overlap.py.
#
# Default INPUTS: cisTopic, PUscOpen, cicero, scbasset, scbasset_puscopen, scbasset_cistopic.
# Override INPUTS or individual *_DIR before sbatch.
#
# Optional env vars:
#   ROOT_DIR, MM_DIR, BINS_TSV, OUT_JSON, CONDA_ENV, INPUTS

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
    ROOT_DIR="${SUBMIT_DIR}"
  elif [[ -f "${SUBMIT_DIR}/bin_pos_neg_baseline_overlap.py" ]]; then
    ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"

DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

BINS_TSV="${BINS_TSV:-${ROOT_DIR}/downstream/bins/pos_neg_bins.tsv}"
OUT_JSON="${OUT_JSON:-${ROOT_DIR}/downstream/bins/baseline_pos_neg_overlap.json}"
CONDA_ENV="${CONDA_ENV:-cistopic}"

# shellcheck source=_pipeline_paths.sh
source "${DOWN_DIR}/slurm/_pipeline_paths.sh"
if [[ -z "${INPUTS:-}" ]]; then
  pipeline_paths_native
  INPUTS="cisTopic=${CISTOPIC_DIR} PUscOpen=${PUSCOPEN_DIR} cicero=${CICERO_DIR} scbasset=${SCBASSET_DIR} scbasset_puscopen=${SCBASSET_PUSCOPEN_DIR} scbasset_cistopic=${SCBASSET_CISTOPIC_DIR}"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(
  python "${DOWN_DIR}/bin_pos_neg_baseline_overlap.py"
  --mm-dir "${MM_DIR}"
  --bins-tsv "${BINS_TSV}"
  --out-json "${OUT_JSON}"
)
for pair in ${INPUTS}; do
  CMD+=(--input "${pair}")
done
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "[$(date)] Done. JSON: ${OUT_JSON}"
