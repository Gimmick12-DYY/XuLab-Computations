#!/bin/bash
#SBATCH --job-name=scb_pu_full
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=36:00:00

# Cascade: scBasset -> PUscOpen
#
#   01_ingest (this dir)
#   02_build_mask.R         (PUscOpen/scripts/, our CFG)
#   03_pu_classifier.py     (PUscOpen/scripts/, our CFG)
#   04_run_scopen.py        (PUscOpen/scripts/, our CFG)
#   05_postfilter.py        (PUscOpen/scripts/, our CFG)
#   06_impute.py            (PUscOpen/scripts/, our CFG)
#
# Outputs land at <work>/impute/{matrix_csr.npz, regions.tsv, barcodes.tsv, meta.json}.
#
# Overrides:
#   sbatch --export=ALL,CFG=/path/to/config.yaml slurm/99_full_pipeline.sbatch
#   sbatch --mem=256G --time=48:00:00            slurm/99_full_pipeline.sbatch

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/scripts" && -d "${SUBMIT_DIR}/configs" ]]; then
  ROOT_DIR="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/../scripts" && -d "${SUBMIT_DIR}/../configs" ]]; then
  ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${ROOT_DIR}"

REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
PUSCOPEN_DIR="${REPO_ROOT}/PUscOpen"

CFG="${CFG:-${ROOT_DIR}/configs/default.yaml}"
CONDA_ENV="${CONDA_ENV:-puscopen}"

echo "[$(date)] Cascade scBasset -> PUscOpen"
echo "ROOT_DIR=${ROOT_DIR}"
echo "PUSCOPEN_DIR=${PUSCOPEN_DIR}"
echo "CFG=${CFG}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "[$(date)] Step 01: ingest scBasset -> mm/"
python "${ROOT_DIR}/scripts/01_ingest_scbasset.py" --config "${CFG}"

echo "[$(date)] Step 02: geometric mask (PUscOpen 02_build_mask.R)"
Rscript "${PUSCOPEN_DIR}/scripts/02_build_mask.R" --config "${CFG}"

echo "[$(date)] Step 03: PU classifier (PUscOpen 03_pu_classifier.py)"
python "${PUSCOPEN_DIR}/scripts/03_pu_classifier.py" --config "${CFG}"

echo "[$(date)] Step 04: scOpen (PUscOpen 04_run_scopen.py)"
python "${PUSCOPEN_DIR}/scripts/04_run_scopen.py" --config "${CFG}"

echo "[$(date)] Step 05: postfilter (PUscOpen 05_postfilter.py)"
python "${PUSCOPEN_DIR}/scripts/05_postfilter.py" --config "${CFG}"

echo "[$(date)] Step 06: persist impute (PUscOpen 06_impute.py)"
python "${PUSCOPEN_DIR}/scripts/06_impute.py" --config "${CFG}"

echo "[$(date)] Cascade done."
echo "Imputed matrix: $(python -c "import yaml; c=yaml.safe_load(open('${CFG}')); print(c['paths']['work_dir'])")/impute/matrix_csr.npz"
echo "Plug into downstream: --input scbasset_puscopen=<work>/impute"
