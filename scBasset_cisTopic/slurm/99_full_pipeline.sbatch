#!/bin/bash
#SBATCH --job-name=scb_cis_full
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=36:00:00

# Cascade: scBasset -> cisTopic
#
#   01_ingest (this dir)
#   02_build_cistopic_obj   (cisTopic/scripts/, our CFG)
#   03_run_lda_mallet       (cisTopic/scripts/, our CFG)
#   04_select_model         (cisTopic/scripts/, our CFG)
#   05_impute               (cisTopic/scripts/, our CFG)
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
CISTOPIC_DIR="${REPO_ROOT}/cisTopic"

CFG="${CFG:-${ROOT_DIR}/configs/default.yaml}"
CONDA_ENV="${CONDA_ENV:-cistopic}"
MALLET_MEMORY="${MALLET_MEMORY:-200G}"

echo "[$(date)] Cascade scBasset -> cisTopic"
echo "ROOT_DIR=${ROOT_DIR}"
echo "CISTOPIC_DIR=${CISTOPIC_DIR}"
echo "CFG=${CFG}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export MALLET_MEMORY

echo "[$(date)] Step 01: ingest scBasset -> mm/"
python "${ROOT_DIR}/scripts/01_ingest_scbasset.py" --config "${CFG}"

echo "[$(date)] Step 02: build CistopicObject (cisTopic 02_build_cistopic_obj.py)"
python "${CISTOPIC_DIR}/scripts/02_build_cistopic_obj.py" --config "${CFG}"

echo "[$(date)] Step 03: LDA (cisTopic 03_run_lda_mallet.py)"
python "${CISTOPIC_DIR}/scripts/03_run_lda_mallet.py" --config "${CFG}"

echo "[$(date)] Step 04: select model (cisTopic 04_select_model.py)"
python "${CISTOPIC_DIR}/scripts/04_select_model.py" --config "${CFG}"

echo "[$(date)] Step 05: impute + persist (cisTopic 05_impute.py)"
python "${CISTOPIC_DIR}/scripts/05_impute.py" --config "${CFG}"

echo "[$(date)] Cascade done."
echo "Imputed matrix: $(python -c "import yaml; c=yaml.safe_load(open('${CFG}')); print(c['paths']['work_dir'])")/impute/matrix_csr.npz"
echo "Plug into downstream: --input scbasset_cistopic=<work>/impute"
