#!/bin/bash
#SBATCH --job-name=cistopic_full
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Run the cisTopic imputation pipeline end-to-end:
#   00_inspect (optional) -> 01_export -> 02_build -> 03_lda
#   -> 04_select -> 05_impute
#
# Post-imputation analyses (06-11) are intentionally NOT run here -- the
# universal downstream/compare_pos_neg.py + downstream/slurm/* scripts handle
# cross-pipeline comparison from the new <work>/impute/ outputs.
#
# Common overrides:
#   sbatch --export=ALL,CFG=/path/to/config.yaml slurm/99_full_pipeline.sbatch
#   sbatch --export=ALL,RUN_INSPECT=1 slurm/99_full_pipeline.sbatch
#   sbatch --mem=256G --time=96:00:00 slurm/99_full_pipeline.sbatch

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

CFG="${CFG:-${ROOT_DIR}/configs/default.yaml}"
CONDA_ENV="${CONDA_ENV:-cistopic}"
MALLET_MEMORY="${MALLET_MEMORY:-200G}"
RUN_INSPECT="${RUN_INSPECT:-0}"

echo "[$(date)] Starting cisTopic imputation pipeline (00 -> 05)"
echo "ROOT_DIR=${ROOT_DIR}"
echo "CFG=${CFG}"
echo "INSPECT=${RUN_INSPECT}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export MALLET_MEMORY

read_yaml() {
  python - <<PY
import yaml
with open("${CFG}") as fh: c = yaml.safe_load(fh)
v = c
for k in "$1".lstrip('.').split('.'): v = v[k]
print(v)
PY
}

INPUT_RDS="$(read_yaml .paths.input_rds)"
WORK_DIR="$(read_yaml  .paths.work_dir)"

if [[ "${RUN_INSPECT}" == "1" ]]; then
  echo "[$(date)] Step 00: inspect input RDS"
  Rscript scripts/00_inspect_rds.R --input "${INPUT_RDS}"
fi

echo "[$(date)] Step 01: export RDS -> mm"
Rscript scripts/01_export_rds_to_mm.R --input "${INPUT_RDS}" --outdir "${WORK_DIR}/mm"

echo "[$(date)] Step 02: build cisTopic object"
python scripts/02_build_cistopic_obj.py --config "${CFG}"

echo "[$(date)] Step 03: run LDA (MALLET)"
python scripts/03_run_lda_mallet.py --config "${CFG}"

echo "[$(date)] Step 04: select best model"
python scripts/04_select_model.py --config "${CFG}"

echo "[$(date)] Step 05: persist factored output (factors.npz + regions.tsv + barcodes.tsv + meta.json)"
python scripts/05_impute.py --config "${CFG}"

echo "[$(date)] cisTopic imputation pipeline completed."
echo "Imputed outputs are under ${WORK_DIR}/impute/"
echo "Run downstream/slurm/compare_pos_neg_all4.sbatch for cross-pipeline comparison."
