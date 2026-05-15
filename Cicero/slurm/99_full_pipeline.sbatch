#!/bin/bash
#SBATCH --job-name=cicero_full
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=36:00:00

# End-to-end Cicero co-accessibility imputation:
#   01_export -> 02_build_cds -> 03_run_cicero -> 05_impute
#
# (00_inspect_rds is optional; set RUN_INSPECT=1 to enable.)
#
# Common overrides:
#   sbatch --export=ALL,CFG=/path/to/config.yaml slurm/99_full_pipeline.sbatch
#   sbatch --mem=384G --time=72:00:00          slurm/99_full_pipeline.sbatch
#
# After completion, the imputed matrix lives at <work>/impute/matrix_csr.npz
# and plugs into downstream/compare_pos_neg.py via:
#   --input cicero=<work>/impute

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
CONDA_ENV="${CONDA_ENV:-cicero}"
RUN_INSPECT="${RUN_INSPECT:-0}"

echo "[$(date)] Starting Cicero pipeline (01 -> 05)"
echo "ROOT_DIR=${ROOT_DIR}"
echo "CFG=${CFG}"
echo "INSPECT=${RUN_INSPECT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

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

echo "[$(date)] Step 02: build Monocle3 CDS (LSI + UMAP)"
Rscript scripts/02_build_cds.R --config "${CFG}"

echo "[$(date)] Step 03: run Cicero (co-accessibility table)"
Rscript scripts/03_run_cicero.R --config "${CFG}"

echo "[$(date)] Step 05: propagate and persist imputed matrix"
python scripts/05_impute.py --config "${CFG}"

echo "[$(date)] Cicero pipeline completed."
echo "Imputed outputs: ${WORK_DIR}/impute/"
echo "Plug into downstream comparators with: --input cicero=${WORK_DIR}/impute"
