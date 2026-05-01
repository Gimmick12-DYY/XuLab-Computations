#!/bin/bash
#SBATCH --job-name=cistopic_full
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00

# Run the full cisTopic pipeline in one SLURM job allocation.
#
# Default order:
#   00_inspect (optional) -> 01_export -> 02_build -> 03_lda
#   -> 04_select -> 05_impute -> 06_downstream (optional)
#   -> 07_eval (optional) -> 08_compare (optional)
#
# Usage:
#   sbatch slurm/99_full_pipeline.sbatch
#
# Common overrides:
#   sbatch --export=ALL,CFG=/path/to/config.yaml slurm/99_full_pipeline.sbatch
#   sbatch --export=ALL,RUN_DOWNSTREAM=0,RUN_EVAL=0,RUN_COMPARE=0 slurm/99_full_pipeline.sbatch
#   sbatch --export=ALL,REGIONS_BED=/path/to/peaks.bed slurm/99_full_pipeline.sbatch

set -euo pipefail

# Resolve project root from where `sbatch` was invoked (SLURM_SUBMIT_DIR),
# not from BASH_SOURCE (which points to /var/spool/slurmd on compute nodes).
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

# Step toggles
RUN_INSPECT="${RUN_INSPECT:-0}"
RUN_DOWNSTREAM="${RUN_DOWNSTREAM:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"

# Optional compare/eval knobs
REGIONS_BED="${REGIONS_BED:-}"
IMPUTED_H5="${IMPUTED_H5:-}"
MM_DIR="${MM_DIR:-}"
N_SAMPLES="${N_SAMPLES:-100000}"
SEED="${SEED:-42}"
OUT_NAME="${OUT_NAME:-heldout_eval}"

echo "[$(date)] Starting full pipeline"
echo "ROOT_DIR=${ROOT_DIR}"
echo "CFG=${CFG}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
module load r/4.4.0 || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export MALLET_MEMORY

if [[ "${RUN_INSPECT}" == "1" ]]; then
  echo "[$(date)] Step 00: inspect input RDS"
  python - <<'PY'
print("Step 00 uses Rscript scripts/00_inspect_rds.R")
PY
  INPUT_RDS="$(python - <<PY
import yaml
with open("${CFG}") as fh:
    c = yaml.safe_load(fh)
print(c["paths"]["input_rds"])
PY
)"
  Rscript scripts/00_inspect_rds.R --input "${INPUT_RDS}"
fi

echo "[$(date)] Step 01: export RDS -> mm"
python - <<'PY'
print("Step 01 uses Rscript scripts/01_export_rds_to_mm.R")
PY
INPUT_RDS="$(python - <<PY
import yaml
with open("${CFG}") as fh:
    c = yaml.safe_load(fh)
print(c["paths"]["input_rds"])
PY
)"
WORK_DIR="$(python - <<PY
import yaml
with open("${CFG}") as fh:
    c = yaml.safe_load(fh)
print(c["paths"]["work_dir"])
PY
)"
Rscript scripts/01_export_rds_to_mm.R --input "${INPUT_RDS}" --outdir "${WORK_DIR}/mm"

echo "[$(date)] Step 02: build cisTopic object"
python scripts/02_build_cistopic_obj.py --config "${CFG}"

echo "[$(date)] Step 03: run LDA (MALLET)"
python scripts/03_run_lda_mallet.py --config "${CFG}"

echo "[$(date)] Step 04: select best model"
python scripts/04_select_model.py --config "${CFG}"

echo "[$(date)] Step 05: impute accessibility"
python scripts/05_impute.py --config "${CFG}"

if [[ "${RUN_DOWNSTREAM}" == "1" ]]; then
  echo "[$(date)] Step 06: downstream"
  python scripts/06_downstream.py --config "${CFG}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "[$(date)] Step 07: heldout-style reconstruction eval"
  python scripts/07_eval_heldout.py \
    --config "${CFG}" \
    --n-samples "${N_SAMPLES}" \
    --seed "${SEED}" \
    --out-name "${OUT_NAME}"
fi

if [[ "${RUN_COMPARE}" == "1" ]]; then
  echo "[$(date)] Step 08: compare imputed vs original matrix"
  CMP_CMD=(
    python scripts/08_compare_imputed.py
    --config "${CFG}"
  )
  if [[ -n "${IMPUTED_H5}" ]]; then CMP_CMD+=(--imputed-h5 "${IMPUTED_H5}"); fi
  if [[ -n "${MM_DIR}" ]]; then CMP_CMD+=(--mm-dir "${MM_DIR}"); fi
  if [[ -n "${REGIONS_BED}" ]]; then CMP_CMD+=(--regions-bed "${REGIONS_BED}"); fi
  echo "Running: ${CMP_CMD[*]}"
  "${CMP_CMD[@]}"
fi

echo "[$(date)] Full pipeline completed."
