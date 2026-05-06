#!/bin/bash
#SBATCH --job-name=fits_full
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00

# Run the full FITS pipeline in one SLURM allocation. Mirrors cisTopic's
# 99_full_pipeline.sbatch step toggles so the two pipelines have the same
# operational shape.
#
# Default order:
#   00_inspect (optional) -> 01_export -> 02_prepare -> 03_phase1
#   -> 04_phase2 -> 05_impute -> 06_downstream (optional)
#   -> 07_eval (optional) -> 08_compare (optional)
#   -> 09_visualize (optional) -> 10_nonzero (optional)
#   -> 11_fragmentation (optional)
#
# Common overrides:
#   sbatch --export=ALL,CFG=/path/to/config.yaml slurm/99_full_pipeline.sbatch
#   sbatch --export=ALL,RUN_DOWNSTREAM=0,RUN_EVAL=0,RUN_COMPARE=0 slurm/99_full_pipeline.sbatch
#   sbatch --export=ALL,REGIONS_BED=/path/to/peaks.bed slurm/99_full_pipeline.sbatch

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
CONDA_ENV="${CONDA_ENV:-fits}"

# Step toggles
RUN_INSPECT="${RUN_INSPECT:-0}"
RUN_DOWNSTREAM="${RUN_DOWNSTREAM:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"
RUN_VISUALIZE="${RUN_VISUALIZE:-1}"
RUN_NONZERO="${RUN_NONZERO:-1}"
RUN_FRAGMENTATION="${RUN_FRAGMENTATION:-1}"

# Optional eval/compare knobs
REGIONS_BED="${REGIONS_BED:-}"
IMPUTED_H5="${IMPUTED_H5:-}"
MM_DIR="${MM_DIR:-}"
COMPARE_SPACE="${COMPARE_SPACE:-counts}"
EXPORT_FILLED_MTX="${EXPORT_FILLED_MTX:-0}"
FILL_THRESHOLD="${FILL_THRESHOLD:-0.0}"
N_SAMPLES="${N_SAMPLES:-100000}"
SEED="${SEED:-42}"
OUT_NAME="${OUT_NAME:-heldout_eval}"
SAMPLE_SPACE="${SAMPLE_SPACE:-modeled}"

# Step 09 (heatmap QC)
N_REGIONS="${N_REGIONS:-1000}"
N_CELLS="${N_CELLS:-5000}"
SPACE="${SPACE:-full}"
REGION_MODE="${REGION_MODE:-original_top_cells}"
VIZ_OUT_NAME="${VIZ_OUT_NAME:-visualize_imputation}"
NOCLUSTER="${NOCLUSTER:-0}"

# Steps 10/11
NONZERO_THRESHOLDS="${NONZERO_THRESHOLDS:-0,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4}"
NONZERO_CHUNK_ROWS="${NONZERO_CHUNK_ROWS:-2000}"
NONZERO_OUT_NAME="${NONZERO_OUT_NAME:-nonzero_complexity}"
FRAG_THRESHOLDS="${FRAG_THRESHOLDS:-1e-7,1e-6,3e-6,1e-5,3e-5,1e-4}"
FRAG_CHUNK_ROWS="${FRAG_CHUNK_ROWS:-10000}"
FRAG_OUT_NAME="${FRAG_OUT_NAME:-per_cell_fragmentation}"

echo "[$(date)] Starting FITS full pipeline"
echo "ROOT_DIR=${ROOT_DIR}"
echo "CFG=${CFG}"
echo "Steps: DOWNSTREAM=${RUN_DOWNSTREAM} EVAL=${RUN_EVAL} COMPARE=${RUN_COMPARE} VIZ=${RUN_VISUALIZE} NONZERO=${RUN_NONZERO} FRAG=${RUN_FRAGMENTATION} INSPECT=${RUN_INSPECT}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
module load r/4.4.0 || true

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

echo "[$(date)] Step 02: prepare FITS input"
python scripts/02_prepare_fits_input.py --config "${CFG}"

echo "[$(date)] Step 03: FITS Phase1"
python scripts/03_run_fits_phase1.py --config "${CFG}"

echo "[$(date)] Step 04: FITS Phase2"
python scripts/04_run_fits_phase2.py --config "${CFG}"

echo "[$(date)] Step 05: wrap to HDF5"
python scripts/05_impute.py --config "${CFG}"

if [[ "${RUN_DOWNSTREAM}" == "1" ]]; then
  echo "[$(date)] Step 06: downstream summary"
  python scripts/06_downstream.py --config "${CFG}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "[$(date)] Step 07: heldout eval"
  python scripts/07_eval_heldout.py \
    --config "${CFG}" \
    --n-samples "${N_SAMPLES}" \
    --seed "${SEED}" \
    --out-name "${OUT_NAME}" \
    --sample-space "${SAMPLE_SPACE}"
fi

if [[ "${RUN_COMPARE}" == "1" ]]; then
  echo "[$(date)] Step 08: compare imputed vs original"
  CMP_CMD=(
    python scripts/08_compare_imputed.py
    --config "${CFG}"
    --compare-space "${COMPARE_SPACE}"
  )
  if [[ -n "${IMPUTED_H5}" ]];  then CMP_CMD+=(--imputed-h5 "${IMPUTED_H5}"); fi
  if [[ -n "${MM_DIR}" ]];      then CMP_CMD+=(--mm-dir "${MM_DIR}"); fi
  if [[ -n "${REGIONS_BED}" ]]; then CMP_CMD+=(--regions-bed "${REGIONS_BED}"); fi
  if [[ "${EXPORT_FILLED_MTX}" == "1" ]]; then CMP_CMD+=(--export-filled-mtx); fi
  if [[ -n "${FILL_THRESHOLD}" ]]; then CMP_CMD+=(--fill-threshold "${FILL_THRESHOLD}"); fi
  echo "Running: ${CMP_CMD[*]}"
  "${CMP_CMD[@]}"
fi

if [[ "${RUN_VISUALIZE}" == "1" ]]; then
  echo "[$(date)] Step 09: visualize imputation"
  VIZ_CMD=(
    python scripts/09_visualize_imputation.py
    --config "${CFG}"
    --n-regions "${N_REGIONS}"
    --n-cells "${N_CELLS}"
    --space "${SPACE}"
    --region-mode "${REGION_MODE}"
    --seed "${SEED}"
    --out-name "${VIZ_OUT_NAME}"
  )
  if [[ -n "${IMPUTED_H5}" ]]; then VIZ_CMD+=(--imputed-h5 "${IMPUTED_H5}"); fi
  if [[ -n "${MM_DIR}" ]];     then VIZ_CMD+=(--mm-dir     "${MM_DIR}");     fi
  if [[ "${NOCLUSTER}" == "1" ]]; then VIZ_CMD+=(--no-cluster); fi
  echo "Running: ${VIZ_CMD[*]}"
  "${VIZ_CMD[@]}"
fi

if [[ "${RUN_NONZERO}" == "1" ]]; then
  echo "[$(date)] Step 10: nonzero complexity vs thresholds"
  NZ_CMD=(
    python scripts/10_nonzero_complexity.py
    --config "${CFG}"
    --thresholds "${NONZERO_THRESHOLDS}"
    --chunk-rows "${NONZERO_CHUNK_ROWS}"
    --out-name "${NONZERO_OUT_NAME}"
  )
  if [[ -n "${IMPUTED_H5}" ]]; then NZ_CMD+=(--imputed-h5 "${IMPUTED_H5}"); fi
  if [[ -n "${MM_DIR}" ]];     then NZ_CMD+=(--mm-dir     "${MM_DIR}");     fi
  echo "Running: ${NZ_CMD[*]}"
  "${NZ_CMD[@]}"
fi

if [[ "${RUN_FRAGMENTATION}" == "1" ]]; then
  echo "[$(date)] Step 11: per-cell fragmentation"
  FR_CMD=(
    python scripts/11_per_cell_fragmentation.py
    --config "${CFG}"
    --thresholds "${FRAG_THRESHOLDS}"
    --chunk-rows "${FRAG_CHUNK_ROWS}"
    --out-name "${FRAG_OUT_NAME}"
  )
  if [[ -n "${IMPUTED_H5}" ]]; then FR_CMD+=(--imputed-h5 "${IMPUTED_H5}"); fi
  if [[ -n "${MM_DIR}" ]];     then FR_CMD+=(--mm-dir     "${MM_DIR}");     fi
  echo "Running: ${FR_CMD[*]}"
  "${FR_CMD[@]}"
fi

echo "[$(date)] FITS full pipeline completed."
