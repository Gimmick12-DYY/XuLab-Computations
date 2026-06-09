#!/bin/bash
#SBATCH --job-name=puscopen_full
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=18:00:00

# Run the full PUscOpen pipeline in one SLURM allocation:
#   00_inspect (optional) -> 01_export -> 02_build_mask -> 03_pu_classifier
#   -> 04_run_scopen -> 05_postfilter -> 06_impute -> 07_validate
#
# Overrides:
#   sbatch --export=ALL,CFG=/path/to/config.yaml slurm/99_full_pipeline.sbatch
#   sbatch --export=ALL,RUN_INSPECT=1,SKIP_VALIDATE=1 slurm/99_full_pipeline.sbatch

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
CONDA_ENV="${CONDA_ENV:-puscopen}"
RUN_INSPECT="${RUN_INSPECT:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"

echo "[$(date)] Starting PUscOpen full pipeline"
echo "ROOT_DIR=${ROOT_DIR}"
echo "CFG=${CFG}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
module load r/4.4.0 || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

read_yaml() {
  python3 - "$CFG" "$1" <<'PY'
import sys, yaml
cfg_path, key = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open(cfg_path)) or {}
v = c
for k in key.lstrip('.').split('.'):
    v = v[k]
print(v)
PY
}

INPUT_RDS=$(read_yaml .paths.input_rds)
WORK_DIR=$(read_yaml  .paths.work_dir)

if [[ "${RUN_INSPECT}" == "1" ]]; then
  echo "[$(date)] Step 00: inspect input RDS"
  Rscript scripts/00_inspect_rds.R --input "${INPUT_RDS}"
fi

echo "[$(date)] Step 01: export RDS -> mm"
Rscript scripts/01_export_rds_to_mm.R --input "${INPUT_RDS}" --outdir "${WORK_DIR}/mm"

echo "[$(date)] Step 02: build geometric candidate mask + features"
Rscript scripts/02_build_mask.R --config "${CFG}"

echo "[$(date)] Step 03: PU classifier (Elkan-Noto)"
python scripts/03_pu_classifier.py --config "${CFG}"

echo "[$(date)] Step 04: scOpen on refined mask"
python scripts/04_run_scopen.py --config "${CFG}"

echo "[$(date)] Step 05: neighborhood post-filter"
python scripts/05_postfilter.py --config "${CFG}"

echo "[$(date)] Step 06: persist final imputed matrix"
python scripts/06_impute.py --config "${CFG}"

if [[ "${SKIP_VALIDATE}" != "1" ]]; then
  echo "[$(date)] Step 07: held-out validation"
  python scripts/07_validate.py --config "${CFG}"
fi

echo "[$(date)] PUscOpen pipeline completed."
echo "Imputed outputs are under ${WORK_DIR}/impute/"
echo "Add to cross-pipeline comparison with:"
echo "   downstream/compare_pos_neg.py --input PUscOpen=${WORK_DIR}/impute"
echo "   downstream/bin_sensitivity_specificity.py --input PUscOpen=${WORK_DIR}/impute"
