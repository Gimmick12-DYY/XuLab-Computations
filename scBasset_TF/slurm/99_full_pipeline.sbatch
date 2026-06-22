#!/bin/bash
#SBATCH --job-name=scbtf_full
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=a100-gpu,l40-gpu,gpu,volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

# End-to-end scBasset:
#   01_export -> 02_prepare_seqs -> 03_train (GPU) -> 04_predict (GPU) -> 05_impute
#
# (00_inspect optional; set RUN_INSPECT=1.)
#
# Imputed matrix lands at <work>/impute/matrix_csr.npz and plugs into
# downstream/compare_pos_neg.py via --input scbasset_tf=<work>/impute.

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
CONDA_ENV="${CONDA_ENV:-scbasset}"
RUN_INSPECT="${RUN_INSPECT:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
  module unload r 2>/dev/null || true   # don't shadow conda R in step 01
  module load cuda || true
  module load cudnn || true
fi

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

echo "[$(date)] Step 02: extract DNA sequences from hg38"
python scripts/02_prepare_seqs.py --config "${CFG}"

echo "[$(date)] Step 03: train scBasset"
python -c "import tensorflow as tf; print('TF', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))"
python scripts/03_train.py --config "${CFG}"

echo "[$(date)] Step 04: predict on every bin"
python scripts/04_predict.py --config "${CFG}"

echo "[$(date)] Step 05: persist imputed matrix"
python scripts/05_impute.py --config "${CFG}"

echo "[$(date)] scBasset pipeline completed."
echo "Imputed outputs: ${WORK_DIR}/impute/"
echo "Plug into downstream comparators with: --input scbasset_tf=${WORK_DIR}/impute"
echo "BigWig for IGV: INPUT_DIR=${WORK_DIR}/impute OUTPUT_BW=${WORK_DIR}/bigwig/scbasset_tf_ctcf_mean.bw \\"
echo "  AGGREGATE=mean NORMALIZE=p99 sbatch ${ROOT_DIR}/../downstream/slurm/imputed_to_bigwig.sbatch"
