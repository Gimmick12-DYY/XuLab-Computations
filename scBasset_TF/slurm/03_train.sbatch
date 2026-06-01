#!/bin/bash
#SBATCH --job-name=scbtf_03_train
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=a100-gpu,l40-gpu,gpu,volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

# scBasset_TF training on any Longleaf GPU partition (A100, L40, generic gpu, V100).
# Common overrides:
#   sbatch --time=48:00:00 --mem=256G --export=ALL,CFG=/path/to/big.yaml slurm/03_train.sbatch

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
CONDA_ENV="${CONDA_ENV:-scbasset_tf}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Cluster-level CUDA, if present, takes precedence over pip wheels.
if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
  module load cuda || true
  module load cudnn || true
fi

python -c "import tensorflow as tf; print('TF', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))"

python scripts/03_train.py --config "${CFG}"
