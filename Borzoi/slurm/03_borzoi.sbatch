#!/bin/bash
#SBATCH --job-name=brz_03_predict
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

# Borzoi inference on Longleaf's volta-gpu partition (Tesla V100 16 GB).
# Use fp16 (default in configs/default.yaml) to fit Borzoi on a 16 GB V100.
# Common overrides:
#   sbatch --time=48:00:00 --export=ALL,CFG=/path/to/big.yaml slurm/03_borzoi.sbatch

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
CONDA_ENV="${CONDA_ENV:-borzoi}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
  module load cuda || true
  module load cudnn || true
fi

python -c "import torch; print('Torch', torch.__version__, 'CUDA', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

python scripts/03_borzoi_predict.py --config "${CFG}"
