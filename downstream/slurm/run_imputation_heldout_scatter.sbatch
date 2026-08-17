#!/bin/bash
#SBATCH --job-name=imp_heldout_scatter
#SBATCH --output=imp_heldout_scatter-%j.out
#SBATCH --error=imp_heldout_scatter-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

# Score held-out (original, imputed) pairs with the unified model, then plot
# tutorial-style scatters (Original vs Imputed, y=x, Pearson r).
#
#   sbatch downstream/slurm/run_imputation_heldout_scatter.sbatch
#   TFS="znf282 maz" sbatch ...

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
  ROOT_DIR="${SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-borzoi}"
conda activate "${CONDA_ENV}"
if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
  module load cuda 2>/dev/null || true
fi

TFS="${TFS:-ctcf maz nfya zbtb7a zic2 znf768 znf777 znf282}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/downstream/plots/imputation_heldout}"
PY="${PY:-python}"

echo "[$(date)] score with ${CONDA_ENV} ($(command -v ${PY}))"
${PY} downstream/imputation_heldout_scatter.py --score \
  --tfs ${TFS} \
  --out-dir "${OUT_DIR}" \
  --n-pos "${N_POS:-25000}" \
  --n-neg "${N_NEG:-25000}" \
  --device cuda

# Plot with whatever has matplotlib (borzoi may lack it)
if ${PY} -c 'import matplotlib' 2>/dev/null; then
  PLOT_PY="${PY}"
else
  PLOT_PY="python3"
fi
echo "[$(date)] plot with ${PLOT_PY}"
${PLOT_PY} downstream/imputation_heldout_scatter.py --plot \
  --tfs ${TFS} \
  --out-dir "${OUT_DIR}"

echo "[$(date)] done -> ${OUT_DIR}"
