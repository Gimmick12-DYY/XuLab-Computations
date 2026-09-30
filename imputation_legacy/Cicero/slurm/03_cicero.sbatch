#!/bin/bash
#SBATCH --job-name=cicero_03_run
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# Heavy step: run_cicero() fits a graphical-Lasso-style model in sliding
# windows across each chromosome. ~119k peaks x ~500kb windows x sample_num=100
# typically takes 6-18h on 16 CPUs and peaks around 150-220 GB.
#
# Knobs that trade speed for stability (configs/default.yaml -> cicero:):
#   sample_num: 100   -> drop to 50 for a faster run, raise to 200 for more
#                       stable estimates.
#   window_bp:  500000 -> Pliner et al. default; halving it linearly cuts time.
#   max_cells:  0     -> set 2000 for a smoke test that finishes in ~1h.

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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

Rscript scripts/check_env.R
Rscript scripts/03_run_cicero.R --config "${CFG}"
