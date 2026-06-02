#!/bin/bash
#SBATCH --job-name=dp_peak_mtx
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# MACS2 + bedtools peak matrix build for one TF.
#
#   TF=CTCF   sbatch data_prep/slurm/build_peak_matrix.sbatch
#   TF=ZNF282 sbatch data_prep/slurm/build_peak_matrix.sbatch

set -euo pipefail

TF="${TF:-CTCF}"
CONDA_ENV="${CONDA_ENV:-data_prep}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/data_prep" ]]; then
  ROOT_DIR="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/../data_prep" ]]; then
  ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

export CONDA_ENV
bash "${ROOT_DIR}/data_prep/run_build_peak_matrix.sh" "${TF}"
