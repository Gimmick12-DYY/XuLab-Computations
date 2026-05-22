#!/bin/bash
#SBATCH --job-name=scb_cis_full
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00

# Single-step pipeline. Run scBasset and standalone cisTopic first; their
# <work>/impute/ outputs are the inputs to this denoiser.
#
# Do not use BASH_SOURCE here: SLURM runs a copy under /var/spool/slurmd/...,
# so sibling scripts must be resolved from SLURM_SUBMIT_DIR (cwd at sbatch).

set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/slurm/01_denoise.sbatch" ]]; then
  DENOISE_SBATCH="${SUBMIT_DIR}/slurm/01_denoise.sbatch"
elif [[ -f "${SUBMIT_DIR}/01_denoise.sbatch" ]]; then
  DENOISE_SBATCH="${SUBMIT_DIR}/01_denoise.sbatch"
else
  echo "ERROR: cannot find 01_denoise.sbatch (SLURM_SUBMIT_DIR=${SUBMIT_DIR})" >&2
  exit 1
fi
exec bash "${DENOISE_SBATCH}" "$@"
