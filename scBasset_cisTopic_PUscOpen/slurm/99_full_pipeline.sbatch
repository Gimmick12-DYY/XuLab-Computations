#!/bin/bash
#SBATCH --job-name=scb_cis_pu_full
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00

# Single-step pipeline. Both scBasset_cisTopic and scBasset_PUscOpen must
# have finished first.

set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/slurm/01_union.sbatch" ]]; then
  UNION_SBATCH="${SUBMIT_DIR}/slurm/01_union.sbatch"
elif [[ -f "${SUBMIT_DIR}/01_union.sbatch" ]]; then
  UNION_SBATCH="${SUBMIT_DIR}/01_union.sbatch"
else
  echo "ERROR: cannot find 01_union.sbatch (SLURM_SUBMIT_DIR=${SUBMIT_DIR})" >&2
  exit 1
fi
exec bash "${UNION_SBATCH}" "$@"
