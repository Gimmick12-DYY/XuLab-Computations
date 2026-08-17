#!/bin/bash
#SBATCH --job-name=bulk_bedcov
#SBATCH --output=bulk_bedcov-%A_%a.out
#SBATCH --error=bulk_bedcov-%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-6
#SBATCH --mail-type=FAIL

# Precompute genome-wide bulk bedcov for the 7-TF panel (parallel array).
# Then run: sbatch --dependency=afterok:<this_jobid> downstream/slurm/plot_vs_bulk_spearman.sbatch

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
CONDA_ENV="${CONDA_ENV:-/nas/longleaf/home/dyy12/.conda/envs/cistopic}"
PY="${CONDA_ENV}/bin/python"
TFS=(ctcf maz zic2 zbtb7a znf777 nfya znf282)

cd "${ROOT_DIR}"
export PATH="/nas/longleaf/rhel9/apps/samtools/1.24/bin:${PATH:-}"

TF="${TFS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} tf=${TF}"
"${PY}" downstream/precompute_bulk_bedcov.py --tf "${TF}"
echo "[$(date)] done ${TF}"
