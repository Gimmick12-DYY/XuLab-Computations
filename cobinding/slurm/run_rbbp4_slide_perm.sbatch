#!/bin/bash
#SBATCH --job-name=rbbp4_perm
#SBATCH --output=/work/users/d/y/dyy12/XuLab/cobinding/logs/rbbp4_perm-%j.out
#SBATCH --error=/work/users/d/y/dyy12/XuLab/cobinding/logs/rbbp4_perm-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#
# Permutation and >25 kb columns for the RBBP4 slide.
# Wang gold pairs and the Our Run (k=50, cap 10,000, seed 555, 1 Mb).
# Same-chromosome null, 500 draws. Runs on a compute node.

set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
PY_ENV="/work/users/d/y/dyy12/conda/envs/cobinding"
mkdir -p "${XULAB}/cobinding/logs"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PY_ENV}"
python "${XULAB}/hic/scripts/fill_slide_loop_table.py" --n-perm 500 --seed 1
echo "[$(date)] done"
