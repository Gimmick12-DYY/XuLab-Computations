#!/bin/bash
#SBATCH -J g1v3_prep
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/gate1/v3_binned/logs/prep_%A_%a.out
# per-TF cell x unit matrices for the four v3 unit types (array over tfs.txt)
set -euo pipefail
V3=/work/users/d/y/dyy12/XuLab/regulatome/gate1/v3_binned
TF=${TF:-$(sed -n "${SLURM_ARRAY_TASK_ID}p" /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2/tfs.txt)}
[ -s $V3/cellmats/$TF.diag.json ] && { echo "$TF done"; exit 0; }
/nas/longleaf/rhel9/apps/anaconda/2024.02/bin/python -I $V3/v3_prep.py $TF
