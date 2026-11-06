#!/bin/bash
#SBATCH -J g1v3
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/gate1/v3_binned/logs/%x_%A_%a.out
# Gate 1 v3: line N of $JOBS = "<unit_type> <tf|module> <name>"
set -euo pipefail
V3=/work/users/d/y/dyy12/XuLab/regulatome/gate1/v3_binned
read UT MODE NAME < <(sed -n "${SLURM_ARRAY_TASK_ID}p" $JOBS)
O=$V3/results; [ $MODE = module ] && O=$V3/results_modules
[ -s $O/$UT.$NAME.tsv ] && { echo "$UT $NAME done"; exit 0; }
/nas/longleaf/rhel9/apps/anaconda/2024.02/bin/python -I $V3/gate1_v3.py $UT $MODE $NAME
