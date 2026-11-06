#!/bin/bash
#SBATCH -J r2
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/r2_targets/logs/%x_%A_%a.out
O=/work/users/d/y/dyy12/XuLab/regulatome/r2_targets
T=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2/tfs.txt)
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python -I $O/r2_scores.py tf $T
