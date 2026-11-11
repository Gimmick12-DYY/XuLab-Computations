#!/bin/bash
# m10 promoter-window counts, one TF per task. usage: sbatch --array=0-78 m10_counts.sbatch
#SBATCH -J m10cnt
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 3:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m10_promoter_regulome/logs/cnt_%A_%a.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m10_promoter_regulome && /work/users/d/y/dyy12/envs/boltz/bin/python m10_counts.py $SLURM_ARRAY_TASK_ID
