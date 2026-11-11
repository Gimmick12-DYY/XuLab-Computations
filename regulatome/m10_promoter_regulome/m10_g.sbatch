#!/bin/bash
# m10 Amendment 2 array runner. usage: sbatch --array=... m10_g.sbatch <fimo|count>
#SBATCH -J m10g
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 4:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m10_promoter_regulome/logs/g_%A_%a.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m10_promoter_regulome && /work/users/d/y/dyy12/envs/boltz/bin/python m10_g.py $1 $SLURM_ARRAY_TASK_ID
