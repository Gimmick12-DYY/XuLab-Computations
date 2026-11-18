#!/bin/bash
#SBATCH -J m46m
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH --array=0-62%20
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/out/macs/m46_macs_%A_%a.log
module load macs3/3.0.3
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/m46_macs_call.py $SLURM_ARRAY_TASK_ID
