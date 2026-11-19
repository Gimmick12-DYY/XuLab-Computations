#!/bin/bash
#SBATCH -J m46p2
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --array=0-377%30
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/out/macs_p2/m46_p2_%A_%a.log
module load macs3/3.0.3
MASKS=(M0 Mzero M5 M50 Mpeak Mdrop5); MK=${MASKS[$((SLURM_ARRAY_TASK_ID / 63))]}; I=$((SLURM_ARRAY_TASK_ID % 63))
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/m46_p2_only.py $MK $I
