#!/bin/bash
#SBATCH -J m46k
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH -t 03:00:00
#SBATCH --array=0-247%24
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/out/macs_mask/m46_macs_mask_%A_%a.log
module load macs3/3.0.3
MASKS=(Mzero M5 M50 Mpeak); N=$(wc -l < /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/work/tfs_mask_ok.txt)
MK=${MASKS[$((SLURM_ARRAY_TASK_ID / 62))]}; I=$((SLURM_ARRAY_TASK_ID % 62))
[ $I -ge $N ] && exit 0
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/m46_macs_mask.py $MK $I
