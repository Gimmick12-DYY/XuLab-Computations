#!/bin/bash
#SBATCH -J hox_bins
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 00:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/hox_check/logs/bins_%A_%a.out
TF=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2/tfs.txt)
/nas/longleaf/rhel9/apps/anaconda/2024.02/bin/python -I /work/users/d/y/dyy12/XuLab/regulatome/hox_check/hox_check.py bins $TF
