#!/bin/bash
#SBATCH -J rep_k9
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 00:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/repressor_matrix/logs/count_%A_%a.out
module load bedtools 2>/dev/null || true
TF=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2/tfs.txt)
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python -I /work/users/d/y/dyy12/XuLab/regulatome/repressor_matrix/repressor_matrix.py count $TF
