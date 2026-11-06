#!/bin/bash
#SBATCH -J rep_build
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/repressor_matrix/logs/build_%j.out
module load bedtools 2>/dev/null || true
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python -I /work/users/d/y/dyy12/XuLab/regulatome/repressor_matrix/repressor_matrix.py build
