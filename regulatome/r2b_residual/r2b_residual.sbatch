#!/bin/bash
#SBATCH -J r2b
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/r2b_residual/logs/%x_%j.out
set -euo pipefail
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python /work/users/d/y/dyy12/XuLab/regulatome/r2b_residual/r2b_residual.py
