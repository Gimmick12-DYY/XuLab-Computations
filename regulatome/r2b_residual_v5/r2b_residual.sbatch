#!/bin/bash
#SBATCH -J r2b_v5
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/r2b_residual_v5/logs/%x_%j.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/regulatome/r2b_residual_v5
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python $O/r2b_residual.py 10000 $O v5
