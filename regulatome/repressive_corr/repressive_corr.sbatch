#!/bin/bash
#SBATCH -J repcorr
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/repressive_corr/logs/repcorr_%j.out
set -euo pipefail
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python /work/users/d/y/dyy12/XuLab/regulatome/repressive_corr/repressive_corr.py
