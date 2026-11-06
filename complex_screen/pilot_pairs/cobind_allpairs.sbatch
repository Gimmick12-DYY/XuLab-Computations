#!/bin/bash
#SBATCH -J cobind_all
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pilot_pairs/logs/cobind_allpairs_%j.out
# Element-level co-binding obs/exp (top-5% residual sets) for all panel TF pairs, distal and promoter
set -euo pipefail
/nas/longleaf/rhel9/apps/anaconda/2024.02/bin/python -I /work/users/d/y/dyy12/XuLab/complex_screen/pilot_pairs/cobind_allpairs.py
