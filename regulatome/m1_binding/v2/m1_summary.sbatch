#!/bin/bash
#SBATCH -J m1v2_sum
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2/logs/summary_%j.out
set -euo pipefail
V=/work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2
cd $V
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python class_summary.py /work/users/d/y/dyy12/XuLab/regulatome/m0_universe/elements_v2.bed $V/counts $V
