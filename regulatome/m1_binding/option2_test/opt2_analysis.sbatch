#!/bin/bash
#SBATCH -J opt2_ana
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/option2_test/logs/analysis_%j.out
set -euo pipefail
module load bedtools
O=/work/users/d/y/dyy12/XuLab/regulatome/m1_binding/option2_test
cd $O
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python opt2_analysis.py $O /work/users/d/y/dyy12/XuLab/regulatome/m1_binding \
  /work/users/d/y/dyy12/XuLab/regulatome/m0_universe/elements_final.bed
