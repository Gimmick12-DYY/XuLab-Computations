#!/bin/bash
#SBATCH -J r5_core
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/r5_core/logs/%x_%j.out
set -euo pipefail
R=/work/users/d/y/dyy12/XuLab/regulatome/r5_core
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python $R/r5_core.py ${R5_TEST:+--test} ${R5_TEST:-}
