#!/bin/bash
#SBATCH -J r3_fit
#SBATCH -p general
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/r3_eb/logs/%x_%j.out
set -euo pipefail
module load r/4.5.0
R3=/work/users/d/y/dyy12/XuLab/regulatome/r3_eb
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python $R3/r3_eb.py fit ${R3_TEST:+--test} ${R3_TEST:-}
