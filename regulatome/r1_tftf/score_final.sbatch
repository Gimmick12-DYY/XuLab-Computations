#!/bin/bash
#SBATCH -J r1_final
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/r1_tftf/logs_final/run_%j.out
X=/work/users/d/y/dyy12/XuLab
/work/users/d/y/dyy12/envs/boltz/bin/python -I $X/complex_screen/scripts/fulllength_score.py
echo fulllength rc $?
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python -I $X/regulatome/r1_tftf/r1_build_v2.py
echo r1 rc $?
