#!/bin/bash
#SBATCH -J g1v3_adjsum
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/gate1/v3_binned/logs/adjctl_%j.out
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python -I /work/users/d/y/dyy12/XuLab/regulatome/gate1/v3_binned/v3_adjcontrol_summary.py 2>&1 | grep -v findfont
