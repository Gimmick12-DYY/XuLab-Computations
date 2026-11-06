#!/bin/bash
#SBATCH -J g1v21_sum
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/gate1/v2_1/logs/summary_%j.out
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python -I /work/users/d/y/dyy12/XuLab/regulatome/gate1/v2_1/gate1_v21_summary.py
