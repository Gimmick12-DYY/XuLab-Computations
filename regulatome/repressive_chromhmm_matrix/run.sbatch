#!/bin/bash
#SBATCH -J rep_chmm
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/repressive_chromhmm_matrix/logs/run_%j.out
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python -I /work/users/d/y/dyy12/XuLab/regulatome/repressive_chromhmm_matrix/repressive_chromhmm.py
