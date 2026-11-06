#!/bin/bash
#SBATCH -J corr_pearson
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/chromhmm_corrected_sets_pearson/run_%j.out
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python -I /work/users/d/y/dyy12/XuLab/regulatome/chromhmm_corrected_sets_pearson/chromhmm_corrected_sets_pearson.py
