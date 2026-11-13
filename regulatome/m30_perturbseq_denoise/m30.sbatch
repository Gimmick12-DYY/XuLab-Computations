#!/bin/bash
#SBATCH -J m30
#SBATCH -p general
#SBATCH -t 06:00:00
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m30_perturbseq_denoise/out/m30_%j.log
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python /work/users/d/y/dyy12/XuLab/regulatome/m30_perturbseq_denoise/m30.py
