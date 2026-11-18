#!/bin/bash
#SBATCH -J m46
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/out/m46_%j.log
PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps /work/users/d/y/dyy12/conda/envs/cobinding/bin/python /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/m46.py
