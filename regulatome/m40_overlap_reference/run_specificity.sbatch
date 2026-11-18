#!/bin/bash
#SBATCH -J m40s
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m40_overlap_reference/out/m40spec_%j.log
PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps /work/users/d/y/dyy12/conda/envs/cobinding/bin/python /work/users/d/y/dyy12/XuLab/regulatome/m40_overlap_reference/m40_specificity.py
