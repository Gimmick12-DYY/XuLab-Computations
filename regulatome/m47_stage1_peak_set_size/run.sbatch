#!/bin/bash
#SBATCH -J m47
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m47_stage1_peak_set_size/out/m47_%j.log
PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps /work/users/d/y/dyy12/conda/envs/cobinding/bin/python /work/users/d/y/dyy12/XuLab/regulatome/m47_stage1_peak_set_size/m47.py
