#!/bin/bash
#SBATCH -J m50v34
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc/out/m50_v3v4_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc
PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps /work/users/d/y/dyy12/conda/envs/cobinding/bin/python m50_v3v4.py
