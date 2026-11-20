#!/bin/bash
#SBATCH -J m50
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc/out/m50_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m50_stage1_log2fc
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python -I m50_stream.py || exit 1
PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps /work/users/d/y/dyy12/conda/envs/cobinding/bin/python m50_stage1.py
