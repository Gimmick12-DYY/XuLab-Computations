#!/bin/bash
#SBATCH -J m49
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m49_z_fold_change/out/m49_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m49_z_fold_change
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python -I m49_stream.py || exit 1
PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps /work/users/d/y/dyy12/conda/envs/cobinding/bin/python m49_tables.py
