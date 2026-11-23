#!/bin/bash
#SBATCH -J m54c
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m54_stage1_percentile_tables/out/m54c_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m54_stage1_percentile_tables; export PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps; PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python
$PY -u m54_47_bulk.py && $PY -u m54_47_paired.py && $PY -u m54_47_coord.py
