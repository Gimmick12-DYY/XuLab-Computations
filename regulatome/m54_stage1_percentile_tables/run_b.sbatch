#!/bin/bash
#SBATCH -J m54b
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m54_stage1_percentile_tables/out/m54b_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m54_stage1_percentile_tables; export PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps; PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python
$PY -u m54_tables5k.py && $PY -u m54_47.py && $PY -u m54_47_precision.py && $PY -u m54_padding.py
