#!/bin/bash
#SBATCH -J m54d
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m54_stage1_percentile_tables/out/m54d_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m54_stage1_percentile_tables; export PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps; PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python
$PY -u m54_counts.py && $PY -u m54_compare.py && $PY -u m54_bulk_side.py && $PY -u m54_plots.py both
