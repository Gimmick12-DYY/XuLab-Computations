#!/bin/bash
#SBATCH -J m53
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m53_stage1_percentile/out/m53_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m53_stage1_percentile; export PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps; PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python
$PY -u m53.py && $PY -u m53_plots.py
