#!/bin/bash
#SBATCH -J m52d
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m52_stage1_paper_score/out/m52d_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m52_stage1_paper_score; export PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps; PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python
$PY -u m52_counts.py && $PY -u m52_compare.py && $PY -u m52_bulk_side.py && $PY -u m52_plots.py both
