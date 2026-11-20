#!/bin/bash
#SBATCH -J m52a
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m52_stage1_paper_score/out/m52a_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m52_stage1_paper_score; export PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps; PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python
$PY -u m52_tables.py && $PY -u m52_shuffle_sets.py && $PY -u m52_tflabel.py && $PY -u m52_direction.py
