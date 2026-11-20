#!/bin/bash
#SBATCH -J m52b
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m52_stage1_paper_score/out/m52b_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m52_stage1_paper_score; export PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps; PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python
$PY -u m52_tables5k.py && $PY -u m52_47.py && $PY -u m52_47_precision.py && $PY -u m52_padding.py
