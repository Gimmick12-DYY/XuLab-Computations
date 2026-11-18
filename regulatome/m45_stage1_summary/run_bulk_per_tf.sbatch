#!/bin/bash
#SBATCH -J m45b
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m45_stage1_summary/out/m45_bulk_per_tf_%j.log
PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps /work/users/d/y/dyy12/conda/envs/cobinding/bin/python /work/users/d/y/dyy12/XuLab/regulatome/m45_stage1_summary/m45_bulk_per_tf.py
