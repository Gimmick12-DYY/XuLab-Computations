#!/bin/bash
#SBATCH -J m51s
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m51_edistance/out/m51_score_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m51_edistance
PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps /work/users/d/y/dyy12/conda/envs/snapatac2/bin/python m51_paper_score.py
