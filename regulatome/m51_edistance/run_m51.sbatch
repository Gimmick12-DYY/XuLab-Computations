#!/bin/bash
#SBATCH -J m51
#SBATCH -p general
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m51_edistance/out/m51_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m51_edistance
[ -s out/ntc_gem_stats.npz ] || /work/users/d/y/dyy12/conda/envs/snapatac2/bin/python -I m51_stream.py || exit 1
PYTHONPATH=/work/users/d/y/dyy12/envs/pptx_pydeps /work/users/d/y/dyy12/conda/envs/snapatac2/bin/python m51_analysis.py
