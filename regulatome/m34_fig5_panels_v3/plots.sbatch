#!/bin/bash
#SBATCH -J m34plot
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m34_fig5_panels_v3/logs/plots_%j.log
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python /work/users/d/y/dyy12/XuLab/regulatome/m34_fig5_panels_v3/m34_plots.py ${1:-abcdefg}
