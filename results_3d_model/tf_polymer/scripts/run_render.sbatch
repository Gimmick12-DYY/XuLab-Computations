#!/bin/bash
#SBATCH -J tfpoly_render
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 00:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer/logs/%x_%j.log
# PyMOL re-render only (representative + gallery). Writes only under tf_polymer/render.
set -euo pipefail
T=/work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer
/work/users/d/y/dyy12/envs/pymol3d/bin/python ${T}/scripts/04_render.py --base ${T}
