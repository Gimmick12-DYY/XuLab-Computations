#!/bin/bash
#SBATCH -J tfpoly_ana
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=12G
#SBATCH -t 01:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer/logs/%x_%j.log
# Validation vs held-out Hi-C / loops / Cicero, strength choice, then PyMOL render. Writes only under tf_polymer/.
set -euo pipefail
T=/work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer
/work/users/d/y/dyy12/envs/polysim/bin/python ${T}/scripts/03_analyze.py --base ${T}
/work/users/d/y/dyy12/envs/pymol3d/bin/python ${T}/scripts/04_render.py --base ${T}
