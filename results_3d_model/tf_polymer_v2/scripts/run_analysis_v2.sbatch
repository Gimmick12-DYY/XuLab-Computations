#!/bin/bash
#SBATCH -J tfpoly2_ana
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=12G
#SBATCH -t 01:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer_v2/logs/%x_%j.log
# v2 validation vs held-out Hi-C / loops / Cicero, strength choice, rosette metrics, then PyMOL renders. Writes only under tf_polymer_v2/.
set -euo pipefail
T=/work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer_v2
/work/users/d/y/dyy12/envs/polysim/bin/python ${T}/scripts/03_analyze_v2.py --base ${T}
/work/users/d/y/dyy12/envs/pymol3d/bin/python ${T}/scripts/04_render_v2.py --base ${T}
