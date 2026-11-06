#!/bin/bash
#SBATCH -J hicfirst_ana
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer_hicfirst/logs/%x_%j.log
# Validation of the three Hi-C-first arms. Writes only under tf_polymer_hicfirst/<arm>/analysis.
set -euo pipefail
T=/work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer_hicfirst
for A in hicfirst hicfirst_tf loops_only; do
  /work/users/d/y/dyy12/envs/polysim/bin/python ${T}/scripts/03_analyze_hicfirst.py --base ${T}/${A}
done
