#!/bin/bash
#SBATCH -J tfpolyhic_ana
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer_hic/logs/%x_%j.log
# Hybrid validation (same tests as v2 + O/E excluding restrained-loop pixels). Writes only under tf_polymer_hic/.
set -euo pipefail
T=/work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer_hic
/work/users/d/y/dyy12/envs/polysim/bin/python ${T}/scripts/03_analyze_hyb.py --base ${T}
