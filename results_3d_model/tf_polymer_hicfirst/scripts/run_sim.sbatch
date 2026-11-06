#!/bin/bash
#SBATCH -J hicfirst_sim
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 00:45:00
#SBATCH --array=0-29
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer_hicfirst/logs/%x_%A_%a.log
# Hi-C-first arms (10 tasks x 20 trajectories each): 0-9 hicfirst (Hi-C pixels + loops + red edges, no TF attraction),
# 10-19 hicfirst_tf (same + TF bridging 16 kT), 20-29 loops_only (165 loops, no TF attraction). Writes only under tf_polymer_hicfirst/.
set -euo pipefail
T=/work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer_hicfirst
ARMS=(hicfirst hicfirst_tf loops_only); EPS=(0 16 0)
g=$((SLURM_ARRAY_TASK_ID / 10)); k=$((SLURM_ARRAY_TASK_ID % 10)); A=${T}/${ARMS[$g]}
/work/users/d/y/dyy12/envs/polysim/bin/python ${T}/scripts/02_simulate_hicfirst.py --beads ${A}/inputs/beads.tsv \
  --loops ${A}/inputs/restraints.tsv --eps ${EPS[$g]} --k-sw 0.001 --traj $((k*20))-$((k*20+19)) \
  --n-eq 50000 --n-keep 50 --n-every 1000 --out ${A}/sim/eps${EPS[$g]}
