#!/bin/bash
#SBATCH -J tfpolyhic_sim
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 00:45:00
#SBATCH --array=0-19
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer_hic/logs/%x_%A_%a.log
# Hybrid: v2 TF-bridging polymer (16 kT, unchanged) + Peakachu 0.7 loop restraints. 20 tasks x 10 trajectories.
# Writes only under results_3d_model/tf_polymer_hic/sim.
set -euo pipefail
T=/work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer_hic
k=${SLURM_ARRAY_TASK_ID}
/work/users/d/y/dyy12/envs/polysim/bin/python ${T}/scripts/02_simulate_hyb.py --beads ${T}/inputs/beads.tsv \
  --loops ${T}/inputs/loop_restraints.tsv --eps 16 --k-sw 0.001 --traj $((k*10))-$((k*10+9)) \
  --n-eq 50000 --n-keep 50 --n-every 1000 --out ${T}/sim/eps16
