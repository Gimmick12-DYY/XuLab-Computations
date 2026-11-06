#!/bin/bash
#SBATCH -J tfpoly_sim
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 00:45:00
#SBATCH --array=0-29
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer/logs/%x_%A_%a.log
# TF-binding polymer, 3 attraction strengths x 10 tasks x 20 trajectories (200 per strength), 50 conformations each.
# Writes only under results_3d_model/tf_polymer/sim.
set -euo pipefail
T=/work/users/d/y/dyy12/XuLab/results_3d_model/tf_polymer
PY=/work/users/d/y/dyy12/envs/polysim/bin/python
EPS=(5 6.5 8)
e=${EPS[$((SLURM_ARRAY_TASK_ID / 10))]}
k=$((SLURM_ARRAY_TASK_ID % 10))
${PY} ${T}/scripts/02_simulate.py --beads ${T}/inputs/beads.tsv --eps ${e} --traj $((k*20))-$((k*20+19)) \
  --n-eq 50000 --n-keep 50 --n-every 1000 --out ${T}/sim/eps${e}
