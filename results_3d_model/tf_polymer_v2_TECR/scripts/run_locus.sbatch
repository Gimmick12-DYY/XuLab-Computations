#!/bin/bash
#SBATCH -J tfpoly2_locus
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 00:45:00
#SBATCH --array=0-9
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/logs_tfpoly2_loci/%x_%A_%a.log
# v2 TF-switching polymer at one extra hub locus (same parameters as chr16 v2, 16 kT only): 10 tasks x 20 trajectories.
# Usage: sbatch --export=T=<locus folder> run_locus.sbatch. Writes only under that folder.
set -euo pipefail
PY=/work/users/d/y/dyy12/envs/polysim/bin/python
k=${SLURM_ARRAY_TASK_ID}
${PY} ${T}/scripts/02_simulate_v2.py --beads ${T}/inputs/beads.tsv --eps 16 --k-sw 0.001 --traj $((k*20))-$((k*20+19)) \
  --n-eq 50000 --n-keep 50 --n-every 1000 --out ${T}/sim/eps16
