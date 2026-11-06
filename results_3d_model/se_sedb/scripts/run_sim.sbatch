#!/bin/bash
#SBATCH -J sedb_sim
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 00:45:00
#SBATCH --array=0-9
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/se_sedb/logs/%x_%A_%a.log
# Hi-C-first model (no TF term) of the exact SEdb SE window, chr19: 10 tasks x 20 trajectories x 50 conformations. Writes only under se_sedb/.
set -euo pipefail
T=/work/users/d/y/dyy12/XuLab/results_3d_model/se_sedb; M=${T}/model; k=${SLURM_ARRAY_TASK_ID}
/work/users/d/y/dyy12/envs/polysim/bin/python ${T}/scripts/02_simulate_hicfirst.py --beads ${M}/inputs/beads.tsv --loops ${M}/inputs/restraints.tsv \
  --eps 0 --k-sw 0.001 --traj $((k*20))-$((k*20+19)) --n-eq 50000 --n-keep 50 --n-every 1000 --out ${M}/sim/eps0
