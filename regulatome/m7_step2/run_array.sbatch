#!/bin/bash
#SBATCH -J m7fit
#SBATCH -p general
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 8:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m7_step2/logs/fit_%A_%a.log
export PYTHONPATH=/work/users/d/y/dyy12/envs/m7_pydeps OMP_NUM_THREADS=4
cd /work/users/d/y/dyy12/XuLab/regulatome/m7_step2 && /work/users/d/y/dyy12/envs/boltz/bin/python m7.py "$1" "$SLURM_ARRAY_TASK_ID"
