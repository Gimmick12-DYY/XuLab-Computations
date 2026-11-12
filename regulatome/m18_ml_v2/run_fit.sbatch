#!/bin/bash
#SBATCH -J m18fit
#SBATCH -p general
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/fit_%A_%a.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m18_ml_v2
export PYTHONPATH=/work/users/d/y/dyy12/envs/m7_pydeps
CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" configs.txt)
/work/users/d/y/dyy12/envs/boltz/bin/python m18_fit.py $CFG
