#!/bin/bash
# m9_abc_diagnosis runner. usage: sbatch run_m9.sbatch <script.py> [args]
#SBATCH -J m9diag
#SBATCH -p general
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m9_abc_diagnosis/logs/m9_%j.log
set -e
export PYTHONPATH=/work/users/d/y/dyy12/envs/m7_pydeps OMP_NUM_THREADS=4
cd /work/users/d/y/dyy12/XuLab/regulatome/m9_abc_diagnosis && /work/users/d/y/dyy12/envs/boltz/bin/python "$@"
