#!/bin/bash
# m7_step2 Amendment 7 (exploratory) runner. usage: sbatch run_wide.sbatch <script.py> [args...]
#SBATCH -J m7wide
#SBATCH -p general
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m7_step2/logs/wide_%j.log
set -e
export PYTHONPATH=/work/users/d/y/dyy12/XuLab/regulatome/m7_step2:/work/users/d/y/dyy12/envs/m7_pydeps OMP_NUM_THREADS=8
cd /work/users/d/y/dyy12/XuLab/regulatome/m7_step2 && /work/users/d/y/dyy12/envs/boltz/bin/python "$@"
