#!/bin/bash
# m8_edge_table runner. usage: sbatch run_m8.sbatch <script.py> [args]
#SBATCH -J m8edge
#SBATCH -p general
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m8_edge_table/logs/m8_%j.log
set -e
cd /work/users/d/y/dyy12/XuLab/regulatome/m8_edge_table && OMP_NUM_THREADS=4 /work/users/d/y/dyy12/envs/boltz/bin/python "$@"
