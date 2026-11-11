#!/bin/bash
#SBATCH -J m7count
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m7_step2/logs/counts_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m7_step2 && /work/users/d/y/dyy12/envs/boltz/bin/python count_canonical.py
