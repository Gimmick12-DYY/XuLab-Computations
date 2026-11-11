#!/bin/bash
#SBATCH -J m7feat
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m7_step2/logs/features_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m7_step2 && /work/users/d/y/dyy12/envs/boltz/bin/python build_features.py "$1" "${2:-powerlaw}"
