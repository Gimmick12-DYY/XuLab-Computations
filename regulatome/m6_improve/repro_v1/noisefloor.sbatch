#!/bin/bash
#SBATCH -J noisefloor
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m6_improve/repro_v1/run2c/logs/noisefloor_%j.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m6_improve/repro_v1 && /work/users/d/y/dyy12/conda/envs/snapatac2/bin/python noisefloor.py
