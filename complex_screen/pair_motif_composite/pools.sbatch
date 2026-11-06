#!/bin/bash
#SBATCH -J pm_pools
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/logs/pools_%j.out
cd /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite
python3 build_pools.py "$@"
