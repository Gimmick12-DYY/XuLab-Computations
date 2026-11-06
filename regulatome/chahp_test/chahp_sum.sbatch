#!/bin/bash
#SBATCH -J chahp_sum
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/chahp_test/logs/chahp_sum_%A_%a.out
module load bedtools 2>/dev/null || true
/nas/longleaf/rhel9/apps/anaconda/2024.02/bin/python -I /work/users/d/y/dyy12/XuLab/regulatome/chahp_test/chahp_test.py summary
