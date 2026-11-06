#!/bin/bash
#SBATCH -J g1v3_units
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/gate1/v3_binned/logs/units_%j.out
set -euo pipefail
module load bedtools 2>/dev/null || true
/nas/longleaf/rhel9/apps/anaconda/2024.02/bin/python -I /work/users/d/y/dyy12/XuLab/regulatome/gate1/v3_binned/v3_units.py
