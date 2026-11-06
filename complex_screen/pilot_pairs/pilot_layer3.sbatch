#!/bin/bash
#SBATCH -J pilot_l3
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pilot_pairs/logs/layer3_%j.out
# Pilot pairs layers 1 and 3 (co-bound elements, GC-matched background, motifs vs split-half shuffles, spacing, GC/CpG)
set -euo pipefail
module load bedtools 2>/dev/null || true
/nas/longleaf/rhel9/apps/anaconda/2024.02/bin/python -I /work/users/d/y/dyy12/XuLab/complex_screen/pilot_pairs/pilot_layer3.py
