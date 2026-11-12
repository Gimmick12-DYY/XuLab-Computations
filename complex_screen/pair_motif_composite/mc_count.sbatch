#!/bin/bash
# Motif-centred read profiles: counts for one library (line of recall/tfs.txt).
#SBATCH -J mc_count
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 3:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/motif_centred/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite
python3 $O/mc_count.py $(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $O/recall/tfs.txt)
