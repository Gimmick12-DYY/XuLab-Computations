#!/bin/bash
# Peak re-calling gate: build pools for one (variant, TF) from recall/tasks_pools_binom.tsv.
#SBATCH -J rc_pools
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/recall/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite
read -r V T < <(awk -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==n' $O/recall/tasks_pools_binom.tsv)
python3 $O/recall_pools.py $V $T
