#!/bin/bash
# NFY depth check (a), step 1 array: one TF per task (line of reversed/depth/tfs.txt).
#SBATCH -J nfy_dep
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 1:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/reversed/depth/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite
T=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $O/reversed/depth/tfs.txt)
python3 $O/rev_depth_counts.py $T
