#!/bin/bash
# Negative-control pseudo-libraries: task k < 10 -> pooled seed k; k >= 10 -> random seed k-10.
#SBATCH -J nc_motif
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 2:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/motif_tables/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; k=$SLURM_ARRAY_TASK_ID
if [ $k -lt 10 ]; then python3 $O/nc_task.py pooled $k; else python3 $O/nc_task.py random $((k-10)); fi
