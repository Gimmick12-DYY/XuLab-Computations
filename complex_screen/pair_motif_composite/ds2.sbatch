#!/bin/bash
# Depth downsampling task array: line of downsample/tasks2.tsv = <TF> <n_cells> <seed>.
#SBATCH -J ds_motif
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 1:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/downsample/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite
read -r T N S < <(awk -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==n' $O/downsample/tasks2.tsv)
python3 $O/ds_task.py $T $N $S
