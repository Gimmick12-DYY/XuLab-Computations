#!/bin/bash
#SBATCH -J g1_stat
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/gate1/logs/stat_%A_%a.out
# Gate 1 per TF: 20 splits, 100 pseudo-TF nulls x 5 splits. TF from $TF or the array index into tfs.txt
set -euo pipefail
G=/work/users/d/y/dyy12/XuLab/regulatome/gate1
TF=${TF:-$(sed -n "${SLURM_ARRAY_TASK_ID}p" /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2/tfs.txt)}
[ -s $G/results/$TF.tsv ] && { echo "$TF done"; exit 0; }
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python $G/gate1_stat.py $TF 20 100 5
