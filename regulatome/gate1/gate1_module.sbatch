#!/bin/bash
#SBATCH -J g1_mod
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/gate1/logs/module_%A_%a.out
# Gate 1 module-pooled test: one R1 provisional module per array task (row of r1_modules_provisional.tsv)
set -euo pipefail
G=/work/users/d/y/dyy12/XuLab/regulatome/gate1
M=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==n{print $1}' /work/users/d/y/dyy12/XuLab/regulatome/r1_tftf/r1_modules_provisional.tsv)
[ -s $G/results_modules/$M.tsv ] && { echo "$M done"; exit 0; }
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python $G/gate1_module.py $M 20 100 5
