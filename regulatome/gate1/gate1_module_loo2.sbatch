#!/bin/bash
#SBATCH -J g1_loo
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/gate1/logs/loo_%A_%a.out
# Gate 1 leave-one-out module test: line N of loo_jobs2.tsv = module <tab> members to remove
set -euo pipefail
G=/work/users/d/y/dyy12/XuLab/regulatome/gate1
read M EX < <(sed -n "${SLURM_ARRAY_TASK_ID}p" $G/loo_jobs2.tsv)
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python $G/gate1_module.py $M 20 100 5 $EX
