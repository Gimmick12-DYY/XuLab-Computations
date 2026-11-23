#!/bin/bash
#SBATCH -J share293T
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=4g
#SBATCH -t 03:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/downstream/share_293T_analysis/logs/copy_%j.log
for m in "$@"; do /work/users/d/y/dyy12/XuLab/downstream/share_293T_analysis/copy_to_293T_analysis.sh $m || exit 1; done
