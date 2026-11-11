#!/bin/bash
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/logs/%x_%A_%a.log
set -euo pipefail
cd /work/users/d/y/dyy12/XuLab/downstream/fig2_qc
export PYTHONUNBUFFERED=1
~/.conda/envs/cistopic/bin/python r11a_read_variants.py "$SLURM_ARRAY_TASK_ID"
