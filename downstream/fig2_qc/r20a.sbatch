#!/bin/bash
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --array=0-78%20
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/logs/%x_%A_%a.log
cd /work/users/d/y/dyy12/XuLab/downstream/fig2_qc
~/.conda/envs/cistopic/bin/python r20a_imp_mdrop5.py $SLURM_ARRAY_TASK_ID
