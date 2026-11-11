#!/bin/bash
#SBATCH -p a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/logs/%x_%A_%a.log
# Usage: sbatch -J f2r9a --array=1-26 r10a.sbatch  (tf_list.txt: lines 1-6 panel TFs, all cells; others 3000 cells)
set -euo pipefail
LIST=/work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/imputed_audit/continuous/tf_list.txt
TF=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
CAP=0; [ "$SLURM_ARRAY_TASK_ID" -gt 6 ] && CAP=3000
cd /work/users/d/y/dyy12/XuLab/downstream/fig2_qc
/work/users/d/y/dyy12/conda/envs/unified/bin/python r10a_rethreshold.py "$TF" "$CAP"
