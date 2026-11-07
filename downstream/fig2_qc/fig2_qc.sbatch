#!/bin/bash
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/logs/%x_%j.log
# Usage: sbatch -J <name> --mem=32G fig2_qc.sbatch <script.py> [args...]
set -euo pipefail
module load bedtools/2.31.1 meme/5.5.7 samtools/1.21 >/dev/null 2>&1 || true
cd /work/users/d/y/dyy12/XuLab/downstream/fig2_qc
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
~/.conda/envs/cistopic/bin/python "$@"
