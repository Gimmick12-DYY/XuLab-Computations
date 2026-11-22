#!/bin/bash
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/logs/%x_%j.log
# Fig 2: source data then panels.
set -euo pipefail
cd /work/users/d/y/dyy12/XuLab
~/.conda/envs/cistopic/bin/python downstream/fig2_qc/r21_fig2_source.py
~/.conda/envs/cistopic/bin/python paper_figures/fig2/make_fig2.py
