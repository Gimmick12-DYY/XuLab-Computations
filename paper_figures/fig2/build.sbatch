#!/bin/bash
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/logs/%x_%j.log
# Fig 2: source data (r21; 1 kb grid arrays r23; accessibility-mask imputed r24) then panels.
set -euo pipefail
cd /work/users/d/y/dyy12/XuLab
~/.conda/envs/cistopic/bin/python downstream/fig2_qc/r21_fig2_source.py
~/.conda/envs/cistopic/bin/python downstream/fig2_qc/r23_fig2_1kb_source.py
~/.conda/envs/cistopic/bin/python downstream/fig2_qc/r24_fig2_mask_source.py
~/.conda/envs/cistopic/bin/python paper_figures/fig2/make_fig2.py
