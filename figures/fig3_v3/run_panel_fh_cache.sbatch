#!/bin/bash
#SBATCH -J fig3v3_hub
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/figures/fig3_v3/logs/%x_%j.log
# Fig 3 v3 panels f/h: cache the chr16p11.2 hub window (reads xujie_lab fragment beds read-only), then draw both panels.
set -euo pipefail
cd /work/users/d/y/dyy12/XuLab/figures/fig3_v3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/users/d/y/dyy12/conda/envs/hic
python panel_fh_hub_browser.py --cache
python panel_fh_hub_browser.py
