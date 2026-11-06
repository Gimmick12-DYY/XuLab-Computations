#!/bin/bash
#SBATCH -J element_counts
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/cobinding/results/hub_browser/%x_%j.log
# Reads xujie_lab fragment beds read-only; writes only to OUT.
set -euo pipefail
OUT=/work/users/d/y/dyy12/XuLab/cobinding/results/hub_browser
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/users/d/y/dyy12/conda/envs/hic
python ${OUT}/element_tf_counts.py ${OUT}/elements.txt /vast/som/xujie_lab/TAGATG-293T/bigwig/fragment ${OUT}
