#!/bin/bash
#SBATCH -J exon_check
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 00:40:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/cobinding/results/hub_browser/%x_%j.log
# Reads xujie_lab fragment beds read-only; writes only to OUT.
set -euo pipefail
OUT=/work/users/d/y/dyy12/XuLab/cobinding/results/hub_browser
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/users/d/y/dyy12/conda/envs/hic
python ${OUT}/exon_check.py ${OUT}/exon_check_genes.tsv /vast/som/xujie_lab/TAGATG-293T/bigwig/fragment ${OUT}
