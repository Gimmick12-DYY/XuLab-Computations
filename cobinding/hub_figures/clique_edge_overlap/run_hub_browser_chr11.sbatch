#!/bin/bash
#SBATCH -J hub_browser11
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/cobinding/results/hub_browser/%x_%j.log
# Genome-browser figure of the TECR/DNAJB1 hub. Reads xujie_lab fragment beds read-only; writes only to OUT.
set -euo pipefail
COB=/work/users/d/y/dyy12/XuLab/cobinding
OUT=${COB}/results/hub_browser
L=/work/users/d/y/dyy12/XuLab/hic/work_wt293t/loops
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/users/d/y/dyy12/conda/envs/hic
python ${OUT}/plot_hub_browser.py --edges ${OUT}/hub_chr11_edges.tsv --genes ${OUT}/genes_chr11_75.1-75.9.tsv \
  --mcool /work/users/d/y/dyy12/XuLab/hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool \
  --loops ${L}/peakachu_5000.loops.0.7.bedpe ${L}/peakachu_10000.loops.0.7.bedpe \
  --frag-dir /vast/som/xujie_lab/TAGATG-293T/bigwig/fragment \
  --zoom chr11:75393000-75416000 chr11:75560000-75580000 \
  --title "chr11:75.18-75.85 Mb hub: shared clique edges, Hi-C, loops and per-TF coverage" \
  --out ${OUT}/hub_chr11_browser
