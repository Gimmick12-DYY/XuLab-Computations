#!/bin/bash
#SBATCH -J hub_browser17
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
python ${OUT}/plot_hub_browser.py --edges ${OUT}/hub_chr17_edges.tsv --genes ${OUT}/genes_chr17_17.1-18.95.tsv \
  --mcool /work/users/d/y/dyy12/XuLab/hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool \
  --loops ${L}/peakachu_5000.loops.0.7.bedpe ${L}/peakachu_10000.loops.0.7.bedpe \
  --frag-dir /vast/som/xujie_lab/TAGATG-293T/bigwig/fragment \
  --zoom chr17:17900000-17930000 chr17:18168000-18198000 \
  --title "chr17p11.2 hub: shared clique edges, Hi-C, loops, per-TF coverage and ChromHMM" --chromhmm /work/users/d/y/dyy12/XuLab/data/HEK293T_chromHMM18.bed.gz \
  --out ${OUT}/hub_chr17_browser
