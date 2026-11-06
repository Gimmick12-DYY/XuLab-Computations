#!/bin/bash
#SBATCH -J hub16_palC
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/figures_paletteC/%x_%j.log
# Palette C re-render of the chr16p11.2 hub browser (same inputs and options as run_hub_browser_chr16.sbatch).
# Reads xujie_lab fragment beds read-only; writes only to OUT.
set -euo pipefail
OUT=/work/users/d/y/dyy12/XuLab/figures_paletteC
HB=/work/users/d/y/dyy12/XuLab/cobinding/results/hub_browser
L=/work/users/d/y/dyy12/XuLab/hic/work_wt293t/loops
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/users/d/y/dyy12/conda/envs/hic
python ${OUT}/plot_hub_browser_paletteC.py --edges ${HB}/hub_chr16_edges.tsv --genes ${HB}/genes_chr16_29.7-31.6.tsv \
  --mcool /work/users/d/y/dyy12/XuLab/hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool \
  --loops ${L}/peakachu_5000.loops.0.7.bedpe ${L}/peakachu_10000.loops.0.7.bedpe \
  --frag-dir /vast/som/xujie_lab/TAGATG-293T/bigwig/fragment \
  --zoom chr16:30008000-30036000 chr16:30640000-30710000 \
  --title "chr16p11.2 hub: shared clique edges, Hi-C, loops, per-TF coverage and ChromHMM" --chromhmm /work/users/d/y/dyy12/XuLab/data/HEK293T_chromHMM18.bed.gz \
  --out ${OUT}/hub_chr16_browser_paletteC
