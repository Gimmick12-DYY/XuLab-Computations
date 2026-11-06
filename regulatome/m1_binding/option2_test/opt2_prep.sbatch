#!/bin/bash
#SBATCH -J opt2_prep
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/option2_test/logs/prep_%j.out
set -euo pipefail
module load meme/5.5.7 bedtools
O=/work/users/d/y/dyy12/XuLab/regulatome/m1_binding/option2_test
X=/work/users/d/y/dyy12/XuLab
MD=$X/motif_analysis/cache/motifdb
cd $O
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python make_motifs.py $MD/H12CORE_meme_format.meme \
  $X/downstream/cache/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme $MD/codebook_top1.meme panel_motifs.meme motif_sources.tsv
cut -f1-4 $X/regulatome/m0_universe/elements_final.bed > elements4.bed
bedtools getfasta -fi $X/downstream/cache/hg38.fa -bed elements4.bed -nameOnly -fo elements.fa
fasta-get-markov -m 0 elements.fa elements.bg
grep "^MOTIF" panel_motifs.meme | awk '{print $2}' > motif_tfs.txt
wc -l motif_tfs.txt
