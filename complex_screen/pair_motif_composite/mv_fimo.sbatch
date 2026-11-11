#!/bin/bash
# Motif validation: FIMO presence (p <= 1e-4, as the reversed design) of one 100-motif chunk over all unique site sequences.
#SBATCH -J mv_fimo
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 3:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/motif_valid/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; G=$O/motif_valid; C=$(printf "c%02d" $SLURM_ARRAY_TASK_ID)
[ -f $G/chunks/$C.meme ] || { echo no chunk; exit 0; }
module load meme/5.5.7 >/dev/null 2>&1
fimo --thresh 1e-4 --text --skip-matched-sequence --max-strand --bfile $O/reversed/bins.bg $G/chunks/$C.meme $G/all.fa 2>/dev/null | awk -F'\t' 'NR>1{print $1"\t"$3}' | sort -u | gzip > $G/fimo/$C.tsv.gz
[ -s $G/fimo/$C.tsv.gz ] || exit 1; echo done $C
