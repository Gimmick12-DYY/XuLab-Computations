#!/bin/bash
# Motif-centred read profiles: FIMO (p <= 1e-5) positions of one 100-motif chunk (motif_valid/chunks, the 2,528-motif DB) in the eligible
# non-HOT union-ATAC bins. Output motif_centred/fimo/<chunk>.tsv.gz: motif, chrom, motif centre (genome), strand.
#SBATCH -J mc_fimo
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 6:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/motif_centred/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; G=$O/motif_centred; C=$(printf "c%02d" $SLURM_ARRAY_TASK_ID)
[ -f $O/motif_valid/chunks/$C.meme ] || { echo no chunk; exit 0; }
module load meme/5.5.7 >/dev/null 2>&1
fimo --thresh 1e-5 --text --skip-matched-sequence --verbosity 1 --bfile $O/reversed/bins.bg $O/motif_valid/chunks/$C.meme $G/bins.fa \
 | awk -F'\t' 'NR>1{split($3,a,"_"); c=int(a[2]+($4+$5)/2); print $1"\t"a[1]"\t"c"\t"$6}' | sort -u | gzip > $G/fimo/$C.tsv.gz
[ -s $G/fimo/$C.tsv.gz ] || exit 1; echo done $C $(zcat $G/fimo/$C.tsv.gz | wc -l)
