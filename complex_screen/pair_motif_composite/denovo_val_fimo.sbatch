#!/bin/bash
# FIMO (p <= 1e-4) of one 100-motif chunk of imputed de novo family representatives over the union-ATAC bin centres, for held-out scoring.
#SBATCH -J dn_fimo
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 2:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/logs_dn/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; B=${TFBASE:-$O/per_tf_matched}; G=$B/denovo_cmp; R=$O/reversed; C=$(printf "c%02d" $SLURM_ARRAY_TASK_ID)
[ -f $G/chunks/$C.meme ] || { echo "no chunk $C"; exit 0; }
module load meme/5.5.7 >/dev/null 2>&1
fimo --thresh 1e-4 --text --skip-matched-sequence --max-strand --bfile $R/bins.bg $G/chunks/$C.meme $G/val.fa 2>/dev/null | awk -F'\t' 'NR>1{print $1"\t"$3}' | sort -u | gzip > $G/fimo/$C.tsv.gz
echo done $C
