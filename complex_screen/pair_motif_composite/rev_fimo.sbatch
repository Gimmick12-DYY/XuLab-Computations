#!/bin/bash
# Step 4: FIMO (p <= 1e-4) of one 100-motif chunk over all eligible bin centres; keeps only sequence and motif ids.
#SBATCH -J pm_fimo
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/reversed/logs/%x_%A_%a.out
set -euo pipefail
R=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/reversed; C=$(printf "c%02d" $SLURM_ARRAY_TASK_ID)
module load meme/5.5.7 >/dev/null 2>&1
fimo --thresh 1e-4 --text --skip-matched-sequence --max-strand --bfile $R/bins.bg $R/chunks/$C.meme $R/bins.fa 2>/dev/null | awk -F'\t' 'NR>1{print $1"\t"$3}' | sort -u | gzip > $R/fimo/$C.tsv.gz
echo done $C
