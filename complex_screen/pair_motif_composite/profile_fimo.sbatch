#!/bin/bash
# Threshold-free motif profile, step 2: FIMO (p <= 1e-4, as the reversed design) of one 100-motif chunk over the summit windows
# (profile/windows.fa) and, for chunks holding de novo family representatives, also over the OCR bin centres (reversed/bins.fa).
#SBATCH -J prof_fimo
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 2:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/profile/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; G=$O/profile; R=$O/reversed; C=$(printf "c%02d" $SLURM_ARRAY_TASK_ID)
module load meme/5.5.7 >/dev/null 2>&1
run() { fimo --thresh 1e-4 --text --skip-matched-sequence --max-strand --bfile $R/bins.bg $G/chunks/$C.meme "$1" 2>/dev/null | awk -F'\t' 'NR>1{print $1"\t"$3}' | sort -u | gzip > "$2"; }
run $G/windows.fa $G/fimo/win_$C.tsv.gz
if grep -q "^MOTIF fam" $G/chunks/$C.meme; then run $R/bins.fa $G/fimo/bins_$C.tsv.gz; fi
echo done $C
