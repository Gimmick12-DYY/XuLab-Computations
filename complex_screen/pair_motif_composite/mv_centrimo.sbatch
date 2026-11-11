#!/bin/bash
# Motif validation: CentriMo (central enrichment around summits, +/- 250 bp; no background set) of all 2,528 known motifs for one TF and part.
#SBATCH -J mv_cm
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 2:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/motif_valid/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; G=$O/motif_valid
read -r T PT < <(awk -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==n' $G/tasks_cm.tsv)
module load meme/5.5.7 >/dev/null 2>&1
centrimo --text --verbosity 1 --ethresh 1e10 --bfile $O/reversed/bins.bg $G/cm_fa/${T}_$PT.fa $O/db/known.meme > $G/cm/${T}_$PT.tsv
[ -s $G/cm/${T}_$PT.tsv ] || exit 1; echo done $T $PT $(wc -l < $G/cm/${T}_$PT.tsv)
