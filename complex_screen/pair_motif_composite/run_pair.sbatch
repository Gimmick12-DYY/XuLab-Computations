#!/bin/bash
# One array task = one (pair, mode) row of the task file given as $1 (default tasks_gate.tsv).
# Modes: homer <vsA|vsB|vsInput> <known|8|10|12> (HOMER known-only, or de novo at one motif length, pair vs A-only / B-only / matched input),
#        sea (SEA + CentriMo + SpaMo, pair vs A-only + B-only), streme (STREME + Tomtom), perm (HOMER known only on label permutations),
#        validate (candidate motifs on even chromosomes, pair vs singles).
#SBATCH -J pm_pair
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite
B=${BASE:-$O}   # BASE = variant folder holding pools/, results/, candidates/ (e.g. $O/gc_matched); db/ and preparsed/ stay in $O
TASKS=${1:-$B/tasks_gate.tsv}
module load homer/5.1 meme/5.5.7 >/dev/null 2>&1
read -r PAIR MODE ARG < <(awk -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==n' "$TASKS")
D=$B/pools/$PAIR/disc; R=$B/results/$PAIR; mkdir -p $R $O/preparsed
DB=$O/db/known.meme
echo "$PAIR $MODE $ARG $(date)"
homer() { findMotifsGenome.pl "$1" hg38 "$2" -size given -bg "$3" -p 1 -preparsedDir $O/preparsed "${@:4}"; }
case $MODE in
  homer)
    read -r BG L <<< "${ARG//,/ }"; F=$D/${BG#vs}only.bed; [ "$BG" = vsInput ] && F=$D/input.bed
    if [ "$L" = known ]; then homer $D/pair.bed $R/homer_$BG/known $F -nomotif
    else homer $D/pair.bed $R/homer_$BG/L$L $F -noknown -len $L; fi ;;
  sea)
    sea --verbosity 1 --oc $R/sea --p $D/pair.fa --n $D/singles.fa --m $DB
    centrimo --verbosity 1 --oc $R/centrimo --neg $D/singles.fa $D/pair.fa $DB
    if [ -n "${ARG:-}" ]; then
      for M in ${ARG//,/ }; do spamo --verbosity 1 --oc $R/spamo_$M --primary $M $D/pair.fa $DB $O/db/jaspar.meme; done
    fi ;;
  streme)
    streme --verbosity 1 --oc $R/streme --dna --p $D/pair.fa --n $D/singles.fa --nmotifs 10 --time 2100
    tomtom -no-ssc -verbosity 1 -oc $R/tomtom_streme -min-overlap 5 -dist pearson -evalue -thresh 1 $R/streme/streme.txt $DB ;;
  perm)
    for K in ${ARG//,/ }; do
      P=$B/pools/$PAIR/perm/$K
      homer $P/pair.bed $R/perm/$K/vsA $P/Aonly.bed -nomotif
      homer $P/pair.bed $R/perm/$K/vsB $P/Bonly.bed -nomotif
      rm -rf $R/perm/$K/vs?/knownResults $R/perm/$K/vs?/*.html
    done ;;
  valknown)
    # held-out chromosomes, known motifs only, pair vs one single pool (ARG = A or B)
    V=$B/pools/$PAIR/val; homer $V/pair.bed $R/val_vs$ARG $V/${ARG}only.bed -nomotif ;;
  validate)
    V=$B/pools/$PAIR/val; C=$B/candidates/$PAIR
    [ -s $C/homer.motifs ] && homer $V/pair.bed $R/val_homer $V/singles.bed -nomotif -mknown $C/homer.motifs -nlen 0
    [ -s $C/meme.meme ] && sea --verbosity 1 --oc $R/val_sea --p $V/pair.fa --n $V/singles.fa --m $C/meme.meme --thresh 1
    true ;;
esac
echo "done $(date)"
