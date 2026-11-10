#!/bin/bash
# Per-TF HOMER on matched pools (per_tf_pools.py). Row of the task file: <TF> <known|8|10|12|val_known>.
#SBATCH -J pm_tf
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/per_tf_matched/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; G=$O/per_tf_matched; TASKS=${1:-$G/tasks_tf.tsv}
module load homer/5.1 >/dev/null 2>&1
read -r TF MODE < <(awk -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==n' "$TASKS")
P=$G/pools/$TF; R=$G/results/$TF; mkdir -p $R
h() { findMotifsGenome.pl "$1" hg38 "$2" -size given -bg "$3" -p 1 -preparsedDir $O/preparsed "${@:4}"; }
echo "$TF $MODE $(date)"
case $MODE in
  known)     h $P/disc/fg.bed $R/known $P/disc/bg.bed -nomotif ;;
  val_known) h $P/val/fg.bed $R/val_known $P/val/bg.bed -nomotif ;;
  *)         h $P/disc/fg.bed $R/L$MODE $P/disc/bg.bed -noknown -len $MODE ;;
esac
rm -rf $R/*/knownResults $R/*/homerResults/*.svg 2>/dev/null || true
echo "done $(date)"
