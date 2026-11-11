#!/bin/bash
# Peak re-calling gate: HOMER known (own library, -nomotif) per (variant, TF, part). Task list recall/tasks_homer.tsv: <variant> <TF> <disc|val>.
#SBATCH -J rc_homer
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 1:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/recall/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; R=$O/recall
read -r V T PT < <(awk -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==n' $R/${TASKS:-tasks_homer.tsv})
PD=${POOLS:-pools}; HD=${HOMERDIR:-homer}; P=$R/$PD/$V/$T/$PT; [ -s $P/fg.bed ] && [ -s $P/bg.bed ] || { echo "empty $V $T $PT"; exit 0; }
[ $(wc -l < $P/fg.bed) -ge 100 ] || { echo "small $V $T $PT"; exit 0; }
module load homer/5.1 >/dev/null 2>&1
findMotifsGenome.pl $P/fg.bed hg38 $R/$HD/$V/$T/$PT -bg $P/bg.bed -size given -nomotif -p 1 -preparsedDir $O/preparsed > /dev/null 2>&1
rm -rf $R/$HD/$V/$T/$PT/knownResults
echo done $V $T $PT
