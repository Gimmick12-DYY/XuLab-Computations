#!/bin/bash
# After the motif and permutation arrays: candidate calls, then the validation array, then the gate summary.
# Usage: sbatch --dependency=afterany:<array>:<perm> chain.sbatch <tasks file>   (pairs taken from the task file)
#SBATCH -J pm_chain
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/logs/%x_%j.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; cd $O; T=$1; S=$(basename $T .tsv)
PAIRS=$(cut -f1 $T | sort -u | tr '\n' ' ')
python3 summarize.py candidates $PAIRS
for p in $PAIRS; do echo -e "$p\tvalidate\t"; done > ${S}_validate.tsv
N=$(($(wc -l < ${S}_validate.tsv) - 1))
V=$(sbatch --parsable --array=0-$N%10 -J pm_val run_pair.sbatch $O/${S}_validate.tsv)
sbatch --dependency=afterany:$V -J pm_final -c 1 --mem=8G -t 30:00 -o $O/logs/pm_final_%j.out --wrap "cd $O && python3 summarize.py final $PAIRS"
echo "validate $V"
