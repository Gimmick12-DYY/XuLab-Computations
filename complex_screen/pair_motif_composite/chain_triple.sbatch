#!/bin/bash
# Candidate run follow-up: candidate calls -> validation array + permutation array (only pairs whose own motif reaches the relaxed
# discovery terms: fold >= 1.3 vs partner-only and p <= 1e-3) -> summary, strict/relaxed levels, pair_summary.tsv.
# Usage: BASE=<run folder> sbatch --export=ALL,BASE=... --dependency=afterany:<array> chain_triple.sbatch <tasks file>
#SBATCH -J pm_chain3
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 45:00
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; B=$BASE; cd $B; T=$1
PAIRS=$(cut -f1 $T | sort -u | tr '\n' ' ')
python3 $O/summarize.py candidates $PAIRS
for p in $PAIRS; do echo -e "$p\tvalidate\t"; done > tasks_validate.tsv
V=$(sbatch --parsable --export=ALL,BASE=$B --array=0-$(($(wc -l < tasks_validate.tsv) - 1))%10 -J pm_val3 $O/run_pair.sbatch $B/tasks_validate.tsv)
python3 - $PAIRS > tasks_perm.tsv <<'PY'
import sys, os; sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite"); import summarize as S
for p in sys.argv[1:]:
    a, b = p.split("-"); R = f"{os.environ['BASE']}/results/{p}"; kA, kB = S.known(f"{R}/homer_vsA/known/knownResults.txt"), S.known(f"{R}/homer_vsB/known/knownResults.txt")
    hit = any(k is not None and any(S.short(n) in own and k.loc[n].fold >= 1.3 and k.loc[n].log10p <= -3 for n in k.index) for k, own in [(kB, S.own_of(a)), (kA, S.own_of(b))])
    if hit:
        for c in range(0, 50, 5): print(f"{p}\tperm\t{','.join(f'{k:02d}' for k in range(c, c + 5))}")
PY
DEP=$V
if [ -s tasks_perm.tsv ]; then Pm=$(sbatch --parsable --export=ALL,BASE=$B --array=0-$(($(wc -l < tasks_perm.tsv) - 1))%10 -J pm_perm3 $O/run_pair.sbatch $B/tasks_perm.tsv); DEP=$V:$Pm; fi
sbatch --dependency=afterany:$DEP -J pm_final3 -c 1 --mem=8G -t 45:00 -o $B/logs/pm_final3_%j.out --wrap "cd $B && export BASE=$B && python3 $O/summarize.py final $PAIRS && python3 $O/gate_levels.py && python3 $O/pair_summary.py"
echo "validate $V perm ${Pm:-none} pairs_with_perm $(cut -f1 tasks_perm.tsv | sort -u | tr '\n' ' ')"
