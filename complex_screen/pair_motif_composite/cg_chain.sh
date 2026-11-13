#!/bin/bash
# Composite + gapped chain driver (run after cg_prep.py): per-pair array, then summary.
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; G=$O/composite_gapped; cd $O
N=$(grep -c . $G/tasks.tsv)
NA=$(python3 -c "import pandas as pd; P=pd.read_csv('$G/pairs.tsv',sep='\\t').set_index('pair'); t=[l.strip() for l in open('$G/tasks.tsv') if l.strip()]; print(sum(P.loc[x,'set'].startswith('af3') for x in t))")
J1=$(sbatch --parsable --array=0-$((NA-1))%150 cg_task.sh)
JS=$(sbatch --parsable --dependency=afterany:$J1 -J cg_sum_af3 -c 1 --mem=8G -t 1:00:00 -o $G/logs/cg_sum_af3_%j.out --wrap "cd $O && SUFFIX=_af3 python3 cg_summary.py")
J1b=$(sbatch --parsable --dependency=afterany:$J1 --array=$NA-$((N-1))%150 cg_task.sh)
J2=$(sbatch --parsable --dependency=afterany:$J1b -J cg_sum -c 1 --mem=8G -t 1:00:00 -o $G/logs/cg_sum_%j.out --wrap "cd $O && python3 cg_summary.py")
echo "pairs $N AF3 $NA array_af3 $J1 summary_af3 $JS array_rest $J1b summary $J2" | tee $G/job_ids.txt
