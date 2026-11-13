#!/bin/bash
# Composite + gapped chain driver (run after cg_prep.py): per-pair array, then summary.
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; G=$O/composite_gapped; cd $O
N=$(grep -c . $G/tasks.tsv)
J1=$(sbatch --parsable --array=0-$((N-1))%150 cg_task.sh)
J2=$(sbatch --parsable --dependency=afterany:$J1 -J cg_sum -c 1 --mem=8G -t 1:00:00 -o $G/logs/cg_sum_%j.out --wrap "cd $O && python3 cg_summary.py")
echo "pairs $N array $J1 summary $J2" | tee $G/job_ids.txt
