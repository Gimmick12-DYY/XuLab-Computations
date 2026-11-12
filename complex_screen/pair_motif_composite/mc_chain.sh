#!/bin/bash
# Motif-centred read profiles chain: prep -> FIMO positions (26) -> per-library counts (79) -> scoring.
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; G=$O/motif_centred; mkdir -p $G/logs; cd $O
J1=$(sbatch --parsable -J mc_prep -c 1 --mem=16G -t 1:00:00 -o $G/logs/mc_prep_%j.out mc_prep.sh)
J2=$(sbatch --parsable --dependency=afterok:$J1 --array=0-25 mc_fimo.sbatch)
J3=$(sbatch --parsable --dependency=afterok:$J2 --array=0-78 mc_count.sbatch)
J4=$(sbatch --parsable --dependency=afterany:$J3 -J mc_score -c 1 --mem=16G -t 1:00:00 -o $G/logs/mc_score_%j.out --wrap "cd $O && python3 mc_score.py")
echo "prep $J1 fimo $J2 counts $J3 score $J4" | tee $G/job_ids.txt
