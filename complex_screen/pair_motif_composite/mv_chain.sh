#!/bin/bash
# Motif validation chain: prep -> FIMO (26 chunks) + CentriMo (158) in parallel -> scoring/calibration.
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; G=$O/motif_valid; cd $O; mkdir -p $G/logs
J1=$(sbatch --parsable -J mv_prep -c 1 --mem=16G -t 2:00:00 -o $G/logs/mv_prep_%j.out --wrap "cd $O && python3 mv_prep.py")
J2=$(sbatch --parsable --dependency=afterok:$J1 --array=0-25 mv_fimo.sbatch)
J3=$(sbatch --parsable --dependency=afterok:$J1 --array=0-157%158 mv_centrimo.sbatch)
J4=$(sbatch --parsable --dependency=afterany:$J2:$J3 -J mv_score -c 1 --mem=32G -t 2:00:00 -o $G/logs/mv_score_%j.out --wrap "cd $O && python3 mv_score.py")
echo "prep $J1 fimo $J2 centrimo $J3 score $J4" | tee $G/job_ids.txt
