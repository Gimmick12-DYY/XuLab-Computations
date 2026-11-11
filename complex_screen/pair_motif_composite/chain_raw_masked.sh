#!/bin/bash
# 2026-11-11 rerun of the raw peak analyses with summit windows masked to union-ATAC bins (overwrites per_tf_raw/ and profile/ in place).
# Step 1 (this script, run as a job after raw_pools.py): submit HOMER per TF, de novo families, held-out FIMO + shared-motif summary, then the profile chain.
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; B=$O/per_tf_raw; cd $O
N=$(wc -l < $B/tasks_tf.tsv)
J2=$(sbatch --parsable --array=0-$((N-1))%80 --time=3:00:00 -o $B/logs/%x_%A_%a.out --export=ALL,BASE=$B run_tf.sbatch $B/tasks_tf.tsv)
J3=$(sbatch --parsable --dependency=afterany:$J2 -J dn_fam_raw -c 1 --mem=16G -t 12:00:00 -o $B/logs/dn_fam_%j.out --wrap "cd $O && TFBASE=$B python3 denovo_share.py && TFBASE=$B python3 denovo_split.py")
J4=$(sbatch --parsable --dependency=afterok:$J3 --array=0-39%40 --export=ALL,TFBASE=$B denovo_val_fimo.sbatch)
J5=$(sbatch --parsable --dependency=afterok:$J4 -J dn_sum_raw -c 1 --mem=32G -t 4:00:00 -o $B/logs/dn_sum_%j.out --wrap "cd $O && TFBASE=$B python3 denovo_imputed_summary.py")
P1=$(sbatch --parsable --dependency=afterok:$J3 -J prof_pools -c 1 --mem=16G -t 3:00:00 -o profile/logs/prof_pools_%j.out --wrap "cd $O && rm -rf profile/pools profile/fimo profile/chunks profile/logos profile/windows.fa profile/tmp && python3 profile_pools.py")
P2=$(sbatch --parsable --dependency=afterok:$P1 --array=0-39%40 profile_fimo.sbatch)
P3=$(sbatch --parsable --dependency=afterany:$P2 -J prof_score -c 1 --mem=48G -t 6:00:00 -o profile/logs/prof_score_%j.out --wrap "cd $O && python3 profile_score.py")
P4=$(sbatch --parsable --dependency=afterok:$P3 -J prof_figs -c 1 --mem=8G -t 1:00:00 -o profile/logs/prof_figs_%j.out --wrap "cd $O && python3 profile_figs.py")
echo "HOMER $J2 ($N tasks); families $J3; held-out FIMO $J4; summary $J5; profile pools $P1, FIMO $P2, score $P3, figs $P4" | tee $O/logs_dn/chain_raw_masked_ids.txt
