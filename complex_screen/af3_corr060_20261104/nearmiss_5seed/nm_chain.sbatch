#!/bin/bash
#SBATCH -J af3nm_chain
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 00:10:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104/nearmiss_5seed/logs/%x_%j.out
# Runs afterany:8667717:8753795 (5-seed pickup; 6-TF imputation summary). Submits the near-miss arrays at 5 seeds (same infer.sbatch, seeds 1,2,3,4,5,
# tag nm5 -> out_nm5/, scores_nm5/), %2, regular then big (128G), after the late big-pair 5-seed array too if the pickup made one; then a pickup
# that resubmits any near-miss pair left without a score at 128G, %2.
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104; N=$O/nearmiss_5seed; cd $O
L=$(awk '$1=="big_late_s5"{print $2}' jobs.txt | tail -1); D=""; [ -n "$L" ] && D="-d afterany:$L"
n1=$(grep -c . $N/pairs_nm.txt); n2=$(grep -c . $N/pairs_nm_big.txt || true)
J1=$(sbatch --parsable $D --array=0-$((n1-1))%2 infer.sbatch nearmiss_5seed/pairs_nm.txt 1,2,3,4,5 nm5); echo "nm5_regular $J1" | tee -a jobs.txt
J2=""; if [ "$n2" -gt 0 ]; then J2=$(sbatch --parsable -d afterany:$J1 --array=0-$((n2-1))%2 --mem=128G infer.sbatch nearmiss_5seed/pairs_nm_big.txt 1,2,3,4,5 nm5); echo "nm5_big $J2" | tee -a jobs.txt; fi
P=$(sbatch --parsable -d afterany:$J1${J2:+:$J2} $N/nm_pickup.sbatch); echo "nm5_pickup $P" | tee -a jobs.txt
