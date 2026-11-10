#!/bin/bash
#SBATCH -J af3nm_pk
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 00:10:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104/nearmiss_5seed/logs/%x_%j.out
# After the near-miss arrays: resubmit any near-miss pair without scores_nm5/<pair>.tsv once, at 128G, %2 (same seeds and tag).
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104; N=$O/nearmiss_5seed; cd $O
cat $N/pairs_nm.txt $N/pairs_nm_big.txt | grep . | while read p; do [ -s scores_nm5/$p.tsv ] || echo $p; done > $N/pairs_nm_retry.txt
n=$(grep -c . $N/pairs_nm_retry.txt || true); echo "unscored: $n"
if [ "$n" -gt 0 ]; then J=$(sbatch --parsable --array=0-$((n-1))%2 --mem=128G infer.sbatch nearmiss_5seed/pairs_nm_retry.txt 1,2,3,4,5 nm5); echo "nm5_retry $J" | tee -a jobs.txt; fi
