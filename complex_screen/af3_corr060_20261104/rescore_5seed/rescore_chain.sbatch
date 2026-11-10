#!/bin/bash
#SBATCH -J af3resc_chain
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 00:10:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104/rescore_5seed/logs/%x_%j.out
# Runs afterok:8667717 (the 5-seed pickup). Submits rescore.py after ALL 5-seed arrays end (8667715, 8667716 and the late big-pair array that
# the pickup records in jobs.txt as big_late_s5, if any).
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104; cd $O
D="afterany:8667715:8667716"; L=$(awk '$1=="big_late_s5"{print $2}' jobs.txt | tail -1); [ -n "$L" ] && D="$D:$L"
J=$(sbatch --parsable -J af3rescore -p general -c 1 --mem=4G -t 00:30:00 -d $D -o $O/rescore_5seed/logs/rescore_%j.out --wrap "/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python $O/rescore_5seed/rescore.py")
echo "rescore $J (dependency $D)"; echo "rescore_5seed $J" >> jobs.txt
