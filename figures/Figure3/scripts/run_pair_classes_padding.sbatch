#!/bin/bash
#SBATCH -p general
#SBATCH -t 6:00:00
#SBATCH --mem=48g
#SBATCH -J pcpad
#SBATCH -o /work/users/d/y/dyy12/logs/pcpad_%j.log
# Pair classes padding sensitivity (2026-11-23): V0 midpoint, V1 overlap, V2 +-500 bp, V3 +-1 kb (E>R and R>E), V4 +-2 kb.
source ~/.bashrc; conda activate hic
cd /work/users/d/y/dyy12/XuLab/figures/Figure3
O=checks/pair_classes_padding
for v in "V0 mid ER" "V1 0 ER" "V2 500 ER" "V3 1000 ER" "V3_RE 1000 RE" "V4 2000 ER"; do
  set -- $v
  PAD=$2 PRIO=$3 PLOT=0 python scripts/pair_classes.py $O/$1 > $O/$1.log 2>&1 &
done
wait
tail -n 12 $O/*.log
