#!/bin/bash
#SBATCH -J abc_hicdump
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m2_abc/logs/hicdump_%j.out
# BUG (found 2026-11-11, left as run): awk $9 below is empty (the balanced value is $8), so inputs/hic5kb has no values; use regulatome/m2c_abc_canonical/dump_hic_fixed.sbatch.
# WT HEK293T Hi-C (GSE216399) 5 kb ICE-balanced cis pixels within 5 Mb -> ABC juicebox layout chrN/chrN.KRobserved.gz (bp bp value)
set -euo pipefail
A=/work/users/d/y/dyy12/XuLab/regulatome/m2_abc
COOL=/work/users/d/y/dyy12/XuLab/hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool::/resolutions/5000
CO=/work/users/d/y/dyy12/conda/envs/hic/bin/cooler
for c in $(cut -f1 $A/inputs/chrom_sizes.tsv); do
  mkdir -p $A/inputs/hic5kb/$c
  $CO dump -r $c --balanced --na-rep nan -t pixels --join $COOL | \
    awk -v OFS="\t" '$9 != "nan" && $5 - $2 <= 5000000 {print $2, $5, $9}' | gzip > $A/inputs/hic5kb/$c/$c.KRobserved.gz
  echo $c $(zcat $A/inputs/hic5kb/$c/$c.KRobserved.gz | wc -l)
done
