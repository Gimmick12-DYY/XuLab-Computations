#!/bin/bash
#SBATCH -J hic_dump_fix
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m2c_abc_canonical/logs/hicdump_%j.out
# WT HEK293T Hi-C (GSE216399) 5 kb ICE-balanced cis pixels within 5 Mb -> ABC juicebox layout chrN/chrN.KRobserved.gz (bp bp value).
# Corrects m2_abc/dump_hic.sh, which printed awk $9; `cooler dump --join --balanced` has 8 columns, the balanced value is $8, so the old files had no values.
# The mcool carries ICE weights only ("weight"); there is no KR vector, so values are ICE-balanced although ABC's layout calls the file KRobserved.
set -euo pipefail
C=/work/users/d/y/dyy12/XuLab/regulatome/m2c_abc_canonical
I=/work/users/d/y/dyy12/XuLab/regulatome/m2_abc/inputs
COOL=/work/users/d/y/dyy12/XuLab/hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool::/resolutions/5000
CO=/work/users/d/y/dyy12/conda/envs/hic/bin/cooler
for c in $(cut -f1 $I/chrom_sizes.tsv); do
  mkdir -p $C/inputs/hic5kb_fixed/$c
  # ABC reads chrN.KRnorm.gz only for the matrix size; values are already balanced, so a vector of 1.0 per bin (same files as m2_abc/inputs/hic5kb)
  cp $I/hic5kb/$c/$c.KRnorm.gz $C/inputs/hic5kb_fixed/$c/
  $CO dump -r $c --balanced --na-rep nan -t pixels --join $COOL | \
    awk -v OFS="\t" '$8 != "nan" && $5 - $2 <= 5000000 {print $2, $5, $8}' | gzip > $C/inputs/hic5kb_fixed/$c/$c.KRobserved.gz
  zcat $C/inputs/hic5kb_fixed/$c/$c.KRobserved.gz | awk -v c=$c 'NF == 3 {n++; s += $3} END {print c, "pixels with value", n + 0, "sum", s}'
done
