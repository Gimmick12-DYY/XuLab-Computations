#!/bin/bash
#SBATCH -J g1v2_cov
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/gate1/v2/logs/cov_%j.out
# per active element: GC fraction (hg38) and fraction of bp uniquely mappable at k50 (Umap)
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; V=$X/regulatome/gate1/v2
awk -F"\t" -v OFS="\t" "\$8==\"promoter\"||\$8==\"distal_active\"||\$8==\"bivalent\"{print \$1,\$2,\$3,\$4}" $X/regulatome/m0_universe/elements_v2.bed > $V/active.bed
bedtools nuc -fi $X/downstream/cache/hg38.fa -bed $V/active.bed | cut -f4,6 | tail -n +2 > $V/gc.tsv
zcat $X/data/mappability/k50.umap.bed.gz | tail -n +2 | cut -f1-3 | sort -k1,1 -k2,2n | bedtools merge > $V/k50.merged.bed
sort -k1,1 -k2,2n $V/active.bed | bedtools coverage -a - -b $V/k50.merged.bed | awk -F"\t" -v OFS="\t" "{print \$4,\$NF}" > $V/map.tsv
wc -l $V/active.bed $V/gc.tsv $V/map.tsv
