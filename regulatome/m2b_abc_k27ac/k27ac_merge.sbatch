#!/bin/bash
#SBATCH -J k27_merge
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac/logs/merge_%j.out
# Merge the 2 deduplicated reps into one ABC tagAlign (one line per mate, like the ATAC input), sorted in the m2_abc chrom_sizes order, bgzip + tabix
set -euo pipefail
B=/work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac
I=/work/users/d/y/dyy12/XuLab/regulatome/m2_abc/inputs
source /nas/longleaf/rhel9/apps/anaconda/2024.02/etc/profile.d/conda.sh
conda activate /work/users/d/y/dyy12/envs/abc
module load samtools/1.25
mkdir -p $B/inputs; T=$B/inputs/h3k27ac_hek293t_srx18002015_16.tagAlign.gz
for R in SRR22020179 SRR22020178; do
  samtools view $B/bam/$R.q30.dedup.bam | awk -v OFS="\t" '{s = and($2, 16) ? "-" : "+"; print $3, $4 - 1, $4 - 1 + length($10), "N", 1000, s}'
done | awk -v OFS="\t" 'NR==FNR{r[$1]=NR; next} ($1 in r){print r[$1], $0}' $I/chrom_sizes.tsv - | \
  LC_ALL=C sort -k1,1n -k3,3n -S 8G -T $B/tmp | cut -f2- | bgzip -@ 2 > $T
tabix -p bed $T
echo "tagAlign lines $(zcat $T | wc -l)"
