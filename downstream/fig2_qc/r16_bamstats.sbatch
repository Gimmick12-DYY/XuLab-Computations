#!/bin/bash
# Reference ChIP BAM stats for r16_chip_selfcheck.py (one TF per call). Primary mapped reads (read 1 only if paired,
# flags 0xF04 excluded), reads flagged duplicate, unique 5' ends (chrom, pos, strand) on chr1-22,X,Y, and dedup FRiP =
# unique 5' ends inside the reference peak file / all unique 5' ends. Writes chip_selfcheck/bamstats_<TF>.json.
# Usage: sbatch -J name r16_bamstats.sbatch TF BAM PEAKS
#SBATCH -p general
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/logs/%x_%j.log
set -euo pipefail
module load samtools/1.21 bedtools/2.31.1 >/dev/null 2>&1
T=$1; B=$2; P=$3
OS=/work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/round6_diffpeaks/chip_selfcheck
TMP=/work/users/d/y/dyy12/tmp/r16_$T; mkdir -p $TMP
CH=$(seq 1 22 | sed 's/^/chr/'; echo chrX chrY)
PAIRED=$( (samtools view -F 0xF04 $B chr1 || true) | head -1000 | awk '{if(and($2,1))p++} END{print (p>500)?1:0}')
F=""; [ "$PAIRED" = 1 ] && F="-f 64"
TOT=$(samtools view -c -@4 -F 0xF04 $F $B $CH)
DUP=$(samtools view -c -@4 -F 0xB04 -f 1024 $F $B $CH)
samtools view -b -@4 -F 0xF04 $F $B $CH | bedtools bamtobed -i stdin \
  | awk 'BEGIN{OFS="\t"}{p=($6=="+")?$2:$3-1; print $1,p,p+1,".",".",$6}' \
  | LC_ALL=C sort -S 12G -T $TMP --parallel=4 -u -k1,1 -k2,2n -k6,6 > $TMP/u.bed
UNIQ=$(wc -l < $TMP/u.bed)
grep -P '^chr([0-9]+|X|Y)\t' $P | cut -f1-3 | LC_ALL=C sort -k1,1 -k2,2n | bedtools merge -i stdin > $TMP/pk.bed
INPK=$(bedtools intersect -sorted -u -a $TMP/u.bed -b $TMP/pk.bed | wc -l)
NPK=$(grep -cP '^chr([0-9]+|X|Y)\t' $P)
cat > $OS/bamstats_$T.json <<J
{"tf": "$T", "bam": "$B", "paired": $([ "$PAIRED" = 1 ] && echo true || echo false), "reads_counted": $TOT, "flagged_duplicates": $DUP,
 "unique_5p_ends": $UNIQ, "dup_rate_5p": $(awk -v u=$UNIQ -v t=$TOT 'BEGIN{printf "%.6f", 1-u/t}'), "frip_dedup": $(awk -v a=$INPK -v u=$UNIQ 'BEGIN{printf "%.6f", a/u}'), "n_peaks": $NPK}
J
rm -rf $TMP
cat $OS/bamstats_$T.json
