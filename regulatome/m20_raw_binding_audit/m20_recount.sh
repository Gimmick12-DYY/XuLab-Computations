#!/bin/bash
# m20 Part 1c: independent recount from the raw files with bedtools + awk only (no m10/m11/m18 code).
# Outputs work/recount_<TF>_prom.tsv (gene, reads in TSS +/- 500 bp), work/recount_<TF>_total.txt, work/recount_<TF>_peakgenes.txt (genes with a q005 summit in a linked element).
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; RG=$X/regulatome; W=$RG/m20_raw_binding_audit/work; BT=/nas/longleaf/rhel9/apps/bedtools/2.31.1/bedtools2/bin/bedtools; V=/vast/som/xujie_lab/TAGATG-293T/bigwig
mkdir -p $W
# promoter windows from the ABC GeneList TSS (name, tss), fresh
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next} $h["tss"]!="" {t=int($h["tss"]); s=t-500; if(s<0)s=0; print $h["chr"]"\t"s"\t"t+500"\t"$h["name"]}' $RG/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt | sort -k1,1 -k2,2n > $W/prom500.bed
# canonical ABC links (ABC >= 0.02 file), element -> gene
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next} {print $h["chr"]"\t"$h["start"]"\t"$h["end"]"\t"$h["TargetGene"]}' $RG/m2c_abc_canonical/results/HEK293T_hicfixed/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv | sort -k1,1 -k2,2n > $W/links.bed
for T in "$@"; do
  zcat $V/fragment/TF.$T.bed.gz | awk 'BEGIN{OFS="\t"} {p=($6=="+")?$2:$3-1; print $1,p,p+1}' | sort -k1,1 -k2,2n > $W/r5_$T.bed
  wc -l < $W/r5_$T.bed > $W/recount_${T}_total.txt
  $BT intersect -a $W/prom500.bed -b $W/r5_$T.bed -c -sorted | awk 'BEGIN{OFS="\t"}{print $4,$5}' > $W/recount_${T}_prom.tsv
  awk 'BEGIN{OFS="\t"} $9>=1.30103 {s=$2+$10; print $1,s,s+1}' $V/macs2/TF.$T/TF.${T}_peaks.narrowPeak | sort -k1,1 -k2,2n > $W/summit_$T.bed
  $BT intersect -a $W/links.bed -b $W/summit_$T.bed -u -sorted | cut -f4 | sort -u > $W/recount_${T}_peakgenes.txt
  echo $T $(cat $W/recount_${T}_total.txt) $(wc -l < $W/summit_$T.bed) $(wc -l < $W/recount_${T}_peakgenes.txt)
  rm -f $W/r5_$T.bed
done
