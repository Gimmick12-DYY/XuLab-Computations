#!/bin/bash
# m11 Amendment 4: V1 (pooled other-78 control downsampled to 3x treatment) and V2 (OmniATAC control, same rule). usage: sbatch --array=0-11 m11_recall_v12.sbatch
#SBATCH -J m11recall
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m11_peak_regulome/logs/recallv12_%A_%a.log
set -e
TFS=(NRF1 CTCF ZNF143 MAZ NFYA E4F1); i=$SLURM_ARRAY_TASK_ID; T=${TFS[$((i % 6))]}; V=$([ $i -lt 6 ] && echo V1 || echo V2)
F=/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment; RG=/work/users/d/y/dyy12/XuLab/regulatome; O=$RG/m11_peak_regulome/recall/$V/$T; mkdir -p $O
NT=$(zcat $F/TF.$T.bed.gz | wc -l)
if [ $V = V1 ]; then
  FILES=$(grep -v "^$" $RG/m1_binding/v2/tfs.txt | grep -vx "$T" | sed "s#^#$F/TF.#; s#\$#.bed.gz#" | tr "\n" " ")
  NC=$(awk -v t=$T 'NR>1 && $1!=t {s+=$2} END{print s}' $RG/m7_step2/features/canon_counts/read_totals.tsv)
  CAT="zcat $FILES"
else
  NC=$(zcat $RG/m2_abc/inputs/omniatac_merged.tn5.tagAlign.gz | wc -l); CAT="zcat $RG/m2_abc/inputs/omniatac_merged.tn5.tagAlign.gz"
fi
P=$(awk -v t=$NT -v c=$NC 'BEGIN{p=3*t/c; if(p>1)p=1; printf "%.8f", p}')
$CAT | awk -v p=$P 'BEGIN{srand(20261111)} rand()<p' | gzip > $O/control.bed.gz
NK=$(zcat $O/control.bed.gz | wc -l); echo -e "tf\tvariant\ttreatment_reads\tcontrol_pool_reads\tkeep_prob\tcontrol_reads_kept" > $O/reads.tsv; echo -e "$T\t$V\t$NT\t$NC\t$P\t$NK" >> $O/reads.tsv; cat $O/reads.tsv
/work/users/d/y/dyy12/conda/envs/atac/bin/macs2 callpeak -t $F/TF.$T.bed.gz -c $O/control.bed.gz -f BED -g hs -q 0.05 --nomodel --shift -75 --extsize 150 --keep-dup all --call-summits --scale-to small -n TF.$T --outdir $O
rm -f $O/control.bed.gz; wc -l $O/TF.${T}_peaks.narrowPeak
