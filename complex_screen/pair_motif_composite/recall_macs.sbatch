#!/bin/bash
# Peak re-calling with a control (2026-11-11). Task k: TF = line (k mod 79) of tfs.txt; control = TF (k < 79: the pooled reads of the
# other 78 TFs, same Tn5 / cells / open chromatin) or ATAC (k >= 79: HEK293T OmniATAC tagAlign, 4 runs). MACS2 settings as the lab call
# (-p 0.01 --nomodel --shift -75 --extsize 150 --keep-dup all --call-summits); MACS2 scales the larger library down.
#SBATCH -J rc_macs
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/recall/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; R=$O/recall; FR=/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment
TA=/work/users/d/y/dyy12/XuLab/atac_omniatac/caper_work/atac/70bfc7e8-3760-496f-bd8e-5a4bc9e02b51/call-bam2ta
k=$SLURM_ARRAY_TASK_ID; T=$(sed -n "$((k % 79 + 1))p" $R/tfs.txt)
if [ $k -lt 79 ]; then C=tf; CTRL=$(ls $FR/TF.*.bed.gz | grep -v "/TF.$T.bed.gz$"); else C=atac; CTRL=$(ls $TA/shard-*/execution/*.tn5.tagAlign.gz); fi
module load macs/2.2.9.1 >/dev/null 2>&1
mkdir -p $R/peaks_$C
macs2 callpeak -t $FR/TF.$T.bed.gz -c $CTRL -f BED -g hs -p 0.01 --nomodel --shift -75 --extsize 150 --keep-dup all --call-summits -n TF.$T --outdir $R/peaks_$C 2> $R/logs/macs_${C}_$T.log
rm -f $R/peaks_$C/TF.${T}_peaks.xls
echo done $T $C $(wc -l < $R/peaks_$C/TF.${T}_peaks.narrowPeak)
