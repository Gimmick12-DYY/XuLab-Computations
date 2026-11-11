#!/bin/bash
# m11 Amendment 3: re-call one TF's peaks with MACS2 2.2.9.1 against the pooled fragments of the other 78 TFs. usage: sbatch --array=0-5 m11_recall.sbatch
#SBATCH -J m11recall
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m11_peak_regulome/logs/recall_%A_%a.log
set -e
TFS=(NRF1 CTCF ZNF143 MAZ NFYA E4F1); T=${TFS[$SLURM_ARRAY_TASK_ID]}
O=/work/users/d/y/dyy12/XuLab/regulatome/m11_peak_regulome/recall/$T; mkdir -p $O; F=/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment
C=$(grep -v "^$" /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2/tfs.txt | grep -vx "$T" | sed "s#^#$F/TF.#; s#\$#.bed.gz#" | tr "\n" " ")
/work/users/d/y/dyy12/conda/envs/atac/bin/macs2 --version
/work/users/d/y/dyy12/conda/envs/atac/bin/macs2 callpeak -t $F/TF.$T.bed.gz -c $C -f BED -g hs -q 0.05 --nomodel --shift -75 --extsize 150 --keep-dup all --call-summits -n TF.$T --outdir $O
wc -l $O/TF.${T}_peaks.narrowPeak
