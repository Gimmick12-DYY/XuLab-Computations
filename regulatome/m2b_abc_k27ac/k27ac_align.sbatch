#!/bin/bash
#SBATCH -J k27_align
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH --array=0-1
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac/logs/align_%A_%a.out
# bowtie2 (hg38 index from atac_validation/ref, same hg38.fa as the repo) -> proper pairs MAPQ>=30 -> samtools markdup -r
set -euo pipefail
B=/work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac
IDX=/work/users/d/y/dyy12/XuLab/atac_validation/ref/hg38
RUNS=(SRR22020179 SRR22020178)
R=${RUNS[$SLURM_ARRAY_TASK_ID]}
source /nas/longleaf/rhel9/apps/anaconda/2024.02/etc/profile.d/conda.sh
conda activate /work/users/d/y/dyy12/conda/envs/atac
module load samtools/1.25
mkdir -p $B/bam $B/tmp; cd $B/bam
bowtie2 -p 8 -X 2000 --no-mixed --no-discordant -x $IDX -1 $B/fastq/${R}_1.fastq.gz -2 $B/fastq/${R}_2.fastq.gz 2> $R.bowtie2.log | \
  samtools view -u -f 2 -F 1804 -q 30 - | samtools sort -n -u -@ 4 -m 1G -T $B/tmp/$R.n - | samtools fixmate -m -u - - | \
  samtools sort -@ 4 -m 1G -T $B/tmp/$R.c -u - | samtools markdup -r -s -f $R.markdup.txt -@ 4 - $R.q30.dedup.bam
samtools index $R.q30.dedup.bam
samtools flagstat $R.q30.dedup.bam > $R.q30.dedup.flagstat
cat $R.bowtie2.log $R.markdup.txt $R.q30.dedup.flagstat
