#!/bin/bash
#SBATCH -J k27_fetch
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 04:00:00
#SBATCH --array=0-1
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac/logs/fetch_%A_%a.out
# Paired-end fastq for ChIP-Atlas HEK293T WT H3K27ac (GSE216xxx, GSM6672008/9): SRX18002015 = SRR22020179, SRX18002016 = SRR22020178, from ENA
set -euo pipefail
B=/work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac
RUNS=(SRR22020179 SRR22020178)
R=${RUNS[$SLURM_ARRAY_TASK_ID]}
mkdir -p $B/fastq; cd $B/fastq
for m in 1 2; do
  f=${R}_${m}.fastq.gz
  [ -e $f.ok ] && continue
  for t in 1 2 3 4 5; do
    curl -fsS -C - --retry 3 -o $f https://ftp.sra.ebi.ac.uk/vol1/fastq/${R:0:6}/0${R: -2}/$R/$f && gzip -t $f && touch $f.ok && break
    sleep $((t * 30))
  done
  [ -e $f.ok ] || { echo "download failed $f"; exit 1; }
done
ls -la ${R}_*
