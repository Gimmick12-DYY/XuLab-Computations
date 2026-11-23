#!/bin/bash
#SBATCH -p general
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/logs/%x_%j.log
# Round 11 step a (SPEC.md 1-2): download ChIP rep 2, align ZNF282 rep 2 like rep 1, markdup both ZNF282 reps, md5 checks.
set -euo pipefail
module load samtools/1.21 bowtie2/2.5.4 >/dev/null 2>&1
X=/work/users/d/y/dyy12/XuLab; R=$X/data/chip_reps; mkdir -p $R; cd $R
md5_of() { curl -s -H "Accept: application/json" "https://www.encodeproject.org/files/$1/?frame=object" | python3 -c "import json,sys; print(json.load(sys.stdin)['md5sum'])"; }
for f in ENCFF474ZKB ENCFF298ZTU ENCFF421CTJ ENCFF042EMQ ENCFF859NFR; do
  [ -s $f.bam ] || curl -sSL -o $f.bam "https://www.encodeproject.org/files/$f/@@download/$f.bam"
  want=$(md5_of $f); got=$(md5sum $f.bam | cut -d" " -f1); echo "$f md5 $([ "$want" = "$got" ] && echo OK || echo MISMATCH)"
  [ "$want" = "$got" ]; [ -s $f.bam.bai ] || samtools index -@ 8 $f.bam
done
# local rep-1 identity check (reported only)
for p in "ENCFF139DCW:$X/data/ENCFF139DCW.bam" "ENCFF085SQS:$X/data/MAZ_HEK293.bam" "ENCFF559CGO:$X/data/ZBTB7A_HEK293.bam" \
         "ENCFF156XLC:$X/data/ZIC2_HEK293.bam" "ENCFF436UNS:$X/data/ZNF777_HEK293.bam"; do
  a=${p%%:*}; b=${p#*:}; want=$(md5_of $a); got=$(md5sum $b | cut -d" " -f1)
  echo "rep1 $a vs $(basename $b): $([ "$want" = "$got" ] && echo SAME FILE || echo DIFFERENT)"
done
# ZNF282 rep 2 (GSM2466516): ENA fastq, aligned exactly as rep 1
if [ ! -s SRR5197099.hg38.bam ]; then
  # resumable download with retries; md5 checked against ENA's file report (execution change after an SSL drop)
  want=$(curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=SRR5197099&result=read_run&fields=fastq_md5" | tail -1 | cut -f1)
  for k in 1 2 3 4 5 6; do
    [ "$(md5sum SRR5197099.fastq.gz 2>/dev/null | cut -d" " -f1)" = "$want" ] && break
    curl -sSL -C - --retry 5 -o SRR5197099.fastq.gz "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR519/009/SRR5197099/SRR5197099.fastq.gz" || sleep 30
  done
  [ "$(md5sum SRR5197099.fastq.gz | cut -d" " -f1)" = "$want" ] && echo "SRR5197099 fastq md5 OK"
  bowtie2 --very-sensitive-local -p 8 -x /proj/seq/data/hg38_UCSC/Sequence/Bowtie2Index/genome -U SRR5197099.fastq.gz 2> SRR5197099.bowtie2.log \
    | samtools view -b -@ 8 - | samtools sort -@ 8 -o SRR5197099.hg38.bam -
  samtools index -@ 8 SRR5197099.hg38.bam; cat SRR5197099.bowtie2.log
fi
for s in SRR5197098:$X/data/geo_znf282/align/SRR5197098.hg38.bam SRR5197099:$R/SRR5197099.hg38.bam; do
  n=${s%%:*}; b=${s#*:}
  [ -s ZNF282_$n.markdup.bam ] || { samtools markdup -@ 8 -s $b ZNF282_$n.markdup.bam 2> ZNF282_$n.markdup.log; samtools index ZNF282_$n.markdup.bam; }
  tail -n +1 ZNF282_$n.markdup.log | head -20
done
echo done
