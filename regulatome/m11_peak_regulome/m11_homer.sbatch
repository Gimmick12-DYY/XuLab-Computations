#!/bin/bash
# m11 Amendment 4: HOMER 5.1 known motifs, one TF x peak set per task. usage: sbatch --array=0-23 m11_homer.sbatch
#SBATCH -J m11homer
#SBATCH -p general
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m11_peak_regulome/logs/homer_%A_%a.log
set -e
module load homer/5.1
H=/work/users/d/y/dyy12/XuLab/regulatome/m11_peak_regulome/recall/homer; J=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $H/jobs.txt); mkdir -p $H/preparsed
findMotifsGenome.pl $H/$J.bed /work/users/d/y/dyy12/XuLab/downstream/cache/hg38.fa $H/$J -size given -nomotif -bg $H/bg.bed -p 4 -preparsedDir $H/preparsed
