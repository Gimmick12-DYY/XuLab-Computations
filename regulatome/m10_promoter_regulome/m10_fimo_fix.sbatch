#!/bin/bash
# m10 Amendment 4: rerun the own-motif FIMO scans (open chromatin and promoters) with failure checking. usage: sbatch --array=0-30 m10_fimo_fix.sbatch
#SBATCH -J m10fix
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m10_promoter_regulome/logs/fimofix_%A_%a.log
set -euo pipefail
W=/work/users/d/y/dyy12/XuLab/regulatome/m10_promoter_regulome/work/g; P=/work/users/d/y/dyy12/XuLab/regulatome/m9_abc_diagnosis/work/promoters.fa; FIMO=/nas/longleaf/rhel9/apps/meme/5.5.7/bin/fimo
M=$(ls $W/motif_*.meme | sed -n "$((SLURM_ARRAY_TASK_ID+1))p"); T=$(basename $M .meme | sed "s/^motif_//")
$FIMO --thresh 1e-4 --text --bfile --uniform-- $M $W/open.fa 2> $W/fimo_open_$T.err | awk -F'\t' 'NR>1{print $3"\t"$4"\t"$5}' > $W/sites_$T.tsv.new
$FIMO --thresh 1e-4 --text --bfile --uniform-- $M $P 2> $W/fimo_prom_$T.err | awk -F'\t' 'NR>1{print $3"\t"$4"\t"$5}' > $W/psites_$T.tsv.new
echo "$T open old $(wc -l < $W/sites_$T.tsv) new $(wc -l < $W/sites_$T.tsv.new) maxidx $(cut -f1 $W/sites_$T.tsv.new | sort -n | tail -1); prom old $(wc -l < $W/psites_$T.tsv) new $(wc -l < $W/psites_$T.tsv.new)"
mv $W/sites_$T.tsv.new $W/sites_$T.tsv; mv $W/psites_$T.tsv.new $W/psites_$T.tsv
