#!/bin/bash
# m9_abc_diagnosis section B1: FIMO (MEME 5.5.7) of each TF's own HOCOMOCO v11 motif + 20 column-permuted copies over promoters (TSS +/- 500 bp).
# Per motif and gene: max FIMO score and min p among sites with p < 1e-3 (genes without such a site get no row). usage: sbatch --array=0-N m9_fimo.sbatch
#SBATCH -J m9fimo
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 2:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m9_abc_diagnosis/logs/fimo_%A_%a.log
set -e
W=/work/users/d/y/dyy12/XuLab/regulatome/m9_abc_diagnosis/work; mkdir -p $W/fimo
F=$(ls $W/motifs/*.meme | sed -n "$((SLURM_ARRAY_TASK_ID+1))p"); T=$(basename $F .meme)
/nas/longleaf/rhel9/apps/meme/5.5.7/bin/fimo --thresh 1e-3 --text --max-strand --bfile --uniform-- $F $W/promoters.fa 2>/dev/null \
 | awk -F'\t' 'NR>1{k=$1"\t"$3; if(!(k in s) || $7>s[k]) s[k]=$7; if(!(k in p) || $8<p[k]) p[k]=$8} END{for(k in s) print k"\t"s[k]"\t"p[k]}' > $W/fimo/$T.tsv
echo "$T $(wc -l < $W/fimo/$T.tsv)"
