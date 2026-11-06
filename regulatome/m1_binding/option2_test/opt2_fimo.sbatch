#!/bin/bash
#SBATCH -J opt2_fimo
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:45:00
#SBATCH --array=1-45%10
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/option2_test/logs/fimo_%A_%a.out
set -euo pipefail
module load meme/5.5.7
O=/work/users/d/y/dyy12/XuLab/regulatome/m1_binding/option2_test
cd $O
TF=$(sed -n "${SLURM_ARRAY_TASK_ID}p" motif_tfs.txt)
mkdir -p fimo
fimo --thresh 1e-4 --text --skip-matched-sequence --bfile elements.bg --motif "$TF" panel_motifs.meme elements.fa \
  | awk -F'\t' 'NR>1{if(!($3 in m)||$8<m[$3])m[$3]=$8}END{for(k in m)print k"\t"m[k]}' > fimo/$TF.best_p.tsv
wc -l fimo/$TF.best_p.tsv
