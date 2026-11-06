#!/bin/bash
#SBATCH -J m1v2_count
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 00:45:00
#SBATCH --array=1-79%10
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2/logs/count_%A_%a.out
set -euo pipefail
M=/work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2
X=/work/users/d/y/dyy12/XuLab
TF=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $M/tfs.txt)
tf=$(echo $TF | tr A-Z a-z)
cd $M
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python count_tf.py $X/regulatome/m0_universe/elements_v2.bed $TF \
  /vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.$TF.bed.gz $X/unified/work/$tf/impute $M/counts/element_bin_map.npz $M/counts
