#!/bin/bash
#SBATCH -J g1_prep
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/gate1/logs/prep_%A_%a.out
set -euo pipefail
G=/work/users/d/y/dyy12/XuLab/regulatome/gate1
X=/work/users/d/y/dyy12/XuLab
TF=${TF:-$(sed -n "${SLURM_ARRAY_TASK_ID}p" $X/regulatome/m1_binding/v2/tfs.txt)}
tf=$(echo $TF | tr A-Z a-z)
[ -s $G/cellmats/$TF.cells.npz ] && { echo "$TF done"; exit 0; }
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python $G/gate1_prep.py $X/regulatome/m0_universe/elements_v2.bed $TF \
  /vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.$TF.bed.gz $X/unified/work/$tf/impute \
  $X/regulatome/m1_binding/v2/counts/element_bin_map.npz $G/cellmats
