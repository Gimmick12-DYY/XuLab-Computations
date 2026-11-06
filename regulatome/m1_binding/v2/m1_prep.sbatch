#!/bin/bash
#SBATCH -J m1v2_prep
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2/logs/prep_%j.out
set -euo pipefail
M=/work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2
X=/work/users/d/y/dyy12/XuLab
TA=$X/atac_omniatac/caper_work/atac/70bfc7e8-3760-496f-bd8e-5a4bc9e02b51/call-bam2ta
cd $M
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python prep_maps.py $X/regulatome/m0_universe/elements_v2.bed \
  $X/unified/work/ctcf/impute/regions.tsv $M/counts $TA/shard-*/execution/*.tn5.tagAlign.gz
