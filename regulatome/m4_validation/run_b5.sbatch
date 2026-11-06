#!/bin/bash
#SBATCH -J b5r2b
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m4_validation/logs_b5/b5_%j.out
# Second Perturb-seq comparison (after the first was seen); checks the frozen v5 lists are unchanged before and after.
B=/work/users/d/y/dyy12/XuLab/regulatome/m4_validation; X=/work/users/d/y/dyy12/XuLab
(cd $X && sha256sum -c --quiet regulatome/r2b_residual_v5/manifest_v5.sha256 && echo manifest_v5_ok_before)
mkdir $B/b5_r2b && cp $B/README_b5.md $B/b5_r2b/README.md || exit 1
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python -I $B/b5_r2b.py; echo rc $?
(cd $X && sha256sum -c --quiet regulatome/r2b_residual_v5/manifest_v5.sha256 && echo manifest_v5_ok_after)
