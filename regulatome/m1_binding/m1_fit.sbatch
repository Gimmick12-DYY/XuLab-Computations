#!/bin/bash
#SBATCH -J m1_fit
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/logs/fit_%j.out
set -euo pipefail
module load bedtools
M=/work/users/d/y/dyy12/XuLab/regulatome/m1_binding
EL=/work/users/d/y/dyy12/XuLab/regulatome/m0_universe/elements_final.bed
PY=/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python
cd $M
$PY fit_binding.py $EL $M/counts $M
$PY sanity.py $EL $M /work/users/d/y/dyy12/XuLab/results_293T_analysis/ctcf_promoter_fp/motif/CTCF_MA0139.2_fimo_p1e-4_hg38.bed
