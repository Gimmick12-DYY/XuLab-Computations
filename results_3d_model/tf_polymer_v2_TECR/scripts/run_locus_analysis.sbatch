#!/bin/bash
#SBATCH -J tfpoly2_locus_ana
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/logs_tfpoly2_loci/%x_%j.log
# v2 validation for one extra hub locus. Usage: sbatch --export=T=<locus folder> run_locus_analysis.sbatch
set -euo pipefail
/work/users/d/y/dyy12/envs/polysim/bin/python ${T}/scripts/03_analyze_v2.py --base ${T}
