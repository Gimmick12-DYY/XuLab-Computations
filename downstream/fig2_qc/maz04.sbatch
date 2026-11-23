#!/bin/bash
#SBATCH -J maz04_xval
#SBATCH -p general
#SBATCH -c 2
#SBATCH --mem=16g
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_293T_analysis/maz_ref_audit/logs/maz04_%j.out
# MAZ reference audit, step 4 (cross-validation) and step 5 (figures). Run after maz03_download.sbatch.
module load bedtools/2.31.1 meme/5.5.7 >/dev/null 2>&1
cd /work/users/d/y/dyy12/XuLab/downstream/fig2_qc
PY=/nas/longleaf/home/dyy12/.conda/envs/scbasset_tf/bin/python
$PY maz04_crossval.py
