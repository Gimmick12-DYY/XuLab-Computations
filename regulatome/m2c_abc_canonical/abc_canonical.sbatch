#!/bin/bash
#SBATCH -J abc_canon
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m2c_abc_canonical/logs/abc_%j.out
# Canonical ABC (EngreitzLab ABC-Enhancer-Gene-Prediction v1.1.2, commit e2a88d2) through the repo's own Snakefile and rules with config/config.yaml
# here (repo defaults + declared changes, see AUDIT.md). Two biosamples: HEK293T_hicfixed (WT Hi-C 5 kb from inputs/hic5kb_fixed + power-law scaling, primary) and
# HEK293T_powerlaw (no Hi-C, sensitivity arm). TSS file = genes_pc_v46 TSS +/-250 (the repo's TSS500bp convention).
set -euo pipefail
C=/work/users/d/y/dyy12/XuLab/regulatome/m2c_abc_canonical
R=/work/users/d/y/dyy12/XuLab/regulatome/m2_abc/abc_v1.1.2
I=/work/users/d/y/dyy12/XuLab/regulatome/m2_abc/inputs
source /nas/longleaf/rhel9/apps/anaconda/2024.02/etc/profile.d/conda.sh
conda activate /work/users/d/y/dyy12/envs/abc
mkdir -p $C/inputs
# TSS +/-250 bp from the m2_abc protein-coding gene file (strand-aware), same columns as the repo TSS500bp file
awk -v OFS="\t" '{t = ($6 == "+") ? $2 : $3 - 1; s = t - 250; if (s < 0) s = 0; print $1, s, t + 250, $4, $5, $6, $7, $8}' $I/genes_pc_v46.bed > $C/inputs/genes_pc_v46.TSS500bp.bed
cd $C
snakemake -s $R/workflow/Snakefile --configfile $C/config/config.yaml --cores 2 --keep-going --rerun-incomplete --rerun-triggers mtime -p
ls -la $C/results/*/Predictions/
