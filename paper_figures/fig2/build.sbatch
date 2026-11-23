#!/bin/bash
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/logs/%x_%j.log
# Fig 2, full rebuild from committed scripts: source data, then every panel into paper_figures/fig2/output/.
#   r21  corr matrix table + heatmap arrays (V1/V2/V3, 50 bp)      r23  1 kb grid arrays (K_*)
#   r24  Mdrop5-masked imputed arrays (K_impM)                      r27  continuous model scores (K_cont)
#   r25  imputed summit-group 6x6 (Mdrop5)                          r28  CTCF clustered example
#   make_fig2.py  all panels (main, supplement, alternatives); files are overwritten in place
# Inputs from earlier analyses (not rerun here): round2 summit tables, round6 bigWigs, input_audit 10 kb matrix,
# inputs/imp79_mdrop5*.npz (r20a), imputed_audit continuous scores (r9a).
set -euo pipefail
module load bedtools/2.31.1 meme/5.5.7 samtools/1.21 >/dev/null 2>&1 || true
cd /work/users/d/y/dyy12/XuLab
PY=~/.conda/envs/cistopic/bin/python
$PY downstream/fig2_qc/r21_fig2_source.py
$PY downstream/fig2_qc/r23_fig2_1kb_source.py
$PY downstream/fig2_qc/r24_fig2_mask_source.py
$PY downstream/fig2_qc/r27_cont_source.py
$PY downstream/fig2_qc/r25_imputed_corr.py
$PY downstream/fig2_qc/r28_ctcf_clusters.py
$PY paper_figures/fig2/make_fig2.py
(cd paper_figures/fig2/output && md5sum *.png) > results_293T_analysis/fig2_qc/logs/fig2_build_md5_${SLURM_JOB_ID}.txt
