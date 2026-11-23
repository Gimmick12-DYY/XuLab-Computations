#!/bin/bash
#SBATCH -p general
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_293T_analysis/fig2_qc/logs/%x_%j.log
# Fig 2, full rebuild from committed scripts: source data, then every panel into paper_figures/fig2/output/.
#   r21  corr matrix table + heatmap arrays (V1/V2/V3, 50 bp)      r23  1 kb grid arrays (K_*)
#   r24  Mdrop5-masked imputed arrays (K_impM)                      r27  continuous model scores (K_cont)
#   r25  imputed summit-group 6x6 (Mdrop5)                          r28  CTCF clustered example
#   r10  peak-presence heatmap source (4 ChIPs, main + ZNF777 variant)
#   make_fig2.py  all panels (main, supplement, alternatives); files are overwritten in place
#   ROUND9=1 (optional, supplement only; ~2-3 h): round 9 DESeq2 differential-peak 6x6 (r9_deseq2_*, spec
#     downstream/fig2_qc/round9_deseq2/SPEC.md), run sequentially here: peaks, 79 count tasks, our matrix, 6 bulk
#     counts + combine, 6 DESeq2 fits (8 cores), 6x6 -> source_data/corr_deseq2_6x6*.tsv (panel a draws it).
#     Usage: sbatch --export=ALL,ROUND9=1 build.sbatch
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
$PY downstream/fig2_qc/r10_presence.py
if [ "${ROUND9:-0}" = "1" ]; then
  module load r/4.5.0 >/dev/null 2>&1 || true
  cd downstream/fig2_qc
  $PY r9_deseq2_a_peaks.py
  for i in $(seq 0 78); do $PY r9_deseq2_b_counts.py $i; done
  $PY - <<'PYEOF'
# our count matrix only (the bulk part of r9_deseq2_c_merge.py is done per BAM below)
import runpy, sys, types
src = open("r9_deseq2_c_merge.py").read().split("out = subprocess.run")[0]
exec(compile(src, "r9_deseq2_c_merge.py (matrix part)", "exec"))
PYEOF
  for i in 0 1 2 3 4 5; do $PY r9_deseq2_c2_bulk.py $i; done
  $PY r9_deseq2_c2_bulk.py combine
  for t in CTCF MAZ ZBTB7A ZIC2 ZNF282 ZNF777; do Rscript r9_deseq2_d.R $t; done
  $PY r9_deseq2_e_corr.py
  cd ../..
fi
$PY paper_figures/fig2/make_fig2.py
(cd paper_figures/fig2/output && md5sum *.png) > results_293T_analysis/fig2_qc/logs/fig2_build_md5_${SLURM_JOB_ID}.txt
