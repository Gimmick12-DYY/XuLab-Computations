#!/bin/bash
# Round 9 chain (SPEC.md): a peaks -> b counts (79-task array) -> c merge + bulk -> d DESeq2 (6 panel TFs) -> e 6x6 -> panels.
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; F=$X/downstream/fig2_qc; L=$X/results_293T_analysis/fig2_qc/logs; PY=~/.conda/envs/cistopic/bin/python
ML="module load bedtools/2.31.1 r/4.5.0 >/dev/null 2>&1 || true"
A=$(sbatch --parsable -J f2r9a -p general --mem=16G -t 01:00:00 -o $L/%x_%j.log --wrap "$ML; cd $F && $PY r9_deseq2_a_peaks.py")
B=$(sbatch --parsable -J f2r9b -p general --mem=24G -t 02:00:00 --array=0-78%25 --dependency=afterok:$A -o $L/%x_%A_%a.log --wrap "cd $F && $PY r9_deseq2_b_counts.py \$SLURM_ARRAY_TASK_ID")
C=$(sbatch --parsable -J f2r9c -p general --mem=32G -t 04:00:00 --dependency=afterok:$B -o $L/%x_%j.log --wrap "$ML; cd $F && $PY r9_deseq2_c_merge.py")
D=""
for t in CTCF MAZ ZBTB7A ZIC2 ZNF282 ZNF777; do
  j=$(sbatch --parsable -J f2r9d_$t -p general --mem=96G -c 2 -t 24:00:00 --dependency=afterok:$C -o $L/%x_%j.log --wrap "$ML; cd $F && Rscript r9_deseq2_d.R $t"); D="$D:$j"
done
E=$(sbatch --parsable -J f2r9e -p general --mem=16G -t 01:00:00 --dependency=afterok$D -o $L/%x_%j.log --wrap "cd $F && $PY r9_deseq2_e_corr.py && cd $X && $PY paper_figures/fig2/make_fig2.py a")
echo "a $A b $B c $C d ${D#:} e $E"
