#!/bin/bash
# Round 9: resubmit steps d (DESeq2, 8 cores, 8 h) and e after cancelling the single-core 24 h jobs (user OK 2026-11-23).
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; F=$X/downstream/fig2_qc; L=$X/results_293T_analysis/fig2_qc/logs; PY=~/.conda/envs/cistopic/bin/python
C=${1:?step c job id}
D=""
for t in CTCF MAZ ZBTB7A ZIC2 ZNF282 ZNF777; do
  j=$(sbatch --parsable -J f2r9d_$t -p general --mem=96G -c 8 -t 08:00:00 --dependency=afterok:$C -o $L/%x_%j.log --wrap "module load r/4.5.0 >/dev/null 2>&1 || true; cd $F && Rscript r9_deseq2_d.R $t"); D="$D:$j"
done
E=$(sbatch --parsable -J f2r9e -p general --mem=16G -t 01:00:00 --dependency=afterok$D -o $L/%x_%j.log --wrap "cd $F && $PY r9_deseq2_e_corr.py && cd $X && $PY paper_figures/fig2/make_fig2.py a")
echo "d ${D#:} e $E"
