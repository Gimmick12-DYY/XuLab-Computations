#!/bin/bash
# Round 9 execution change (user OK 2026-11-23; no method change): DESeq2 on 8 cores with an 8 h limit, started right away
# (it needs only counts.mtx / samples.tsv from steps b-c); bulk ChIP counted as 6 parallel per-BAM jobs + combine
# (r9_deseq2_c2_bulk.py, submitted separately; pass its combine job id); the 6x6 waits for all of them.
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; F=$X/downstream/fig2_qc; L=$X/results_293T_analysis/fig2_qc/logs; PY=~/.conda/envs/cistopic/bin/python
D=""
for t in CTCF MAZ ZBTB7A ZIC2 ZNF282 ZNF777; do
  j=$(sbatch --parsable -J f2r9d_$t -p general --mem=96G -c 8 -t 08:00:00 -o $L/%x_%j.log --wrap "module load r/4.5.0 >/dev/null 2>&1 || true; cd $F && Rscript r9_deseq2_d.R $t"); D="$D:$j"
done
BC=${1:?bulk combine job id}
E=$(sbatch --parsable -J f2r9e -p general --mem=16G -t 01:00:00 --dependency=afterok$D:$BC -o $L/%x_%j.log --wrap "cd $F && $PY r9_deseq2_e_corr.py && cd $X && $PY paper_figures/fig2/make_fig2.py a")
echo "d ${D#:} e $E"
