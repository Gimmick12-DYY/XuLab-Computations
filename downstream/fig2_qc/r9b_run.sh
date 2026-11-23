#!/bin/bash
# Round 9b driver: submits the steps with dependencies. Usage: bash r9b_run.sh
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; D=$X/downstream/fig2_qc; L=$X/results_293T_analysis/fig2_qc/logs; PY=~/.conda/envs/cistopic/bin/python
SB="sbatch --parsable -p general -o $L/%x_%j.log"
fitA=$($SB -J r9b_fitA -c 16 --mem=160G -t 24:00:00 --wrap "module load r/4.5.0; cd $D && Rscript r9b_d.R consensus")
a=$($SB -J r9b_a -c 1 --mem=16G -t 01:00:00 --wrap "module load bedtools/2.31.1; cd $D && $PY r9b_a_regions.py")
b=$($SB -J r9b_b -c 1 --mem=48G -t 04:00:00 --array=0-78%40 -d afterok:$a --wrap "cd $D && $PY r9b_b_counts.py \$SLURM_ARRAY_TASK_ID")
k=$($SB -J r9b_bulk -c 1 --mem=16G -t 08:00:00 --array=0-5 -d afterok:$a --wrap "module load bedtools/2.31.1; cd $D && $PY r9b_c_merge.py \$SLURM_ARRAY_TASK_ID")
c=$($SB -J r9b_c -c 1 --mem=32G -t 01:00:00 -d afterok:$b:$k --wrap "cd $D && $PY r9b_c_merge.py matrix && $PY r9b_c_merge.py combine")
fitB=$($SB -J r9b_fitB -c 16 --mem=128G -t 24:00:00 -d afterok:$c --wrap "module load r/4.5.0; cd $D && Rscript r9b_d.R chip1kb")
e=$($SB -J r9b_e -c 1 --mem=32G -t 01:00:00 -d afterok:$fitA:$fitB --wrap "module load bedtools/2.31.1; cd $D && $PY r9b_e_corr.py && cd $X && $PY paper_figures/fig2/make_fig2.py a")
echo "fitA=$fitA regions=$a counts=$b bulk=$k merge=$c fitB=$fitB corr+render=$e"
