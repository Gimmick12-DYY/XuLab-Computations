#!/bin/bash
# Round 11 driver (SPEC.md): submits all steps with dependencies. Usage: bash r11_run.sh
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; D=$X/downstream/fig2_qc; L=$X/results_293T_analysis/fig2_qc/logs; PY=~/.conda/envs/cistopic/bin/python
SB="sbatch --parsable -p general -o $L/%x_%j.log"
a=$(sbatch --parsable -J r11_data $D/r11_a_data.sh)
b=$($SB -J r11_regions -c 1 --mem=8G -t 00:30:00 --wrap "module load bedtools/2.31.1; cd $D && $PY r11_b_regions.py")
c=$($SB -J r11_count -c 2 --mem=24G -t 04:00:00 --array=0-11 -d afterok:$a:$b --wrap "module load samtools/1.21 bedtools/2.31.1; cd $D && $PY r11_c_count.py \$SLURM_ARRAY_TASK_ID")
d=$($SB -J r11_qc -c 1 --mem=16G -t 00:30:00 -d afterok:$c --wrap "cd $D && $PY r11_d_qc.py")
e=$($SB -J r11_deseq2 -c 1 --mem=32G -t 04:00:00 -d afterok:$d --wrap "module load r/4.5.0; cd $D && Rscript r11_e_deseq2.R")
f=$($SB -J r11_corr -c 1 --mem=96G -t 06:00:00 -d afterok:$e --wrap "cd $D && $PY r11_f_corr.py && cd $X && $PY paper_figures/fig2/make_fig2.py a g")
echo "data=$a regions=$b counts=$c qc=$d deseq2=$e corr+render=$f"
