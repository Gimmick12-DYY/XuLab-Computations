#!/bin/bash
#SBATCH -J dump_reads
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/cobinding/results/hub_browser/%x_%j.log
set -euo pipefail
OUT=/work/users/d/y/dyy12/XuLab/cobinding/results/hub_browser
python3 ${OUT}/dump_reads.py ${OUT}/zoom_reads_chr16_chr17.tsv chr17:17880000-17950000 chr17:18150000-18220000 chr16:29990000-30060000 chr16:30620000-30730000
