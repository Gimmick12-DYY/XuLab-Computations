#!/bin/bash
#SBATCH -J abc_k27_hicfix
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac/followups/logs/bench_hicfix_%j.out
# Re-benchmark of the Hi-C arms with real Hi-C contacts (dev-set tuning, development TFs only). The earlier oldhic / newhic arms had an empty Hi-C dump and are invalid.
set -euo pipefail
A=/work/users/d/y/dyy12/XuLab/regulatome/m2_abc/results
B=/work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac
F=$B/followups
PY=/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python
$PY $F/bench_multi.py $F/bench_hicfix \
  old=$A/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv new=$B/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv \
  oldhicfix=$F/oldhicfix/EnhancerPredictionsFull_thr0.02.tsv newhicfix=$F/newhicfix/EnhancerPredictionsFull_thr0.02.tsv
$PY $F/bench_summary_multi.py $F/bench_hicfix oldS_raw@1000 new:old oldhicfix:old newhicfix:oldhicfix newhicfix:new
