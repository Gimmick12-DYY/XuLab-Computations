#!/bin/bash
#SBATCH -J abc_k27_fubench
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac/followups/logs/bench_%j.out
# Follow-up dev-TF benchmark (descriptive dev-set tuning, development TFs only) over all link-set variants; reference = list e over old links.
set -euo pipefail
A=/work/users/d/y/dyy12/XuLab/regulatome/m2_abc/results
B=/work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac
F=$B/followups
PY=/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python
$PY $F/bench_multi.py $F/bench \
  old=$A/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv new=$B/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv \
  oldhic=$A/Predictions_hic5kb/EnhancerPredictionsFull_thr0.02.tsv newhic=$F/k27_hic5kb/EnhancerPredictionsFull_thr0.02.tsv \
  kon=$F/k27only/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv \
  oldt01=$A/Predictions_powerlaw/EnhancerPredictionsFull_thr0.01.tsv oldt05=$A/Predictions_powerlaw/EnhancerPredictionsFull_thr0.05.tsv \
  newt01=$B/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.01.tsv newt05=$B/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.05.tsv
$PY $F/bench_summary_multi.py $F/bench oldS_raw@1000 new:old oldhic:old newhic:oldhic newhic:new kon:old kon:new oldt01:old oldt05:old newt01:new newt05:new
