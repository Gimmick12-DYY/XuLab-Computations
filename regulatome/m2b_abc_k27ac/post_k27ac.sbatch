#!/bin/bash
#SBATCH -J abc_k27_post
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac/logs/post_%j.out
# After the m2b ABC run: old-vs-new link comparison, then the dev-TF benchmark and its summary (development TFs only)
set -euo pipefail
B=/work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac
PY=/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python  # env used by the m6_improve scripts
$PY $B/compare_abc.py
$PY $B/bench_k27ac.py
$PY $B/bench_summary.py
