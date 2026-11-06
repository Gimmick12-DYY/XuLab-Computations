#!/bin/bash
#SBATCH -J model3d_v2
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/v2/logs/%x_%j.log
# v2 figure of the chr16p11.2 model: PyMOL render + 4-panel figure. No refit; writes only under results_3d_model/v2.
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab
R=${X}/results_3d_model
PY=/work/users/d/y/dyy12/envs/pymol3d/bin/python
STAGE=${STAGE:-all}
${PY} ${R}/v2/scripts/v2_figure.py --base ${R} --stage ${STAGE} \
  --hub-edges ${X}/cobinding/results/hub_browser/hub_chr16_edges.tsv \
  --genes ${X}/cobinding/results/hub_browser/genes_chr16_29.7-31.6.tsv \
  --out ${R}/v2
