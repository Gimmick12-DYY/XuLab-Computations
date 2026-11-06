#!/bin/bash
#SBATCH -J model3d_chr16
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/logs/%x_%j.log
# 3D model of chr16p11.2 (steps 1-4). Reads xujie_lab peaks read-only; writes only under results_3d_model.
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab
R=${X}/results_3d_model
PY=/work/users/d/y/dyy12/envs/pastis3d/bin/python
OUT=${OUT:-${R}/out}
N_INIT=${N_INIT:-50}
L=${X}/hic/work_wt293t/loops
${PY} ${R}/scripts/01_prep_locus.py \
  --mcool ${X}/hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool \
  --loops ${L}/peakachu_5000.loops.0.7.bedpe ${L}/peakachu_10000.loops.0.7.bedpe \
  --shared-edges ${R}/inputs/top_shared_clique_edges.tsv \
  --hub-cores ${R}/inputs/anchor_hub_cores_promoter_labels.tsv \
  --peaks-dir /vast/som/xujie_lab/TAGATG-293T/bigwig/peaks \
  --cicero-dir ${X}/cobinding/work/ourrun_k50 \
  --rbbp4-edges ${X}/cobinding/work/RBBP4_labpmat/RBBP4.peak_edges.tsv \
  --out ${OUT}
for Z in include pastis; do
  ${PY} ${R}/scripts/02_pastis_pm2.py --prep ${OUT} --n-init ${N_INIT} --zeros ${Z} > ${OUT}/pastis_run_${Z}.log 2>&1
  grep -E "^fit|^seed|^zeros|^n_|^best|^alpha|^beta|^loglik|^rmsd|^rg|^spearman|Error|Traceback" ${OUT}/pastis_run_${Z}.log | grep -v "^seed [1-9]" 
  ${PY} ${R}/scripts/03_quant_check.py --prep ${OUT} --model ${OUT}/model_${Z}
  ${PY} ${R}/scripts/04_render.py --prep ${OUT} --model ${OUT}/model_${Z} --out ${OUT}/chr16p11.2_3d_model_${Z}
done
