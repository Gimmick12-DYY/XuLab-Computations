#!/bin/bash
#SBATCH -J sedb_ana
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/se_sedb/logs/%x_%j.log
# Ensemble analysis + direct Hi-C test (chr19-wide nulls) for the exact SE. Writes only under se_sedb/.
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; T=/work/users/d/y/dyy12/XuLab/results_3d_model/se_sedb; PY=/work/users/d/y/dyy12/envs/polysim/bin/python
${PY} ${T}/scripts/04_analyze_se.py --base ${T}/model
${PY} ${T}/scripts/05_hic_test.py --mcool ${X}/hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool --peaks-dir /vast/som/xujie_lab/TAGATG-293T/bigwig/peaks --gtf ${X}/data/gencode_v46/gencode.v46.basic.annotation.gtf.gz --hub-edges ${T}/model/inputs/hub_edges.tsv --shared-edges ${T}/model/inputs/hub_edges_shared.tsv --chrom chr19 --l0 430000 --l1 2250000 --res 10000 --out ${T}/hic_test
