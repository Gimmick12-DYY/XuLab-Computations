#!/bin/bash
#SBATCH -J hic_hub_test
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/hic_hub_test/%x_%j.log
# Direct Hi-C test of the chr16p11.2 hub (10 kb and 5 kb). Reads xujie_lab peaks read-only; writes only to hic_hub_test/.
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; T=${X}/results_3d_model/hic_hub_test; PY=/work/users/d/y/dyy12/envs/polysim/bin/python
for R in 10000 5000; do
${PY} ${T}/hic_hub_test_chr16.py --mcool ${X}/hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool \
  --peaks-dir /vast/som/xujie_lab/TAGATG-293T/bigwig/peaks --gtf ${X}/data/gencode_v46/gencode.v46.basic.annotation.gtf.gz \
  --hub-edges ${X}/cobinding/results/hub_browser/hub_chr16_edges.tsv --shared-edges ${X}/results_3d_model/inputs/top_shared_clique_edges.tsv \
  --res ${R} --out ${T}
done
