#!/bin/bash
#SBATCH -J sedb_prep
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_3d_model/se_sedb/logs/%x_%j.log
# Model inputs for the exact SEdb SE on chr19. Writes only under se_sedb/.
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; T=/work/users/d/y/dyy12/XuLab/results_3d_model/se_sedb; L=${X}/hic/work_wt293t/loops
/work/users/d/y/dyy12/envs/polysim/bin/python ${T}/scripts/03_prep_model_se.py --edges ${T}/edges_exact_se.tsv --se-region chr19:1248039-1254347 --chrom chr19 --start 430000 --end 2250000 --mcool ${X}/hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool --loops ${L}/peakachu_5000.loops.0.7.bedpe ${L}/peakachu_10000.loops.0.7.bedpe --peaks-dir /vast/som/xujie_lab/TAGATG-293T/bigwig/peaks --out ${T}/model/inputs
