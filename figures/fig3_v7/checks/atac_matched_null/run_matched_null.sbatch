#!/bin/bash
#SBATCH -J atacnull
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=6G
#SBATCH -t 06:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/figures/fig3_v7/checks/atac_matched_null/logs/%x_%A_%a.log
# 40 TFs x 2 variants x 10 chunks of 1,000 shuffles; task = (variant * 40 + TF index) * 10 + chunk. Writes only under this folder.
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; COB=$X/cobinding; L=$X/hic/work_wt293t/loops; D=$X/figures/fig3_v7/checks/atac_matched_null
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /work/users/d/y/dyy12/conda/envs/cobinding
mapfile -t TFS < $X/figures/fig3_v6/checks/tfs40.txt; VARS=(atac atac_density)
V=${VARS[$((SLURM_ARRAY_TASK_ID / 400))]}; R=$((SLURM_ARRAY_TASK_ID % 400)); TF=${TFS[$((R / 10))]}; K=$((R % 10))
if [[ $TF == RBBP4 ]]; then E=$COB/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv; C=$COB/results/RBBP4_sweep_k50_i10000_s555/cliques.tsv; A=$COB/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_annot.tsv
else E=$COB/work/ourrun_k50/$TF/$TF.peak_edges.tsv; C=$COB/results/ourrun_k50/$TF/cliques.tsv; A=$COB/work/ourrun_k50/$TF/$TF.peak_annot.tsv; fi
[[ -s $D/null/${V}_${TF}_${K}.npz ]] && exit 0
python $D/matched_null.py --tf $TF --variant $V --edges $E --cliques $C --annot $A --fine $L/peakachu_5000.loops.0.7.bedpe --coarse $L/peakachu_10000.loops.0.7.bedpe \
  --n-perm 1000 --chunk $K --out $D/null/${V}_${TF}_${K}.npz
