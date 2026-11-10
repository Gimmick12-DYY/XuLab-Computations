#!/bin/bash
#SBATCH -J modnull_chk
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/figures/fig3_v6/checks/logs/%x_%A_%a.log
# Per-edge shuffle null for module edges, the 40 TFs with >=100 module edges >25 kb; task = TF index * 10 + chunk (1,000 shuffles each).
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; COB=$X/cobinding; L=$X/hic/work_wt293t/loops; D=$X/figures/fig3_v6/checks
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /work/users/d/y/dyy12/conda/envs/cobinding
mapfile -t TFS < $D/tfs40.txt
T=$((SLURM_ARRAY_TASK_ID / 10)); K=$((SLURM_ARRAY_TASK_ID % 10)); TF=${TFS[$T]}
if [[ $TF == RBBP4 ]]; then E=$COB/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv; C=$COB/results/RBBP4_sweep_k50_i10000_s555/cliques.tsv
else E=$COB/work/ourrun_k50/$TF/$TF.peak_edges.tsv; C=$COB/results/ourrun_k50/$TF/cliques.tsv; fi
[[ -s $D/null_per_edge/${TF}_${K}.npz ]] && exit 0
python $D/module_edge_null_check.py --tf $TF --edges $E --cliques $C --fine $L/peakachu_5000.loops.0.7.bedpe --coarse $L/peakachu_10000.loops.0.7.bedpe \
  --n-perm 1000 --chunk $K --out $D/null_per_edge/${TF}_${K}.npz
