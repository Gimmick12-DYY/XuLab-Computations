#!/bin/bash
#SBATCH -J perm_tf_long
#SBATCH -p general
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 08:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/hic/work_wt293t/loops/ourrun_tf_loops_0.7_pad30kb_perm10000/logs/%x_%A_%a.log
# 10 chunks x 1,000 shuffles per TF; task = TF index * 10 + chunk. Writes only to OUT.
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; COB=$X/cobinding; L=$X/hic/work_wt293t/loops
OUT=$L/ourrun_tf_loops_0.7_pad30kb_perm10000
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate /work/users/d/y/dyy12/conda/envs/cobinding
mapfile -t TFS < $OUT/tfs.txt
T=$((SLURM_ARRAY_TASK_ID / 10)); K=$((SLURM_ARRAY_TASK_ID % 10)); TF=${TFS[$T]}
if [[ $TF == RBBP4 ]]; then E=$COB/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_edges.tsv; C=$COB/results/RBBP4_sweep_k50_i10000_s555/cliques.tsv
else E=$COB/work/ourrun_k50/$TF/$TF.peak_edges.tsv; C=$COB/results/ourrun_k50/$TF/cliques.tsv; fi
[[ -s $OUT/null/${TF}_${K}.npz ]] && exit 0
python $OUT/perm_tf_long.py --tf $TF --edges $E --cliques $C --fine $L/peakachu_5000.loops.0.7.bedpe --coarse $L/peakachu_10000.loops.0.7.bedpe \
  --n-perm 1000 --chunk $K --out $OUT/null/${TF}_${K}.npz
