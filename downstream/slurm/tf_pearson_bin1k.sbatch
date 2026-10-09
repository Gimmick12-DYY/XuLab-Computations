#!/bin/bash
#SBATCH --job-name=tf_pearson_1kb
#SBATCH --output=/work/users/d/y/dyy12/XuLab/downstream/tf_hitag_pearson/bin1k/logs/%x-%j.out
#SBATCH --error=/work/users/d/y/dyy12/XuLab/downstream/tf_hitag_pearson/bin1k/logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

# Pearson on the native 1 kb unified bins. No 10 kb reduction.
#
#   cd /work/users/d/y/dyy12/XuLab
#   sbatch downstream/slurm/tf_pearson_bin1k.sbatch
#
# Tracks, in order, omitting any that do not exist:
#   Raw      HiTAG aggregate  (mtx2bw/*_HiTAG_unweighted_smooth.bw)
#   Imputed  log1p of the 5-bin smooth of the union-masked impute cell-sum
#   Pred     Chromnitron SA bigWig
#   Bulk     bulk ChIP CPM bigWig
#
# Outputs under downstream/tf_hitag_pearson/bin1k/:
#   <TF>_pearson.{tsv,png}
#   raw_vs_imputed_pearson.{tsv,png}
#   imputed_vs_pred_pearson.{tsv,png}   (TFs with a Chromnitron bigWig only)
#   tfs_without_pred.txt

set -euo pipefail

ROOT_DIR=/work/users/d/y/dyy12/XuLab
cd "${ROOT_DIR}"
mkdir -p downstream/tf_hitag_pearson/bin1k/logs

PY=/nas/longleaf/home/dyy12/.conda/envs/cistopic/bin/python
export N_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

echo "[$(date)] host=$(hostname) cpus=${N_WORKERS} job=${SLURM_JOB_ID:-local}"
"${PY}" downstream/tf_hitag_pearson/build_bin1k_correlations.py
echo "[$(date)] done"
