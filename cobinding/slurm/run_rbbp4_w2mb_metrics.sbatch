#!/bin/bash
#SBATCH --job-name=rbbp4_w2mb_met
#SBATCH --output=/work/users/d/y/dyy12/XuLab/cobinding/logs/rbbp4_w2mb_metrics-%j.out
#SBATCH --error=/work/users/d/y/dyy12/XuLab/cobinding/logs/rbbp4_w2mb_metrics-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#
# Loop metrics and permutation for RBBP4 pairs that were selected both by the
# 1 Mb Our Run and by the 2 Mb second search. Submit with
# --dependency=afterok:<2 Mb cicero job>.

set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
COB="${XULAB}/cobinding"
PY_ENV="/work/users/d/y/dyy12/conda/envs/cobinding"
EDGES="${COB}/work/RBBP4_sweep_k50_i10000_s555_w2mb/RBBP4.peak_edges.both_windows.tsv"
CLIQUES="${COB}/results/RBBP4_sweep_k50_i10000_s555_w2mb/cliques.tsv"
OUT="${COB}/results/RBBP4_sweep_k50_i10000_s555_w2mb/slide"
mkdir -p "${COB}/logs" "${OUT}"

[[ -s "${EDGES}" ]] || { echo "missing ${EDGES}" >&2; exit 1; }
[[ -s "${CLIQUES}" ]] || { echo "missing ${CLIQUES}" >&2; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PY_ENV}"
python "${XULAB}/hic/scripts/fill_slide_loop_table.py" \
  --n-perm 500 --seed 1 \
  --label both_windows --edges "${EDGES}" --cliques "${CLIQUES}" \
  --out-dir "${OUT}"
echo "[$(date)] done"
