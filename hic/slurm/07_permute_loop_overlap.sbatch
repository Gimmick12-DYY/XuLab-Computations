#!/bin/bash
#SBATCH --job-name=loop_perm
#SBATCH --output=loop_perm-%A_%a.out
#SBATCH --error=loop_perm-%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --mail-type=FAIL
#
# 100,000 genome-wide shuffles of the locked 5 kb ∪ 10 kb, cutoff 0.7 tests.
# Submit as an 16-task array (8 shards × ±10 kb and ±20 kb), then a merge job:
#
#   jid=$(sbatch --parsable --array=0-15 hic/slurm/07_permute_loop_overlap.sbatch)
#   RUN=merge sbatch --parsable --dependency=afterok:${jid} \
#     hic/slurm/07_permute_loop_overlap.sbatch
#
# Each shard runs 12,500 independent shuffles (seed + shard). Upper-tail
# p = (1 + #{null >= obs}) / (1 + 100000).

set -euo pipefail
source "${XULAB:-/work/users/d/y/dyy12/XuLab}/hic/scripts/lib.sh"

CUTOFF="${CUTOFF:-0.7}"
N_SHARDS="${N_SHARDS:-8}"
PER_SHARD="${PER_SHARD:-12500}"
SEED="${SEED:-1}"
OUT="${WORK}/loops/union_5k_10k_${CUTOFF}"
mkdir -p "${OUT}"

COMMON=(
  python3 "${HIC}/scripts/permute_loop_overlap.py"
  --fine "${WORK}/loops/peakachu_5000.loops.${CUTOFF}.bedpe"
  --coarse "${WORK}/loops/peakachu_10000.loops.${CUTOFF}.bedpe"
  --edges "${XULAB}/cobinding/work/RBBP4.peak_edges.tsv"
  --cliques "${XULAB}/cobinding/results/RBBP4/cliques.tsv"
  --chrom-sizes "${CHROMSIZES}"
  --cutoff "${CUTOFF}"
  --seed "${SEED}"
  --out-dir "${OUT}"
)

if [[ "${RUN:-perm}" == "merge" ]]; then
  for SLOP in 10000 20000; do
    echo "[$(date)] merge slop=${SLOP}"
    "${COMMON[@]}" --merge --n-shards "${N_SHARDS}" --slop "${SLOP}"
  done
  exit 0
fi

TID="${SLURM_ARRAY_TASK_ID:?submit with --array=0-15}"
if (( TID < N_SHARDS )); then
  SLOP=10000
  SHARD=$TID
else
  SLOP=20000
  SHARD=$((TID - N_SHARDS))
fi

echo "[$(date)] shard=${SHARD}/${N_SHARDS} slop=${SLOP} per_shard=${PER_SHARD}"
"${COMMON[@]}" \
  --slop "${SLOP}" \
  --n-perm "${PER_SHARD}" \
  --shard "${SHARD}" \
  --n-shards "${N_SHARDS}"
