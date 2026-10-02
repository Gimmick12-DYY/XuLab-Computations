#!/bin/bash
#SBATCH --job-name=loop_perm
#SBATCH --output=loop_perm-%j.out
#SBATCH --error=loop_perm-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --mail-type=FAIL
#
# Genome-wide permutation p-values (random chrom + start) for RBBP4 pairs /
# clique edges vs Peakachu 5 kb ∪ 10 kb loops at 0.7.
#
#   sbatch hic/slurm/07_permute_loop_overlap.sbatch
#   N_PERM=5000 sbatch hic/slurm/07_permute_loop_overlap.sbatch

set -euo pipefail
source "${XULAB:-/work/users/d/y/dyy12/XuLab}/hic/scripts/lib.sh"

CUTOFF="${CUTOFF:-0.7}"
SLOP="${SLOP:-10000}"
N_PERM="${N_PERM:-1000}"
SEED="${SEED:-1}"
OUT="${WORK}/loops/union_5k_10k_${CUTOFF}"
mkdir -p "${OUT}"

python3 "${HIC}/scripts/permute_loop_overlap.py" \
  --fine  "${WORK}/loops/peakachu_5000.loops.${CUTOFF}.bedpe" \
  --coarse "${WORK}/loops/peakachu_10000.loops.${CUTOFF}.bedpe" \
  --edges "${XULAB}/cobinding/work/RBBP4.peak_edges.tsv" \
  --cliques "${XULAB}/cobinding/results/RBBP4/cliques.tsv" \
  --chrom-sizes "${CHROMSIZES}" \
  --cutoff "${CUTOFF}" \
  --slop "${SLOP}" \
  --n-perm "${N_PERM}" \
  --seed "${SEED}" \
  --out-dir "${OUT}"
