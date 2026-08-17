#!/bin/bash
#SBATCH --job-name=vs_bulk_spearman
#SBATCH --output=vs_bulk_spearman-%j.out
#SBATCH --error=vs_bulk_spearman-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL

# Spearman raw×bulk and imputed×bulk on all 1 kb bins (or FEATURES=ccre|union_peaks|all).
#
# Usage:
#   sbatch downstream/slurm/plot_vs_bulk_spearman.sbatch
#   FEATURES=ccre sbatch downstream/slurm/plot_vs_bulk_spearman.sbatch
#   FEATURES=union_peaks OPEN=1 sbatch downstream/slurm/plot_vs_bulk_spearman.sbatch

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
CONDA_ENV="${CONDA_ENV:-/nas/longleaf/home/dyy12/.conda/envs/cistopic}"
FEATURES="${FEATURES:-all}"
OPEN="${OPEN:-0}"
PY="${CONDA_ENV}/bin/python"

cd "${ROOT_DIR}"
export PATH="/nas/longleaf/apps/samtools/1.21/bin:${PATH:-}"
command -v samtools >/dev/null
command -v bedtools >/dev/null || export PATH="/nas/longleaf/rhel9/apps/bedtools/2.31.1/bedtools2/bin:${PATH}"

extra=()
if [[ "${OPEN}" == "1" || "${OPEN}" == "true" || "${OPEN}" == "yes" ]]; then
  extra+=(--open-chromatin)
fi

echo "[$(date)] starting plot_vs_bulk_spearman_heatmap.py --features ${FEATURES} ${extra[*]:-}"
"${PY}" downstream/plot_vs_bulk_spearman_heatmap.py --features "${FEATURES}" "${extra[@]+"${extra[@]}"}"
echo "[$(date)] done"
