#!/bin/bash
#SBATCH --job-name=vs_bulk_qn_pearson
#SBATCH --output=vs_bulk_qn_pearson-%j.out
#SBATCH --error=vs_bulk_qn_pearson-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --mail-type=FAIL

# Pearson raw×bulk and imputed×bulk after QN (cols then rows by default).
#
# Usage:
#   sbatch downstream/slurm/plot_vs_bulk_qn_pearson.sbatch
#   FEATURES=union_peaks QN=both sbatch downstream/slurm/plot_vs_bulk_qn_pearson.sbatch
#   FEATURES=all QN=cols sbatch downstream/slurm/plot_vs_bulk_qn_pearson.sbatch

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
CONDA_ENV="${CONDA_ENV:-/nas/longleaf/home/dyy12/.conda/envs/cistopic}"
FEATURES="${FEATURES:-union_peaks}"
QN="${QN:-both}"
PY="${CONDA_ENV}/bin/python"

cd "${ROOT_DIR}"

echo "[$(date)] correlation_analysis.py vs-bulk-qn-pearson --features ${FEATURES} --qn ${QN}"
"${PY}" downstream/correlation_analysis.py vs-bulk-qn-pearson --features "${FEATURES}" --qn "${QN}"
echo "[$(date)] done"
