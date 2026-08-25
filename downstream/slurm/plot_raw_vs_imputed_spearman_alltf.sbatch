#!/bin/bash
#SBATCH --job-name=raw_imp_spearman_alltf
#SBATCH --output=raw_imp_spearman_alltf-%j.out
#SBATCH --error=raw_imp_spearman_alltf-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL

# Giant raw × imputed Spearman matrix for every TF with mm + impute.
#
# Usage:
#   sbatch downstream/slurm/plot_raw_vs_imputed_spearman_alltf.sbatch
#   FEATURES=all sbatch downstream/slurm/plot_raw_vs_imputed_spearman_alltf.sbatch

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
CONDA_ENV="${CONDA_ENV:-/nas/longleaf/home/dyy12/.conda/envs/cistopic}"
FEATURES="${FEATURES:-ccre}"
TFS="${TFS:-all}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/downstream/plots/tf_bulk_pearson/all_tfs}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/downstream/plots/tf_bulk_pearson/_cache}"
PY="${CONDA_ENV}/bin/python"

cd "${ROOT_DIR}"
export PATH="/nas/longleaf/rhel9/apps/bedtools/2.31.1/bedtools2/bin:${PATH:-}"
command -v bedtools >/dev/null

echo "[$(date)] correlation_analysis.py raw-vs-imputed --tfs ${TFS} --features ${FEATURES}"
echo "[$(date)] OUT_DIR=${OUT_DIR}"
"${PY}" downstream/correlation_analysis.py raw-vs-imputed \
  --tfs "${TFS}" \
  --features "${FEATURES}" \
  --method spearman \
  --annot no \
  --out-dir "${OUT_DIR}" \
  --cache-dir "${CACHE_DIR}"
echo "[$(date)] done"
