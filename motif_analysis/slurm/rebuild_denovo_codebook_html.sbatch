#!/bin/bash
#SBATCH --job-name=denovo_cb_html
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --mail-type=FAIL
#
# Rematch HOMER de-novo motifs vs Codebook top1 (+ artifact flag) with TOMTOM,
# then rebuild the giant TF × best-match HTML tables for motif_pmat and
# motif_imputed. Runs on a compute node — do not run the rebuild on login.
#
#   sbatch motif_analysis/slurm/rebuild_denovo_codebook_html.sbatch
#
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
MOTIF_DIR="${MOTIF_DIR:-${ROOT_DIR}/motif_analysis}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-data_prep}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

MEME_BIN="${MEME_BIN:-/nas/longleaf/rhel9/apps/meme/5.5.7/bin}"
export PATH="${MEME_BIN}:${PATH}"
command -v tomtom >/dev/null 2>&1 || { echo "ERROR: tomtom not on PATH" >&2; exit 1; }

echo "[$(date)] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-8}"
echo "[$(date)] codebook_top1=$(ls -lh ${MOTIF_DIR}/cache/motifdb/codebook_top1.meme)"
echo "[$(date)] codebook_artifacts=$(ls -lh ${MOTIF_DIR}/cache/motifdb/codebook_artifacts.meme)"

# ThreadPool in the rebuild script uses this many workers
export REBUILD_WORKERS="${REBUILD_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

python "${MOTIF_DIR}/scripts/rebuild_denovo_codebook_html.py"

echo "[$(date)] outputs:"
ls -lh "${MOTIF_DIR}/motif_pmat/homer_denovo_all.html" \
       "${MOTIF_DIR}/motif_imputed/homer_denovo_all.html"
echo "[$(date)] done"
