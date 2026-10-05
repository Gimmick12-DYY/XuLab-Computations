#!/bin/bash
#SBATCH --job-name=panel_matrix
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=8:00:00
#SBATCH --mail-type=FAIL
#
# Build the per-bin x per-TF prevalence matrix for the whole panel (once).
# Feeds tf_specific_regions.py, which defines the foreground/background used for
# TF-specific motif discovery.
#
#   sbatch motif_analysis/slurm/build_panel_matrix.sbatch

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
MOTIF_DIR="${MOTIF_DIR:-${ROOT_DIR}/motif_analysis}"
DOWN_DIR="${DOWN_DIR:-${ROOT_DIR}/downstream}"  # shared: call_imputed_peaks, hg38.fa, HOCOMOCOv11
OUT_NPZ="${OUT_NPZ:-${MOTIF_DIR}/motif/panel_prevalence.npz}"

CONDA_ENV="${CONDA_ENV:-data_prep}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "[$(date)] building panel matrix -> ${OUT_NPZ}"
python "${MOTIF_DIR}/scripts/build_panel_matrix.py" \
  --work-root "${ROOT_DIR}/unified/work" \
  --out-npz "${OUT_NPZ}"
echo "[$(date)] done"
