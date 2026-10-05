#!/bin/bash
#SBATCH --job-name=pmat_panel
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=FAIL
#
# Peak-level occupancy panel from TF1000cells.pmat.mtx.rds (median ~339 bp
# MACS peaks, all 79 TFs). This is the original CUT&Tag resolution -- not 1 kb.
#
#   sbatch motif_analysis/slurm/build_pmat_panel.sbatch

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
MOTIF_DIR="${MOTIF_DIR:-${ROOT_DIR}/motif_analysis}"
DOWN_DIR="${DOWN_DIR:-${ROOT_DIR}/downstream}"  # shared: call_imputed_peaks, hg38.fa, HOCOMOCOv11
PREFIX="${PREFIX:-${MOTIF_DIR}/motif/pmat_panel}"
OUT_NPZ="${OUT_NPZ:-${MOTIF_DIR}/motif/pmat_panel.npz}"

CONDA_ENV="${CONDA_ENV:-data_prep}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PATH="/nas/longleaf/rhel9/apps/r/4.4.0/bin:${PATH}"

echo "[$(date)] build_pmat_panel.R -> ${PREFIX}"
Rscript "${MOTIF_DIR}/scripts/build_pmat_panel.R" \
  --matrix "${ROOT_DIR}/data/TF1000cells.pmat.mtx.rds" \
  --meta "${ROOT_DIR}/data/TF1000cells.meta.csv" \
  --out-prefix "${PREFIX}"

echo "[$(date)] pack npz -> ${OUT_NPZ}"
python "${MOTIF_DIR}/scripts/pack_pmat_panel.py" \
  --prefix "${PREFIX}" --out-npz "${OUT_NPZ}"
echo "[$(date)] done"
