#!/bin/bash
#SBATCH --job-name=tf_agg_pearson
#SBATCH --output=tf_agg_pearson-%A_%a.out
#SBATCH --error=tf_agg_pearson-%A_%a.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-78

# Pearson 10 kb matrices for all unified TFs:
#   HiTAG_aggregate + Imputed_aggregate + Chromnitron (if present) + Bulk (if present)
#
#   cd /work/users/d/y/dyy12/XuLab
#   sbatch downstream/slurm/tf_aggregate_pearson.sbatch
#   # one TF:
#   TF=ctcf sbatch --array=0 downstream/slurm/tf_aggregate_pearson.sbatch
#
# Outputs: downstream/tf_hitag_pearson/all_tfs_aggregate/<TF>_pearson_10kb.{tsv,png}

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-cistopic}"
SCRIPT="${ROOT_DIR}/downstream/tf_hitag_pearson/build_aggregate_matrix.py"
OUT_DIR="${ROOT_DIR}/downstream/tf_hitag_pearson/all_tfs_aggregate"
mkdir -p "${OUT_DIR}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if ! python -c "import pyBigWig, numpy, scipy, matplotlib" 2>/dev/null; then
  echo "ERROR: need pyBigWig/numpy/scipy/matplotlib in ${CONDA_ENV}" >&2
  exit 1
fi

if [[ -n "${TF:-}" ]]; then
  echo "[$(date)] TF=${TF} (override)"
  python "${SCRIPT}" "${TF}"
else
  IDX="${SLURM_ARRAY_TASK_ID:-0}"
  echo "[$(date)] array index=${IDX}"
  python "${SCRIPT}" --index "${IDX}"
fi

echo "[$(date)] done"
