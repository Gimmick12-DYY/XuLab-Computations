#!/bin/bash
#SBATCH --job-name=tf_rawpeaks
#SBATCH --output=tf_rawpeaks-%A_%a.out
#SBATCH --error=tf_rawpeaks-%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-199%20
#SBATCH --mail-type=FAIL
#
# Step 1: per-TF MACS3 q<0.05 peaks from RAW pseudobulk (unified/work/<tf>/mm).
# Array job -- one TF per task. Sizing: set --array=0-<N-1> for N TFs (extra tasks
# no-op). Peaks land in tf_complex/work/peaks/<tf>.bed.
#
#   sbatch tf_complex/slurm/01_call_raw_peaks.sbatch
#
set -euo pipefail

XULAB="${XULAB:-/work/users/d/y/dyy12/XuLab}"
TFC="${XULAB}/tf_complex"
WORK_ROOT="${WORK_ROOT:-${XULAB}/unified/work}"
PEAKS_DIR="${PEAKS_DIR:-${TFC}/work/peaks}"
MACS_Q="${MACS_Q:-0.05}"
MIN_PEAKS="${MIN_PEAKS:-100}"
CONDA_ENV="${CONDA_ENV:-tf_complex}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
mkdir -p "${PEAKS_DIR}" "${TFC}/work/peaks_tmp"

# discover TFs (dirs with mm/matrix.mtx.gz), pick this task's TF
mapfile -t TFS < <(for d in "${WORK_ROOT}"/*/; do
  [[ -f "${d}mm/matrix.mtx.gz" ]] && basename "${d}"; done | sort)
echo "[info] ${#TFS[@]} TFs available"
i="${SLURM_ARRAY_TASK_ID:-0}"
if (( i >= ${#TFS[@]} )); then echo "[skip] task $i >= ${#TFS[@]} TFs"; exit 0; fi
TF="${TFS[$i]}"
echo "[$(date)] TF=${TF} (task ${i})"

python "${TFC}/scripts/call_raw_peaks.py" \
  --raw-dir "${WORK_ROOT}/${TF}" --tf "${TF}" \
  --out-bed "${PEAKS_DIR}/${TF}.bed" \
  --out-dir "${TFC}/work/peaks_tmp" \
  --macs-q "${MACS_Q}" --min-peaks "${MIN_PEAKS}"
echo "[$(date)] done ${TF}"
