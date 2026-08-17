#!/bin/bash
#SBATCH --job-name=unified_openmask
#SBATCH --output=unified_openmask-%A_%a.out
#SBATCH --error=unified_openmask-%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL

# Post-hoc open-chromatin mask on existing unified impute CSRs (CPU, no GPU).
#
# Usage:
#   # all TFs with impute/matrix_csr.npz
#   N=$(ls -d /work/users/d/y/dyy12/XuLab/unified/work/*/impute/matrix_csr.npz | wc -l)
#   sbatch --array=0-$((N-1)) unified/slurm/02b_apply_open_chromatin_mask.sbatch
#
#   # panel only
#   TFS="ctcf,maz,zic2,zbtb7a,znf777,nfya,znf282" sbatch --array=0-6 \
#     unified/slurm/02b_apply_open_chromatin_mask.sbatch

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab/unified}"
XULAB="$(cd "${ROOT_DIR}/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-/work/users/d/y/dyy12/conda/envs/unified}"
PY="${CONDA_ENV}/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="/nas/longleaf/home/dyy12/.conda/envs/unified/bin/python"
fi

cd "${ROOT_DIR}"

if [[ -n "${TFS:-}" ]]; then
  IFS=',' read -r -a TF_ARR <<< "${TFS}"
else
  mapfile -t TF_ARR < <(
    for d in "${ROOT_DIR}"/work/*/impute/matrix_csr.npz; do
      [[ -f "$d" ]] || continue
      basename "$(dirname "$(dirname "$d")")"
    done | sort
  )
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#TF_ARR[@]} )); then
  echo "array index ${IDX} out of range (n=${#TF_ARR[@]})" >&2
  exit 1
fi
TF="${TF_ARR[$IDX]}"
CFG="${ROOT_DIR}/configs/${TF}.yaml"
WORK="${ROOT_DIR}/work/${TF}"

if [[ ! -f "${CFG}" ]]; then
  # Fall back to default.yaml + --work-dir
  CFG="${ROOT_DIR}/configs/default.yaml"
fi

echo "[$(date)] TF=${TF} work=${WORK}"
"${PY}" scripts/02b_apply_open_chromatin_mask.py \
  --config "${CFG}" \
  --work-dir "${WORK}"
echo "[$(date)] done ${TF}"
