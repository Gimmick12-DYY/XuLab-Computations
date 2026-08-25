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
#   sbatch --array=0-$((N-1)) \
#     --export=ALL,OPEN_CHROMATIN_BED=/path/to/union.bed.gz,RESTORE_PRE_OPENMASK=1 \
#     unified/slurm/02b_apply_open_chromatin_mask.sbatch
#
#   # panel only (colon-separated: Slurm --export splits on commas)
#   TFS="ctcf:maz:zic2:zbtb7a:znf777:nfya:znf282" sbatch --array=0-6 \
#     --export=ALL,TFS,OPEN_CHROMATIN_BED=...,RESTORE_PRE_OPENMASK=1 \
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
  # Prefer colon/space separators: Slurm --export=ALL,TFS=a,b,c splits on commas.
  TFS_NORM="${TFS//:/,}"
  TFS_NORM="${TFS_NORM// /,}"
  IFS=',' read -r -a TF_ARR <<< "${TFS_NORM}"
  _tmp=()
  for t in "${TF_ARR[@]}"; do
    [[ -n "${t}" ]] && _tmp+=("${t}")
  done
  TF_ARR=("${_tmp[@]}")
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

IMPUTE="${WORK}/impute"
PRE="${IMPUTE}/matrix_csr.npz.pre_openmask"
CSR="${IMPUTE}/matrix_csr.npz"
# Re-applying a wider mask on an already-masked CSR cannot restore dropped
# imputed calls. Restore the unmasked backup first when present.
if [[ "${RESTORE_PRE_OPENMASK:-1}" == "1" && -s "${PRE}" ]]; then
  echo "[$(date)] ${TF}: restoring ${PRE} -> ${CSR}"
  cp -f "${PRE}" "${CSR}"
elif [[ ! -s "${CSR}" ]]; then
  echo "missing ${CSR} (and no usable pre_openmask)" >&2
  exit 1
fi

OPEN_BED="${OPEN_CHROMATIN_BED:-}"
echo "[$(date)] TF=${TF} work=${WORK} open_bed=${OPEN_BED:-<config>}"
CMD=("${PY}" scripts/02b_apply_open_chromatin_mask.py --config "${CFG}" --work-dir "${WORK}")
if [[ -n "${OPEN_BED}" ]]; then
  CMD+=(--open-chromatin-bed "${OPEN_BED}")
fi
"${CMD[@]}"
echo "[$(date)] done ${TF}"
