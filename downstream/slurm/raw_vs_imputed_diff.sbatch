#!/bin/bash
#SBATCH --job-name=ds_rvi_diff
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# Raw vs imputed coordinate-level diff: at every raw-nonzero (region, cell)
# pair, look up the imputed value from each pipeline; report lost-positive
# counts at multiple thresholds, per-cell + per-bin loss counts, and a
# context view of imputed values at sampled raw-zero coordinates.
#
# Defaults: each pipeline's native <work>/impute/. UNION_WITH_RAW=0 (strict
# imputed-vs-raw deltas). Set UNION_WITH_RAW=1 to score max(raw, imputed) at raw-positive sites.
#
# Optional env vars (defaults shown):
#   ROOT_DIR       repo root (default: auto from submit dir)
#   MM_DIR         raw matrix dir
#                  (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm)
#   OUT_DIR        output directory
#                  (default: ${ROOT_DIR}/downstream/raw_vs_imputed_diff)
#   INPUTS         space-separated LABEL=PATH pairs; if unset, built from below.
#   CISTOPIC_DIR / FITS_DIR / SCOPEN_DIR / MAGIC_DIR / PUSCOPEN_DIR
#   CONDA_ENV      conda env (default: cistopic)
#   OUT_NAME       basename for output files (default: raw_vs_imputed_diff)
#   THRESHOLDS     comma-separated thresholds (default: 0,0.001,0.01,0.05,0.1,0.5,1.0)
#   N_ZERO_SAMPLE  number of raw=0 coordinates to sample (default: 200000;
#                  pass 0 to disable the imp@raw=0 plot)
#   SEED           RNG seed for the zero sampler (default: 42)
#   UNION_WITH_RAW if 1/true (default), pass --union-with-raw (max(raw, imputed)
#                  at raw-positive sites; writes added-per-cell tables). Set
#                  0/false/no/off for legacy diff-only semantics.

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
    ROOT_DIR="${SUBMIT_DIR}"
  elif [[ -f "${SUBMIT_DIR}/raw_vs_imputed_diff.py" ]]; then
    ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"

DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/downstream/raw_vs_imputed_diff}"
DOWN_DIR="${ROOT_DIR}/downstream"
# shellcheck source=_pipeline_paths.sh
source "${DOWN_DIR}/slurm/_pipeline_paths.sh"
if [[ -z "${INPUTS:-}" ]]; then
  default_inputs imputed
fi
pipeline_paths_native

CONDA_ENV="${CONDA_ENV:-cistopic}"
OUT_NAME="${OUT_NAME:-raw_vs_imputed_diff}"
THRESHOLDS="${THRESHOLDS:-0,0.001,0.01,0.05,0.1,0.5,1.0}"
N_ZERO_SAMPLE="${N_ZERO_SAMPLE:-200000}"
SEED="${SEED:-42}"
UNION_WITH_RAW="${UNION_WITH_RAW:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(
  python "${DOWN_DIR}/raw_vs_imputed_diff.py"
  --mm-dir        "${MM_DIR}"
  --out-dir       "${OUT_DIR}"
  --out-name      "${OUT_NAME}"
  --thresholds    "${THRESHOLDS}"
  --n-zero-sample "${N_ZERO_SAMPLE}"
  --seed          "${SEED}"
)
case "${UNION_WITH_RAW}" in
  0|false|FALSE|no|No|NO|off|OFF) ;;
  *) CMD+=(--union-with-raw) ;;
esac
for pair in ${INPUTS}; do
  CMD+=(--input "${pair}")
done

echo "Running: ${CMD[*]}"
"${CMD[@]}"
