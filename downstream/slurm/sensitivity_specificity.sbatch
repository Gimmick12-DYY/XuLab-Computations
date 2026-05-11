#!/bin/bash
#SBATCH --job-name=ds_sens_spec
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# Compute sensitivity / specificity / AUROC / AUPR / max-F1 for each imputation
# pipeline against the raw observed (binarised) matrix.
#
# Required env vars:
#   MM_DIR        directory with matrix.mtx.gz + regions.tsv.gz + barcodes.tsv.gz
#                 (the raw observed matrix used as ground truth)
#   OUT_DIR       output directory for plots / JSON / TSV
#   INPUTS        space-separated LABEL=PATH pairs, e.g.
#                 INPUTS="cisTopic=/work/.../cistopic_ctcf/impute \
#                         FITS=/work/.../FITS/work/ctcf/impute \
#                         scOpen=/work/.../scOpen/work/ctcf/impute \
#                         MAGIC=/work/.../MAGIC/work/ctcf/impute"
# Optional env vars:
#   CONDA_ENV     conda env (default: cistopic; FITS/scopen/magic envs work too)
#   OUT_NAME      basename for output files (default: sensitivity_specificity)
#   THRESHOLDS    comma-separated absolute thresholds (default: auto per-pipeline)
#   N_THRESHOLDS  number of auto thresholds per pipeline (default 48)
#   CHUNK_ROWS    streaming chunk size in rows (default 2000)
#   SEED          RNG seed for the threshold-sample step (default 42)

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
  ROOT_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/sensitivity_specificity.py" ]]; then
  ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

if [[ -z "${MM_DIR:-}" || -z "${OUT_DIR:-}" || -z "${INPUTS:-}" ]]; then
  echo "Required: MM_DIR=<mm/>, OUT_DIR=<dir>, INPUTS=\"label1=path1 label2=path2 ...\"" >&2
  exit 1
fi
CONDA_ENV="${CONDA_ENV:-cistopic}"
OUT_NAME="${OUT_NAME:-sensitivity_specificity}"
N_THRESHOLDS="${N_THRESHOLDS:-48}"
CHUNK_ROWS="${CHUNK_ROWS:-2000}"
SEED="${SEED:-42}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(
  python "${DOWN_DIR}/sensitivity_specificity.py"
  --mm-dir       "${MM_DIR}"
  --out-dir      "${OUT_DIR}"
  --out-name     "${OUT_NAME}"
  --n-thresholds "${N_THRESHOLDS}"
  --chunk-rows   "${CHUNK_ROWS}"
  --seed         "${SEED}"
)
if [[ -n "${THRESHOLDS:-}" ]]; then
  CMD+=(--thresholds "${THRESHOLDS}")
fi
for pair in ${INPUTS}; do
  CMD+=(--input "${pair}")
done

echo "Running: ${CMD[*]}"
"${CMD[@]}"
