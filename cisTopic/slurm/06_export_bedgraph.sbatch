#!/bin/bash
#SBATCH --job-name=cistopic_06_bedgraph
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Pseudobulk the cisTopic imputed CSR into a sorted bedGraph for bigWig
# conversion (browser visualisation).
#
# Optional env vars (defaults shown):
#   IMPUTE_DIR  cisTopic <work>/impute/ directory
#               (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/impute)
#   OUT_DIR     output directory
#               (default: ${IMPUTE_DIR}/../bigwig)
#   OUT_NAME    basename for output files (default: pseudobulk)
#   MODE        sum (default), count, or cell:<barcode>
#   GROUPS      optional TSV (header barcode<TAB>group); emits one bedGraph per group
#   CONDA_ENV   conda env (default: cistopic)

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/scripts" && -d "${SUBMIT_DIR}/configs" ]]; then
  ROOT_DIR="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/../scripts" && -d "${SUBMIT_DIR}/../configs" ]]; then
  ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${ROOT_DIR}"

CFG="${CFG:-${ROOT_DIR}/configs/default.yaml}"
CONDA_ENV="${CONDA_ENV:-cistopic}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

read_yaml() {
  python - <<PY
import yaml
c = yaml.safe_load(open("$CFG")) or {}
v = c
for k in "$1".lstrip('.').split('.'): v = v[k]
print(v)
PY
}

WORK_DIR=$(read_yaml .paths.work_dir)
IMPUTE_DIR="${IMPUTE_DIR:-${WORK_DIR}/impute}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/bigwig}"
OUT_NAME="${OUT_NAME:-pseudobulk}"
MODE="${MODE:-sum}"

CMD=(
  python scripts/06_export_bedgraph.py
  --impute-dir "${IMPUTE_DIR}"
  --out-dir    "${OUT_DIR}"
  --out-name   "${OUT_NAME}"
  --mode       "${MODE}"
)
if [[ -n "${GROUPS:-}" ]]; then
  CMD+=(--groups "${GROUPS}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo
echo "To turn a bedGraph into a bigWig (UCSC tools must be on PATH):"
echo "  bedGraphToBigWig ${OUT_DIR}/${OUT_NAME}.bedgraph hg38.chrom.sizes ${OUT_DIR}/${OUT_NAME}.bw"
