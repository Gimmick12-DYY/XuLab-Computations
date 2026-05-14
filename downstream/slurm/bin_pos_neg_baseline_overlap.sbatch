#!/bin/bash
#SBATCH --job-name=ds_bin_posneg_bl
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# Baseline overlap: per-mm-row "any signal" + total stored nnz; bin-rows where
# imputed CSR has signal but raw row was all-zero (see JSON imputed_vs_raw).
# Runs downstream/bin_pos_neg_baseline_overlap.py.
#
# Optional env vars (override before sbatch or use sbatch --export=ALL,...):
#   ROOT_DIR            repo root (default: auto from submit dir)
#   MM_DIR              raw mm (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm)
#   BINS_TSV            pos_neg_bins.tsv (default: ${ROOT_DIR}/downstream/bins/pos_neg_bins.tsv)
#   CISTOPIC_DIR        cisTopic impute/ with matrix_csr.npz
#   PUSCOPEN_DIR        PUscOpen impute/ with matrix_csr.npz
#   OUT_JSON            output JSON path (default: ${ROOT_DIR}/downstream/bins/baseline_pos_neg_overlap.json)
#   CONDA_ENV           default: cistopic
#
# Example:
#   sbatch downstream/slurm/bin_pos_neg_baseline_overlap.sbatch
#   OUT_JSON=/work/.../my_overlap.json sbatch downstream/slurm/bin_pos_neg_baseline_overlap.sbatch

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
    ROOT_DIR="${SUBMIT_DIR}"
  elif [[ -f "${SUBMIT_DIR}/bin_pos_neg_baseline_overlap.py" ]]; then
    ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"

DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

MM_DIR="${MM_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm}"
BINS_TSV="${BINS_TSV:-${ROOT_DIR}/downstream/bins/pos_neg_bins.tsv}"
CISTOPIC_DIR="${CISTOPIC_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/impute}"
PUSCOPEN_DIR="${PUSCOPEN_DIR:-${ROOT_DIR}/PUscOpen/work/ctcf/impute}"
OUT_JSON="${OUT_JSON:-${ROOT_DIR}/downstream/bins/baseline_pos_neg_overlap.json}"
CONDA_ENV="${CONDA_ENV:-cistopic}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python "${DOWN_DIR}/bin_pos_neg_baseline_overlap.py" \
  --mm-dir "${MM_DIR}" \
  --bins-tsv "${BINS_TSV}" \
  --cis-topic-impute "${CISTOPIC_DIR}" \
  --puscopen-impute "${PUSCOPEN_DIR}" \
  --out-json "${OUT_JSON}"

echo "[$(date)] Done. JSON: ${OUT_JSON}"
