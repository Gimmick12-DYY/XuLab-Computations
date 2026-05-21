#!/bin/bash
#SBATCH --job-name=ds_bin_sens_spec
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00

# Bin-level coverage of positive / negative bin sets (from prepare_pos_neg_bins.R).
# Sparsity-matched by default: the reference input (MATCH_TO, default "raw") is
# scored at THRESHOLD; every other pipeline picks its threshold so the total
# number of "calls" (covered pos+neg bins) equals the reference's. Set
# MATCH_TO="" to apply THRESHOLD uniformly instead.
#
# Imputed pipelines default to each pipeline's native <work>/impute/ (CSR,
# factors, or dense per bin_sensitivity_specificity.py). That keeps top-K
# bin coverage sensitive to imputer-specific rankings. Union CSRs from
# Override INPUTS or *_DIR to point at alternate impute locations if needed.
#
# Optional env vars (defaults shown):
#   ROOT_DIR      repo root (default: auto from submit dir, else /work/users/d/y/dyy12/XuLab)
#   BINS_TSV      output of prepare_pos_neg_bins.R
#                 (default: ${ROOT_DIR}/downstream/bins/pos_neg_bins.tsv)
#   OUT_DIR       output directory
#                 (default: ${ROOT_DIR}/downstream/bin_sensitivity_specificity)
#   INPUTS        space-separated LABEL=PATH pairs; if unset, built from the dirs below
#   MM_DIR        raw matrix dir for the "raw" input
#                 (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm)
#   CISTOPIC_DIR  (default: ${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/impute)
#   FITS_DIR      (default: ${ROOT_DIR}/FITS/work/ctcf/impute)
#   SCOPEN_DIR    (default: ${ROOT_DIR}/scOpen/work/ctcf/impute)
#   MAGIC_DIR     (default: ${ROOT_DIR}/MAGIC/work/ctcf/impute)
#   PUSCOPEN_DIR  (default: ${ROOT_DIR}/PUscOpen/work/ctcf/impute)
#   CICERO_DIR    (default: ${ROOT_DIR}/Cicero/work/ctcf/impute)
#   SCBASSET_DIR  (default: ${ROOT_DIR}/scBasset/work/ctcf/impute)
#   SCBASSET_PUSCOPEN_DIR  (default: ${ROOT_DIR}/scBasset_PUscOpen/work/ctcf/impute)
#   SCBASSET_CISTOPIC_DIR  (default: ${ROOT_DIR}/scBasset_cisTopic/work/ctcf/impute)
#   CONDA_ENV     conda env (default: cistopic; FITS/scopen/magic envs work too)
#   OUT_NAME      basename for output files (default: bin_coverage)
#   THRESHOLD     reference signal threshold (default 0; "any nonzero signal")
#   MATCH_TO      reference input label for sparsity matching (default "raw";
#                 empty string disables matching)
#   MATCH_MULT    multiplier on raw's base_K when MATCH_TO is set (default 1.35
#                 → K ≈ 1.35× #(bins with ref signal>t), capped at #bins).
#                 Set to 1.0 for the old ~9k K; raise further or set TOP_K to fix K.
#   TOP_K         explicit K for every pipeline; overrides MATCH_TO / MATCH_MULT

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
    ROOT_DIR="${SUBMIT_DIR}"
  elif [[ -f "${SUBMIT_DIR}/bin_sensitivity_specificity.py" ]]; then
    ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"

DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

BINS_TSV="${BINS_TSV:-${ROOT_DIR}/downstream/bins/pos_neg_bins.tsv}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/downstream/bin_sensitivity_specificity}"

DOWN_DIR="${ROOT_DIR}/downstream"
# shellcheck source=_pipeline_paths.sh
source "${DOWN_DIR}/slurm/_pipeline_paths.sh"
if [[ -z "${INPUTS:-}" ]]; then
  default_inputs all
fi
pipeline_paths_native

if [[ -z "${BINS_TSV:-}" || -z "${OUT_DIR:-}" || -z "${INPUTS:-}" ]]; then
  echo "Required: BINS_TSV=<tsv>, OUT_DIR=<dir>, INPUTS=\"label1=path1 ...\" (or rely on defaults under ROOT_DIR)" >&2
  exit 1
fi
CONDA_ENV="${CONDA_ENV:-cistopic}"
OUT_NAME="${OUT_NAME:-bin_coverage}"
THRESHOLD="${THRESHOLD:-0.01}"
MATCH_TO="${MATCH_TO-raw}"
MATCH_MULT="${MATCH_MULT:-1.0}"
TOP_K="${TOP_K:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CMD=(
  python "${DOWN_DIR}/bin_sensitivity_specificity.py"
  --bins-tsv         "${BINS_TSV}"
  --out-dir          "${OUT_DIR}"
  --out-name         "${OUT_NAME}"
  --threshold        "${THRESHOLD}"
  --match-multiplier "${MATCH_MULT}"
)
if [[ -n "${TOP_K}" ]]; then
  CMD+=(--top-k "${TOP_K}")
fi
if [[ -z "${MATCH_TO}" ]]; then
  CMD+=(--no-match)
else
  CMD+=(--match-to "${MATCH_TO}")
fi
for pair in ${INPUTS}; do
  CMD+=(--input "${pair}")
done

echo "Running: ${CMD[*]}"
"${CMD[@]}"
