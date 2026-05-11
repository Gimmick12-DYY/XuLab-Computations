#!/bin/bash
#SBATCH --job-name=ds_bin_sens_spec
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Bin-level coverage of positive / negative bin sets (from prepare_pos_neg_bins.R).
# Sparsity-matched by default: the reference input (MATCH_TO, default "raw") is
# scored at THRESHOLD; every other pipeline picks its threshold so the total
# number of "calls" (covered pos+neg bins) equals the reference's. Set
# MATCH_TO="" to apply THRESHOLD uniformly instead.
#
# Defaults match compare_pos_neg_all4.sbatch (this XuLab tree). Override any path
# by exporting the variable before sbatch.
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
#   FITS_DIR      (default: ${ROOT_DIR}/FITS/work/ctcf/impute) — disabled below during full FITS rerun
#   SCOPEN_DIR    (default: ${ROOT_DIR}/scOpen/work/ctcf/impute)
#   MAGIC_DIR     (default: ${ROOT_DIR}/MAGIC/work/ctcf/impute)
#   PUSCOPEN_DIR  (default: ${ROOT_DIR}/PUscOpen/work/ctcf/impute)
#   CONDA_ENV     conda env (default: cistopic; FITS/scopen/magic envs work too)
#   OUT_NAME      basename for output files (default: bin_coverage)
#   THRESHOLD     reference signal threshold (default 0; "any nonzero signal")
#   MATCH_TO      reference input label for sparsity matching (default "raw";
#                 empty string disables matching)
#   MATCH_MULT    multiplier on the reference K (default 1.0; e.g. 2 for 2×K)
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

MM_DIR="${MM_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/mm}"
CISTOPIC_DIR="${CISTOPIC_DIR:-${ROOT_DIR}/cisTopic/scripts/cistopic_ctcf/impute}"
# FITS_DIR="${FITS_DIR:-${ROOT_DIR}/FITS/work/ctcf/impute}"  # re-enable after FITS 05_impute export exists
SCOPEN_DIR="${SCOPEN_DIR:-${ROOT_DIR}/scOpen/work/ctcf/impute}"
MAGIC_DIR="${MAGIC_DIR:-${ROOT_DIR}/MAGIC/work/ctcf/impute}"
PUSCOPEN_DIR="${PUSCOPEN_DIR:-${ROOT_DIR}/PUscOpen/work/ctcf/impute}"

if [[ -z "${INPUTS:-}" ]]; then
  # FITS temporarily omitted (full FITS rerun / empty impute/). Restore:
  # INPUTS="raw=${MM_DIR} cisTopic=${CISTOPIC_DIR} FITS=${FITS_DIR} scOpen=${SCOPEN_DIR} MAGIC=${MAGIC_DIR} PUscOpen=${PUSCOPEN_DIR}"
  INPUTS="raw=${MM_DIR} cisTopic=${CISTOPIC_DIR} scOpen=${SCOPEN_DIR} MAGIC=${MAGIC_DIR} PUscOpen=${PUSCOPEN_DIR}"
fi

if [[ -z "${BINS_TSV:-}" || -z "${OUT_DIR:-}" || -z "${INPUTS:-}" ]]; then
  echo "Required: BINS_TSV=<tsv>, OUT_DIR=<dir>, INPUTS=\"label1=path1 ...\" (or rely on defaults under ROOT_DIR)" >&2
  exit 1
fi
CONDA_ENV="${CONDA_ENV:-cistopic}"
OUT_NAME="${OUT_NAME:-bin_coverage}"
THRESHOLD="${THRESHOLD:-0}"
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
