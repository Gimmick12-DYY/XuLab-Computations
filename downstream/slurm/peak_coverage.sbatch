#!/bin/bash
#SBATCH --job-name=ds_peak_cov
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# MACS3 peak calling on pseudo-bulk BEDs from each pipeline matrix, then
# reference pos/neg bin coverage. Requires macs3 (data_prep env: pip-built
# macs3 on Python 3.11; same wheels work on Longleaf).
#
# Default INPUTS: raw + all ten imputed pipelines (same as compare_pos_neg_all10).
# Override INPUTS or individual *_DIR before sbatch.
#
# Optional env vars:
#   ROOT_DIR, CONDA_ENV, MM_DIR, BINS_TSV, PEAK_COV_OUT_DIR,
#   OUT_NAME, TARGET_FRAGMENTS, INPUTS,
#   CISTOPIC_DIR, FITS_DIR, SCOPEN_DIR, MAGIC_DIR, PUSCOPEN_DIR, CICERO_DIR,
#   SCBASSET_DIR, SCBASSET_PUSCOPEN_DIR, SCBASSET_CISTOPIC_DIR,
#   SCBASSET_CISTOPIC_PUSCOPEN_DIR

set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
  if [[ -d "${SUBMIT_DIR}/downstream" ]]; then
    ROOT_DIR="${SUBMIT_DIR}"
  elif [[ -f "${SUBMIT_DIR}/peak_coverage.py" ]]; then
    ROOT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
ROOT_DIR="${ROOT_DIR:-/work/users/d/y/dyy12/XuLab}"

DOWN_DIR="${ROOT_DIR}/downstream"
cd "${ROOT_DIR}"

# shellcheck source=_pipeline_paths.sh
source "${DOWN_DIR}/slurm/_pipeline_paths.sh"

BINS_TSV="${BINS_TSV:-${ROOT_DIR}/downstream/bins/pos_neg_bins.tsv}"
PEAK_COV_OUT_DIR="${PEAK_COV_OUT_DIR:-${ROOT_DIR}/downstream/peak_coverage_ctcf}"
OUT_NAME="${OUT_NAME:-peak_coverage}"
# TARGET_FRAGMENTS is now a CAP on total simulated fragments (scaling is
# per-bin, see REF_QUANTILE/FRAGS_AT_REF), not a normalization target.
TARGET_FRAGMENTS="${TARGET_FRAGMENTS:-150000000}"
# NORMALIZE_SIGNAL: keep at 'none'. 'rank' (percentile-normalise each method's
# per-bin signal) is INCOMPATIBLE with the fold-based caller: it forces every
# method to a fixed 2x median->max range, and collapses tie-heavy (near-binary)
# tracks (raw/scBasset) to a single constant -> every bin's pile-up equals the
# median background bar -> fold 1.0 -> MACS calls 0 peaks (only the genuinely
# continuous 'unified' survived). Cross-method fairness comes from the
# median-anchored per-bin scaling, not from flattening the distribution.
NORMALIZE_SIGNAL="${NORMALIZE_SIGNAL:-none}"
REF_QUANTILE="${REF_QUANTILE:-0.5}"
FRAGS_AT_REF="${FRAGS_AT_REF:-100}"
BG_QUANTILE="${BG_QUANTILE:-0.5}"   # peak-call background bar = this quantile of nonzero signal
BG_SCALE="${BG_SCALE:-1.0}"         # bar multiplier (<1 = more coverage). Sweep e.g. 1.0,0.8,0.6
MAX_FRAGS_PER_BIN="${MAX_FRAGS_PER_BIN:-500}"
NOLAMBDA="${NOLAMBDA:-1}"          # 1 -> --nolambda (default), 0 -> --no-nolambda
SPREAD_BP="${SPREAD_BP:-150}"
MACS_Q="${MACS_Q:-0.05}"
EXTSIZE="${EXTSIZE:-200}"
# Optional sweeps (comma lists). When set, override the single value above and
# the BED is generated once per pipeline then reused across the grid.
# e.g.  Q_SWEEP=0.01,0.05,0.1  BG_SCALE_SWEEP=1.0,0.9,0.8  sbatch ...
Q_SWEEP="${Q_SWEEP:-}"
BG_SCALE_SWEEP="${BG_SCALE_SWEEP:-}"
EXTSIZE_SWEEP="${EXTSIZE_SWEEP:-}"
CONDA_ENV="${CONDA_ENV:-cistopic}"

# Peak-coverage-specific default subset: raw + cisTopic + PUscOpen + scBasset
# and its cascades, omitting FITS / scOpen / MAGIC / cicero to keep MACS3
# runtime manageable. Override by exporting INPUTS=... before sbatch to run
# the full set.
pipeline_paths_native
if [[ -z "${INPUTS:-}" ]]; then
  INPUTS="raw=${MM_DIR} \
cisTopic=${CISTOPIC_DIR} \
PUscOpen=${PUSCOPEN_DIR} \
scbasset=${SCBASSET_DIR} \
scbasset_puscopen=${SCBASSET_PUSCOPEN_DIR} \
scbasset_cistopic=${SCBASSET_CISTOPIC_DIR} \
scbasset_cistopic_puscopen=${SCBASSET_CISTOPIC_PUSCOPEN_DIR} \
unified=${UNIFIED_DIR}"
fi

if [[ ! -f "${BINS_TSV}" ]]; then
  echo "ERROR: BINS_TSV not found: ${BINS_TSV}" >&2
  echo "Run: sbatch downstream/slurm/prepare_pos_neg_bins.sbatch" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if ! command -v macs3 >/dev/null 2>&1; then
  echo "ERROR: macs3 not on PATH in env ${CONDA_ENV}" >&2
  exit 1
fi

mkdir -p "${PEAK_COV_OUT_DIR}"

echo "[$(date)] peak_coverage: NORMALIZE_SIGNAL=${NORMALIZE_SIGNAL} TARGET_FRAGMENTS(cap)=${TARGET_FRAGMENTS} REF_QUANTILE=${REF_QUANTILE} FRAGS_AT_REF=${FRAGS_AT_REF} BG_QUANTILE=${BG_QUANTILE} MAX_FRAGS_PER_BIN=${MAX_FRAGS_PER_BIN} NOLAMBDA=${NOLAMBDA} SPREAD_BP=${SPREAD_BP} MACS_Q=${MACS_Q} EXTSIZE=${EXTSIZE} Q_SWEEP=${Q_SWEEP:-none} BG_SCALE_SWEEP=${BG_SCALE_SWEEP:-none} EXTSIZE_SWEEP=${EXTSIZE_SWEEP:-none}"
echo "BINS_TSV=${BINS_TSV}"
echo "OUT_DIR=${PEAK_COV_OUT_DIR}"
echo "INPUTS=${INPUTS}"

if [[ "${NOLAMBDA}" == "1" ]]; then NOLAMBDA_FLAG="--nolambda"; else NOLAMBDA_FLAG="--no-nolambda"; fi

CMD=(
  python "${DOWN_DIR}/peak_coverage.py"
  --bins-tsv "${BINS_TSV}"
  --out-dir "${PEAK_COV_OUT_DIR}"
  --out-name "${OUT_NAME}"
  --target-fragments "${TARGET_FRAGMENTS}"
  --normalize-signal "${NORMALIZE_SIGNAL}"
  --ref-quantile "${REF_QUANTILE}"
  --frags-at-ref "${FRAGS_AT_REF}"
  --bg-quantile "${BG_QUANTILE}"
  --bg-scale "${BG_SCALE}"
  --max-frags-per-bin "${MAX_FRAGS_PER_BIN}"
  "${NOLAMBDA_FLAG}"
  --spread-bp "${SPREAD_BP}"
  --macs-q "${MACS_Q}"
  --extsize "${EXTSIZE}"
)
[[ -n "${Q_SWEEP}" ]]        && CMD+=(--q-sweep "${Q_SWEEP}")
[[ -n "${BG_SCALE_SWEEP}" ]] && CMD+=(--bg-scale-sweep "${BG_SCALE_SWEEP}")
[[ -n "${EXTSIZE_SWEEP}" ]]  && CMD+=(--extsize-sweep "${EXTSIZE_SWEEP}")
for pair in ${INPUTS}; do
  CMD+=(--input "${pair}")
done
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "[$(date)] Done."
echo "  ${PEAK_COV_OUT_DIR}/${OUT_NAME}.tsv"
if [[ -n "${Q_SWEEP}${BG_SCALE_SWEEP}${EXTSIZE_SWEEP}" ]]; then
  echo "  ${PEAK_COV_OUT_DIR}/${OUT_NAME}_sweep_table.txt"
else
  echo "  ${PEAK_COV_OUT_DIR}/${OUT_NAME}_table.txt"
fi
