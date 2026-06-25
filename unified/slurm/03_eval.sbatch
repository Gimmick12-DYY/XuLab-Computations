#!/bin/bash
#SBATCH --job-name=unified_eval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

# Step 03 (eval): peak_coverage metrics for the unified imputed matrix.
#
# Runs downstream/peak_coverage.py scoped to the single `unified` pipeline, so
# you get the signal ceilings (Pos.cov / Neg.cov), pos/neg peak coverage, peak
# counts and AUROC right after impute -- no need to run the full-set
# downstream/slurm/peak_coverage.sbatch every iteration.
#
# Needs macs3 + bedtools -> the `cistopic` analysis env (NOT the GPU `unified`
# env; macs3/bedtools/samtools were merged into cistopic from data_prep). This
# is a CPU-only job: 99_full_pipeline.sbatch auto-submits it with an afterok
# dependency (RUN_EVAL=1, the default) so the GPU is freed after impute. Or run
# it standalone any time the impute matrix exists:
#   sbatch unified/slurm/03_eval.sbatch
#
# Knob defaults below mirror downstream/slurm/peak_coverage.sbatch -- keep them
# in sync so the numbers match the full benchmark. Optional env overrides:
#   CFG, CONDA_ENV_EVAL, BINS_TSV, EVAL_OUT_DIR, OUT_NAME,
#   NORMALIZE_SIGNAL, REF_QUANTILE, FRAGS_AT_REF, BG_QUANTILE, BG_SCALE,
#   MAX_FRAGS_PER_BIN, NOLAMBDA, SPREAD_BP, MACS_Q, EXTSIZE, TARGET_FRAGMENTS,
#   Q_SWEEP, BG_SCALE_SWEEP, EXTSIZE_SWEEP

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
XULAB_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CFG="${CFG:-${ROOT_DIR}/configs/default.yaml}"
CONDA_ENV_EVAL="${CONDA_ENV_EVAL:-cistopic}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_EVAL}"

for tool in macs3 bedtools; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "ERROR: ${tool} not on PATH in env ${CONDA_ENV_EVAL}" >&2
    exit 1
  fi
done

read_yaml() {
  python - "$1" <<PY
import sys, yaml
with open("${CFG}") as fh:
    c = yaml.safe_load(fh)
v = c
for k in sys.argv[1].lstrip('.').split('.'):
    v = v[k]
print(v)
PY
}

WORK_DIR="$(read_yaml .paths.work_dir)"
IMPUTE_DIR="${WORK_DIR}/impute"

if [[ ! -f "${IMPUTE_DIR}/matrix_csr.npz" ]]; then
  echo "ERROR: no imputed matrix at ${IMPUTE_DIR}/matrix_csr.npz -- run 02_impute first" >&2
  exit 1
fi

BINS_TSV="${BINS_TSV:-${XULAB_ROOT}/downstream/bins/pos_neg_bins.tsv}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${WORK_DIR}/eval}"
OUT_NAME="${OUT_NAME:-unified_peak_coverage}"

if [[ ! -f "${BINS_TSV}" ]]; then
  echo "ERROR: BINS_TSV not found: ${BINS_TSV}" >&2
  echo "Run: sbatch ${XULAB_ROOT}/downstream/slurm/prepare_pos_neg_bins.sbatch" >&2
  exit 1
fi

# --- peak-calling knobs (mirror downstream/slurm/peak_coverage.sbatch) --------
TARGET_FRAGMENTS="${TARGET_FRAGMENTS:-150000000}"
NORMALIZE_SIGNAL="${NORMALIZE_SIGNAL:-none}"
REF_QUANTILE="${REF_QUANTILE:-0.5}"
FRAGS_AT_REF="${FRAGS_AT_REF:-100}"
BG_QUANTILE="${BG_QUANTILE:-0.5}"
BG_SCALE="${BG_SCALE:-1.0}"
MAX_FRAGS_PER_BIN="${MAX_FRAGS_PER_BIN:-500}"
NOLAMBDA="${NOLAMBDA:-1}"
SPREAD_BP="${SPREAD_BP:-150}"
MACS_Q="${MACS_Q:-0.05}"
EXTSIZE="${EXTSIZE:-200}"
Q_SWEEP="${Q_SWEEP:-}"
BG_SCALE_SWEEP="${BG_SCALE_SWEEP:-}"
EXTSIZE_SWEEP="${EXTSIZE_SWEEP:-}"

mkdir -p "${EVAL_OUT_DIR}"

echo "[$(date)] unified eval: peak_coverage scoped to unified"
echo "  IMPUTE_DIR=${IMPUTE_DIR}"
echo "  BINS_TSV=${BINS_TSV}"
echo "  OUT_DIR=${EVAL_OUT_DIR}"
echo "  NORMALIZE_SIGNAL=${NORMALIZE_SIGNAL} FRAGS_AT_REF=${FRAGS_AT_REF} BG_SCALE=${BG_SCALE} MACS_Q=${MACS_Q} EXTSIZE=${EXTSIZE} Q_SWEEP=${Q_SWEEP:-none} BG_SCALE_SWEEP=${BG_SCALE_SWEEP:-none}"

if [[ "${NOLAMBDA}" == "1" ]]; then NOLAMBDA_FLAG="--nolambda"; else NOLAMBDA_FLAG="--no-nolambda"; fi

CMD=(
  python "${XULAB_ROOT}/downstream/peak_coverage.py"
  --input "unified=${IMPUTE_DIR}"
  --bins-tsv "${BINS_TSV}"
  --out-dir "${EVAL_OUT_DIR}"
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

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "[$(date)] Done. Metrics:"
echo "  ${EVAL_OUT_DIR}/${OUT_NAME}.tsv"
if [[ -n "${Q_SWEEP}${BG_SCALE_SWEEP}${EXTSIZE_SWEEP}" ]]; then
  echo "  ${EVAL_OUT_DIR}/${OUT_NAME}_sweep_table.txt"
else
  echo "  ${EVAL_OUT_DIR}/${OUT_NAME}_table.txt"
fi
