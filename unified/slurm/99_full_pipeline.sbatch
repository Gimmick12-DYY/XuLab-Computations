#!/bin/bash
#SBATCH --job-name=unified_ctcf
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

# End-to-end unified model on the CTCF 1 kb bin matrix:
#   export RDS -> mm
#   00_ingest.py   mm + hg38.fa -> seqs.h5 + aligned matrix.npz
#   01_train.py    GPU, focal loss + negative sampling
#   02_impute.py   GPU, sparsity_match threshold -> standard impute schema
#
# Imputed matrix lands at <work>/impute/matrix_csr.npz and plugs into
# downstream/compare_pos_neg.py via:
#   --input unified=<work>/impute
#
# Submit from anywhere under XuLab/unified:
#   sbatch slurm/99_full_pipeline.sbatch
#
# Optional env overrides:
#   CFG=configs/default.yaml
#   CONDA_ENV=unified          Python steps (00-02)
#   CONDA_ENV_R=cistopic       RDS -> mm export (needs R + Matrix)
#   SKIP_EXPORT=1              reuse existing <work>/mm if already exported

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
CONDA_ENV="${CONDA_ENV:-unified}"
CONDA_ENV_R="${CONDA_ENV_R:-cistopic}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
EXPORT_RSCRIPT="${EXPORT_RSCRIPT:-${XULAB_ROOT}/cisTopic/scripts/01_export_rds_to_mm.R}"

read_yaml() {
  python - <<PY
import yaml
with open("${CFG}") as fh:
    c = yaml.safe_load(fh)
v = c
for k in "$1".lstrip('.').split('.'):
    v = v[k]
print(v)
PY
}

INPUT_RDS="$(read_yaml .paths.input_rds)"
WORK_DIR="$(read_yaml  .paths.work_dir)"
MM_DIR="${WORK_DIR}/mm"
IMPUTE_DIR="${WORK_DIR}/impute"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -f "${EXPORT_RSCRIPT}" ]]; then
  echo "ERROR: export script not found: ${EXPORT_RSCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${INPUT_RDS}" ]]; then
  echo "ERROR: input RDS not found: ${INPUT_RDS}" >&2
  exit 1
fi

if [[ "${SKIP_EXPORT}" != "1" ]]; then
  echo "[$(date)] Step export: RDS -> mm"
  conda activate "${CONDA_ENV_R}"
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh
    module unload r 2>/dev/null || true
  fi
  Rscript "${EXPORT_RSCRIPT}" \
    --input  "${INPUT_RDS}" \
    --outdir "${MM_DIR}"
  conda deactivate
else
  echo "[$(date)] Step export: skipped (SKIP_EXPORT=1)"
fi

for f in matrix.mtx.gz regions.tsv.gz barcodes.tsv.gz; do
  if [[ ! -f "${MM_DIR}/${f}" ]]; then
    echo "ERROR: missing ${MM_DIR}/${f} -- run export or unset SKIP_EXPORT" >&2
    exit 1
  fi
done

conda activate "${CONDA_ENV}"
if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
  module load cuda || true
  module load cudnn || true
fi

echo "[$(date)] Step 00: ingest mm + hg38 -> seqs.h5"
python scripts/00_ingest.py --config "${CFG}"

echo "[$(date)] Step 01: train unified model (GPU)"
python -c "import torch; print('Torch', torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
python scripts/01_train.py --config "${CFG}"

echo "[$(date)] Step 02: impute -> CSR"
python scripts/02_impute.py --config "${CFG}"

# Step 03 (eval): peak_coverage metrics. Submitted as a DEPENDENT CPU job
# (afterok) rather than run inline, so this scarce GPU is released the moment
# impute finishes -- macs3/bedtools need the data_prep env, not the GPU
# `unified` env. The eval job reuses whatever knobs are in this job's
# environment (e.g. a BG_SCALE_SWEEP you exported). Set RUN_EVAL=0 to skip.
if [[ "${RUN_EVAL:-1}" == "1" ]]; then
  echo "[$(date)] Step 03: submitting peak_coverage eval (dependent CPU job)"
  if command -v sbatch >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    sbatch --dependency="afterok:${SLURM_JOB_ID}" \
           --export=ALL,CFG="${CFG}" \
           "${ROOT_DIR}/slurm/03_eval.sbatch" \
      && echo "  eval queued (runs on CPU after this job succeeds)" \
      || echo "  WARN: could not submit eval; run 'sbatch slurm/03_eval.sbatch' manually"
  else
    echo "  WARN: not under sbatch; run 'sbatch slurm/03_eval.sbatch' manually after this finishes"
  fi
fi

echo "[$(date)] unified CTCF pipeline completed."
echo "Imputed outputs: ${IMPUTE_DIR}/"
echo "Downstream compare_pos_neg:"
echo "  --input unified=${IMPUTE_DIR}"
echo "Downstream peak_coverage (add to INPUTS):"
echo "  unified=${IMPUTE_DIR}"
echo "BigWig for IGV:"
echo "  INPUT_DIR=${IMPUTE_DIR} OUTPUT_BW=${WORK_DIR}/bigwig/unified_ctcf_mean.bw \\"
echo "    AGGREGATE=mean NORMALIZE=p99 sbatch ${XULAB_ROOT}/downstream/slurm/imputed_to_bigwig.sbatch"
