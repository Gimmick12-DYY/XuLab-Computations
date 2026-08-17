#!/bin/bash
# Extract ZNF282 matrix, then unified ingest/train/impute (no eval).
# Must sbatch 99_full_pipeline.sbatch FROM unified/ so SLURM_SUBMIT_DIR resolves.
set -euo pipefail
ROOT=/work/users/d/y/dyy12/XuLab
cd "$ROOT"

if [[ ! -f data/ZNF282_bin1000_mtx.rds ]]; then
  EXTRACT_JOB=$(sbatch --parsable --export=ALL,TF=ZNF282,MAKE_UNIFIED_CONFIG=1 \
    data_prep/slurm/extract_tf_matrix.sbatch)
  echo "[submit] extract TF=ZNF282 job=${EXTRACT_JOB}"
  DEP=(--dependency="afterok:${EXTRACT_JOB}")
else
  echo "[submit] reuse existing data/ZNF282_bin1000_mtx.rds"
  DEP=()
fi

cd "${ROOT}/unified"
PIPE_JOB=$(sbatch --parsable "${DEP[@]}" --job-name=unified_znf282 \
  --export=ALL,CFG=${ROOT}/unified/configs/znf282.yaml,RUN_EVAL=0 \
  slurm/99_full_pipeline.sbatch)
echo "[submit] unified pipeline job=${PIPE_JOB} (RUN_EVAL=0)"
echo "Outputs -> ${ROOT}/unified/work/znf282/impute/"
squeue -u "$USER" -j "${PIPE_JOB}" 2>/dev/null || true
