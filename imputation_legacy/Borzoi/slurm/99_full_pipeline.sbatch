#!/bin/bash
#SBATCH --job-name=brz_99_full
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:10:00

# Submit the Borzoi pipeline end-to-end as a chain of dependent jobs.
# Each step's resource requests come from its own .sbatch file. This wrapper
# only owns the dependency wiring.
#
# Usage:
#   sbatch slurm/99_full_pipeline.sbatch
#   CFG=/path/to/big.yaml sbatch slurm/99_full_pipeline.sbatch

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
EXPORT="ALL,CFG=${CFG}"

J1=$(sbatch --parsable --export="${EXPORT}" slurm/01_export.sbatch)
echo "01_export: ${J1}"

J2=$(sbatch --parsable --dependency=afterok:${J1} --export="${EXPORT}" slurm/02_windows.sbatch)
echo "02_windows: ${J2}"

J3=$(sbatch --parsable --dependency=afterok:${J2} --export="${EXPORT}" slurm/03_borzoi.sbatch)
echo "03_borzoi: ${J3}"

J4=$(sbatch --parsable --dependency=afterok:${J3} --export="${EXPORT}" slurm/04_tracks.sbatch)
echo "04_tracks: ${J4}"

J5=$(sbatch --parsable --dependency=afterok:${J4} --export="${EXPORT}" slurm/05_distribute.sbatch)
echo "05_distribute: ${J5}"

J6=$(sbatch --parsable --dependency=afterok:${J5} --export="${EXPORT}" slurm/06_impute.sbatch)
echo "06_impute: ${J6}"

echo "All jobs submitted. Final output will appear under:"
python - <<PY
import yaml
with open("${CFG}") as fh: c = yaml.safe_load(fh)
print(" ", c["paths"]["work_dir"] + "/impute/")
PY
