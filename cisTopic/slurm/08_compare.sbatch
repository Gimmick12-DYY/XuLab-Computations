#!/bin/bash
#SBATCH --job-name=cistopic_08_compare
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# Compare imputed P(r|c) HDF5 against the *original* observed matrix.
#
# Usage:
#   sbatch slurm/08_compare.sbatch
#
# Optional env vars:
#   REGIONS_BED=/path/to/peaks.bed   restrict comparison to TF-bin overlaps
#   IMPUTED_H5=/path/to/imputed.h5   override imputed file (defaults to
#                                    <impute>/imputed_Prc_hdf5_all.h5)
#   MM_DIR=/path/to/mm               override original-matrix directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CFG="${CFG:-${ROOT_DIR}/configs/default.yaml}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic

CMD=(
  python scripts/08_compare_imputed.py
  --config "$CFG"
)

if [[ -n "${IMPUTED_H5:-}" ]];   then CMD+=(--imputed-h5  "$IMPUTED_H5");   fi
if [[ -n "${MM_DIR:-}" ]];       then CMD+=(--mm-dir      "$MM_DIR");       fi
if [[ -n "${REGIONS_BED:-}" ]];  then CMD+=(--regions-bed "$REGIONS_BED");  fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
