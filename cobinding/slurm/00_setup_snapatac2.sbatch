#!/bin/bash
#SBATCH --job-name=snapatac2_env
#SBATCH --output=snapatac2_env-%j.out
#SBATCH --error=snapatac2_env-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --mail-type=FAIL

set -euo pipefail

ENV="${SNAP_ENV:-/work/users/d/y/dyy12/conda/envs/snapatac2}"
PYTHON="${BASE_PYTHON:-python3}"

if [[ ! -x "${ENV}/bin/python" ]]; then
  "${PYTHON}" -m venv "${ENV}"
fi
"${ENV}/bin/python" -m pip install --upgrade pip
"${ENV}/bin/python" -m pip install "snapatac2==2.9.0" scipy anndata
"${ENV}/bin/python" -c 'import snapatac2 as s; print("SnapATAC2", s.__version__)'
