#!/bin/bash
#SBATCH --job-name=cistopic_01_export
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

# Resolve project root from SLURM_SUBMIT_DIR (BASH_SOURCE points to
# /var/spool/slurmd on compute nodes and won't find scripts/configs).
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

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
module load r/4.4.0 || true

# yq is the most portable way to read a YAML path; falls back to python.
read_yaml() {
  if command -v yq >/dev/null 2>&1; then
    yq -r "$1" "$CFG"
  else
    python - <<PY
import sys, yaml
with open("$CFG") as fh: c = yaml.safe_load(fh)
keys = "$1".lstrip('.').split('.')
v = c
for k in keys: v = v[k]
print(v)
PY
  fi
}

INPUT_RDS=$(read_yaml .paths.input_rds)
WORK_DIR=$(read_yaml  .paths.work_dir)

Rscript scripts/01_export_rds_to_mm.R \
  --input  "$INPUT_RDS" \
  --outdir "$WORK_DIR/mm"
