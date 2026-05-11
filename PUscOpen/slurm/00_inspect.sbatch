#!/bin/bash
#SBATCH --job-name=pusc_00_inspect
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

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

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
module load r/4.4.0 || true

read_yaml() {
  python - <<PY
import yaml
c=yaml.safe_load(open("$CFG"))
v=c
for k in "$1".lstrip('.').split('.'): v=v[k]
print(v)
PY
}

INPUT_RDS=$(read_yaml .paths.input_rds)
Rscript scripts/00_inspect_rds.R --input "$INPUT_RDS"
