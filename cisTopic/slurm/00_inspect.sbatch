#!/bin/bash
#SBATCH --job-name=cistopic_00_inspect
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# Read-only inspection of the .rds before running the pipeline. Safe to run on
# a login node too (single CPU, no writes).

set -euo pipefail
CFG="${CFG:-configs/default.yaml}"

read_yaml() {
  if command -v yq >/dev/null 2>&1; then
    yq -r "$1" "$CFG"
  else
    python - <<PY
import sys, yaml
with open("$CFG") as fh: c = yaml.safe_load(fh)
v = c
for k in "$1".lstrip('.').split('.'): v = v[k]
print(v)
PY
  fi
}

INPUT_RDS=$(read_yaml .paths.input_rds)

module load R/4.3 || true
Rscript cisTopic/scripts/00_inspect_rds.R --input "$INPUT_RDS"
