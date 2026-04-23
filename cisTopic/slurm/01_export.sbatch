#!/bin/bash
#SBATCH --job-name=cistopic_01_export
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
CFG="${CFG:-configs/default.yaml}"

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

module load R/4.3 || true
Rscript cisTopic/scripts/01_export_rds_to_mm.R \
  --input  "$INPUT_RDS" \
  --outdir "$WORK_DIR/mm"
