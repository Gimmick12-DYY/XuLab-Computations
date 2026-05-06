#!/bin/bash
#SBATCH --job-name=fits_full
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CFG="${CFG:-$ROOT/configs/default.yaml}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic
module load r/4.4.0 || true
INPUT_RDS="$(python - <<PY2
import yaml
c=yaml.safe_load(open("$CFG"))
print(c["paths"]["input_rds"])
PY2
)"
WORK_DIR="$(python - <<PY2
import yaml
c=yaml.safe_load(open("$CFG"))
print(c["paths"]["work_dir"])
PY2
)"
Rscript scripts/01_export_rds_to_mm.R --input "$INPUT_RDS" --outdir "$WORK_DIR/mm"
python scripts/02_prepare_fits_csv.py --config "$CFG"
python scripts/03_run_fits_phase1.py --config "$CFG"
python scripts/04_run_fits_phase2.py --config "$CFG"
python scripts/05_compare_fits.py --config "$CFG"
