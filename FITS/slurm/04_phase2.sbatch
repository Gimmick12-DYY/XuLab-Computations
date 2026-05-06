#!/bin/bash
#SBATCH --job-name=fits_04_phase2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CFG="${CFG:-$ROOT/configs/default.yaml}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic
python scripts/04_run_fits_phase2.py --config "$CFG"
