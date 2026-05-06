#!/bin/bash
#SBATCH --job-name=fits_03_phase1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CFG="${CFG:-$ROOT/configs/default.yaml}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic
python scripts/03_run_fits_phase1.py --config "$CFG"
