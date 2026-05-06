#!/bin/bash
#SBATCH --job-name=fits_05_compare
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CFG="${CFG:-$ROOT/configs/default.yaml}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic
python scripts/05_compare_fits.py --config "$CFG"
