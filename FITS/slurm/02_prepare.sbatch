#!/bin/bash
#SBATCH --job-name=fits_02_prepare
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CFG="${CFG:-$ROOT/configs/default.yaml}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic
python scripts/02_prepare_fits_csv.py --config "$CFG"
