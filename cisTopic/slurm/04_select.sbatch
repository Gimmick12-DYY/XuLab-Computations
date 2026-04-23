#!/bin/bash
#SBATCH --job-name=cistopic_04_select
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail
CFG="${CFG:-configs/default.yaml}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic

python cisTopic/scripts/04_select_model.py --config "$CFG"
