#!/bin/bash
#SBATCH --job-name=cistopic_06_down
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail
CFG="${CFG:-configs/default.yaml}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic

python cisTopic/scripts/06_downstream.py --config "$CFG"
