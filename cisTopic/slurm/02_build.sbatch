#!/bin/bash
#SBATCH --job-name=cistopic_02_build
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail
CFG="${CFG:-configs/default.yaml}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic

python cisTopic/scripts/02_build_cistopic_obj.py --config "$CFG"
