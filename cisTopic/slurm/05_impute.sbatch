#!/bin/bash
#SBATCH --job-name=cistopic_05_impute
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --time=08:00:00

# Memory-heavy step if impute.mode is "hdf5_all" or "hdf5_variable":
# materialising phi^T @ theta for ~1M regions x 10K cells takes ~40 GB in
# float32, peaking much higher in pycisTopic's internal copies. Adjust --mem
# to your actual surviving region count.

set -euo pipefail
CFG="${CFG:-configs/default.yaml}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic

python cisTopic/scripts/05_impute.py --config "$CFG"
