#!/bin/bash
#SBATCH --job-name=cistopic_03_lda
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00

# NOTE: MALLET training on ~1M regions x 10K cells, 500 iterations, K in [10..60]
# is genuinely long. If your scheduler allows it, consider submitting one job
# per K (--n-topics <K>) as a job array instead of training the whole grid
# serially in one job.

set -euo pipefail
CFG="${CFG:-configs/default.yaml}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic

export MALLET_MEMORY="${MALLET_MEMORY:-200G}"

python cisTopic/scripts/03_run_lda_mallet.py --config "$CFG"
