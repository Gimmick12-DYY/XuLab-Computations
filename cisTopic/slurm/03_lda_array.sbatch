#!/bin/bash
#SBATCH --job-name=cistopic_03_lda_array
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --array=0-5   # edit to match the length of K_LIST below

# One MALLET training per K, parallel across the cluster. Edit K_LIST to the
# same values as lda.n_topics in your config.

set -euo pipefail
CFG="${CFG:-configs/default.yaml}"

K_LIST=(10 20 30 40 50 60)
K=${K_LIST[$SLURM_ARRAY_TASK_ID]}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cistopic
export MALLET_MEMORY="${MALLET_MEMORY:-200G}"

python cisTopic/scripts/03_run_lda_mallet.py --config "$CFG" --n-topics "$K"
