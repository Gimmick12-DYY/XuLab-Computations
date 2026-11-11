#!/bin/bash
# Peak re-calling QC per TF (recall_qc_tf.py), one task per line of recall/tfs.txt.
#SBATCH -J rc_qc
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/recall/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite
python3 $O/recall_qc_tf.py $(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $O/recall/tfs.txt)
