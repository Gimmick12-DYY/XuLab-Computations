#!/bin/bash
#SBATCH --job-name=se_tf_cov
#SBATCH --output=/work/users/d/y/dyy12/XuLab/paper_figures/figure3/source_scripts/se_tf_cov-%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2:00:00
# Pseudobulk TF read coverage for Figure 3g at 1 kb (used) and 2 kb (check against the earlier table).
set -euo pipefail
cd /work/users/d/y/dyy12/XuLab/paper_figures/figure3
PY=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python
$PY source_scripts/se_tf_coverage.py /tmp/se_tf_coverage_2kb_check_$SLURM_JOB_ID.tsv 2000 &
$PY source_scripts/se_tf_coverage.py source_data/se_tf_coverage_1kb.tsv 1000 &
wait
sort source_data/se_tf_coverage_2kb.tsv > /tmp/a_$SLURM_JOB_ID; sort /tmp/se_tf_coverage_2kb_check_$SLURM_JOB_ID.tsv > /tmp/b_$SLURM_JOB_ID
cmp /tmp/a_$SLURM_JOB_ID /tmp/b_$SLURM_JOB_ID && echo "2 kb check: identical to the earlier table" || echo "2 kb check: DIFFERS"
