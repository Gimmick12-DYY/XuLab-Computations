#!/bin/bash
#SBATCH -J m6_matrix
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m6_improve/matrix_v1/logs/%x_%j.out
# DESCRIPTIVE ONLY: full-matrix comparison (SPEC_MATRIX.md). usage: sbatch matrix_compare.sbatch
set -euo pipefail
cd /work/users/d/y/dyy12/XuLab/regulatome/m6_improve/matrix_v1
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python matrix_compare.py
