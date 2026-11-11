#!/bin/bash
#SBATCH -J abc_canon_ov
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m2c_abc_canonical/logs/overlap_%x_%j.out
# usage: sbatch overlap.sbatch HEK293T_hic|HEK293T_powerlaw
set -euo pipefail
C=/work/users/d/y/dyy12/XuLab/regulatome/m2c_abc_canonical
PY=/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python
$PY $C/tf_targets_overlap.py $1
$PY $C/pooled.py $1
