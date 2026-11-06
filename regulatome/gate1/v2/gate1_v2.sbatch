#!/bin/bash
#SBATCH -J g1v2
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/gate1/v2/logs/%x_%A_%a.out
# Gate 1 v2: line N of $JOBS = "<mode> <name> [excluded members]"; per-TF jobs use tfs.txt when JOBS is unset
set -euo pipefail
V=/work/users/d/y/dyy12/XuLab/regulatome/gate1/v2
if [ -n "${JOBS:-}" ]; then read MODE NAME EX < <(sed -n "${SLURM_ARRAY_TASK_ID}p" $JOBS)
else MODE=tf; NAME=${TF:-$(sed -n "${SLURM_ARRAY_TASK_ID}p" /work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2/tfs.txt)}; EX=""; fi
/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python $V/gate1_v2.py $MODE $NAME $EX
