#!/bin/bash
# repro_v1 Run 2c chain (README Amendment 4): sbatch --array=0-78%20 run2c.sbatch peaks ; then run2c.sbatch score (afterok)
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m6_improve/repro_v1/run2c/logs/%x_%A_%a.log
cd /work/users/d/y/dyy12/XuLab/regulatome/m6_improve/repro_v1
PY=/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python
if [ "$1" = peaks ]; then $PY run2c_peaks.py $SLURM_ARRAY_TASK_ID; else $PY run2c_score.py; fi
