#!/bin/bash
#SBATCH -J m32fit
#SBATCH -p general
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --array=0-5
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m32_ml_tf_split/logs/fit_%A_%a.log
export PYTHONPATH=/work/users/d/y/dyy12/envs/m7_pydeps
C=(full:lgb baseline:lgb hitag_only:lgb full:enet baseline:enet hitag_only:enet); x=${C[$SLURM_ARRAY_TASK_ID]}
/work/users/d/y/dyy12/envs/boltz/bin/python /work/users/d/y/dyy12/XuLab/regulatome/m32_ml_tf_split/m32_fit.py ${x%%:*} ${x##*:}
