#!/bin/bash
# m7_step2: one-time test scoring after the freeze (Amendment 5.10). usage: sbatch --export=ALL,M7_LINK=<hicfixed|powerlaw> run_score.sbatch <raw|imp>
#   then: sbatch --dependency=afterok:<ids> run_score.sbatch summarise
#SBATCH -J m7score
#SBATCH -p general
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 3:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m7_step2/logs/score_%j.log
set -e
O=/work/users/d/y/dyy12/XuLab/regulatome/m7_step2; PY=/work/users/d/y/dyy12/envs/boltz/bin/python
export PYTHONPATH=$O:/work/users/d/y/dyy12/envs/m7_pydeps OMP_NUM_THREADS=4; cd $O
if [ "$1" = summarise ]; then $PY summarise.py > logs/summarise.out; $PY weights.py; else $PY score.py "$1"; fi
