#!/bin/bash
#SBATCH -J af3_jie
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda-570.86.15
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/af3_local/logs/jie_%A_%a.out
source /work/users/d/y/dyy12/XuLab/complex_screen/af3_local/af3_common.sh
N=(EZH2_MTF2 EZH2_SUZ12_MTF2); n=${N[$SLURM_ARRAY_TASK_ID]}; echo $n $(hostname)
ENVS=(--env XLA_PYTHON_CLIENT_PREALLOCATE=false --env TF_FORCE_UNIFIED_MEMORY=true --env XLA_CLIENT_MEM_FRACTION=3.2)
run_af3 --json_path=/root/af/jie_in/$n.json --output_dir=/root/af/out --norun_data_pipeline --jax_compilation_cache_dir=/root/af/jaxcache
echo rc $?
