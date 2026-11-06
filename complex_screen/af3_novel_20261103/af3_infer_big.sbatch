#!/bin/bash
#SBATCH -J af3nsrc
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda-570.86.15
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/af3_novel_20261103/logs/infer_big_%A_%a.out
source /work/users/d/y/dyy12/XuLab/complex_screen/af3_novel_20261103/af3_common.sh
n=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $O/pairs.txt); echo $n $(hostname); nvidia-smi -L
ENVS=(--env XLA_PYTHON_CLIENT_PREALLOCATE=false --env TF_FORCE_UNIFIED_MEMORY=true --env XLA_CLIENT_MEM_FRACTION=3.2)
run_af3 --json_path=/root/af/pair_in/$n.json --output_dir=/root/af/out --norun_data_pipeline --jax_compilation_cache_dir=/root/af/jaxcache
echo rc $?; ls $O/out/${n,,}/ | head
