#!/bin/bash
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J boltz_pilot
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pilot_pairs/logs/boltz_%j.out
# Pilot pairs not yet scored by the screen (screen chains/MSAs; 1 seed, 5 samples, low-memory settings)
C=/work/users/d/y/dyy12/XuLab/complex_screen; P=$C/pilot_pairs
export HF_HOME=$C/cache/hf BOLTZ_CACHE=$C/cache/boltz XDG_CACHE_HOME=$C/cache/xdg TORCH_HOME=$C/cache/torch TRITON_CACHE_DIR=$C/cache/triton CUEQ_CACHE_DIR=$C/cache/cueq
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /work/users/d/y/dyy12/envs/boltz/bin/activate
boltz predict $P/boltz/inputs --out_dir $P/boltz/out --cache $C/cache/boltz --diffusion_samples 5 --seed 1 --output_format mmcif \
  --num_workers 2 --max_parallel_samples 1 > $P/logs/boltz_predict.log 2>&1
echo rc $?
